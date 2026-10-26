"""
Prueba de carga: N peticiones CONCURRENTES contra la API y lectura de las
métricas de esa corrida en Phoenix.

    python scripts/prueba_carga.py --limpiar             # 5 concurrentes (la de la Pre-Entrega 7)
    python scripts/prueba_carga.py --n 20 --limpiar      # alta concurrencia (entrega final)
    python scripts/prueba_carga.py --api http://localhost:8000 --phoenix http://localhost:6006

Entrega final (devolución de la Pre-Entrega 7: "monitorear latencia y costo
bajo alta concurrencia"): con --n mayor que los slots de los workers, los
jobs esperan en la cola. Por eso además de la latencia de punta a punta se
separa la ESPERA EN COLA (latencia - ejecución), se mide la cola máxima, el
throughput, el costo por job (p50/p95) y cuántos jobs corrió cada worker.
Comparar --n 20 con "docker compose up -d --scale worker=1" y con
"--scale worker=3" muestra el efecto del escalado horizontal.

Qué hace:
1. Lanza las 5 consultas a la vez (asyncio.gather) y mide cuánto tarda cada
   POST: tiene que ser de milisegundos, porque la API solo encola.
2. Hace polling de los 5 jobs hasta que terminan y calcula la latencia de
   punta a punta (p50 y p95) desde los tiempos que guarda la API en Redis.
3. Le pide a Phoenix (API GraphQL) el costo de cada ejecución y la latencia
   p95 de las trazas de esa misma ventana de tiempo. Phoenix calcula el
   costo solo, con los tokens de entrada y salida de cada llamada al modelo.
4. Arma el desglose por agente (para la "lectura del dashboard"): cuántas
   llamadas al LLM hizo cada uno, cuántos tokens y cuánto costo consumió, y
   qué parte de su tiempo fue espera del modelo (red/API) y qué parte
   procesamiento propio (herramientas, validación, Redis).
5. Guarda todo en evidencia/prueba_carga_<fecha>.json.

Las capturas para /screenshots se sacan del dashboard de Phoenix (ver README).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from itertools import cycle, islice
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()  # PHOENIX_PROJECT_NAME del .env, el mismo que usa la API

# En Windows, redirigir la salida a un archivo usa cp1252 y no tiene emojis (lección de la Pre-Entrega 6).
sys.stdout.reconfigure(encoding="utf-8")

CONSULTAS = [
    "¿Cuántos CV y cuántos Nm más tiene el Virtus Exclusive que el Tera High, y cuántas unidades de esas "
    "dos versiones hay en total en la sucursal Palermo?",
    "Según Latin NCAP, ¿cuántos puntos porcentuales de diferencia hay entre el Tera y el Nivus en protección "
    "de ocupante adulto?",
    "Según el catálogo de la concesionaria, ¿cuánto torque más tiene el Taos que el Nivus Sense?",
    "¿Qué porcentaje más de potencia tiene el Vento GLI que el Polo Highline?",
    "¿Cuántas unidades de la Saveiro hay en total, sumando todas sus versiones, en la sucursal Córdoba?",
]
FINALES = {"DONE", "FAILED", "WAITING_APPROVAL"}
PROYECTO = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-vw")  # el mismo nombre que usa la API (ver .env)


def percentil(valores: list[float], p: float) -> float:
    """Percentil con interpolación lineal (el método por defecto de numpy)."""
    ordenados = sorted(valores)
    posicion = (len(ordenados) - 1) * p
    abajo = int(posicion)
    arriba = min(abajo + 1, len(ordenados) - 1)
    return ordenados[abajo] + (ordenados[arriba] - ordenados[abajo]) * (posicion - abajo)


async def encolar(cliente: httpx.AsyncClient, tarea: str) -> tuple[str, float]:
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"tarea": tarea})
    respuesta.raise_for_status()
    return respuesta.json()["job_id"], (time.perf_counter() - inicio) * 1000


async def esperar(cliente: httpx.AsyncClient, job_id: str, numero: int, limite_s: float, verbose: bool) -> dict:
    ultimo, inicio = None, time.perf_counter()
    while time.perf_counter() - inicio < limite_s:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] != ultimo:
            ultimo = job["estado"]
            if verbose or ultimo in FINALES:
                print(f"   job {numero} ({job_id[:8]}) -> {ultimo}" + (f" [{job.get('worker')}]" if ultimo in FINALES else ""))
        if ultimo in FINALES:
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"el job {job_id} no terminó en {limite_s} s")


async def consultar_phoenix(url: str, consulta: str, variables: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=15) as cliente:
        r = await cliente.post(f"{url}/graphql", json={"query": consulta, "variables": variables or {}})
        r.raise_for_status()
        cuerpo = r.json()
        if cuerpo.get("errors"):
            raise RuntimeError(cuerpo["errors"])
        return cuerpo["data"]


async def limpiar_proyecto(url: str) -> None:
    """
    Error a evitar de la clase ("Mezclar datos"): sin limpiar el historial, las
    trazas de pruebas anteriores entran en el p95 de esta corrida.
    """
    datos = await consultar_phoenix(url, "query($n: String!) { getProjectByName(name: $n) { id traceCount } }",
                                    {"n": PROYECTO})
    proyecto = datos["getProjectByName"]
    if proyecto is None:
        print(f"   el proyecto '{PROYECTO}' todavía no existe en Phoenix: nada que limpiar")
        return
    await consultar_phoenix(url, "mutation($id: ID!) { clearProject(input: {id: $id}) { __typename } }",
                            {"id": proyecto["id"]})
    print(f"🧹 Proyecto '{PROYECTO}' vaciado en Phoenix ({proyecto['traceCount']} trazas anteriores borradas)")


async def vigilar_cola(cliente: httpx.AsyncClient, muestras: list[int], fin: asyncio.Event) -> None:
    """Muestrea el largo de la cola cada medio segundo: con más jobs que slots, ahí se ve la espera."""
    while not fin.is_set():
        try:
            muestras.append((await cliente.get("/health")).json()["en_cola"])
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)


async def metricas_de_phoenix(url: str, job_ids: list[str], desde: datetime, hasta: datetime) -> dict:
    """
    Costo y latencia de cada job y el p95 de la ventana de la prueba. Cada job de
    la prueba empieza una conversación nueva, así que su sesión de Phoenix
    (session.id = conversacion_id) tiene el mismo ID que el job.

    Ojo: el exportador manda los spans en lotes cada pocos segundos, y el span
    raíz ("orquestador") es el último en cerrarse. Si se leen las métricas
    antes de que llegue, Phoenix calcula la latencia con la traza incompleta
    (pasó en las pruebas: daba 6,2 s para un job de 9,1 s). Por eso se espera
    a que TODAS las trazas tengan su span raíz.
    """
    por_job, completas = {}, False
    for _ in range(20):
        for job_id in job_ids:
            datos = await consultar_phoenix(url, """
                query($id: String!) { getProjectSessionById(sessionId: $id) {
                    numTraces costSummary { total { cost tokens } }
                    traces(first: 10) { edges { node { traceId latencyMs rootSpan { name } } } } } }""", {"id": job_id})
            sesion = datos["getProjectSessionById"]
            if sesion:
                trazas = [e["node"] for e in sesion["traces"]["edges"]]
                por_job[job_id] = {
                    "trazas": sesion["numTraces"],
                    "costo_usd": sesion["costSummary"]["total"]["cost"],
                    "tokens": sesion["costSummary"]["total"]["tokens"],
                    "latencia_ms": [t["latencyMs"] for t in trazas],
                    "trace_ids": [t["traceId"] for t in trazas],
                    "completa": bool(trazas) and all((t["rootSpan"] or {}).get("name") == "orquestador" for t in trazas),
                }
        completas = len(por_job) == len(job_ids) and all(m["completa"] for m in por_job.values())
        if completas:
            break
        await asyncio.sleep(2)
    if not completas:
        print("   ⚠️ Phoenix todavía no recibió todas las trazas completas: las métricas pueden estar incompletas")
    ventana = {"start": desde.isoformat(), "end": hasta.isoformat()}
    proyecto = (await consultar_phoenix(url, """
        query($nombre: String!, $ventana: TimeRange!) { getProjectByName(name: $nombre) {
            traceCount(timeRange: $ventana)
            p50: latencyMsQuantile(probability: 0.5, timeRange: $ventana)
            p95: latencyMsQuantile(probability: 0.95, timeRange: $ventana)
            costSummary(timeRange: $ventana) { total { cost tokens } } } }""",
        {"nombre": PROYECTO, "ventana": ventana}))["getProjectByName"]
    return {"por_job": por_job, "ventana": ventana, "proyecto": proyecto}


async def desglose_por_agente(url: str, trace_ids: list[str]) -> dict:
    """
    Recorre los spans de cada traza y asigna cada llamada al LLM al agente que
    la hizo (el span de tipo AGENT más cercano hacia arriba: supervisor,
    investigador, analista, gestor_reservas, sintesis).
    """
    agentes: dict[str, dict] = {}
    tiempo_trazas = tiempo_llm = 0.0
    for trace_id in trace_ids:
        datos = await consultar_phoenix(url, """
            query($p: String!, $t: ID!) { getProjectByName(name: $p) { trace(traceId: $t) {
                latencyMs spans(first: 1000) { edges { node {
                    name spanKind spanId parentId latencyMs tokenCountTotal costSummary { total { cost } } } } } } } }""",
            {"p": PROYECTO, "t": trace_id})
        traza = datos["getProjectByName"]["trace"]
        if traza is None:
            continue
        spans = {e["node"]["spanId"]: e["node"] for e in traza["spans"]["edges"]}
        tiempo_trazas += traza["latencyMs"]
        for span in spans.values():
            if span["spanKind"] == "agent" and span["parentId"] in spans:  # agentes (no el span raíz)
                fila = agentes.setdefault(span["name"], {"ejecuciones": 0, "llamadas_llm": 0, "tokens": 0,
                                                         "costo_usd": 0.0, "tiempo_ms": 0.0, "tiempo_llm_ms": 0.0})
                fila["ejecuciones"] += 1
                fila["tiempo_ms"] += span["latencyMs"]
            if span["spanKind"] != "llm":
                continue
            tiempo_llm += span["latencyMs"]
            padre = spans.get(span["parentId"])
            while padre is not None and not (padre["spanKind"] == "agent" and padre["parentId"] in spans):
                padre = spans.get(padre["parentId"])
            if padre is None:
                continue
            fila = agentes.setdefault(padre["name"], {"ejecuciones": 0, "llamadas_llm": 0, "tokens": 0,
                                                       "costo_usd": 0.0, "tiempo_ms": 0.0, "tiempo_llm_ms": 0.0})
            fila["llamadas_llm"] += 1
            fila["tokens"] += span["tokenCountTotal"] or 0
            fila["costo_usd"] += (span["costSummary"]["total"]["cost"] or 0) if span["costSummary"] else 0
            fila["tiempo_llm_ms"] += span["latencyMs"]
    return {"agentes": agentes, "tiempo_trazas_ms": tiempo_trazas, "tiempo_llm_ms": tiempo_llm}


def imprimir_desglose(desglose: dict) -> None:
    agentes = desglose["agentes"]
    if not agentes:
        return
    print(f"\n🧩 Desglose por agente (todas las ejecuciones de la prueba)")
    print(f"   {'agente':<22}{'llamadas LLM':>13}{'tokens':>9}{'costo USD':>11}{'tiempo':>9}{'en el LLM':>11}")
    for nombre, f in sorted(agentes.items(), key=lambda x: -x[1]["tokens"]):
        print(f"   {nombre:<22}{f['llamadas_llm']:>13}{f['tokens']:>9}{f['costo_usd']:>11.5f}"
              f"{f['tiempo_ms'] / 1000:>8.1f}s{f['tiempo_llm_ms'] / 1000:>10.1f}s")
    mayor = max(agentes.items(), key=lambda x: x[1]["tokens"])
    total_tokens = sum(f["tokens"] for f in agentes.values()) or 1
    print(f"   -> el que más tokens consume: {mayor[0]} ({mayor[1]['tokens'] / total_tokens:.0%} del total)")
    if desglose["tiempo_trazas_ms"]:
        en_llm = desglose["tiempo_llm_ms"] / desglose["tiempo_trazas_ms"]
        print(f"   -> {en_llm:.0%} del tiempo de las trazas es espera del LLM (red/API); "
              f"{1 - en_llm:.0%} es procesamiento propio (herramientas, validación, Redis, orquestación)")


def resumen_de_concurrencia(jobs: list[dict], total_s: float, muestras_cola: list[int], workers: list[dict]) -> dict:
    """Las métricas que importan bajo carga: espera en cola, throughput, reparto entre workers, errores."""
    terminados = [j for j in jobs if j["estado"] == "DONE"]
    esperas = [j["latencia_total_s"] - j["duracion_ejecucion_s"] for j in terminados]
    ejecuciones = [j["duracion_ejecucion_s"] for j in terminados]
    por_worker = Counter((j.get("worker") or "?").split("/")[0] for j in jobs)
    slots = sum(w["jobs_en_paralelo"] for w in workers)
    r = {
        "jobs": len(jobs), "done": len(terminados), "fallidos": sum(j["estado"] == "FAILED" for j in jobs),
        "procesos_worker": len(workers), "slots_en_paralelo": slots,
        "throughput_jobs_por_min": len(terminados) / total_s * 60 if total_s else 0,
        "cola_maxima": max(muestras_cola, default=0),
        "espera_en_cola_s": {"p50": percentil(esperas, 0.5), "p95": percentil(esperas, 0.95)} if esperas else None,
        "ejecucion_s": {"p50": percentil(ejecuciones, 0.5), "p95": percentil(ejecuciones, 0.95)} if ejecuciones else None,
        "jobs_por_worker": dict(por_worker),
    }
    print(f"\n📈 Concurrencia: {r['jobs']} jobs, {r['procesos_worker']} proceso(s) worker con {slots} slots en total")
    print(f"   throughput: {r['throughput_jobs_por_min']:.1f} jobs/min · cola máxima: {r['cola_maxima']} · "
          f"fallidos: {r['fallidos']}")
    if esperas:
        print(f"   espera en cola: p50 = {r['espera_en_cola_s']['p50']:.1f} s · p95 = {r['espera_en_cola_s']['p95']:.1f} s"
              f"   |   ejecución del grafo: p50 = {r['ejecucion_s']['p50']:.1f} s · p95 = {r['ejecucion_s']['p95']:.1f} s")
    print(f"   jobs por proceso worker: {dict(por_worker)}")
    if r["cola_maxima"] > 0:
        print("   -> hubo más jobs que slots: la latencia extra es espera en cola, no lentitud del grafo. "
              "Se reduce escalando workers (docker compose up -d --scale worker=N).")
    return r


async def main(api: str, phoenix: str, limite_s: float, limpiar: bool, n: int) -> None:
    if limpiar:
        await limpiar_proyecto(phoenix)
    consultas = list(islice(cycle(CONSULTAS), n))
    async with httpx.AsyncClient(base_url=api, timeout=30) as cliente:
        salud = (await cliente.get("/health")).json()
        workers = salud.get("workers_vivos", [])
        print(f"API OK — en cola: {salud['en_cola']}, procesos worker vivos: {len(workers)} "
              f"({sum(w['jobs_en_paralelo'] for w in workers)} slots), workers en la API: {salud['workers_en_proceso']}")

        desde = datetime.now(timezone.utc) - timedelta(seconds=2)
        print(f"\n🚀 Enviando {n} peticiones concurrentes a {api}/tasks")
        inicio = time.perf_counter()
        encolados = await asyncio.gather(*(encolar(cliente, t) for t in consultas))
        print(f"   los {len(encolados)} POST respondieron en {(time.perf_counter() - inicio) * 1000:.0f} ms en total "
              f"(el más lento: {max(ms for _, ms in encolados):.0f} ms): la API encola y no espera al grafo\n")

        job_ids = [job_id for job_id, _ in encolados]
        muestras_cola, fin = [], asyncio.Event()
        vigia = asyncio.create_task(vigilar_cola(cliente, muestras_cola, fin))
        jobs = await asyncio.gather(*(esperar(cliente, j, i + 1, limite_s, verbose=n <= 10)
                                      for i, j in enumerate(job_ids)))
        fin.set()
        await vigia
        hasta = datetime.now(timezone.utc) + timedelta(seconds=2)
        total_s = time.perf_counter() - inicio

    latencias = [j["latencia_total_s"] for j in jobs if j["estado"] == "DONE"]
    print(f"\n{'#':<3}{'job':<10}{'estado':<18}{'latencia':>10}{'ejecución':>11}  respuesta")
    for i, j in enumerate(jobs, 1):
        respuesta = (j.get("resultado") or {}).get("respuesta") or j.get("error") or ""
        print(f"{i:<3}{j['job_id'][:8]:<10}{j['estado']:<18}{j.get('latencia_total_s', 0):>9.1f}s"
              f"{j['duracion_ejecucion_s']:>10.1f}s  {respuesta[:70].replace(chr(10), ' ')}")
    print(f"\nLos {len(jobs)} jobs terminaron en {total_s:.1f} s (en serie hubieran sumado "
          f"{sum(j['duracion_ejecucion_s'] for j in jobs):.1f} s de ejecución)")
    if latencias:
        print(f"Latencia de punta a punta medida por la API: p50 = {percentil(latencias, 0.5):.1f} s · "
              f"p95 = {percentil(latencias, 0.95):.1f} s")
    concurrencia = resumen_de_concurrencia(jobs, total_s, muestras_cola, workers)

    print(f"\n🔭 Métricas de Phoenix ({phoenix}, proyecto '{PROYECTO}')")
    try:
        phoenix_metricas = await metricas_de_phoenix(phoenix, job_ids, desde, hasta)
    except (httpx.HTTPError, RuntimeError) as error:
        phoenix_metricas = {"error": str(error)}
        print(f"   No se pudieron leer las métricas de Phoenix: {error}")
    else:
        for i, job_id in enumerate(job_ids, 1):
            m = phoenix_metricas["por_job"].get(job_id)
            if m and m["costo_usd"] is not None:
                print(f"   job {i} ({job_id[:8]}): costo USD {m['costo_usd']:.6f} · {m['tokens']:.0f} tokens · "
                      f"{m['trazas']} traza(s)")
            else:
                print(f"   job {i} ({job_id[:8]}): todavía sin datos en Phoenix")
        p = phoenix_metricas["proyecto"]
        costos = [m["costo_usd"] for m in phoenix_metricas["por_job"].values() if m["costo_usd"] is not None]
        if costos:
            print(f"   costo por ejecución: promedio USD {sum(costos) / len(costos):.6f} · "
                  f"p50 USD {percentil(costos, 0.5):.6f} · p95 USD {percentil(costos, 0.95):.6f} · "
                  f"total USD {sum(costos):.6f}")
            print(f"   proyección: USD {sum(costos) / len(costos) * 1000:.2f} cada 1.000 consultas")
        if p["p95"] is not None:
            print(f"   latencia de las trazas en la ventana de la prueba: p50 = {p['p50'] / 1000:.1f} s · "
                  f"p95 = {p['p95'] / 1000:.1f} s ({p['traceCount']} trazas)")
        ids = [t for m in phoenix_metricas["por_job"].values() for t in m["trace_ids"]]
        phoenix_metricas["desglose"] = await desglose_por_agente(phoenix, ids)
        imprimir_desglose(phoenix_metricas["desglose"])

    destino = Path("evidencia") / f"prueba_carga_n{n}_{datetime.now():%Y%m%d_%H%M%S}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({
        "api": api, "n": n, "consultas": consultas, "post_ms": [ms for _, ms in encolados],
        "duracion_total_s": total_s, "concurrencia": concurrencia, "workers": workers,
        "latencia_api_s": {"p50": percentil(latencias, 0.5), "p95": percentil(latencias, 0.95)} if latencias else None,
        "jobs": jobs, "phoenix": phoenix_metricas,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 Reporte guardado en {destino}")
    print(f"📸 Capturas: abrí {phoenix} -> proyecto '{PROYECTO}' (ver README, sección 'Capturas')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N peticiones concurrentes + métricas de Phoenix")
    parser.add_argument("--n", type=int, default=5, help="cantidad de peticiones concurrentes (default 5)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--phoenix", default="http://localhost:6006")
    parser.add_argument("--limite", type=float, default=300, help="segundos máximos de espera por job")
    parser.add_argument("--limpiar", action="store_true",
                        help="vacía el proyecto de Phoenix antes de la prueba (así el p95 es solo de esta corrida)")
    argumentos = parser.parse_args()
    asyncio.run(main(argumentos.api, argumentos.phoenix, argumentos.limite, argumentos.limpiar, argumentos.n))
