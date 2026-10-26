"""
Las 5 pruebas del sistema completo, con el LLM real y trazas en Phoenix
(entrega final: "al menos 5 pruebas con trazas en Phoenix y capturas").

    python scripts/pruebas_sistema.py               # las 5 pruebas
    python scripts/pruebas_sistema.py --reinicio    # + la 6: la conversación sigue después de reiniciar todo

Cada prueba manda pedidos reales a la API (que corre con docker compose),
espera el resultado y verifica algo concreto del sistema, no solo que
"respondió". Al final imprime, por prueba, el link a su sesión en Phoenix
(session.id = conversacion_id): ese link es el que se abre para la captura.

| # | Prueba | Qué demuestra |
|---|--------|---------------|
| 1 | RAG híbrido con paráfrasis | la búsqueda vectorial encuentra "qué tan seguro" en la sección Latin NCAP |
| 2 | Supervisor + especialistas | investigador -> analista -> síntesis, con números verificados |
| 3 | Conversación de dos turnos | el turno 2 resuelve "ese" con la memoria del checkpointer |
| 4 | Human-in-the-loop | la reserva espera la aprobación y recién ahí descuenta stock |
| 5 | Puerta de calidad | un cálculo corrompido a propósito lo detecta el validador y se repara |
| 6 | Reinicio (--reinicio) | docker compose restart a mitad de la conversación: el turno 2 la recuerda |

Guarda el resultado en evidencia/pruebas_sistema_<fecha>.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

PROYECTO = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-vw")
FINALES = {"DONE", "FAILED"}


class Prueba:
    def __init__(self, numero: int, nombre: str):
        self.numero, self.nombre = numero, nombre
        self.chequeos: list[tuple[str, bool]] = []
        self.jobs: list[dict] = []
        self.conversacion_id: str | None = None

    def chequear(self, descripcion: str, ok: bool) -> None:
        self.chequeos.append((descripcion, bool(ok)))
        print(f"   {'✅' if ok else '❌'} {descripcion}")

    @property
    def ok(self) -> bool:
        return bool(self.chequeos) and all(ok for _, ok in self.chequeos)


async def pedir(c: httpx.AsyncClient, tarea: str, conversacion_id: str | None = None,
                simulacion: dict | None = None) -> dict:
    cuerpo = {"tarea": tarea, "conversacion_id": conversacion_id, "simulacion": simulacion}
    r = await c.post("/tasks", json={k: v for k, v in cuerpo.items() if v is not None})
    r.raise_for_status()
    return r.json()


async def esperar(c: httpx.AsyncClient, job_id: str, finales: set[str] = FINALES, limite_s: float = 300) -> dict:
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite_s:
        job = (await c.get(f"/tasks/{job_id}")).json()
        if job["estado"] in finales:
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"el job {job_id} no terminó en {limite_s} s")


def mostrar(job: dict) -> None:
    resultado = job.get("resultado") or {}
    print(f"   job {job['job_id'][:8]} -> {job['estado']} en {job.get('latencia_total_s') or 0:.1f} s "
          f"[{job.get('worker')}]")
    if resultado.get("respuesta"):
        print("   respuesta: " + resultado["respuesta"].replace("\n", " ")[:300])
    if job.get("error"):
        print(f"   error: {job['error']}")


def respuesta(job: dict) -> str:
    return (job.get("resultado") or {}).get("respuesta", "")


def sin_escalar(job: dict) -> bool:
    """Una respuesta correcta que igual terminó escalada a revisión humana no es un OK
    (pasó en la primera corrida real de la prueba 3, y el chequeo del 49 no lo detectaba)."""
    resultado = job.get("resultado") or {}
    return job["estado"] == "DONE" and not resultado.get("escalado_humano", True)


def agentes(job: dict) -> list[str]:
    return [d["siguiente"] for d in (job.get("resultado") or {}).get("decisiones", [])]


# --- Las pruebas -------------------------------------------------------------------------------
async def prueba_rag(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(1, "RAG híbrido con paráfrasis")
    creado = await pedir(c, "¿Qué tan seguro es el Tera en un choque? Quiero el puntaje de protección "
                            "de ocupante adulto.")
    job = await esperar(c, creado["job_id"])
    p.jobs.append(job), mostrar(job)
    p.conversacion_id = job["conversacion_id"]
    p.chequear("terminó DONE", job["estado"] == "DONE")
    p.chequear("el dato validado sale de la ficha tera.md", "tera.md" in (job.get("resultado") or {}).get("fuentes", []))
    p.chequear("la respuesta cita Latin NCAP", "ncap" in respuesta(job).lower())
    p.chequear("sin escalado a revisión humana", sin_escalar(job))
    return p


async def prueba_multiagente(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(2, "Supervisor + especialistas")
    creado = await pedir(c, "¿Cuántos CV y cuántos Nm más tiene el Virtus Exclusive que el Tera High?")
    job = await esperar(c, creado["job_id"])
    p.jobs.append(job), mostrar(job)
    p.conversacion_id = job["conversacion_id"]
    p.chequear("terminó DONE", job["estado"] == "DONE")
    p.chequear("intervinieron investigador y analista", {"investigador", "analista"} <= set(agentes(job)))
    p.chequear("la diferencia de potencia es 49 CV (150 - 101)", "49" in respuesta(job))
    p.chequear("sin escalado a revisión humana", sin_escalar(job))
    return p


async def prueba_conversacion(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(3, "Conversación de dos turnos")
    primero = await pedir(c, "¿Cuántos CV y cuántos Nm tiene el Nivus Sense según el catálogo?")
    ocupada = await c.post("/tasks", json={"tarea": "¿Y el Taos?", "conversacion_id": primero["conversacion_id"]})
    job1 = await esperar(c, primero["job_id"])
    p.jobs.append(job1), mostrar(job1)
    segundo = await pedir(c, "¿Y cuántos CV más que ese tiene el Taos?", primero["conversacion_id"])
    job2 = await esperar(c, segundo["job_id"])
    p.jobs.append(job2), mostrar(job2)
    p.conversacion_id = primero["conversacion_id"]
    p.chequear("un segundo turno mientras corre el primero se rechaza (409)", ocupada.status_code == 409)
    p.chequear("los dos turnos terminaron DONE", job1["estado"] == job2["estado"] == "DONE")
    p.chequear("el segundo es el turno 2 de la misma conversación", (job2.get("resultado") or {}).get("turno") == 2)
    p.chequear("resolvió 'ese' = Nivus Sense: 150 - 101 = 49 CV", "49" in respuesta(job2))
    p.chequear("ningún turno terminó escalado a revisión humana", sin_escalar(job1) and sin_escalar(job2))
    conversacion = (await c.get(f"/conversaciones/{p.conversacion_id}")).json()
    p.chequear("GET /conversaciones devuelve los 2 turnos", len(conversacion["turnos"]) == 2)
    return p


async def reponer_stock_de_la_demo() -> bool:
    """
    El stock de la concesionaria es ficticio, pero el control es real: Redis lleva la cuenta de
    las unidades reservadas. La Saveiro Extreme tiene 2 en Córdoba, así que la tercera corrida
    de esta prueba fallaba con "no quedan unidades" (pasó en la corrida real: el control de stock
    funcionó, la prueba no era repetible). Antes de la prueba se borran las reservas de la demo.
    """
    proceso = await asyncio.create_subprocess_exec(
        "docker", "compose", "exec", "-T", "redis", "sh", "-c",
        "redis-cli --scan --pattern 'vw:reserva*' | xargs -r redis-cli del",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, error = await proceso.communicate()
    if proceso.returncode != 0:
        print(f"   ⚠️ no se pudo reponer el stock de la demo: {error.decode(errors='replace')[:200]}")
    return proceso.returncode == 0


async def prueba_hitl(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(4, "Human-in-the-loop")
    if await reponer_stock_de_la_demo():
        print("   stock ficticio de la demo repuesto (se borraron las reservas anteriores)")
    creado = await pedir(c, "Reservá una Saveiro Extreme cabina doble en la sucursal Córdoba para el cliente "
                            "Juan Pérez.")
    pausado = await esperar(c, creado["job_id"], FINALES | {"WAITING_APPROVAL"})
    p.conversacion_id = pausado["conversacion_id"]
    p.chequear("se detuvo en WAITING_APPROVAL antes de reservar", pausado["estado"] == "WAITING_APPROVAL")
    if pausado["estado"] == "WAITING_APPROVAL":
        print(f"   pedido de aprobación: {pausado['aprobacion']['pregunta']}")
        r = await c.post(f"/tasks/{creado['job_id']}/approve",
                         json={"aprobado": True, "revisor": "Damián", "comentario": "prueba del sistema"})
        p.chequear("POST /approve aceptado (202)", r.status_code == 202)
    job = await esperar(c, creado["job_id"])
    p.jobs.append(job), mostrar(job)
    reserva = (job.get("resultado") or {}).get("reserva") or {}
    p.chequear("la reserva se ejecutó después de la aprobación", job["estado"] == "DONE" and reserva.get("ok"))
    return p


async def prueba_calidad(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(5, "Puerta de calidad")
    creado = await pedir(c, "¿Cuántos CV más tiene el Virtus Exclusive que el Tera High?",
                         simulacion={"corromper_analisis": 1})
    job = await esperar(c, creado["job_id"])
    p.jobs.append(job), mostrar(job)
    p.conversacion_id = job["conversacion_id"]
    resultado = job.get("resultado") or {}
    p.chequear("el validador rechazó el cálculo corrompido", resultado.get("validaciones_fallidas", 0) >= 1)
    p.chequear("el analista volvió a intervenir (bucle de reparación)", agentes(job).count("analista") >= 2)
    p.chequear("la respuesta final tiene el valor correcto (49 CV)", job["estado"] == "DONE" and "49" in respuesta(job))
    p.chequear("se reparó sin escalar a revisión humana", sin_escalar(job))
    return p


async def arranque_de_workers(c: httpx.AsyncClient) -> set[str]:
    """Cuándo arrancó cada proceso worker. Con docker compose restart el contenedor conserva su
    nombre (hostname-pid), así que lo que prueba que el proceso es nuevo es este momento."""
    return {w["desde"] for w in (await c.get("/health")).json()["workers_vivos"]}


async def reiniciar_servicios(c: httpx.AsyncClient) -> bool:
    print("   docker compose restart api worker ...")
    proceso = await asyncio.create_subprocess_exec("docker", "compose", "restart", "api", "worker",
                                                   stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, error = await proceso.communicate()
    if proceso.returncode != 0:
        print(f"   no se pudo reiniciar: {error.decode(errors='replace')[:200]}")
        return False
    for _ in range(60):
        try:
            if (await c.get("/health")).json()["workers_vivos"]:
                return True
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        await asyncio.sleep(2)
    return False


async def prueba_reinicio(c: httpx.AsyncClient) -> Prueba:
    p = Prueba(6, "La conversación sobrevive a un reinicio")
    primero = await pedir(c, "¿Cuántas unidades del Taos 1.4 TSI hay en la sucursal Palermo?")
    job1 = await esperar(c, primero["job_id"])
    p.jobs.append(job1), mostrar(job1)
    antes = await arranque_de_workers(c)
    p.chequear("la API y los workers volvieron después del reinicio", await reiniciar_servicios(c))
    despues = await arranque_de_workers(c)
    print(f"   workers arrancados: antes {sorted(antes)} · después {sorted(despues)}")
    p.chequear("los workers son procesos nuevos (arrancaron después del reinicio)", bool(despues) and not despues & antes)
    segundo = await pedir(c, "¿Y en la sucursal Córdoba?", primero["conversacion_id"])
    job2 = await esperar(c, segundo["job_id"])
    p.jobs.append(job2), mostrar(job2)
    p.conversacion_id = primero["conversacion_id"]
    p.chequear("el turno 2 corrió en un proceso nuevo y es el turno 2 de la conversación",
               job2["estado"] == "DONE" and (job2.get("resultado") or {}).get("turno") == 2)
    p.chequear("entendió que se sigue hablando del Taos", "taos" in respuesta(job2).lower())
    p.chequear("ningún turno terminó escalado a revisión humana", sin_escalar(job1) and sin_escalar(job2))
    return p


# --- Links a Phoenix ----------------------------------------------------------------------------
async def links_de_phoenix(phoenix: str, pruebas: list[Prueba]) -> dict[int, str]:
    links = {}
    async with httpx.AsyncClient(base_url=phoenix, timeout=10) as c:
        try:
            r = await c.post("/graphql", json={"query": "query($n: String!) { getProjectByName(name: $n) { id } }",
                                               "variables": {"n": PROYECTO}})
            proyecto = r.json()["data"]["getProjectByName"]["id"]
            for p in pruebas:
                r = await c.post("/graphql", json={
                    "query": "query($s: String!) { getProjectSessionById(sessionId: $s) { id numTraces } }",
                    "variables": {"s": p.conversacion_id}})
                sesion = r.json()["data"]["getProjectSessionById"]
                if sesion:
                    links[p.numero] = f"{phoenix}/projects/{proyecto}/sessions/{sesion['id']}"
        except (httpx.HTTPError, KeyError, TypeError):
            pass
    return links


async def main(api: str, phoenix: str, reinicio: bool) -> int:
    pruebas_a_correr = [prueba_rag, prueba_multiagente, prueba_conversacion, prueba_hitl, prueba_calidad]
    if reinicio:
        pruebas_a_correr.append(prueba_reinicio)
    pruebas: list[Prueba] = []
    async with httpx.AsyncClient(base_url=api, timeout=30) as c:
        salud = (await c.get("/health")).json()
        print(f"API OK · workers vivos: {[w['nombre'] for w in salud['workers_vivos']]} · "
              f"RAG: {salud['workers_vivos'][0]['rag'] if salud['workers_vivos'] else '?'}")
        for funcion in pruebas_a_correr:
            numero = len(pruebas) + 1
            print(f"\n[{numero}] {funcion.__doc__ or funcion.__name__}")
            try:
                prueba = await funcion(c)
            except Exception as error:  # una prueba que explota no corta las demás
                prueba = Prueba(numero, funcion.__name__)
                prueba.chequear(f"la prueba terminó sin errores ({type(error).__name__}: {error})", False)
            pruebas.append(prueba)

    print("\n⏳ Esperando a que Phoenix reciba las trazas...")
    await asyncio.sleep(6)
    links = await links_de_phoenix(phoenix, pruebas)
    print("\n" + "=" * 72)
    for p in pruebas:
        print(f"{'✅' if p.ok else '❌'} {p.numero}. {p.nombre}  (conversación {p.conversacion_id})")
        print(f"     Phoenix: {links.get(p.numero, phoenix + ' -> Sessions -> buscar ' + str(p.conversacion_id))}")
    aprobadas = sum(p.ok for p in pruebas)
    print(f"\n{aprobadas}/{len(pruebas)} pruebas OK")

    destino = Path("evidencia") / f"pruebas_sistema_{datetime.now():%Y%m%d_%H%M%S}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps([{
        "numero": p.numero, "nombre": p.nombre, "ok": p.ok, "conversacion_id": p.conversacion_id,
        "phoenix": links.get(p.numero), "chequeos": [{"chequeo": d, "ok": ok} for d, ok in p.chequeos],
        "jobs": p.jobs} for p in pruebas], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 {destino}")
    return 0 if aprobadas == len(pruebas) else 1


prueba_rag.__doc__ = "RAG híbrido con paráfrasis"
prueba_multiagente.__doc__ = "Supervisor + especialistas (investigador -> analista -> síntesis)"
prueba_conversacion.__doc__ = "Conversación de dos turnos con memoria"
prueba_hitl.__doc__ = "Human-in-the-loop antes de una acción crítica"
prueba_calidad.__doc__ = "Puerta de calidad: cálculo corrompido -> validador -> reparación"
prueba_reinicio.__doc__ = "La conversación sobrevive a docker compose restart"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Las 5 pruebas del sistema con trazas en Phoenix")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--phoenix", default="http://localhost:6006")
    parser.add_argument("--reinicio", action="store_true", help="suma la prueba 6 (reinicia la API y los workers)")
    argumentos = parser.parse_args()
    sys.exit(asyncio.run(main(argumentos.api, argumentos.phoenix, argumentos.reinicio)))
