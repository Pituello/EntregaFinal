"""
Tests de la Pre-Entrega 7: API, worker, estados en Redis y Human-in-the-loop.

Corren sin API key y sin Docker: Redis se reemplaza por fakeredis (mismos
comandos, incluidos los scripts Lua) y los agentes por los dobles de la
Pre-Entrega 6. Los tests marcados con "redis" usan un Redis 8 real (el de
docker compose) y se saltean solos si no hay uno levantado.

Uso:
    python -m pytest -v
"""
import asyncio
import json
import os
import time
import uuid

import fakeredis
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app import worker
from app.graph import construir_grafo, tipos_del_estado
from app.jobs import COLA, Estado, RepositorioJobs
from app.main import crear_app
from orquestador import config
from orquestador.state import DecisionSupervisor, PropuestaReserva
from orquestador.tools import reservas
from orquestador.validation import validar_reserva
from tests.test_orquestador import (
    ANALISIS_OK,
    INVESTIGACION_OK,
    AnalistaFalso,
    InvestigadorFalso,
    SintetizadorFalso,
    SupervisorFalso,
    decision,
)

TAREA = "¿Cuánta más potencia tiene el Virtus Exclusive que el Tera High?"
TAREA_RESERVA = "Reservá una Saveiro Extreme en Córdoba para Juan Pérez"


def correr(coro):
    return asyncio.run(coro)


class GestorFalso:
    def __init__(self, propuesta: PropuestaReserva | None = None):
        self.propuesta = propuesta or PropuestaReserva(
            version_id="SAVEIRO-EXTREME-CD", sucursal="Córdoba", cliente="Juan Pérez",
            resumen="1 Saveiro Extreme cabina doble en Córdoba para Juan Pérez")

    async def ejecutar(self, instruccion: str) -> PropuestaReserva:
        return self.propuesta


class SupervisorPorTarea:
    """Decide según la tarea y el estado, como un supervisor real (sirve para varios jobs a la vez)."""

    async def decidir(self, estado: dict) -> DecisionSupervisor:
        if "Reservá" in estado["tarea"] and estado.get("propuesta_reserva") is None:
            return decision("reservas", estado["tarea"])
        if "Reservá" in estado["tarea"] or estado.get("analisis") is not None:
            return decision("FINALIZAR")
        if estado.get("investigacion") is None:
            return decision("investigador", "Buscá las potencias")
        return decision("analista", "Calculá la diferencia")


def nuevo_grafo(supervisor=None, checkpointer=None):
    serde = JsonPlusSerializer(allowed_msgpack_modules=tipos_del_estado())
    return construir_grafo(
        supervisor or SupervisorPorTarea(), InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]),
        SintetizadorFalso(), reservas=GestorFalso(),
    ).compile(checkpointer=checkpointer or InMemorySaver(serde=serde))


@pytest.fixture(autouse=True)
def entorno(monkeypatch: pytest.MonkeyPatch):
    """Tests independientes del .env de cada máquina y sin enviar trazas."""
    monkeypatch.setenv("PHOENIX_HABILITADO", "false")
    for nombre, valor in {"MAX_PASOS": "8", "MAX_REPARACIONES": "2", "RECURSION_LIMIT": "40",
                          "TIMEOUT_JOB_S": "60"}.items():
        monkeypatch.setenv(nombre, valor)
    reservas.usar_registro(reservas.RegistroEnMemoria())
    yield
    reservas.usar_registro(None)


def nuevo_repositorio() -> RepositorioJobs:
    return RepositorioJobs(fakeredis.FakeAsyncRedis(decode_responses=True))


def cliente(jobs: RepositorioJobs) -> httpx.AsyncClient:
    app = crear_app(ciclo=None)
    app.state.jobs = jobs
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def procesar_siguiente(grafo, jobs: RepositorioJobs) -> None:
    mensaje = await jobs.tomar(timeout_s=1)
    assert mensaje is not None, "no había nada en la cola"
    await worker.procesar(grafo, jobs, mensaje)


# --- API asíncrona ---------------------------------------------------------------
def test_post_devuelve_el_job_id_al_instante_y_deja_el_estado_antes_de_encolar():
    async def caso():
        jobs = nuevo_repositorio()
        async with cliente(jobs) as c:
            inicio = time.perf_counter()
            r = await c.post("/tasks", json={"tarea": TAREA})
            demora = time.perf_counter() - inicio
            assert r.status_code == 202 and r.json()["estado"] == "PENDING"
            job_id = r.json()["job_id"]
            assert demora < 0.5  # no corre el grafo: solo escribe en Redis
            assert (await c.get(f"/tasks/{job_id}")).json()["estado"] == "PENDING"
        assert json.loads((await jobs.redis.lrange(COLA, 0, -1))[0]) == {"job_id": job_id, "accion": "iniciar"}
    correr(caso())


def test_job_inexistente_da_404_y_tarea_invalida_da_422():
    async def caso():
        async with cliente(nuevo_repositorio()) as c:
            assert (await c.get("/tasks/no-existe")).status_code == 404
            assert (await c.post("/tasks", json={"tarea": "hi"})).status_code == 422
            assert (await c.post("/tasks/no-existe/approve", json={"aprobado": True})).status_code == 404
    correr(caso())


# --- Worker y estados en Redis -------------------------------------------------------
def test_worker_procesa_el_job_y_deja_done_con_la_traza_de_decisiones():
    async def caso():
        jobs = nuevo_repositorio()
        job = await jobs.crear(TAREA)
        await procesar_siguiente(nuevo_grafo(), jobs)
        final = await jobs.obtener(job["job_id"])
        assert final["estado"] == "DONE"
        assert final["resultado"]["respuesta"] == "Respuesta final de prueba."
        assert [d["siguiente"] for d in final["resultado"]["decisiones"]] == ["investigador", "analista", "FINALIZAR"]
        assert [e["estado"] for e in final["historial"]] == ["PENDING", "RUNNING", "DONE"]
        assert final["duracion_ejecucion_s"] > 0 and final["latencia_total_s"] > 0
    correr(caso())


def test_una_excepcion_no_controlada_deja_el_job_en_failed():
    """Error común de la consigna: si el agente falla en segundo plano, el estado tiene que pasar a FAILED."""
    async def caso():
        jobs = nuevo_repositorio()
        job = await jobs.crear(TAREA, {"romper_grafo": True})
        await procesar_siguiente(nuevo_grafo(), jobs)
        final = await jobs.obtener(job["job_id"])
        assert final["estado"] == "FAILED"
        assert "RuntimeError" in final["error"] and "Falla simulada" in final["error"]
        assert [e["estado"] for e in final["historial"]] == ["PENDING", "RUNNING", "FAILED"]
    correr(caso())


def test_un_job_que_supera_el_timeout_queda_en_failed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "timeout_job_s", lambda: 0.2)

    async def caso():
        jobs = nuevo_repositorio()
        job = await jobs.crear(TAREA, {"demora_s": 5})
        await procesar_siguiente(nuevo_grafo(), jobs)
        final = await jobs.obtener(job["job_id"])
        assert final["estado"] == "FAILED" and final["error"].startswith("Timeout")
    correr(caso())


def test_cinco_jobs_concurrentes_no_se_bloquean_entre_si():
    """
    Cada job pasa 3 veces por el supervisor, con un LLM simulado de 0,3 s: ~0,9 s por job.
    En serie, 5 jobs tardarían ~4,5 s; con 5 workers async tienen que terminar en ~1 s.
    """
    async def caso():
        jobs = nuevo_repositorio()
        ids = [(await jobs.crear(TAREA, {"demora_s": 0.3}))["job_id"] for _ in range(5)]

        async def estados() -> list[str]:
            return [(await jobs.obtener(i))["estado"] for i in ids]

        inicio = time.perf_counter()
        tareas = worker.lanzar_workers(nuevo_grafo(), jobs, 5)
        while await estados() != ["DONE"] * 5:
            assert time.perf_counter() - inicio < 6, "los jobs no terminaron"
            await asyncio.sleep(0.05)
        total = time.perf_counter() - inicio
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
        assert total < 2.5, f"tardaron {total:.1f} s: se están ejecutando en serie"
    correr(caso())


def test_un_job_interrumpido_se_retoma_desde_el_ultimo_checkpoint():
    async def caso():
        jobs = nuevo_repositorio()
        grafo = nuevo_grafo()
        job = await jobs.crear(TAREA)
        await jobs.tomar(timeout_s=1)
        assert await jobs.reclamar(job["job_id"], "worker-muerto", "iniciar", lease_s=30)  # y se "cortó" acá
        assert await jobs.recuperar_interrumpidos() == []  # su lease sigue vigente: nadie se lo saca
        await jobs.redis.delete(f"vw:job:{job['job_id']}:lease")  # pasaron LEASE_S sin renovarlo
        assert await jobs.recuperar_interrumpidos() == [job["job_id"]]
        assert await jobs.recuperar_interrumpidos() == []  # un segundo vigilante no lo encola otra vez
        await procesar_siguiente(grafo, jobs)
        assert (await jobs.obtener(job["job_id"]))["estado"] == "DONE"
    correr(caso())


# --- Human-in-the-loop ------------------------------------------------------------
def test_reserva_pausa_hasta_la_aprobacion_y_recien_ahi_reserva():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        async with cliente(jobs) as c:
            job_id = (await c.post("/tasks", json={"tarea": TAREA_RESERVA})).json()["job_id"]
            await procesar_siguiente(grafo, jobs)

            pausado = (await c.get(f"/tasks/{job_id}")).json()
            assert pausado["estado"] == "WAITING_APPROVAL"
            assert pausado["aprobacion"]["accion"] == "reservar_unidad"
            assert "Saveiro Extreme cabina doble" in pausado["aprobacion"]["pregunta"]
            assert reservas._registro.reservas == {}  # nada se reservó todavía

            r = await c.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "Damián"})
            assert r.status_code == 202 and r.json()["estado"] == "PENDING"
            await procesar_siguiente(grafo, jobs)

            final = (await c.get(f"/tasks/{job_id}")).json()
        assert final["estado"] == "DONE"
        assert final["resultado"]["reserva"]["ok"] and final["resultado"]["decision_humana"]["revisor"] == "Damián"
        assert list(reservas._registro.reservas) == [f"vw:reserva:{job_id}"]
        assert [e["estado"] for e in final["historial"]] == [
            "PENDING", "RUNNING", "WAITING_APPROVAL", "PENDING", "RUNNING", "DONE"]
    correr(caso())


def test_reserva_rechazada_no_toca_el_stock():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        async with cliente(jobs) as c:
            job_id = (await c.post("/tasks", json={"tarea": TAREA_RESERVA})).json()["job_id"]
            await procesar_siguiente(grafo, jobs)
            await c.post(f"/tasks/{job_id}/approve",
                         json={"aprobado": False, "revisor": "Damián", "comentario": "cliente sin seña"})
            await procesar_siguiente(grafo, jobs)
            final = (await c.get(f"/tasks/{job_id}")).json()
        assert final["estado"] == "DONE" and final["resultado"]["reserva"] is None
        assert final["resultado"]["decisiones"][-1]["origen"] == "regla"  # finaliza una regla, no el LLM
        assert reservas._registro.reservas == {}
    correr(caso())


def test_solo_se_aprueba_un_job_que_espera_aprobacion():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        async with cliente(jobs) as c:
            job_id = (await c.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
            assert (await c.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409

            reserva_id = (await c.post("/tasks", json={"tarea": TAREA_RESERVA})).json()["job_id"]
            await procesar_siguiente(grafo, jobs)  # el primero (no crítico)
            await procesar_siguiente(grafo, jobs)  # la reserva: queda en pausa
            assert (await c.post(f"/tasks/{reserva_id}/approve", json={"aprobado": True})).status_code == 202
            # aprobar dos veces no reanuda dos veces
            assert (await c.post(f"/tasks/{reserva_id}/approve", json={"aprobado": True})).status_code == 409
    correr(caso())


def test_el_llm_no_puede_repetir_una_reserva_ya_resuelta():
    async def caso():
        sup = SupervisorFalso([decision("reservas", TAREA_RESERVA), decision("reservas", "otra vez")])
        jobs, grafo = nuevo_repositorio(), nuevo_grafo(supervisor=sup)
        job = await jobs.crear(TAREA_RESERVA)
        await procesar_siguiente(grafo, jobs)
        await jobs.registrar_decision(job["job_id"], {"aprobado": True, "revisor": "Damián"})
        await procesar_siguiente(grafo, jobs)
        final = await jobs.obtener(job["job_id"])
        ultima = final["resultado"]["decisiones"][-1]
        assert ultima["siguiente"] == "FINALIZAR" and ultima["origen"] == "regla" and "no se repite" in ultima["razon"]
        assert len(reservas._registro.reservas) == 1
    correr(caso())


def test_la_propuesta_se_valida_contra_el_catalogo_antes_de_llegar_al_humano():
    def propuesta(version_id="SAVEIRO-EXTREME-CD", sucursal="Córdoba", cliente="Juan"):
        return PropuestaReserva(version_id=version_id, sucursal=sucursal, cliente=cliente, resumen="x")

    assert validar_reserva(propuesta()).ok
    assert "no existe en el catálogo" in validar_reserva(propuesta(version_id="SAVEIRO-INVENTADA")).errores[0]
    assert "no existe" in validar_reserva(propuesta(sucursal="Mendoza")).errores[0]
    sin_stock = validar_reserva(propuesta(sucursal="Palermo")).errores[0]  # Extreme: 0 en Palermo
    assert "no tiene stock en Palermo" in sin_stock and "Córdoba" in sin_stock


def test_reservar_unidad_es_idempotente_y_respeta_el_stock():
    async def caso():
        def pedir(clave):
            return reservas.reservar_unidad.ainvoke(
                {"version_id": "VENTO-GLI-DSG7", "sucursal": "palermo", "cliente": "Ana", "clave_idempotencia": clave})
        primera = json.loads(await pedir("job-a"))  # el Vento GLI tiene 1 unidad en Palermo
        repetida = json.loads(await pedir("job-a"))  # el mismo job re-ejecutado: no descuenta otra vez
        otra = await pedir("job-b")
        assert primera["estado"] == "nueva" and repetida["estado"] == "existente"
        assert primera["reserva_id"] == repetida["reserva_id"]
        assert otra.startswith("ERROR") and "no quedan unidades" in otra
    correr(caso())


# --- Integración con Redis 8 real (docker compose up -d) ----------------------------
# Base 0: los índices de búsqueda de Redis (que usa el checkpointer) solo existen ahí. Por eso los tests
# no borran la base: usan claves únicas por corrida y limpian solo lo suyo.
REDIS_DE_PRUEBA = os.getenv("REDIS_URL_TESTS", "redis://localhost:6379/0")


def _hay_redis() -> bool:
    import redis

    try:
        return bool(redis.Redis.from_url(REDIS_DE_PRUEBA, socket_connect_timeout=0.5).ping())
    except Exception:
        return False


con_redis = pytest.mark.skipif(not _hay_redis(), reason="no hay Redis en REDIS_URL_TESTS (docker compose up -d)")


@con_redis
def test_redis_real_la_pausa_sobrevive_a_un_reinicio_y_los_tipos_vuelven_bien():
    """
    Regresión: con AsyncRedisSaver, sin declarar los tipos del estado, al
    reanudar volvían como diccionarios y aprobacion_humana fallaba con
    "'dict' object has no attribute 'version_id'". Con el checkpointer en
    memoria no se veía: hacía falta Redis de verdad.
    """
    from langgraph.types import Command
    from redis.asyncio import from_url

    from app.graph import checkpointer_redis, config_de_ejecucion
    from orquestador.state import estado_inicial

    async def caso():
        hilo = f"test-reinicio-{uuid.uuid4()}"
        cfg = config_de_ejecucion(thread_id=hilo)
        async with checkpointer_redis(REDIS_DE_PRUEBA) as saver:  # "proceso" 1: queda en pausa
            await nuevo_grafo(checkpointer=saver).ainvoke(estado_inicial(TAREA_RESERVA), cfg)
        async with checkpointer_redis(REDIS_DE_PRUEBA) as saver:  # "proceso" 2: después de reiniciar
            grafo = nuevo_grafo(checkpointer=saver)
            pausado = await grafo.aget_state(cfg)
            assert pausado.next == ("aprobacion_humana",)
            assert type(pausado.values["propuesta_reserva"]).__name__ == "PropuestaReserva"
            final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "Damián"}), cfg)
            await saver.adelete_thread(hilo)
        assert final["resultado_reserva"].ok
    correr(caso())


@con_redis
def test_redis_real_el_script_lua_reserva_una_sola_vez():
    async def caso():
        from redis.asyncio import from_url

        prefijo = f"test:{uuid.uuid4().hex[:8]}"
        reservas.usar_registro(reservas.RegistroRedis(REDIS_DE_PRUEBA, prefijo=prefijo))
        pedidos = [reservas.reservar_unidad.ainvoke({"version_id": "VENTO-GLI-DSG7", "sucursal": "Palermo",
                                                     "cliente": f"c{i}", "clave_idempotencia": f"job-{i}"})
                   for i in range(5)]
        resultados = await asyncio.gather(*pedidos)  # 5 a la vez por la única unidad
        r = from_url(REDIS_DE_PRUEBA)
        claves = [k async for k in r.scan_iter(match=f"{prefijo}*")]
        await r.delete(*claves)
        await r.aclose()
        assert sum(not x.startswith("ERROR") for x in resultados) == 1
    correr(caso())
