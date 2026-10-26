"""
Tests de la entrega final: conversaciones persistentes, workers independientes
(leases), progreso en vivo (SSE) y contratos Pydantic de la API.

Como los de la Pre-Entrega 7, corren sin API key y sin Docker (fakeredis +
agentes dobles). Los marcados con con_redis usan el Redis 8 de docker compose
y se saltean solos si no está levantado.

Uso:
    python -m pytest -v
"""
import asyncio
import json
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app import worker
from app.graph import config_de_ejecucion, construir_grafo, tipos_del_estado
from app.jobs import COLA, ConversacionOcupada, Estado
from app.main import crear_app
from orquestador.agents.supervisor import resumir_estado
from orquestador.state import DecisionSupervisor, estado_del_turno
from orquestador.tools import reservas
from tests.test_api import (
    REDIS_DE_PRUEBA,
    TAREA,
    TAREA_RESERVA,
    GestorFalso,
    SupervisorPorTarea,
    cliente,
    con_redis,
    correr,
    nuevo_grafo,
    nuevo_repositorio,
    procesar_siguiente,
)
from tests.test_orquestador import ANALISIS_OK, INVESTIGACION_OK, AnalistaFalso, InvestigadorFalso, SintetizadorFalso

SEGUIMIENTO = "¿Y cuánto torque más tiene?"


@pytest.fixture(autouse=True)
def entorno(monkeypatch: pytest.MonkeyPatch):
    for nombre, valor in {"MAX_PASOS": "8", "MAX_REPARACIONES": "2", "RECURSION_LIMIT": "40",
                          "TIMEOUT_JOB_S": "60", "LEASE_S": "30"}.items():
        monkeypatch.setenv(nombre, valor)
    reservas.usar_registro(reservas.RegistroEnMemoria())
    yield
    reservas.usar_registro(None)


class SupervisorQueAnota(SupervisorPorTarea):
    """Decide como SupervisorPorTarea y guarda el resumen que vio en cada decisión."""

    def __init__(self):
        self.resumenes: list[str] = []

    async def decidir(self, estado: dict) -> DecisionSupervisor:
        self.resumenes.append(resumir_estado(estado))
        return await super().decidir(estado)


def grafo_con(supervisor, checkpointer=None):
    serde = JsonPlusSerializer(allowed_msgpack_modules=tipos_del_estado())
    return construir_grafo(
        supervisor, InvestigadorFalso([INVESTIGACION_OK] * 5), AnalistaFalso([ANALISIS_OK] * 5),
        SintetizadorFalso(), reservas=GestorFalso(),
    ).compile(checkpointer=checkpointer or InMemorySaver(serde=serde))


# --- Conversaciones -------------------------------------------------------------------------
def test_el_segundo_turno_ve_el_primero_y_empieza_con_logs_limpios():
    async def caso():
        jobs, supervisor = nuevo_repositorio(), SupervisorQueAnota()
        grafo = grafo_con(supervisor)
        primero = await jobs.crear(TAREA)
        await procesar_siguiente(grafo, jobs)
        segundo = await jobs.crear(SEGUIMIENTO, conversacion_id=primero["conversacion_id"])
        await procesar_siguiente(grafo, jobs)

        final = await jobs.obtener(segundo["job_id"])
        assert final["estado"] == "DONE" and final["resultado"]["turno"] == 2
        # Las decisiones son solo las de este turno (el reducer acumular las reinició)...
        assert [d["siguiente"] for d in final["resultado"]["decisiones"]] == ["investigador", "analista", "FINALIZAR"]
        # ...pero el supervisor vio la conversación previa para resolver "¿Y cuánto torque más?"
        assert "CONVERSACIÓN PREVIA" not in supervisor.resumenes[0]
        assert "CONVERSACIÓN PREVIA" in supervisor.resumenes[-1] and f"Usuario: {TAREA}" in supervisor.resumenes[-1]

        estado = await grafo.aget_state(config_de_ejecucion(thread_id=primero["conversacion_id"]))
        assert [t.pregunta for t in estado.values["historial"]] == [TAREA, SEGUIMIENTO]
        assert [t.job_id for t in estado.values["historial"]] == [primero["job_id"], segundo["job_id"]]
    correr(caso())


def test_una_conversacion_acepta_un_turno_a_la_vez():
    async def caso():
        jobs = nuevo_repositorio()
        async with cliente(jobs) as c:
            primero = (await c.post("/tasks", json={"tarea": TAREA})).json()
            conversacion = primero["conversacion_id"]
            ocupada = await c.post("/tasks", json={"tarea": SEGUIMIENTO, "conversacion_id": conversacion})
            assert ocupada.status_code == 409 and ocupada.json()["detail"]["job_activo"] == primero["job_id"]
            assert (await c.post("/tasks", json={"tarea": TAREA})).status_code == 202  # otra conversación, sí

            await procesar_siguiente(nuevo_grafo(), jobs)  # termina el primer turno
            libre = await c.post("/tasks", json={"tarea": SEGUIMIENTO, "conversacion_id": conversacion})
            assert libre.status_code == 202 and libre.json()["conversacion_id"] == conversacion

            assert (await c.post("/tasks", json={"tarea": TAREA, "conversacion_id": "con espacios"})).status_code == 422
    correr(caso())


def test_un_job_terminado_mal_no_deja_trabada_la_conversacion():
    async def caso():
        jobs = nuevo_repositorio()
        job = await jobs.crear(TAREA, {"romper_grafo": True})
        await procesar_siguiente(nuevo_grafo(), jobs)
        assert (await jobs.obtener(job["job_id"]))["estado"] == "FAILED"
        siguiente = await jobs.crear(TAREA, conversacion_id=job["conversacion_id"])  # no lanza ConversacionOcupada
        assert siguiente["conversacion_id"] == job["conversacion_id"]
        with pytest.raises(ConversacionOcupada):
            await jobs.crear(TAREA, conversacion_id=job["conversacion_id"])
    correr(caso())


def test_get_conversacion_devuelve_los_turnos_en_orden():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        async with cliente(jobs) as c:
            conversacion = (await c.post("/tasks", json={"tarea": TAREA})).json()["conversacion_id"]
            await procesar_siguiente(grafo, jobs)
            await c.post("/tasks", json={"tarea": SEGUIMIENTO, "conversacion_id": conversacion})
            r = (await c.get(f"/conversaciones/{conversacion}")).json()
            assert [t["pregunta"] for t in r["turnos"]] == [TAREA, SEGUIMIENTO]
            assert r["turnos"][0]["respuesta"] == "Respuesta final de prueba." and r["turnos"][1]["respuesta"] is None
            assert r["job_activo"] == r["turnos"][1]["job_id"]
            assert (await c.get("/conversaciones/no-existe")).status_code == 404
    correr(caso())


def test_cada_turno_reserva_con_su_propia_clave_de_idempotencia():
    """En la Pre-Entrega 7 la clave era el thread_id; ahora el hilo es la conversación y la clave, el job."""
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        primero = await jobs.crear(TAREA_RESERVA)
        conversacion = primero["conversacion_id"]
        for job in (primero, None):
            if job is None:
                job = await jobs.crear(TAREA_RESERVA, conversacion_id=conversacion)
            await procesar_siguiente(grafo, jobs)
            await jobs.registrar_decision(job["job_id"], {"aprobado": True, "revisor": "Damián"})
            await procesar_siguiente(grafo, jobs)
            assert (await jobs.obtener(job["job_id"]))["resultado"]["reserva"]["ok"]
        assert len(reservas._registro.reservas) == 2
    correr(caso())


# --- Workers independientes -------------------------------------------------------------------
def test_un_mensaje_duplicado_no_ejecuta_dos_veces_el_job():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        job = await jobs.crear(TAREA)
        await jobs.redis.rpush(COLA, json.dumps({"job_id": job["job_id"], "accion": "iniciar"}))  # duplicado
        await procesar_siguiente(grafo, jobs)
        await procesar_siguiente(grafo, jobs)  # lo toma otro worker: reclamar() lo rechaza
        final = await jobs.obtener(job["job_id"])
        assert [e["estado"] for e in final["historial"]] == ["PENDING", "RUNNING", "DONE"]
    correr(caso())


def test_apagado_ordenado_devuelve_el_job_a_la_cola_y_otro_worker_lo_termina():
    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        job = await jobs.crear(TAREA, {"demora_s": 0.5})
        mensaje = await jobs.tomar(timeout_s=1)
        corriendo = asyncio.create_task(worker.procesar(grafo, jobs, mensaje, "worker-a"))
        await asyncio.sleep(0.2)
        corriendo.cancel()  # SIGTERM de docker a mitad del job
        await asyncio.gather(corriendo, return_exceptions=True)

        devuelto = await jobs.obtener(job["job_id"])
        assert devuelto["estado"] == "PENDING" and "apagado ordenado" in devuelto["historial"][-1]["detalle"]
        assert not await jobs.redis.exists(f"vw:job:{job['job_id']}:lease")
        mensaje = await jobs.tomar(timeout_s=1)
        assert mensaje == {"job_id": job["job_id"], "accion": "continuar"}
        await worker.procesar(grafo, jobs, mensaje, "worker-b")
        final = await jobs.obtener(job["job_id"])
        assert final["estado"] == "DONE" and final["worker"] == "worker-b"
    correr(caso())


def test_el_worker_renueva_el_lease_mientras_trabaja(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEASE_S", "3")  # se renueva cada 1 s

    async def caso():
        jobs, grafo = nuevo_repositorio(), nuevo_grafo()
        job = await jobs.crear(TAREA, {"demora_s": 1.2})  # 3 decisiones x 1,2 s = 3,6 s > LEASE_S
        corriendo = asyncio.create_task(procesar_siguiente(grafo, jobs))
        clave = f"vw:job:{job['job_id']}:lease"
        for _ in range(6):
            await asyncio.sleep(0.5)
            if (await jobs.obtener(job["job_id"]))["estado"] == "RUNNING":
                assert await jobs.redis.ttl(clave) > 0
                assert await jobs.recuperar_interrumpidos() == []  # está vivo: no se recupera
        await corriendo
        assert (await jobs.obtener(job["job_id"]))["estado"] == "DONE"
        assert not await jobs.redis.exists(clave)
    correr(caso())


def test_health_muestra_los_workers_vivos():
    async def caso():
        jobs = nuevo_repositorio()
        tareas = worker.lanzar_workers(nuevo_grafo(), jobs, 2, proceso="contenedor-1")
        await asyncio.sleep(0.1)
        async with cliente(jobs) as c:
            salud = (await c.get("/health")).json()
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
        assert [w["nombre"] for w in salud["workers_vivos"]] == ["contenedor-1"]
        assert salud["workers_vivos"][0]["jobs_en_paralelo"] == 2 and salud["workers_vivos"][0]["rag"]
    correr(caso())


# --- Progreso en vivo y contratos de la API ----------------------------------------------------
def test_eventos_sse_muestran_cada_paso_del_grafo():
    async def caso():
        jobs = nuevo_repositorio()
        async with cliente(jobs) as c:
            job_id = (await c.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
            await procesar_siguiente(nuevo_grafo(), jobs)
            r = await c.get(f"/tasks/{job_id}/eventos")
        assert r.headers["content-type"].startswith("text/event-stream")
        eventos = [(b.split("\n")[0].removeprefix("event: "), json.loads(b.split("\n")[1].removeprefix("data: ")))
                   for b in r.text.strip().split("\n\n")]
        pasos = [d["nodo"] for tipo, d in eventos if tipo == "paso"]
        assert pasos[:3] == ["supervisor", "investigador", "validador"] and pasos[-1] == "sintesis"
        assert "decide 'investigador'" in next(d["detalle"] for t, d in eventos if t == "paso")
        assert eventos[-1] == ("fin", {"estado": "DONE", "consultar": f"/tasks/{job_id}",
                                       "respuesta": "Respuesta final de prueba."})
    correr(caso())


def test_las_respuestas_de_la_api_estan_validadas_por_modelos_pydantic():
    app = crear_app(ciclo=None)
    esquema = app.openapi()
    modelos = {
        ("/tasks", "post"): "TareaCreada",
        ("/tasks/{job_id}", "get"): "EstadoJob",
        ("/tasks/{job_id}/approve", "post"): "RespuestaAprobacion",
        ("/conversaciones/{conversacion_id}", "get"): "Conversacion",
        ("/health", "get"): "Salud",
    }
    for (ruta, metodo), modelo in modelos.items():
        respuestas = esquema["paths"][ruta][metodo]["responses"]
        exito = respuestas.get("200") or respuestas.get("202")
        assert exito["content"]["application/json"]["schema"]["$ref"].endswith(f"/{modelo}"), ruta
    assert "PedidoTarea" in esquema["components"]["schemas"]


def test_el_resultado_del_job_trae_turno_y_fuentes():
    async def caso():
        jobs = nuevo_repositorio()
        async with cliente(jobs) as c:
            job_id = (await c.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
            await procesar_siguiente(nuevo_grafo(), jobs)
            final = (await c.get(f"/tasks/{job_id}")).json()
        assert final["resultado"]["turno"] == 1 and final["resultado"]["fuentes"] == ["catalogo_concesionaria"]
        assert "creado_ts" not in final  # el modelo de salida filtra los campos internos
    correr(caso())


# --- Con Redis 8 real ----------------------------------------------------------------------------
@con_redis
def test_redis_real_la_conversacion_sobrevive_a_un_reinicio():
    """El criterio de la consigna: el turno 2 corre en un "proceso" nuevo y recuerda el turno 1."""
    from app.graph import checkpointer_redis

    async def caso():
        conversacion = f"test-conversacion-{uuid.uuid4()}"
        async with checkpointer_redis(REDIS_DE_PRUEBA) as saver:  # proceso 1
            await grafo_con(SupervisorPorTarea(), saver).ainvoke(
                estado_del_turno(TAREA, "job-1"), config_de_ejecucion(thread_id=conversacion, job_id="job-1"))
        async with checkpointer_redis(REDIS_DE_PRUEBA) as saver:  # proceso 2, después de reiniciar
            supervisor = SupervisorQueAnota()
            final = await grafo_con(supervisor, saver).ainvoke(
                estado_del_turno(SEGUIMIENTO, "job-2"), config_de_ejecucion(thread_id=conversacion, job_id="job-2"))
            await saver.adelete_thread(conversacion)
        assert [t.job_id for t in final["historial"]] == ["job-1", "job-2"]
        assert type(final["historial"][0]).__name__ == "Turno"  # volvió como modelo, no como dict
        assert f"Usuario: {TAREA}" in supervisor.resumenes[0]
    correr(caso())
