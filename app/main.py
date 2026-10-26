"""
API REST del sistema (entrega final).

    POST /tasks                    encola un turno y devuelve el job_id AL INSTANTE (202)
    GET  /tasks/{job_id}           estado del job: PENDING, RUNNING, WAITING_APPROVAL, DONE o FAILED
    GET  /tasks/{job_id}/eventos   progreso en vivo, nodo por nodo (Server-Sent Events)
    GET  /tasks                    últimos jobs (para revisar una prueba de carga)
    POST /tasks/{job_id}/approve   aprobación humana de una acción crítica (app/hitl.py)
    GET  /conversaciones/{id}      los turnos de una conversación
    GET  /health                   Redis, cola y workers vivos (de todos los contenedores)

Ningún endpoint corre el grafo: solo leen y escriben en Redis con el cliente
async (redis.asyncio), así que nunca bloquean el event loop. Un flujo de
agentes tarda decenas de segundos; mantener el request abierto todo ese
tiempo se corta por timeout y satura el servidor. Por eso el cliente recibe
un job_id y consulta el estado (polling), o se suscribe a los eventos (SSE).

Todas las entradas y salidas son modelos Pydantic (app/esquemas.py).

Levantar todo:  docker compose up -d        (API en http://localhost:8000/docs)
Desarrollo:     uvicorn app.main:app --port 8000   (con WORKER_EN_PROCESO=true)
"""
import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger
from redis.exceptions import ConnectionError as ErrorDeConexionRedis

from app.esquemas import (
    PATRON_ID,
    Conversacion,
    EstadoJob,
    PedidoTarea,
    ResumenJob,
    Salud,
    TareaCreada,
    TurnoConversacion,
    WorkerVivo,
)
from app.hitl import router as router_hitl
from app.jobs import TERMINALES, ConversacionOcupada, Estado, RepositorioJobs, conectar
from orquestador import config

INTERVALO_EVENTOS_S = 0.5


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    from app.graph import checkpointer_redis, crear_orquestador
    from app.observability import apagar_observabilidad, iniciar_observabilidad
    from app.worker import lanzar_workers
    from orquestador.rag.servicio import preparar_rag

    iniciar_observabilidad("api")
    redis = conectar(config.redis_url())
    try:
        await redis.ping()
    except ErrorDeConexionRedis:
        logger.error(f"No se pudo conectar a Redis en {config.redis_url()}. ¿Corriste 'docker compose up -d'?")
        raise
    app.state.jobs = RepositorioJobs(redis)
    app.state.workers, app.state.cantidad_workers = [], 0
    async with AsyncExitStack() as pila:
        if config.worker_en_proceso():
            await preparar_rag()
            checkpointer = await pila.enter_async_context(checkpointer_redis(config.redis_url()))
            app.state.workers = lanzar_workers(crear_orquestador(checkpointer), app.state.jobs, config.workers())
            app.state.cantidad_workers = config.workers()
            logger.info(f"WORKER_EN_PROCESO=true: {config.workers()} workers en el event loop de la API")
        else:
            logger.info("WORKER_EN_PROCESO=false: la API solo encola; los workers corren aparte "
                        "(servicio 'worker' de docker compose, o 'python -m app.worker')")
        yield
        for tarea in app.state.workers:
            tarea.cancel()
        await asyncio.gather(*app.state.workers, return_exceptions=True)
    await redis.aclose()
    apagar_observabilidad()


def _sin_telemetria_http() -> dict:
    """
    Las versiones nuevas de FastAPI generan una traza por cada request HTTP si
    hay un TracerProvider global (el de Phoenix). Se apaga: si no, los ~5 ms de
    cada GET /tasks/{id} del polling se mezclan con las ejecuciones del grafo
    (decenas de segundos) y el p95 del dashboard deja de medir a los agentes.
    En versiones de FastAPI sin ese parámetro no hace falta (no generan trazas).
    """
    if "telemetry" not in inspect.signature(FastAPI.__init__).parameters:
        return {}
    return {"telemetry": {"tracing": False, "metrics": False, "logs": False, "operation_spans": False}}


async def _job_o_404(jobs: RepositorioJobs, job_id: str) -> dict:
    job = await jobs.obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job '{job_id}'")
    return job


def _sse(evento: str, datos: dict) -> str:
    return f"event: {evento}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


async def _flujo_de_eventos(jobs: RepositorioJobs, job_id: str, request: Request) -> AsyncIterator[str]:
    """
    Emite cada cambio de estado y cada paso del grafo hasta que el job termina
    o queda esperando aprobación. Lee Redis cada medio segundo con el cliente
    async: un cliente conectado no ocupa un hilo ni bloquea a los demás.
    """
    ultimo_estado, vistos = None, 0
    while not await request.is_disconnected():
        job = await jobs.obtener(job_id)
        for paso in await jobs.progreso(job_id, desde=vistos):
            vistos += 1
            yield _sse("paso", paso)
        if job["estado"] != ultimo_estado:
            ultimo_estado = job["estado"]
            yield _sse("estado", job["historial"][-1])
        if ultimo_estado in TERMINALES or ultimo_estado == Estado.WAITING_APPROVAL:
            final = {"estado": ultimo_estado, "consultar": f"/tasks/{job_id}"}
            if job.get("resultado"):
                final["respuesta"] = job["resultado"]["respuesta"]
            yield _sse("fin", final)
            return
        await asyncio.sleep(INTERVALO_EVENTOS_S)


def crear_app(ciclo=ciclo_de_vida) -> FastAPI:
    """Fábrica de la app: los tests la crean sin ciclo de vida y le inyectan un Redis falso."""
    app = FastAPI(
        **_sin_telemetria_http(),
        title="Sistema Intelligence VW — API de producción",
        description="Sistema multi-agente (Supervisor + especialistas + RAG híbrido) expuesto como API "
                    "asíncrona, con conversaciones persistentes en Redis, workers escalables, aprobación "
                    "humana para acciones críticas y trazas en Phoenix.",
        version="8.0",
        lifespan=ciclo,
    )
    app.include_router(router_hitl)

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TareaCreada, tags=["tareas"],
              responses={409: {"description": "La conversación ya tiene un turno en curso"}})
    async def crear_tarea(pedido: PedidoTarea, request: Request) -> TareaCreada:
        """Guarda el job en Redis (PENDING), lo encola y devuelve el job_id sin esperar al grafo."""
        try:
            job = await request.app.state.jobs.crear(pedido.tarea, pedido.simulacion, pedido.conversacion_id)
        except ConversacionOcupada as ocupada:
            raise HTTPException(status.HTTP_409_CONFLICT, {
                "detalle": "La conversación ya tiene un turno en curso: esperá a que termine.",
                "job_activo": ocupada.job_activo, "consultar": f"/tasks/{ocupada.job_activo}"})
        return TareaCreada(job_id=job["job_id"], conversacion_id=job["conversacion_id"], estado=job["estado"],
                           consultar=f"/tasks/{job['job_id']}", eventos=f"/tasks/{job['job_id']}/eventos")

    @app.get("/tasks/{job_id}", response_model=EstadoJob, tags=["tareas"])
    async def estado_de_tarea(job_id: str, request: Request) -> EstadoJob:
        job = await _job_o_404(request.app.state.jobs, job_id)
        if job["estado"] == Estado.WAITING_APPROVAL:
            job["siguiente_paso"] = f"POST /tasks/{job_id}/approve con la decisión del revisor"
        return EstadoJob.model_validate(job)

    @app.get("/tasks/{job_id}/eventos", tags=["tareas"],
             response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}},
                              "description": "Eventos 'estado', 'paso' (un nodo del grafo) y 'fin'"}})
    async def eventos_de_tarea(job_id: str, request: Request) -> StreamingResponse:
        """Progreso en vivo. Probar con: curl -N http://localhost:8000/tasks/{job_id}/eventos"""
        await _job_o_404(request.app.state.jobs, job_id)
        return StreamingResponse(_flujo_de_eventos(request.app.state.jobs, job_id, request),
                                 media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/tasks", response_model=list[ResumenJob], tags=["tareas"])
    async def listar_tareas(request: Request, cantidad: int = Query(20, ge=1, le=200)) -> list[ResumenJob]:
        return [ResumenJob.model_validate(j) for j in await request.app.state.jobs.listar(cantidad)]

    @app.get("/conversaciones/{conversacion_id}", response_model=Conversacion, tags=["conversaciones"])
    async def conversacion(request: Request,
                           conversacion_id: Annotated[str, Path(pattern=PATRON_ID)]) -> Conversacion:
        jobs, activo = await request.app.state.jobs.turnos(conversacion_id)
        if not jobs:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación '{conversacion_id}'")
        return Conversacion(
            conversacion_id=conversacion_id,
            turnos=[TurnoConversacion(job_id=j["job_id"], estado=j["estado"], pregunta=j["tarea"],
                                      respuesta=(j.get("resultado") or {}).get("respuesta"), creado_en=j["creado_en"])
                    for j in jobs],
            job_activo=activo,
            como_seguir=f'POST /tasks con {{"tarea": "...", "conversacion_id": "{conversacion_id}"}}',
        )

    @app.get("/health", response_model=Salud, tags=["operación"])
    async def salud(request: Request) -> Salud:
        jobs: RepositorioJobs = request.app.state.jobs
        await jobs.redis.ping()
        vivos = await jobs.workers_vivos()
        return Salud(estado="ok", redis="ok", en_cola=await jobs.largo_de_cola(),
                     workers_en_proceso=getattr(request.app.state, "cantidad_workers", 0),
                     workers_vivos=[WorkerVivo(nombre=n, **d) for n, d in sorted(vivos.items())])

    return app


app = crear_app()
