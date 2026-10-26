"""
Worker: toma jobs de la cola de Redis, corre el grafo y actualiza el estado.

Patrón de la clase: FastAPI (recepcionista) -> Redis (buzón) -> Worker
(especialista). La API nunca corre el grafo dentro del request: solo encola.

Dos formas de correrlo:
- Como proceso aparte, el modo de producción (WORKER_EN_PROCESO=false en la
  API, el que usa docker compose):
      python -m app.worker
  La API y el procesamiento escalan por separado: docker compose up --scale
  worker=3 levanta 3 procesos que leen la misma cola.
- Dentro de la API (WORKER_EN_PROCESO=true): app/main.py lanza las tareas en
  el mismo event loop que uvicorn. Cómodo para desarrollo con un solo comando.

Cada proceso corre WORKERS tareas asyncio (no hilos): el grafo es async de
punta a punta (LLM, herramientas, Redis), así que mientras un job espera al
modelo, el event loop atiende a los demás. Ninguna llamada bloqueante corre
en el event loop (error común: "bloquear el event loop").

Varios procesos a la vez (entrega final):
- reclamar() pasa el job de PENDING a RUNNING de forma atómica y le da un
  LEASE: un mensaje duplicado nunca ejecuta dos veces el mismo job;
- mientras corre, una tarea renueva el lease cada LEASE_S/3;
- el VIGILANTE de cada proceso publica su latido (GET /health lo muestra) y
  devuelve a la cola los jobs cuyo lease venció (su worker murió);
- apagado ordenado (SIGTERM de docker, Ctrl+C): el job en curso vuelve a la
  cola al instante y otro worker lo retoma desde su último checkpoint.

Manejo de errores: CUALQUIER excepción del grafo (o un timeout) deja el job
en FAILED con el error, así el cliente que hace polling sabe que terminó.
"""
import asyncio
import os
import signal
import socket
import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from app.esquemas import DecisionResumida, ResultadoJob
from app.graph import config_de_ejecucion
from app.jobs import Estado, RepositorioJobs, ahora, conectar
from app.observability import descripcion_del_llm, span_de_job
from orquestador import config, prompts
from orquestador.state import estado_del_turno


# --- Resultado y progreso ---------------------------------------------------------------------
def resumir_resultado(valores: dict[str, Any]) -> dict[str, Any]:
    """Lo que devuelve GET /tasks/{id} al terminar: la respuesta y quién decidió qué y por qué."""
    investigacion = valores.get("investigacion")
    resultado = ResultadoJob(
        respuesta=valores.get("respuesta_final", ""),
        turno=max(len(valores.get("historial", [])), 1),
        fuentes=sorted({d.fuente for d in investigacion.datos}) if investigacion else [],
        intervenciones_de_especialistas=valores.get("pasos", 0),
        escalado_humano=valores.get("escalado_humano", False),
        validaciones_fallidas=sum(not v.ok for v in valores.get("validaciones", [])),
        decisiones=[DecisionResumida(paso=d.paso, origen=d.origen, siguiente=d.siguiente, razon=d.razon)
                    for d in valores.get("decisiones", [])],
        decision_humana=valores.get("decision_humana"),
        reserva=valores.get("resultado_reserva"),
        prompts_version=prompts.PROMPTS_VERSION,
    )
    return resultado.model_dump(mode="json")


def describir_paso(nodo: str, cambios: Any) -> str | None:
    """Una línea legible de lo que hizo un nodo (para GET /tasks/{id}/eventos)."""
    if nodo == "__interrupt__":
        return "pausa: la acción crítica espera aprobación humana"
    if not isinstance(cambios, dict):
        return None
    if nodo == "supervisor" and cambios.get("decisiones"):
        d = cambios["decisiones"][-1]
        return f"decide '{d.siguiente}' ({d.origen}): {d.razon}"
    if cambios.get("contribuciones"):
        c = cambios["contribuciones"][-1]
        return f"{c.resumen} ({c.duracion_s:.1f} s)" if c.ok else f"falló: {c.error}"
    if nodo == "validador" and cambios.get("validacion"):
        v = cambios["validacion"]
        return f"{v.agente}: OK" if v.ok else f"{v.agente}: FALLÓ: {' | '.join(v.errores)}"
    if nodo == "aprobacion_humana" and cambios.get("decision_humana"):
        h = cambios["decision_humana"]
        return f"{'aprobada' if h.aprobado else 'rechazada'} por {h.revisor}"
    if nodo == "ejecutar_reserva" and cambios.get("resultado_reserva"):
        return cambios["resultado_reserva"].detalle
    if nodo == "sintesis":
        return "respuesta lista"
    return None


async def _correr_grafo(grafo: CompiledStateGraph, entrada, cfg: dict, jobs: RepositorioJobs, job_id: str) -> None:
    """Como ainvoke, pero registra en Redis lo que hace cada nodo a medida que termina."""
    async for actualizacion in grafo.astream(entrada, cfg, stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            detalle = describir_paso(nodo, cambios)
            if detalle:
                await jobs.registrar_progreso(job_id, nodo, detalle)


async def _entrada_del_grafo(grafo: CompiledStateGraph, job: dict, accion: str, cfg: dict):
    if accion == "reanudar":
        return Command(resume=job["decision"])  # retoma el hilo pausado con la decisión del humano
    if accion == "continuar":
        # El worker anterior murió a mitad de camino. Si el último checkpoint del hilo es de ESTE
        # turno, se sigue desde ahí (None); si no, el turno nunca llegó a guardarse y empieza de cero.
        guardado = await grafo.aget_state(cfg)
        if guardado.values.get("job_id") == job["job_id"]:
            if guardado.interrupts and job.get("decision"):  # murió justo al reanudar una aprobación
                return Command(resume=job["decision"])
            return None
    return estado_del_turno(job["tarea"], job["job_id"])


# --- Un job ---------------------------------------------------------------------------------
async def _renovar_lease(jobs: RepositorioJobs, job_id: str, nombre: str) -> None:
    while True:
        await asyncio.sleep(max(1.0, config.lease_s() / 3))
        try:
            await jobs.renovar_lease(job_id, nombre, config.lease_s())
        except Exception as error:  # Redis caído un momento: el próximo intento lo renueva
            logger.warning(f"[{nombre}] no se pudo renovar el lease de {job_id[:8]}: {error}")


async def procesar(grafo: CompiledStateGraph, jobs: RepositorioJobs, mensaje: dict, nombre: str = "worker") -> None:
    job_id, accion = mensaje["job_id"], mensaje.get("accion", "iniciar")
    job = await jobs.obtener(job_id)
    if job is None:
        logger.warning(f"[{nombre}] job {job_id} no existe: se descarta el mensaje")
        return
    if not await jobs.reclamar(job_id, nombre, accion, config.lease_s()):
        logger.warning(f"[{nombre}] job {job_id[:8]} está {job['estado']}: mensaje duplicado o ya lo tiene "
                       "otro worker; se ignora")
        return

    cfg = config_de_ejecucion(job["simulacion"], thread_id=job["conversacion_id"], job_id=job_id)
    logger.info(f"[{nombre}] {accion} job {job_id[:8]} (conversación {job['conversacion_id'][:8]}): "
                f"{job['tarea'][:70]}")
    inicio = time.perf_counter()
    latido = asyncio.create_task(_renovar_lease(jobs, job_id, nombre))
    try:
        entrada = await _entrada_del_grafo(grafo, job, accion, cfg)
        with span_de_job(job_id, job["conversacion_id"], accion, job["tarea"], job["simulacion"]) as span:
            await asyncio.wait_for(_correr_grafo(grafo, entrada, cfg, jobs, job_id), timeout=config.timeout_job_s())
            snapshot = await grafo.aget_state(cfg)
            # La salida de la traza en Phoenix: la respuesta, o el pedido de aprobación si quedó en pausa.
            span.set_output(snapshot.interrupts[0].value["pregunta"] if snapshot.interrupts
                            else snapshot.values.get("respuesta_final", ""))
    except TimeoutError:
        await _fallar(jobs, job, f"Timeout: la ejecución superó los {config.timeout_job_s()} s (TIMEOUT_JOB_S)")
        return
    except asyncio.CancelledError:
        # Apagado ordenado: el job vuelve a la cola YA (no espera a que venza el lease) y otro worker
        # lo retoma desde el último checkpoint.
        await jobs.devolver_a_la_cola(job_id, f"apagado ordenado de {nombre}: se retoma desde el último "
                                              "checkpoint", solo_si_vencio=False)
        logger.info(f"[{nombre}] apagado: job {job_id[:8]} devuelto a la cola")
        raise
    except Exception as error:  # cualquier error del grafo, del LLM o de Redis: el job NO queda colgado
        logger.exception(f"[{nombre}] job {job_id[:8]} falló")
        await _fallar(jobs, job, f"{type(error).__name__}: {error}")
        return
    finally:
        latido.cancel()
        await jobs.sumar_duracion(job_id, time.perf_counter() - inicio)

    if snapshot.interrupts:
        pedido = snapshot.interrupts[0].value
        await jobs.actualizar(job_id, Estado.WAITING_APPROVAL, pedido["pregunta"], aprobacion=pedido)
        logger.info(f"[{nombre}] job {job_id[:8]} en pausa: espera aprobación humana")
        return
    await jobs.actualizar(
        job_id, Estado.DONE, "respuesta lista",
        resultado=resumir_resultado(snapshot.values),
        finalizado_en=ahora(), latencia_total_s=_latencia(job),
    )
    logger.info(f"[{nombre}] job {job_id[:8]} DONE en {time.perf_counter() - inicio:.1f} s")


async def _fallar(jobs: RepositorioJobs, job: dict, error: str) -> None:
    await jobs.actualizar(job["job_id"], Estado.FAILED, error, error=error,
                          finalizado_en=ahora(), latencia_total_s=_latencia(job))


def _latencia(job: dict) -> str:
    """Desde que se creó el job hasta ahora (incluye la espera en la cola y, si hubo, la espera del humano)."""
    return f"{time.time() - float(job['creado_ts']):.3f}"


# --- Bucles ---------------------------------------------------------------------------------
async def bucle(grafo: CompiledStateGraph, jobs: RepositorioJobs, nombre: str) -> None:
    """Un worker: saca mensajes de la cola para siempre. Un error inesperado no lo mata."""
    logger.info(f"[{nombre}] esperando jobs en la cola de Redis")
    while True:
        try:
            mensaje = await jobs.tomar()
            if mensaje is not None:
                await procesar(grafo, jobs, mensaje, nombre)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[{nombre}] error inesperado (¿se cayó Redis?); reintento en 2 s")
            await asyncio.sleep(2)


def intervalo_del_vigilante() -> float:
    return min(10.0, config.lease_s() / 2)


async def vigilar(jobs: RepositorioJobs, proceso: str, cantidad: int) -> None:
    """Latido del proceso + recuperación de jobs con lease vencido (de este o de otro worker)."""
    from orquestador.rag.servicio import modo_rag

    datos = {"jobs_en_paralelo": cantidad, "rag": modo_rag(), "llm": descripcion_del_llm(), "desde": ahora()}
    try:
        while True:
            try:
                await jobs.latido_de_worker(proceso, datos, ttl_s=int(intervalo_del_vigilante() * 3))
                recuperados = await jobs.recuperar_interrumpidos()
                if recuperados:
                    logger.info(f"[{proceso}] se retoman {len(recuperados)} jobs con lease vencido: "
                                f"{[j[:8] for j in recuperados]}")
            except Exception:
                logger.exception(f"[{proceso}] el vigilante no pudo revisar los leases; reintento")
            await asyncio.sleep(intervalo_del_vigilante())
    finally:  # apagado ordenado: deja de figurar en GET /health
        await jobs.baja_de_worker(proceso)


def nombre_del_proceso() -> str:
    """En docker, el hostname es el ID del contenedor: distingue a cada réplica de --scale."""
    return f"{socket.gethostname()}-{os.getpid()}"


def lanzar_workers(grafo: CompiledStateGraph, jobs: RepositorioJobs, cantidad: int,
                   proceso: str | None = None) -> list[asyncio.Task]:
    proceso = proceso or nombre_del_proceso()
    return [
        *(asyncio.create_task(bucle(grafo, jobs, f"{proceso}/w{i + 1}")) for i in range(cantidad)),
        asyncio.create_task(vigilar(jobs, proceso, cantidad)),
    ]


async def main() -> None:
    """Worker como proceso separado: python -m app.worker"""
    from app.graph import checkpointer_redis, crear_orquestador
    from app.observability import apagar_observabilidad, iniciar_observabilidad
    from orquestador.rag.servicio import preparar_rag

    iniciar_observabilidad("worker")
    redis = conectar(config.redis_url())
    await redis.ping()
    jobs = RepositorioJobs(redis)
    try:
        await preparar_rag()  # indexa las fichas si hace falta (un solo proceso indexa: lock en Redis)
        async with checkpointer_redis(config.redis_url()) as checkpointer:
            tareas = lanzar_workers(crear_orquestador(checkpointer), jobs, config.workers())
            loop = asyncio.get_running_loop()
            for senal in (signal.SIGTERM, signal.SIGINT):
                try:  # docker stop manda SIGTERM: se cancelan las tareas y cada job en curso vuelve a la cola
                    loop.add_signal_handler(senal, lambda: [t.cancel() for t in tareas])
                except NotImplementedError:  # Windows: Ctrl+C llega como KeyboardInterrupt y asyncio.run cancela
                    pass
            await asyncio.gather(*tareas, return_exceptions=True)
    finally:
        await redis.aclose()
        apagar_observabilidad()
        logger.info("Worker detenido")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
