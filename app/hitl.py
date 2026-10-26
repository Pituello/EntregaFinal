"""
Human-in-the-loop (HITL): pausa obligatoria antes de una acción crítica.

Qué es "crítico" en este sistema: ejecutar una herramienta con EFECTO
SECUNDARIO. Reservar una unidad compromete stock real (si sale mal, otro
cliente se queda sin el auto), así que nunca la decide sola una IA.

Cómo se garantiza la pausa (no depende de que el LLM "se acuerde"):
1. El Gestor de reservas solo PREPARA la reserva; la herramienta que reserva
   no está entre sus herramientas.
2. El nodo que la ejecuta (ejecutar_reserva) solo es alcanzable desde
   aprobacion_humana: es una propiedad de las aristas del grafo, no de un
   prompt.
3. aprobacion_humana llama a interrupt(): el grafo se detiene y su estado
   queda guardado en Redis (AsyncRedisSaver) bajo el hilo de la conversación. El
   proceso no queda esperando en memoria: puede pasar una hora, o reiniciarse
   la API, y la reserva sigue pausada.
4. POST /tasks/{id}/approve guarda la decisión y encola la reanudación; el
   worker reanuda el MISMO hilo con Command(resume=decision).

Detalle de la clase: al reanudar, el nodo se vuelve a ejecutar desde el
principio. Por eso aprobacion_humana no tiene efectos antes del interrupt():
la acción con efectos vive en el nodo siguiente.
"""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.esquemas import PedidoAprobacion, RespuestaAprobacion
from app.jobs import Estado
from orquestador.state import DecisionHumana, EstadoOrquestador, ResultadoReserva
from orquestador.tools.catalogo_datos import STOCK
from orquestador.tools.reservas import buscar_version, reservar_unidad, resolver_sucursal

# Herramientas que exigen aprobación humana, y por qué. Es la "lista de acciones críticas" del sistema.
ACCIONES_CRITICAS: dict[str, str] = {
    reservar_unidad.name: "Efecto secundario: compromete una unidad del stock real de la sucursal.",
}


def requiere_aprobacion(herramienta: str) -> bool:
    return herramienta in ACCIONES_CRITICAS


# --- Nodos del grafo ----------------------------------------------------------
def armar_pedido_de_aprobacion(state: EstadoOrquestador) -> dict:
    """Lo que ve el revisor: qué se va a hacer, por qué es crítico y con qué datos."""
    propuesta = state["propuesta_reserva"]
    version = buscar_version(propuesta.version_id)
    sucursal = resolver_sucursal(propuesta.sucursal) or propuesta.sucursal
    pedido = PedidoAprobacion(
        accion=reservar_unidad.name,
        motivo=ACCIONES_CRITICAS[reservar_unidad.name],
        pregunta=(f"¿Aprobás reservar 1 unidad de {version.modelo} {version.version} en la sucursal "
                  f"{sucursal} para {propuesta.cliente}?"),
        propuesta=propuesta,
        stock_en_catalogo=STOCK[version.version_id].get(sucursal, 0),
        tarea_original=state["tarea"],
        como_responder='POST /tasks/{job_id}/approve con {"aprobado": true|false, "revisor": "...", "comentario": "..."}',
    )
    return pedido.model_dump(mode="json")  # el checkpointer lo guarda como dict en Redis


async def nodo_aprobacion_humana(state: EstadoOrquestador, config: RunnableConfig) -> dict:
    # interrupt() detiene el grafo acá y le pasa el pedido al exterior (el worker lo guarda en el job).
    # Cuando se reanuda con Command(resume=...), interrupt() devuelve esa decisión y el nodo sigue.
    respuesta = interrupt(armar_pedido_de_aprobacion(state))
    return {"decision_humana": DecisionHumana.model_validate(respuesta)}


async def nodo_ejecutar_reserva(state: EstadoOrquestador, config: RunnableConfig) -> dict:
    """La acción crítica. Solo se llega acá con una aprobación humana registrada en el estado."""
    decision = state["decision_humana"]
    if decision is None or not decision.aprobado:  # defensa en profundidad: las aristas ya lo impiden
        raise RuntimeError("Se intentó ejecutar una reserva sin aprobación humana")
    propuesta = state["propuesta_reserva"]
    salida = await reservar_unidad.ainvoke({
        "version_id": propuesta.version_id,
        "sucursal": propuesta.sucursal,
        "cliente": propuesta.cliente,
        # idempotencia: si el nodo se re-ejecuta para el mismo job (turno), no se reserva dos veces.
        # Es el job_id y no el thread_id: el hilo es la conversación, que puede tener varias reservas.
        "clave_idempotencia": _clave_idempotencia(state, config),
    })
    if salida.startswith("ERROR"):
        resultado = ResultadoReserva(ok=False, detalle=salida.removeprefix("ERROR: "))
    else:
        reserva = json.loads(salida)
        resultado = ResultadoReserva(
            ok=True, reserva_id=reserva["reserva_id"],
            detalle=f"Reserva {reserva['reserva_id']} confirmada: {reserva['modelo']} {reserva['version']} en "
                    f"{reserva['sucursal']} para {reserva['cliente']} (aprobada por {decision.revisor}).",
        )
    return {"resultado_reserva": resultado, "messages": [AIMessage(resultado.detalle, name="ejecutar_reserva")]}


def _clave_idempotencia(state: EstadoOrquestador, config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    return configurable.get("job_id") or state.get("job_id") or configurable.get("thread_id", "sin-thread")


def despues_de_validar(state: EstadoOrquestador) -> Literal["aprobacion_humana", "supervisor"]:
    """Una propuesta de reserva VALIDADA va a la aprobación humana; todo lo demás vuelve al supervisor."""
    val = state["validacion"]
    return "aprobacion_humana" if val is not None and val.agente == "reservas" and val.ok else "supervisor"


def despues_de_aprobar(state: EstadoOrquestador) -> Literal["ejecutar_reserva", "supervisor"]:
    decision = state["decision_humana"]
    return "ejecutar_reserva" if decision is not None and decision.aprobado else "supervisor"


# --- Endpoint de aprobación -----------------------------------------------------
router = APIRouter(tags=["human-in-the-loop"])


@router.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=RespuestaAprobacion,
             responses={404: {"description": "No existe el job"},
                        409: {"description": "El job no está esperando aprobación"}})
async def aprobar(job_id: str, decision: DecisionHumana, request: Request) -> RespuestaAprobacion:
    """
    Registra la decisión del revisor y encola la reanudación del grafo.
    Responde al instante: la reserva (si se aprobó) la ejecuta el worker.
    """
    jobs = request.app.state.jobs
    job = await jobs.obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job '{job_id}'")
    # Atómico en Redis: solo se acepta si el job está esperando aprobación (evita aprobar dos veces).
    if not await jobs.registrar_decision(job_id, decision.model_dump()):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en estado {job['estado']}: solo se puede aprobar uno en {Estado.WAITING_APPROVAL}",
        )
    return RespuestaAprobacion(job_id=job_id, estado=Estado.PENDING, decision=decision,
                               detalle="Decisión registrada: un worker reanuda el grafo desde el punto de pausa.")
