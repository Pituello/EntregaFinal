"""
El grafo del orquestador (topología jerárquica, patrón Supervisor), el de la
Pre-Entrega 6 más la rama de reservas con aprobación humana (Pre-Entrega 7):

    START -> supervisor --(Literal)--> investigador -> validador -> supervisor
                        \\-----------> analista     -> validador -> supervisor
                        \\-----------> reservas     -> validador -> aprobacion_humana (interrupt)
                        \\                                            |-- aprobada  -> ejecutar_reserva -> supervisor
                        \\                                            '-- rechazada -> supervisor
                         \\----------> sintesis -> END

- Se compila con un checkpointer de Redis (AsyncRedisSaver): el estado se
  guarda en cada paso bajo thread_id = conversacion_id (entrega final; en la
  Pre-Entrega 7 era el job_id). Eso permite pausar en aprobacion_humana y
  reanudar más tarde, y que una conversación siga después de reiniciar todo:
  cada turno corre sobre el mismo hilo y la síntesis agrega el turno al
  historial. La acción crítica (ejecutar_reserva) solo es alcanzable desde
  aprobacion_humana: ver app/hitl.py.

- Los especialistas nunca hablan entre sí: todo pasa por el supervisor.
- Todo lo que produce un especialista pasa por el VALIDADOR determinístico
  antes de volver al supervisor (la "puerta de calidad").
- El supervisor decide con dos capas:
    1. Reglas determinísticas, que se evalúan primero y no gastan LLM: límite
       de pasos, bucle de reparación con el error específico, escalado a
       revisión humana y guardas de suficiencia.
    2. Si ninguna regla aplica, decide el LLM con su rúbrica
       (with_structured_output -> Literal de nombres de nodo).
  Cada decisión queda registrada con su origen ("regla" o "llm") y su razón.
- Aislamiento de escritura: con_parcela() verifica en cada paso que un nodo
  solo escriba las claves de su parcela (state.PARCELAS).
"""
import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel

from app.hitl import despues_de_aprobar, despues_de_validar, nodo_aprobacion_humana, nodo_ejecutar_reserva
from orquestador import config
from orquestador import prompts
from orquestador import state as modulo_estado
from orquestador.llm import es_error_transitorio, get_llm
from orquestador.state import (
    PARCELAS,
    Contribucion,
    DecisionRegistrada,
    DecisionSupervisor,
    EstadoOrquestador,
    ResultadoValidacion,
    Turno,
    analisis_rechazado,
)
from orquestador.validation import validar_analisis, validar_investigacion, validar_reserva

Nodo = Callable[[EstadoOrquestador, RunnableConfig], Awaitable[dict]]

# Reintento ante fallas TRANSITORIAS del LLM (503, rate limit por minuto).
# Una cuota diaria agotada o una API key inválida no se reintentan.
REINTENTO_LLM = RetryPolicy(initial_interval=2.0, backoff_factor=2.0, max_attempts=4, retry_on=es_error_transitorio)


class ParcelaInvadida(RuntimeError):
    """Un nodo intentó escribir una clave del estado que no es suya."""


def con_parcela(nombre: str, nodo: Nodo) -> Nodo:
    async def envuelto(state: EstadoOrquestador, config: RunnableConfig) -> dict:
        actualizacion = await nodo(state, config)
        ajenas = set(actualizacion) - PARCELAS[nombre]
        if ajenas:
            raise ParcelaInvadida(f"El nodo '{nombre}' intentó escribir fuera de su parcela: {sorted(ajenas)}")
        return actualizacion

    return envuelto


# --- Reglas determinísticas del supervisor -----------------------------------
def _instruccion_original(state: EstadoOrquestador, agente: str) -> str:
    """La última instrucción 'de fondo' que recibió el agente (no un prompt de reparación)."""
    for d in reversed(state["decisiones"]):
        if d.siguiente == agente and d.origen == "llm":
            return d.instruccion
    return state["tarea"]


def _aporte_anterior(state: EstadoOrquestador, agente: str) -> str:
    aporte = {"investigador": state["investigacion"], "analista": state["analisis"],
              "reservas": state["propuesta_reserva"]}[agente]
    return aporte.model_dump_json(indent=2) if aporte else "(no produjo un resultado)"


DATOS_PARA_ANALISTA = "datos_para_analista"  # clave en "reparaciones": la regla corre una vez por turno


def _buscar_datos_faltantes(state: EstadoOrquestador, val: ResultadoValidacion) -> DecisionSupervisor:
    """
    Al análisis le faltan datos validados: reparar al analista con los mismos
    datos no sirve (lo único que puede hacer es repetir el número o
    inventar otro). Se manda al investigador a buscarlos. Pasó en la corrida
    real: en el turno 2 de una conversación ("¿y cuántos CV más que ese tiene
    el Taos?"), el investigador trajo solo el Taos, el analista usó el 101 del
    Nivus que venía del turno 1, y dos reparaciones del analista terminaron en
    escalado a revisión humana.
    """
    numeros = ", ".join(f"{n:g}" for n in val.numeros_sin_respaldo)
    ya_validados = state["investigacion"].datos if state["investigacion"] else []
    instruccion = prompts.DATOS_FALTANTES.format(
        analisis=_instruccion_original(state, "analista"),
        numeros=numeros,
        ya_validados="; ".join(f"{d.descripcion} = {d.valor:g} {d.unidad}" for d in ya_validados) or "(ninguno)",
    )
    return DecisionSupervisor(
        siguiente="investigador", instruccion=instruccion,
        razon=f"Al análisis le faltan datos validados en este turno (los números {numeros} no provienen de la "
              "investigación): se manda al investigador a buscarlos en vez de reparar al analista con los "
              "mismos datos.",
    )


def decidir_por_reglas(state: EstadoOrquestador) -> tuple[DecisionSupervisor | None, dict]:
    """Devuelve (decisión, cambios extra al estado), o (None, {}) si decide el LLM."""
    if state["pasos"] >= config.max_pasos():
        return DecisionSupervisor(
            siguiente="FINALIZAR",
            razon=f"Se alcanzó el límite de {config.max_pasos()} intervenciones (MAX_PASOS): "
                  "se finaliza con lo obtenido para evitar un supervisor infinito.",
        ), {}

    decision = state.get("decision_humana")
    if decision is not None and not decision.aprobado:
        return DecisionSupervisor(
            siguiente="FINALIZAR",
            razon=f"El revisor humano ({decision.revisor}) rechazó la reserva: la acción crítica no se ejecuta "
                  "y se finaliza informando el rechazo.",
        ), {}

    val = state["validacion"]
    if val is not None and not val.ok and val.numeros_sin_respaldo and not state["reparaciones"].get(DATOS_PARA_ANALISTA):
        return _buscar_datos_faltantes(state, val), {
            "reparaciones": {**state["reparaciones"], DATOS_PARA_ANALISTA: 1}
        }

    if val is not None and not val.ok:
        agente = val.agente
        usadas = state["reparaciones"].get(agente, 0)
        if usadas >= config.max_reparaciones():
            return DecisionSupervisor(
                siguiente="FINALIZAR",
                razon=f"El {agente} no pasó la validación después de {usadas} reparaciones: se escala a "
                      "revisión humana en vez de seguir reintentando (evita el bucle infinito de corrección).",
            ), {"escalado_humano": True}

        original = _instruccion_original(state, agente)
        if val.fallo_de_ejecucion:
            instruccion = prompts.REINTENTO_POR_FALLO.format(error=" | ".join(val.errores), instruccion_original=original)
            razon = f"El {agente} falló al ejecutarse: se reintenta ({usadas + 1}/{config.max_reparaciones()})."
        else:
            instruccion = prompts.REPARACION.format(
                aporte_anterior=_aporte_anterior(state, agente),
                errores="\n".join(f"- {e}" for e in val.errores),
                instruccion_original=original,
            )
            razon = (f"Bucle de reparación ({usadas + 1}/{config.max_reparaciones()}): se reenvía al {agente} su "
                     "output con los errores específicos del validador, no un reintento ciego.")
        return DecisionSupervisor(siguiente=agente, instruccion=instruccion, razon=razon), {
            "reparaciones": {**state["reparaciones"], agente: usadas + 1}
        }

    return None, {}


def aplicar_guardas(decision: DecisionSupervisor, state: EstadoOrquestador) -> tuple[DecisionSupervisor, bool]:
    """Corrige decisiones del LLM que violan el criterio de suficiencia. Devuelve (decisión, si se corrigió)."""
    reserva_resuelta = state.get("decision_humana") is not None or state.get("resultado_reserva") is not None
    if decision.siguiente == "reservas" and reserva_resuelta:
        return DecisionSupervisor(
            siguiente="FINALIZAR",
            razon="Guarda: la reserva ya se resolvió (aprobada o rechazada por un humano); una acción crítica "
                  "no se repite por decisión del LLM.",
        ), True
    if decision.siguiente == "FINALIZAR" and analisis_rechazado(state) and state["investigacion"] is not None:
        return DecisionSupervisor(
            siguiente="analista",
            instruccion=_instruccion_original(state, "analista"),
            razon="Guarda: el último análisis no pasó la validación; no se finaliza con un resultado rechazado. "
                  "El analista lo rehace con la investigación validada actual.",
        ), True
    hubo_reserva = state.get("propuesta_reserva") is not None
    if state["investigacion"] is None and (
        decision.siguiente == "analista" or (decision.siguiente == "FINALIZAR" and not hubo_reserva)
    ):
        return DecisionSupervisor(
            siguiente="investigador",
            instruccion=f"Buscá todos los datos necesarios para resolver esta tarea: {state['tarea']}",
            razon=f"Guarda: el LLM eligió '{decision.siguiente}' sin ninguna investigación hecha; "
                  "primero hacen falta datos.",
        ), True
    if decision.siguiente != "FINALIZAR" and not decision.instruccion.strip():
        return decision.model_copy(update={"instruccion": state["tarea"]}), True
    return decision, False


# --- Construcción del grafo ---------------------------------------------------
def _veces_que_corrio(state: EstadoOrquestador, agente: str) -> int:
    return sum(1 for c in state["contribuciones"] if c.agente == agente)


def construir_grafo(supervisor, investigador, analista, sintetizador, reservas=None) -> StateGraph:
    """Los componentes se inyectan: graph.py no sabe si son agentes reales o dobles de test."""

    async def nodo_supervisor(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        simulacion = config_.get("configurable", {}).get("simulacion", {})
        if simulacion.get("demora_s"):  # simula un LLM lento (para probar el timeout y la concurrencia)
            await asyncio.sleep(min(float(simulacion["demora_s"]), 600))
        if simulacion.get("romper_grafo"):
            # Excepción NO controlada (a diferencia de las de los especialistas): prueba que el worker
            # la captura y deja el job en FAILED en vez de dejar al cliente esperando para siempre.
            raise RuntimeError("Falla simulada no controlada del grafo (prueba del estado FAILED)")
        decision, extra = decidir_por_reglas(state)
        origen = "regla"
        if decision is None:
            decision = await supervisor.decidir(state)
            decision, corregida = aplicar_guardas(decision, state)
            origen = "regla" if corregida else "llm"

        actualizacion = {
            "siguiente": decision.siguiente,
            "instruccion": decision.instruccion,
            "decisiones": [DecisionRegistrada(**decision.model_dump(), paso=state["pasos"], origen=origen)],
            **extra,
        }
        if decision.siguiente != "FINALIZAR":
            actualizacion["pasos"] = state["pasos"] + 1
        return actualizacion

    async def nodo_investigador(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        inicio, paso = time.perf_counter(), state["pasos"]
        simulacion = config_.get("configurable", {}).get("simulacion", {})
        try:
            if _veces_que_corrio(state, "investigador") < simulacion.get("fallar_investigador", 0):
                raise RuntimeError("Fallo simulado del investigador (prueba del camino de error)")
            aporte, evidencia = await investigador.ejecutar(state["instruccion"])
        except Exception as e:
            if es_error_transitorio(e):
                raise  # lo reintenta la RetryPolicy del nodo
            return {"contribuciones": [Contribucion(
                paso=paso, agente="investigador", ok=False, resumen="(falló)",
                error=f"{type(e).__name__}: {e}", duracion_s=time.perf_counter() - inicio)]}
        return {
            "investigacion": aporte,
            "evidencia": evidencia,
            "contribuciones": [Contribucion(paso=paso, agente="investigador", ok=True, resumen=aporte.resumen,
                                            duracion_s=time.perf_counter() - inicio)],
            "messages": [AIMessage(aporte.resumen, name="investigador")],
        }

    async def nodo_analista(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        inicio, paso = time.perf_counter(), state["pasos"]
        simulacion = config_.get("configurable", {}).get("simulacion", {})
        try:
            aporte = await analista.ejecutar(state["instruccion"], state["investigacion"])
        except Exception as e:
            if es_error_transitorio(e):
                raise
            return {"contribuciones": [Contribucion(
                paso=paso, agente="analista", ok=False, resumen="(falló)",
                error=f"{type(e).__name__}: {e}", duracion_s=time.perf_counter() - inicio)]}
        resumen = aporte.conclusion
        if _veces_que_corrio(state, "analista") < simulacion.get("corromper_analisis", 0):
            # Simula un error aritmético del LLM ("suena bien pero está mal") para probar la puerta de calidad.
            # Se arma un objeto NUEVO: mutar el aporte en el lugar es estado mutable compartido (si
            # alguien más tiene una referencia, también lo vería alterado).
            primero = aporte.calculos[0].model_copy(update={"resultado": aporte.calculos[0].resultado + 10})
            aporte = aporte.model_copy(update={"calculos": [primero, *aporte.calculos[1:]]})
            resumen += " [SIMULACIÓN: resultado alterado a propósito para probar el validador]"
        return {
            "analisis": aporte,
            "contribuciones": [Contribucion(paso=paso, agente="analista", ok=True, resumen=resumen,
                                            duracion_s=time.perf_counter() - inicio)],
            "messages": [AIMessage(aporte.conclusion, name="analista")],
        }

    async def nodo_reservas(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        inicio, paso = time.perf_counter(), state["pasos"]
        try:
            if reservas is None:
                raise RuntimeError("no hay un gestor de reservas configurado")
            propuesta = await reservas.ejecutar(state["instruccion"])
        except Exception as e:
            if es_error_transitorio(e):
                raise
            return {"contribuciones": [Contribucion(
                paso=paso, agente="reservas", ok=False, resumen="(falló)",
                error=f"{type(e).__name__}: {e}", duracion_s=time.perf_counter() - inicio)]}
        return {
            "propuesta_reserva": propuesta,
            "contribuciones": [Contribucion(paso=paso, agente="reservas", ok=True, resumen=propuesta.resumen,
                                            duracion_s=time.perf_counter() - inicio)],
            "messages": [AIMessage(f"Reserva propuesta: {propuesta.resumen}", name="reservas")],
        }

    async def nodo_validador(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        agente = state["siguiente"]
        ultima = next((c for c in reversed(state["contribuciones"]) if c.agente == agente), None)
        if ultima is None or not ultima.ok:
            resultado = ResultadoValidacion(
                agente=agente, ok=False, fallo_de_ejecucion=True,
                errores=[ultima.error if ultima else "el especialista no produjo ningún resultado"],
            )
        elif agente == "investigador":
            resultado = validar_investigacion(state["investigacion"], state["evidencia"])
        elif agente == "reservas":
            resultado = validar_reserva(state["propuesta_reserva"])
        else:
            resultado = validar_analisis(state["analisis"], state["investigacion"])
        return {"validacion": resultado, "validaciones": [resultado]}

    async def nodo_sintesis(state: EstadoOrquestador, config_: RunnableConfig) -> dict:
        respuesta = await sintetizador.redactar(state)
        turno = Turno(job_id=state.get("job_id", ""), pregunta=state["tarea"], respuesta=respuesta)
        return {"respuesta_final": respuesta, "messages": [AIMessage(respuesta, name="sintesis")],
                "historial": [turno]}

    def enrutar(state: EstadoOrquestador) -> Literal["investigador", "analista", "reservas", "sintesis"]:
        return "sintesis" if state["siguiente"] == "FINALIZAR" else state["siguiente"]

    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("supervisor", con_parcela("supervisor", nodo_supervisor), retry_policy=REINTENTO_LLM)
    grafo.add_node("investigador", con_parcela("investigador", nodo_investigador), retry_policy=REINTENTO_LLM)
    grafo.add_node("analista", con_parcela("analista", nodo_analista), retry_policy=REINTENTO_LLM)
    grafo.add_node("reservas", con_parcela("reservas", nodo_reservas), retry_policy=REINTENTO_LLM)
    grafo.add_node("validador", con_parcela("validador", nodo_validador))
    grafo.add_node("aprobacion_humana", con_parcela("aprobacion_humana", nodo_aprobacion_humana))
    grafo.add_node("ejecutar_reserva", con_parcela("ejecutar_reserva", nodo_ejecutar_reserva))
    grafo.add_node("sintesis", con_parcela("sintesis", nodo_sintesis), retry_policy=REINTENTO_LLM)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar, ["investigador", "analista", "reservas", "sintesis"])
    grafo.add_edge("investigador", "validador")
    grafo.add_edge("analista", "validador")
    grafo.add_edge("reservas", "validador")
    # Pre-Entrega 7: una propuesta de reserva validada no vuelve al supervisor, va a la aprobación humana.
    grafo.add_conditional_edges("validador", despues_de_validar, ["aprobacion_humana", "supervisor"])
    grafo.add_conditional_edges("aprobacion_humana", despues_de_aprobar, ["ejecutar_reserva", "supervisor"])
    grafo.add_edge("ejecutar_reserva", "supervisor")
    grafo.add_edge("sintesis", END)
    return grafo


def crear_orquestador(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """
    Orquestador con los agentes reales, usando el LLM configurado en el .env.
    En la API se compila con AsyncRedisSaver (ver app/main.py y app/worker.py).
    """
    from orquestador.agents.analyst_agent import AgenteAnalista
    from orquestador.agents.research_agent import AgenteInvestigador
    from orquestador.agents.reservations_agent import AgenteReservas
    from orquestador.agents.supervisor import Supervisor
    from orquestador.agents.synthesis_agent import Sintetizador

    llm = get_llm()
    grafo = construir_grafo(Supervisor(llm), AgenteInvestigador(llm), AgenteAnalista(llm), Sintetizador(llm),
                            reservas=AgenteReservas(llm))
    return grafo.compile(checkpointer=checkpointer)


def config_de_ejecucion(simulacion: dict | None = None, thread_id: str | None = None,
                        job_id: str | None = None) -> RunnableConfig:
    """
    recursion_limit (techo del grafo) + simulación opcional de fallos para probar el camino de error.
    thread_id: con checkpointer, identifica el hilo (en la API, la conversación).
    job_id: el turno en curso; es la clave de idempotencia de la reserva (una conversación puede
    tener varias reservas, una por turno, pero un mismo turno nunca reserva dos veces).
    """
    configurable = {"simulacion": simulacion or {}}
    if thread_id:
        configurable["thread_id"] = thread_id
    if job_id:
        configurable["job_id"] = job_id
    return {"recursion_limit": config.recursion_limit(), "configurable": configurable}


# --- Checkpointer de Redis (Pre-Entrega 7) -------------------------------------
def tipos_del_estado() -> list[tuple[str, str]]:
    """
    Los modelos Pydantic del estado (DecisionRegistrada, PropuestaReserva...).
    El checkpointer los guarda en Redis y, al reanudar, los vuelve a construir.
    LangGraph pide declararlos explícitamente: deserializar cualquier clase que
    venga de la base de datos sería un riesgo de seguridad.
    """
    return [
        (obj.__module__, obj.__name__) for obj in vars(modulo_estado).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == modulo_estado.__name__
    ]


@asynccontextmanager
async def checkpointer_redis(redis_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """AsyncRedisSaver listo para usar: crea sus índices en Redis la primera vez (asetup)."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        tipos = tipos_del_estado()
        saver.serde = JsonPlusRedisSerializer(
            allowed_msgpack_modules=tipos,
            # en JSON la clase se guarda como ruta partida: ["orquestador", "state", "PropuestaReserva"]
            allowed_json_modules=[(*modulo.split("."), nombre) for modulo, nombre in tipos],
        )
        await saver.asetup()
        yield saver
