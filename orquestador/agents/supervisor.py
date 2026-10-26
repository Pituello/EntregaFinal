"""
Supervisor: el router del sistema. Decide quién interviene, con qué
instrucción y por qué, usando with_structured_output(DecisionSupervisor):
la salida es un Literal de los nombres de nodo, así el ruteo nunca depende de
interpretar texto libre.

Ve un RESUMEN del estado (datos validados, cálculos, errores), no las salidas
crudas de las herramientas: así su prompt no crece con cada paso. En una
conversación ve además los últimos turnos (pregunta y respuesta), recortados,
para resolver referencias como "¿y el Nivus?" o "reservame ese".

Es una cadena LCEL (prompt | modelo con salida estructurada): el prompt es
una plantilla y no un string armado a mano, y la cadena es un Runnable más
del árbol de LangChain (se ve en Phoenix como ChatPromptTemplate -> LLM).
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from orquestador import prompts
from orquestador.state import TURNOS_DE_CONTEXTO, DecisionSupervisor, analisis_rechazado
from orquestador.trazado import agente

LARGO_RESPUESTA_EN_CONTEXTO = 400


def resumir_conversacion(estado: dict) -> list[str]:
    turnos = (estado.get("historial") or [])[-TURNOS_DE_CONTEXTO:]
    if not turnos:
        return []
    lineas = ["CONVERSACIÓN PREVIA (turnos anteriores, del más viejo al más nuevo):"]
    for t in turnos:
        respuesta = t.respuesta if len(t.respuesta) <= LARGO_RESPUESTA_EN_CONTEXTO \
            else t.respuesta[:LARGO_RESPUESTA_EN_CONTEXTO] + "…"
        lineas += [f"  Usuario: {t.pregunta}", f"  Respuesta: {respuesta}"]
    return [*lineas, ""]


def resumir_estado(estado: dict) -> str:
    lineas = [*resumir_conversacion(estado), f"TAREA: {estado['tarea']}", ""]

    inv = estado.get("investigacion")
    if inv:
        lineas.append(f"INVESTIGACIÓN: {inv.resumen}")
        lineas += [f"  - {d.descripcion}: {d.valor:g} {d.unidad} (fuente: {d.fuente})" for d in inv.datos]
    else:
        lineas.append("INVESTIGACIÓN: (todavía no hay)")

    an = estado.get("analisis")
    if an and analisis_rechazado(estado):
        # Lo ven el supervisor y la síntesis: un análisis rechazado no puede leerse como resultado.
        # Pasó en la corrida real: después de buscar los datos faltantes, el LLM finalizó y la
        # síntesis respondió con la cuenta rechazada ("150 CV más", que era 150 - 0).
        lineas.append(f"ANÁLISIS RECHAZADO POR EL VALIDADOR (no es un resultado; hay que rehacerlo): {an.conclusion}")
    elif an:
        lineas.append(f"ANÁLISIS: {an.conclusion}")
        lineas += [f"  - {c.descripcion}: {c.expresion} = {c.resultado:g} {c.unidad}" for c in an.calculos]
    else:
        lineas.append("ANÁLISIS: (todavía no hay)")

    propuesta = estado.get("propuesta_reserva")
    if propuesta:
        lineas.append(f"RESERVA PROPUESTA: {propuesta.resumen} (versión {propuesta.version_id}, "
                      f"sucursal {propuesta.sucursal}, cliente {propuesta.cliente})")
    decision = estado.get("decision_humana")
    if decision:
        veredicto = "APROBADA" if decision.aprobado else "RECHAZADA"
        lineas.append(f"REVISIÓN HUMANA: {veredicto} por {decision.revisor}"
                      + (f" — comentario: {decision.comentario}" if decision.comentario else ""))
    resultado = estado.get("resultado_reserva")
    if resultado:
        lineas.append(f"RESERVA EJECUTADA: {'OK' if resultado.ok else 'NO SE PUDO'} — {resultado.detalle}")

    val = estado.get("validacion")
    if val:
        estado_val = "OK" if val.ok else "FALLÓ: " + " | ".join(val.errores)
        lineas.append(f"ÚLTIMA VALIDACIÓN ({val.agente}): {estado_val}")

    lineas.append(f"PASOS USADOS: {estado.get('pasos', 0)}")
    return "\n".join(lineas)


PLANTILLA = ChatPromptTemplate.from_messages([("system", prompts.SUPERVISOR), ("human", "{resumen}")])


class Supervisor:
    def __init__(self, llm: BaseChatModel):
        self._cadena = PLANTILLA | llm.with_structured_output(DecisionSupervisor)

    async def decidir(self, estado: dict) -> DecisionSupervisor:
        return await self._decidir(resumir_estado(estado))

    @agente("supervisor")
    async def _decidir(self, resumen: str) -> DecisionSupervisor:
        # El span de Phoenix muestra como entrada el resumen que vio el supervisor y como salida su decisión.
        return await self._cadena.ainvoke({"resumen": resumen})
