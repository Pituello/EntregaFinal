"""
Agente Gestor de reservas (Pre-Entrega 7): PREPARA la reserva de una unidad.

Tiene una sola herramienta, y es de solo lectura (consultar_catalogo): la usa
para encontrar el version_id exacto y verificar el stock. La herramienta que
reserva de verdad (reservar_unidad) NO está en su lista: la ejecuta el nodo
ejecutar_reserva del grafo, y a ese nodo solo se llega con la aprobación de
una persona. Así, aunque el LLM se equivoque o "se entusiasme", no puede
comprometer stock por su cuenta.
"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from orquestador import prompts
from orquestador.state import PropuestaReserva
from orquestador.tools.catalogo import consultar_catalogo
from orquestador.trazado import agente

LIMITE_PASOS_INTERNOS = 8


class AgenteReservas:
    nombre = "reservas"

    def __init__(self, llm: BaseChatModel):
        self._agente = create_agent(
            model=llm,
            tools=[consultar_catalogo],
            system_prompt=prompts.GESTOR_RESERVAS,
            response_format=PropuestaReserva,
        )

    @agente("gestor_reservas")
    async def ejecutar(self, instruccion: str) -> PropuestaReserva:
        resultado = await self._agente.ainvoke(
            {"messages": [HumanMessage(instruccion)]},
            config={"recursion_limit": LIMITE_PASOS_INTERNOS},
        )
        return resultado["structured_response"]
