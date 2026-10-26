"""
Agente Analista: hace cálculos con una única herramienta (la calculadora
segura) y devuelve un AporteAnalisis validado por Pydantic.

Contexto acotado: recibe la instrucción del supervisor y SOLO los datos
validados de la investigación (no los fragmentos crudos de las fichas ni la
conversación).
"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from orquestador import prompts
from orquestador.state import AporteAnalisis, AporteInvestigacion
from orquestador.tools.calculadora import calculadora
from orquestador.trazado import agente

LIMITE_PASOS_INTERNOS = 12


class AgenteAnalista:
    nombre = "analista"

    def __init__(self, llm: BaseChatModel):
        self._agente = create_agent(
            model=llm,
            tools=[calculadora],
            system_prompt=prompts.ANALISTA,
            response_format=AporteAnalisis,
        )

    @staticmethod
    def armar_mensaje(instruccion: str, investigacion: AporteInvestigacion | None) -> str:
        datos = "\n".join(
            f"- {d.descripcion}: {d.valor:g} {d.unidad} (fuente: {d.fuente})"
            for d in (investigacion.datos if investigacion else [])
        ) or "(no hay datos)"
        return f"{instruccion}\n\nDatos validados disponibles:\n{datos}"

    async def ejecutar(self, instruccion: str, investigacion: AporteInvestigacion | None) -> AporteAnalisis:
        return await self._ejecutar(self.armar_mensaje(instruccion, investigacion))

    @agente("analista")
    async def _ejecutar(self, mensaje: str) -> AporteAnalisis:
        resultado = await self._agente.ainvoke(
            {"messages": [HumanMessage(mensaje)]},
            config={"recursion_limit": LIMITE_PASOS_INTERNOS},
        )
        return resultado["structured_response"]
