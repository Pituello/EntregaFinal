"""
Agente Investigador: busca datos con herramientas acotadas a su rol
(consultar_catalogo y buscar_en_fichas) y devuelve un AporteInvestigacion
validado por Pydantic (response_format de create_agent).

Contexto acotado (error a evitar "Contaminación de Contexto"): recibe SOLO la
instrucción puntual del supervisor, no la conversación ni la metadata del
sistema. También devuelve la "evidencia": las salidas crudas de sus
herramientas, contra las que el Validador chequea cada dato.
"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from orquestador import prompts
from orquestador.state import AporteInvestigacion
from orquestador.tools.busqueda import buscar_en_fichas
from orquestador.tools.catalogo import consultar_catalogo
from orquestador.trazado import agente

HERRAMIENTAS = [consultar_catalogo, buscar_en_fichas]
LIMITE_PASOS_INTERNOS = 12  # techo del ciclo ReAct interno del especialista


class AgenteInvestigador:
    nombre = "investigador"

    def __init__(self, llm: BaseChatModel):
        self._agente = create_agent(
            model=llm,
            tools=HERRAMIENTAS,
            system_prompt=prompts.INVESTIGADOR,
            response_format=AporteInvestigacion,
        )
        self._nombres_herramientas = {h.name for h in HERRAMIENTAS}

    @agente("investigador")
    async def ejecutar(self, instruccion: str) -> tuple[AporteInvestigacion, list[str]]:
        resultado = await self._agente.ainvoke(
            {"messages": [HumanMessage(instruccion)]},
            config={"recursion_limit": LIMITE_PASOS_INTERNOS},
        )
        evidencia = [
            str(m.content) for m in resultado["messages"]
            if isinstance(m, ToolMessage) and m.name in self._nombres_herramientas
        ]
        return resultado["structured_response"], evidencia
