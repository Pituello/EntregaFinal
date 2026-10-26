"""
Síntesis final: redacta la respuesta al usuario con lo validado. Solo se
ejecuta cuando el supervisor decide FINALIZAR (o cuando un límite lo fuerza).

Cadena LCEL: plantilla | modelo | StrOutputParser. El parser normaliza la
salida a texto (Gemini, por ejemplo, puede devolver bloques en vez de un string).
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from orquestador import prompts
from orquestador.agents.supervisor import resumir_estado
from orquestador.trazado import agente


PLANTILLA = ChatPromptTemplate.from_messages([("system", prompts.SINTESIS), ("human", "{estado}")])


class Sintetizador:
    def __init__(self, llm: BaseChatModel):
        self._cadena = PLANTILLA | llm | StrOutputParser()

    async def redactar(self, estado: dict) -> str:
        extra = []
        if estado.get("escalado_humano"):
            extra.append("ATENCIÓN: el proceso se escaló a REVISIÓN HUMANA porque un resultado no pasó la "
                         "validación después de los reintentos permitidos.")
        val = estado.get("validacion")
        if val and not val.ok:
            extra.append("La última validación FALLÓ: " + " | ".join(val.errores))
        return await self._redactar(resumir_estado(estado) + ("\n\n" + "\n".join(extra) if extra else ""))

    @agente("sintesis")
    async def _redactar(self, mensaje: str) -> str:
        return (await self._cadena.ainvoke({"estado": mensaje})).strip()
