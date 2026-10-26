"""
Decorador para trazar a cada agente como un span propio en Phoenix.

    @agente("investigador")
    async def ejecutar(self, instruccion: str) -> ...:

El instrumentador de LangChain (app/observability.py) registra solo cada nodo
del grafo, cada llamada al modelo y cada herramienta, pero arma la jerarquía
con el árbol de ejecuciones de LangChain, no con el contexto de OpenTelemetry.
Por eso un decorador de OpenTelemetry "por fuera" quedaba colgado del span
raíz, vacío y separado de sus propias llamadas al LLM (pasó en las pruebas).

Este decorador envuelve la función en un Runnable de LangChain llamado
"agente_<nombre>". Así entra en el mismo árbol: las llamadas al modelo y a las
herramientas que hace adentro quedan como hijas, y OpenInference lo marca
como AGENT porque el nombre contiene "agent". En Phoenix se lee:

    investigador (nodo) -> agente_investigador (AGENT) -> ChatOpenAI (LLM)
                                                       -> consultar_catalogo (TOOL)

Sin Phoenix (por ejemplo, en los tests) funciona igual: es solo un Runnable.
"""
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.runnables import RunnableLambda

R = TypeVar("R")


def agente(nombre: str) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    def decorar(funcion: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(funcion)
        async def envuelta(*args: Any, **kwargs: Any) -> R:
            async def llamar(_entrada: Any) -> R:
                return await funcion(*args, **kwargs)

            # La entrada visible del span es el primer texto que recibe la función (instrucción o resumen).
            entrada = next((a for a in [*args, *kwargs.values()] if isinstance(a, str)), "")
            return await RunnableLambda(llamar, name=f"agente_{nombre}").ainvoke(entrada)

        return envuelta

    return decorar
