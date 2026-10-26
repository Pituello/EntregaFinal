"""
Herramienta del Investigador: búsqueda en las fichas técnicas de los 7
modelos, con el recuperador híbrido de orquestador/rag (BM25 + vectores en
Redis + metadatos, fusionados con RRF).

Entrada y salida validadas con Pydantic (entrega final: "contratos de datos en
cada frontera"): BuscarEnFichasInput es el esquema que ve el LLM, y la salida
se arma desde ResultadoBusqueda antes de pasarla a texto. El formato del
texto, "[Fuente: archivo.md]" antes de cada fragmento, es el que lee el
Validador de la Pre-Entrega 6 para comprobar de dónde sale cada número.
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from orquestador.rag.servicio import obtener_recuperador


class BuscarEnFichasInput(BaseModel):
    consulta: str = Field(min_length=3, max_length=300,
                          description="Qué buscar, incluyendo el modelo, ej: 'Latin NCAP ocupante adulto Tera'")


class FragmentoRecuperado(BaseModel):
    fuente: str = Field(pattern=r"^[a-z0-9_\-]+\.md$")
    seccion: str
    texto: str = Field(min_length=1)


class ResultadoBusqueda(BaseModel):
    modo: str = Field(description="hibrido o solo_bm25")
    fragmentos: list[FragmentoRecuperado]

    def como_texto(self) -> str:
        if not self.fragmentos:
            return "ERROR: no se encontró nada relevante en las fichas. Probá con otras palabras."
        return "\n\n".join(f"[Fuente: {f.fuente}]\n{f.texto}" for f in self.fragmentos)


@tool(args_schema=BuscarEnFichasInput)
async def buscar_en_fichas(consulta: str) -> str:
    """Busca en las fichas técnicas de los modelos Volkswagen (Tera, Polo, Virtus, Nivus,
    Taos, Vento, Saveiro): seguridad y puntajes Latin NCAP, dimensiones, historia, plantas
    de fabricación, colores, rivales. Incluí el nombre del modelo en la consulta. Para
    versiones, potencia, torque o stock usá consultar_catalogo.
    Devuelve los fragmentos más relevantes, cada uno precedido por [Fuente: archivo]."""
    documentos = await obtener_recuperador().ainvoke(consulta)
    resultado = ResultadoBusqueda(
        modo=documentos[0].metadata["modo"] if documentos else "sin resultados",
        fragmentos=[FragmentoRecuperado(fuente=d.metadata["fuente"], seccion=d.metadata["seccion"],
                                        texto=d.page_content) for d in documentos],
    )
    return resultado.como_texto()
