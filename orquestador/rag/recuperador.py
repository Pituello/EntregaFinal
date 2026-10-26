"""
Recuperador híbrido de las fichas: el RAG de las Pre-Entregas 3 y 4 como
herramienta del Investigador.

Tres rankings, fusionados por posición con RRF (Reciprocal Rank Fusion):

1. BM25 (léxico): encuentra los términos exactos ("Latin NCAP", "baúl",
   "TSI"), con el tokenizador para español de la Pre-Entrega 4.
2. Vectorial global (semántico): encuentra lo que se dice con otras
   palabras ("¿qué tan seguro es?" -> la sección de Latin NCAP).
3. Vectorial filtrado por METADATOS: si la consulta nombra un modelo, una
   búsqueda vectorial restringida a los fragmentos de ese modelo
   (filtro TAG "modelo" en Redis). Reduce alucinaciones: el contexto del
   Nivus ya no se llena de fragmentos del Tera que lo mencionan de pasada.

RRF combina por POSICIÓN, no por puntaje: BM25 y la distancia coseno están en
escalas distintas. Después, un re-ranking suave refuerza los fragmentos del
modelo nombrado. Es un refuerzo y no un filtro duro, porque en la Pre-Entrega
4 el filtro duro fallaba con "el sedán del Polo", cuya respuesta está en la
ficha del Virtus (por eso el ranking 2 sigue siendo global).

Es un BaseRetriever de LangChain: en Phoenix aparece como un span RETRIEVER
con los documentos que devolvió y sus metadatos (la clase advierte que no
instrumentar la base vectorial deja una "mancha ciega" en la traza).

Si no hay embeddings (EMBEDDINGS_PROVIDER=ninguno) o fallan, degrada a solo
BM25 y lo deja registrado en los metadatos: el sistema sigue respondiendo.
"""
import re
import unicodedata
from functools import cached_property
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from orquestador.rag.documentos import Fragmento

RRF_C = 60
CANDIDATOS_POR_RANKING = 8
# Re-ranking: multiplica el puntaje RRF de los fragmentos del modelo que nombra la consulta.
REFUERZO_MODELO_NOMBRADO = 1.5

_STOPWORDS = set(
    "el la los las un una unos unas de del al a en y o que con por para es son se su sus como cual cuales "
    "cuanto cuantos cuanta cuantas tiene tienen hay le lo me mi te tu mas pero sin sobre entre este esta".split()
)


def tokenizar(texto: str) -> list[str]:
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"\w+", sin_tildes) if t not in _STOPWORDS]


# Peso de cada ranking en la fusión, calibrado con scripts/evaluar_rag.py (20 preguntas, k=5):
# pesos iguales y refuerzo 1.2-1.5 dieron el mejor resultado (acierto@5 contra 75 % de BM25 solo:
# 95 % con un modelo local, 85 % con text-embedding-3-small en la corrida real). Ninguna
# combinación recupera la trampa "sedán del Polo" sin perder aciertos en el resto:
# limitación conocida, documentada en el README.
PESOS = {"bm25": 1.0, "vectorial": 1.0, "vectorial_modelo": 1.0}


def fusionar_rrf(rankings: dict[str, list[str]], c: int = RRF_C, pesos: dict[str, float] | None = None) -> dict[str, float]:
    """{nombre del ranking: [ids en orden]} -> {id: puntaje RRF ponderado}."""
    pesos = pesos or PESOS
    puntos: dict[str, float] = {}
    for nombre, ranking in rankings.items():
        peso = pesos.get("vectorial_modelo" if nombre.startswith("vectorial_") else nombre, 1.0)
        for posicion, item in enumerate(ranking, start=1):
            puntos[item] = puntos.get(item, 0.0) + peso / (c + posicion)
    return puntos


class RecuperadorHibrido(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fragmentos: tuple[Fragmento, ...]
    k: int = 5
    indice: Any = None  # un IndiceVectorial (IndiceRedis o IndiceEnMemoria); Pydantic no valida Protocols
    embeddings: Embeddings | None = None

    @cached_property
    def _por_id(self) -> dict[str, Fragmento]:
        return {f.id: f for f in self.fragmentos}

    @cached_property
    def _bm25(self) -> BM25Okapi:
        return BM25Okapi([tokenizar(f.texto) for f in self.fragmentos])

    @cached_property
    def _modelos(self) -> set[str]:
        return {f.modelo for f in self.fragmentos}

    def modelos_nombrados(self, consulta: str) -> set[str]:
        return self._modelos & set(tokenizar(consulta))

    def ranking_bm25(self, consulta: str, n: int = CANDIDATOS_POR_RANKING) -> list[str]:
        # BM25 es cálculo en memoria sobre ~70 fragmentos (microsegundos): no hace I/O ni bloquea el loop.
        puntajes = self._bm25.get_scores(tokenizar(consulta))
        orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
        return [self.fragmentos[i].id for i in orden[:n] if puntajes[i] > 0]

    async def rankings(self, consulta: str) -> tuple[dict[str, list[str]], str]:
        """Los rankings disponibles y el modo en que se buscó."""
        rankings = {"bm25": self.ranking_bm25(consulta)}
        if self.indice is None or self.embeddings is None:
            return rankings, "solo_bm25"
        try:
            vector = await self.embeddings.aembed_query(consulta)
            rankings["vectorial"] = await self.indice.buscar(vector, CANDIDATOS_POR_RANKING)
            nombrados = self.modelos_nombrados(consulta)
            if len(nombrados) == 1:  # metadatos: búsqueda vectorial restringida al modelo nombrado
                (modelo,) = nombrados
                rankings[f"vectorial_{modelo}"] = await self.indice.buscar(vector, CANDIDATOS_POR_RANKING, modelo)
            return rankings, "hibrido"
        except Exception as error:  # embeddings o Redis caídos: se degrada en vez de fallar
            logger.warning(f"Búsqueda vectorial no disponible ({type(error).__name__}: {error}); se usa solo BM25")
            return rankings, "solo_bm25 (búsqueda vectorial caída)"

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        rankings, modo = await self.rankings(query)
        puntos = fusionar_rrf(rankings)
        nombrados = self.modelos_nombrados(query)
        for fragmento_id in puntos:  # re-ranking suave por el modelo nombrado
            if self._por_id[fragmento_id].modelo in nombrados:
                puntos[fragmento_id] *= REFUERZO_MODELO_NOMBRADO
        elegidos = sorted(puntos, key=puntos.get, reverse=True)[: self.k]
        return [
            Document(
                page_content=self._por_id[i].texto,
                metadata={
                    "fragmento_id": i, "fuente": self._por_id[i].fuente, "modelo": self._por_id[i].modelo,
                    "seccion": self._por_id[i].seccion, "puntaje_rrf": round(puntos[i], 5), "modo": modo,
                    "posiciones": {nombre: r.index(i) + 1 for nombre, r in rankings.items() if i in r},
                },
            )
            for i in elegidos
        ]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError("El recuperador es solo async: usá ainvoke()")
