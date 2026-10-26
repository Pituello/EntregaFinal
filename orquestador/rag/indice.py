"""
Índice vectorial de las fichas, en Redis (la misma instancia del estado y los
checkpoints: Redis 8 trae el módulo Search con índices vectoriales).

Por qué Redis y no el índice de Pinecone de la Pre-Entrega 4: el sistema
completo tiene que levantar con un solo comando en una máquina limpia, sin
cuentas externas salvo la del LLM. Con Redis ya en el stack, sumar una base
vectorial más no aporta nada; y los mismos metadatos (modelo, sección,
fuente) permiten filtrar igual que en Pinecone.

Versionado (lección de la Pre-Entrega 4: un cambio de esquema dejó vectores
viejos mezclados): el índice guarda una HUELLA que combina la versión del
esquema, el modelo de embeddings y el contenido de los fragmentos. Al
arrancar, si la huella no coincide, el índice se reconstruye solo.

Es async de punta a punta (redisvl.AsyncSearchIndex sobre redis.asyncio), y
la carga corre una sola vez aunque arranquen varios workers a la vez: un
lock en Redis hace que uno indexe y los demás esperen.
"""
import asyncio
from typing import Protocol

import numpy as np
from langchain_core.embeddings import Embeddings
from loguru import logger

from orquestador.rag.documentos import Fragmento, huella_del_corpus

VERSION_ESQUEMA = "1"
TAMANO_LOTE = 64


class IndiceVectorial(Protocol):
    async def asegurar(self, fragmentos: tuple[Fragmento, ...], embeddings: Embeddings, modelo: str) -> None: ...
    async def buscar(self, vector: list[float], k: int, modelo: str | None = None) -> list[str]: ...


def huella_del_indice(fragmentos: tuple[Fragmento, ...], modelo_embeddings: str) -> str:
    return f"v{VERSION_ESQUEMA}:{modelo_embeddings}:{huella_del_corpus(fragmentos)}"


class IndiceRedis:
    """Índice HNSW de coseno en Redis, con 'modelo' y 'fuente' como TAG para filtrar."""

    def __init__(self, redis_url: str, nombre: str):
        self.redis_url, self.nombre = redis_url, nombre
        self._indice = None

    def _esquema(self, dimensiones: int):
        from redisvl.schema import IndexSchema

        return IndexSchema.from_dict({
            "index": {"name": self.nombre, "prefix": f"{self.nombre}:frag", "storage_type": "hash"},
            "fields": [
                {"name": "fragmento_id", "type": "tag"},
                {"name": "modelo", "type": "tag"},
                {"name": "fuente", "type": "tag"},
                {"name": "embedding", "type": "vector",
                 "attrs": {"dims": dimensiones, "distance_metric": "cosine", "algorithm": "hnsw",
                           "datatype": "float32"}},
            ],
        })

    async def asegurar(self, fragmentos: tuple[Fragmento, ...], embeddings: Embeddings, modelo: str) -> None:
        from redis.asyncio import from_url
        from redisvl.index import AsyncSearchIndex

        huella = huella_del_indice(fragmentos, modelo)
        redis = from_url(self.redis_url, decode_responses=True)
        clave_huella, clave_lock = f"{self.nombre}:huella", f"{self.nombre}:lock"
        try:
            for _ in range(120):  # hasta 2 minutos esperando a otro proceso que esté indexando
                if await redis.get(clave_huella) == huella:
                    dimensiones = int(await redis.get(f"{self.nombre}:dimensiones"))
                    self._indice = AsyncSearchIndex(self._esquema(dimensiones), redis_url=self.redis_url)
                    logger.info(f"Índice vectorial '{self.nombre}' al día ({len(fragmentos)} fragmentos)")
                    return
                if await redis.set(clave_lock, "1", nx=True, ex=300):
                    try:
                        await self._reconstruir(fragmentos, embeddings, huella, redis)
                    finally:
                        await redis.delete(clave_lock)
                    return
                await asyncio.sleep(1)
            raise TimeoutError(f"Otro proceso tiene tomado el lock de indexación de '{self.nombre}'")
        finally:
            await redis.aclose()

    async def _reconstruir(self, fragmentos, embeddings: Embeddings, huella: str, redis) -> None:
        from redisvl.index import AsyncSearchIndex

        logger.info(f"Indexando {len(fragmentos)} fragmentos en '{self.nombre}' (huella {huella})")
        vectores: list[list[float]] = []
        for i in range(0, len(fragmentos), TAMANO_LOTE):
            vectores += await embeddings.aembed_documents([f.texto for f in fragmentos[i:i + TAMANO_LOTE]])
        indice = AsyncSearchIndex(self._esquema(len(vectores[0])), redis_url=self.redis_url)
        await indice.create(overwrite=True, drop=True)  # drop: borra también los vectores viejos
        await indice.load(
            [{"fragmento_id": f.id, "modelo": f.modelo, "fuente": f.fuente,
              "embedding": np.asarray(v, dtype=np.float32).tobytes()} for f, v in zip(fragmentos, vectores)],
            id_field="fragmento_id",
        )
        await redis.set(f"{self.nombre}:dimensiones", len(vectores[0]))
        await redis.set(f"{self.nombre}:huella", huella)
        self._indice = indice
        logger.info(f"Índice '{self.nombre}' listo: {len(vectores)} vectores de {len(vectores[0])} dimensiones")

    async def buscar(self, vector: list[float], k: int, modelo: str | None = None) -> list[str]:
        """IDs de los k fragmentos más parecidos (opcionalmente, solo de un modelo)."""
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        if self._indice is None:
            raise RuntimeError("El índice vectorial no está preparado (falta asegurar())")
        consulta = VectorQuery(
            vector=vector, vector_field_name="embedding", return_fields=["fragmento_id"], num_results=k,
            filter_expression=(Tag("modelo") == modelo) if modelo else None,
        )
        return [r["fragmento_id"] for r in await self._indice.query(consulta)]


class IndiceEnMemoria:
    """La misma interfaz con numpy: para los tests (fakeredis no implementa búsqueda vectorial)."""

    def __init__(self):
        self._ids: list[str] = []
        self._modelos: list[str] = []
        self._matriz: np.ndarray | None = None

    async def asegurar(self, fragmentos: tuple[Fragmento, ...], embeddings: Embeddings, modelo: str) -> None:
        vectores = np.asarray(await embeddings.aembed_documents([f.texto for f in fragmentos]), dtype=np.float32)
        self._matriz = vectores / np.linalg.norm(vectores, axis=1, keepdims=True)
        self._ids, self._modelos = [f.id for f in fragmentos], [f.modelo for f in fragmentos]

    async def buscar(self, vector: list[float], k: int, modelo: str | None = None) -> list[str]:
        q = np.asarray(vector, dtype=np.float32)
        similitud = self._matriz @ (q / np.linalg.norm(q))
        orden = [i for i in np.argsort(-similitud) if modelo is None or self._modelos[i] == modelo]
        return [self._ids[i] for i in orden[:k]]
