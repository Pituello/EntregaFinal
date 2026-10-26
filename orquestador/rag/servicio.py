"""
Arranque del RAG: crea los embeddings, asegura el índice vectorial en Redis
y deja listo el recuperador que usa la herramienta buscar_en_fichas.

Se llama una vez al arrancar la API o el worker (preparar_rag). Si los
embeddings no están configurados o fallan al indexar, el sistema arranca igual
con búsqueda solo BM25, y lo informa en el log y en GET /health.
"""
from langchain_core.embeddings import Embeddings
from loguru import logger

from orquestador import config
from orquestador.rag.documentos import cargar_fragmentos
from orquestador.rag.indice import IndiceRedis, IndiceVectorial
from orquestador.rag.recuperador import RecuperadorHibrido

_recuperador: RecuperadorHibrido | None = None


def crear_embeddings() -> Embeddings | None:
    proveedor = config.embeddings_proveedor()
    if proveedor == "ninguno":
        return None
    modelo = config.embeddings_modelo()
    if proveedor == "openai":
        from langchain_openai import OpenAIEmbeddings

        # check_embedding_ctx_length=False: los fragmentos (<= 1500 caracteres) entran de sobra en el
        # contexto del modelo, y así no hace falta descargar el tokenizador de tiktoken al arrancar.
        return OpenAIEmbeddings(model=modelo, api_key=config.requerido("OPENAI_API_KEY", "sk-..."),
                                check_embedding_ctx_length=False)
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=modelo, base_url=config.requerido("OLLAMA_BASE_URL", "http://localhost:11434"))


async def preparar_rag(indice: IndiceVectorial | None = None, embeddings: Embeddings | None = None) -> RecuperadorHibrido:
    """Indexa las fichas si hace falta (huella distinta) y deja el recuperador listo."""
    global _recuperador
    fragmentos = cargar_fragmentos()
    if embeddings is None:
        embeddings = crear_embeddings()
        modelo = config.embeddings_modelo() if embeddings is not None else ""
    else:
        modelo = "inyectado"  # embeddings de prueba
    if embeddings is None:
        logger.warning("EMBEDDINGS_PROVIDER=ninguno: la búsqueda en las fichas usa solo BM25")
        _recuperador = RecuperadorHibrido(fragmentos=fragmentos, k=config.rag_top_k())
        return _recuperador
    indice = indice or IndiceRedis(config.redis_url(), config.rag_indice())
    try:
        await indice.asegurar(fragmentos, embeddings, modelo)
    except Exception as error:
        logger.error(f"No se pudo preparar el índice vectorial ({type(error).__name__}: {error}). "
                     "La búsqueda en las fichas sigue con solo BM25.")
        indice, embeddings = None, None
    _recuperador = RecuperadorHibrido(fragmentos=fragmentos, k=config.rag_top_k(), indice=indice,
                                      embeddings=embeddings)
    return _recuperador


def obtener_recuperador() -> RecuperadorHibrido:
    """El recuperador preparado; si nadie llamó a preparar_rag() (por ejemplo, un test), uno solo BM25."""
    global _recuperador
    if _recuperador is None:
        _recuperador = RecuperadorHibrido(fragmentos=cargar_fragmentos(), k=config.rag_top_k())
    return _recuperador


def usar_recuperador(recuperador: RecuperadorHibrido | None) -> None:
    global _recuperador
    _recuperador = recuperador


def modo_rag() -> str:
    r = obtener_recuperador()
    return "hibrido (BM25 + vectorial en Redis)" if r.indice is not None else "solo BM25"
