"""
Configuración central, leída del .env.

Entrega final: ningún nombre de modelo ni de índice está escrito en el código
(error a evitar del módulo: "hard-coding"). Si falta una variable obligatoria,
requerido() corta el arranque con un mensaje que dice qué agregar al .env;
.env.example tiene todos los valores con los que se probó el sistema.

Se carga el .env apenas se importa el módulo, y los valores ajustables se leen
con funciones al momento de usarlos (no como constantes al importar): la
corrección de TEMPERATURE que marcó la devolución de la Pre-Entrega 3.
"""
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")


def _entero(nombre: str, default: int, minimo: int = 1) -> int:
    valor = os.getenv(nombre, str(default))
    try:
        numero = int(valor)
    except ValueError:
        raise ValueError(f"{nombre}='{valor}' en el .env no es un número entero")
    if numero < minimo:
        raise ValueError(f"{nombre}={numero} tiene que ser >= {minimo}")
    return numero


def requerido(nombre: str, ejemplo: str) -> str:
    """Variable obligatoria: sin default en el código."""
    valor = os.getenv(nombre, "").strip()
    if not valor:
        raise RuntimeError(f"Falta {nombre} en el .env (ejemplo: {nombre}={ejemplo}). Ver .env.example")
    return valor


def max_pasos() -> int:
    """
    Error a evitar "Supervisor Infinito": cantidad máxima de intervenciones de
    especialistas por tarea. Al llegar, el supervisor finaliza con lo que haya.
    """
    return _entero("MAX_PASOS", 8)


def max_reparaciones() -> int:
    """
    Error a evitar "Bucle Infinito de Corrección": cuántas veces se le devuelve
    a un mismo especialista un output que no pasó la validación. Superado ese
    límite, se escala a revisión humana en vez de seguir reintentando.
    """
    return _entero("MAX_REPARACIONES", 2, minimo=0)


def recursion_limit() -> int:
    """Techo de LangGraph (cada vuelta especialista -> validador -> supervisor usa 3 pasos)."""
    return _entero("RECURSION_LIMIT", 40, minimo=5)


# --- RAG (entrega final) ----------------------------------------------------
def embeddings_proveedor() -> str:
    """openai (default) | ollama | ninguno. Con "ninguno" el RAG funciona solo con BM25."""
    proveedor = os.getenv("EMBEDDINGS_PROVIDER", "openai").strip().lower()
    if proveedor not in ("openai", "ollama", "ninguno"):
        raise ValueError(f"EMBEDDINGS_PROVIDER='{proveedor}' no válido: openai, ollama o ninguno")
    return proveedor


def embeddings_modelo() -> str:
    return requerido("EMBEDDINGS_MODEL", "text-embedding-3-small")


def rag_indice() -> str:
    """Nombre del índice vectorial en Redis (también es el prefijo de sus claves)."""
    return requerido("RAG_INDICE", "vw-fichas")


def rag_top_k() -> int:
    """Cuántos fragmentos devuelve la búsqueda en las fichas."""
    return _entero("RAG_TOP_K", 5)


# --- Pre-Entrega 7: API, Redis y observabilidad -------------------------------
def redis_url() -> str:
    """Redis guarda el estado de los jobs, la cola, los checkpoints de LangGraph y las reservas."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def workers() -> int:
    """
    Cuántos jobs corre en paralelo cada proceso worker. Son tareas asyncio, no
    hilos: mientras un job espera al LLM, el event loop atiende a los otros.
    Con 5 alcanzan para las 5 peticiones concurrentes de la prueba de carga.
    """
    return _entero("WORKERS", 5)


def worker_en_proceso() -> bool:
    """true: la API levanta los workers en su propio event loop. false: se corren aparte (python -m app.worker)."""
    return os.getenv("WORKER_EN_PROCESO", "true").strip().lower() in ("1", "true", "si", "sí", "yes")


def lease_s() -> int:
    """
    Vencimiento del "contrato" de un worker sobre un job (entrega final). El
    worker lo renueva cada LEASE_S/3 mientras trabaja; si muere, a los LEASE_S
    segundos otro worker retoma el job desde su último checkpoint.
    """
    return _entero("LEASE_S", 30, minimo=3)


def timeout_job_s() -> int:
    """Techo de tiempo de una ejecución del grafo. Si se pasa, el job queda FAILED (no en un polling infinito)."""
    return _entero("TIMEOUT_JOB_S", 300, minimo=10)


def phoenix_habilitado() -> bool:
    return os.getenv("PHOENIX_HABILITADO", "true").strip().lower() in ("1", "true", "si", "sí", "yes")


def phoenix_endpoint() -> str:
    """Colector OTLP de Phoenix (mismo nombre de variable que usa phoenix.otel)."""
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")


def phoenix_proyecto() -> str:
    return os.getenv("PHOENIX_PROJECT_NAME", "orquestador-vw")
