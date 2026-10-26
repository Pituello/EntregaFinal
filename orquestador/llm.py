"""
LLM del sistema.

El proveedor se elige por .env: LLM_PROVIDER si está definido; si no, el
primero que tenga API key en este orden: OpenAI, Anthropic, Gemini. La
consigna pide OpenAI o Anthropic; Gemini se suma porque es el que usa la
notebook de la clase (y tiene capa gratuita).

Ollama (modelo local) solo se usa si se pide explícitamente con
LLM_PROVIDER=ollama: no necesita API key, así que si entrara en la elección
automática ganaría siempre. Sirve para practicar sin gastar cuota, pero los
modelos locales chicos eligen herramientas peor que los de la nube, así que
la traza que se entrega conviene generarla con OpenAI, Anthropic o Gemini.
"""
import os

from langchain_core.exceptions import ModelAPIError, ModelConnectionError, ModelRateLimitError, ModelTimeoutError
from langchain_core.language_models import BaseChatModel
from loguru import logger

from orquestador import config  # importarlo carga el .env

# (variable de la API key, variable del modelo, ejemplo para el mensaje de error).
# El modelo NO tiene default en el código: se lee siempre del .env (entrega final).
PROVEEDORES: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-flash-latest"),
}

OLLAMA = "ollama"
TIMEOUT_OLLAMA_SEGUNDOS = 120  # un modelo local puede tardar en cargar la primera vez


def leer_temperature() -> float:
    """Se lee al crear el modelo (no al importar) y se valida."""
    valor = os.getenv("TEMPERATURE", "0")
    try:
        temperature = float(valor)
    except ValueError:
        raise ValueError(f"TEMPERATURE='{valor}' en el .env no es un número válido")
    if not 0 <= temperature <= 2:
        raise ValueError(f"TEMPERATURE={temperature} fuera de rango (tiene que estar entre 0 y 2)")
    return temperature


def elegir_proveedor() -> str:
    elegido = os.getenv("LLM_PROVIDER", "").strip().lower()
    if elegido == OLLAMA:
        return OLLAMA
    if elegido:
        if elegido not in PROVEEDORES:
            raise ValueError(f"LLM_PROVIDER='{elegido}' no soportado. Opciones: {[*PROVEEDORES, OLLAMA]}")
        if not os.getenv(PROVEEDORES[elegido][0]):
            raise RuntimeError(f"LLM_PROVIDER={elegido}, pero falta {PROVEEDORES[elegido][0]} en el .env")
        return elegido
    for nombre, (variable_key, _, _) in PROVEEDORES.items():
        if os.getenv(variable_key):
            return nombre
    raise RuntimeError(
        "Falta la API key de un LLM en el .env: OPENAI_API_KEY, ANTHROPIC_API_KEY o GEMINI_API_KEY "
        "(o LLM_PROVIDER=ollama para usar un modelo local)"
    )


def get_llm(proveedor: str | None = None) -> BaseChatModel:
    proveedor = proveedor or elegir_proveedor()
    temperature = leer_temperature()

    if proveedor == OLLAMA:
        from langchain_ollama import ChatOllama

        modelo = config.requerido("OLLAMA_MODEL", "llama3.2")
        base_url = config.requerido("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"LLM: ollama / {modelo} en {base_url} (temperature={temperature})")
        timeout = {"timeout": TIMEOUT_OLLAMA_SEGUNDOS}  # sin esto, ChatOllama espera para siempre
        return ChatOllama(model=modelo, base_url=base_url, temperature=temperature,
                          client_kwargs=timeout, async_client_kwargs=timeout)

    variable_key, variable_modelo, ejemplo = PROVEEDORES[proveedor]
    api_key = os.getenv(variable_key)
    modelo = config.requerido(variable_modelo, ejemplo)
    logger.info(f"LLM: {proveedor} / {modelo} (temperature={temperature})")

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=modelo, api_key=api_key, temperature=temperature, timeout=30)
    if proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=modelo, api_key=api_key, temperature=temperature, timeout=30)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=modelo, google_api_key=api_key, temperature=temperature, timeout=30)


def describir_llm() -> str:
    """'gemini / gemini-flash-latest': para dejar registrado en la traza qué modelo respondió."""
    proveedor = elegir_proveedor()
    if proveedor == OLLAMA:
        return f"ollama / {config.requerido('OLLAMA_MODEL', 'llama3.2')}"
    _, variable_modelo, ejemplo = PROVEEDORES[proveedor]
    return f"{proveedor} / {config.requerido(variable_modelo, ejemplo)}"


# --- Errores transitorios del proveedor (lección de la Pre-Entrega 5) ---------
# Un 503 "high demand" o un rate limit por minuto se arreglan reintentando; una
# cuota DIARIA agotada (mismo código 429) no, así que se distingue por el mensaje.
ERRORES_TRANSITORIOS_LLM = (ModelAPIError, ModelRateLimitError, ModelTimeoutError, ModelConnectionError)
_CUOTA_DIARIA = ("perday", "per day", "daily")


def es_error_transitorio(error: Exception) -> bool:
    if not isinstance(error, ERRORES_TRANSITORIOS_LLM):
        return False
    return not any(senal in str(error).lower() for senal in _CUOTA_DIARIA)
