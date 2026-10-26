"""Errores del proveedor del LLM, con un mensaje claro en vez de un traceback."""
import httpx
from langchain_core.exceptions import ModelError

try:
    from ollama import ResponseError as _ErrorDeOllama
except ImportError:  # langchain-ollama no instalado
    _ErrorDeOllama = ModelError

ERRORES_DEL_PROVEEDOR: tuple[type[Exception], ...] = (ModelError, httpx.HTTPError, _ErrorDeOllama)


def avisar_error_del_llm(error: Exception) -> None:
    print(f"\n❌ El proveedor del LLM rechazó el pedido: {type(error).__name__}")
    if isinstance(error, httpx.ConnectError):
        print("   No se pudo conectar. Si usás Ollama: abrilo (o corré 'ollama serve') y revisá OLLAMA_BASE_URL.")
    elif isinstance(error, _ErrorDeOllama) and "not found" in str(error).lower():
        print("   Ollama no tiene ese modelo descargado: corré 'ollama pull <modelo>' (el de OLLAMA_MODEL).")
    elif "perday" in str(error).lower():
        print("   Se agotó la cuota DIARIA del modelo (capa gratuita). Opciones: esperar a que se reinicie,")
        print("   usar otro modelo (GEMINI_MODEL) o usar otro proveedor (LLM_PROVIDER=openai u anthropic).")
    print(f"\n   Detalle: {str(error)[:400]}")
