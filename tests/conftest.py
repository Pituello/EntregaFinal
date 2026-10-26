"""
Configuración común de los tests.

Las variables obligatorias (sin default en el código, ver orquestador/config.py)
se completan con valores de prueba si la máquina no tiene un .env: así los
tests corren en un clon limpio, sin API keys y sin Docker. Los tests no
llaman a ningún LLM real (usan dobles) ni a un servicio de embeddings real.
"""
import os

VALORES_DE_PRUEBA = {
    "LLM_PROVIDER": "openai",
    "OPENAI_MODEL": "gpt-4o-mini",
    "EMBEDDINGS_PROVIDER": "ninguno",
    "EMBEDDINGS_MODEL": "text-embedding-3-small",
    "RAG_INDICE": "vw-fichas-test",
    "PHOENIX_HABILITADO": "false",
}
for nombre, valor in VALORES_DE_PRUEBA.items():
    os.environ.setdefault(nombre, valor)
