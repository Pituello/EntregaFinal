"""
Tests del RAG híbrido (entrega final): fragmentación con metadatos, fusión
RRF, filtro por modelo, degradación a BM25 y versionado del índice.

Sin API key: los embeddings son DeterministicFakeEmbedding de LangChain
(vectores pseudoaleatorios estables por texto) y el índice es IndiceEnMemoria,
la misma interfaz que IndiceRedis. El test marcado con_redis indexa en el
Redis 8 real (búsqueda vectorial HNSW con filtro TAG).
"""
import asyncio
import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from pydantic import ValidationError

from orquestador import config
from orquestador.rag import servicio
from orquestador.rag.documentos import MAX_CARACTERES, cargar_fragmentos, huella_del_corpus
from orquestador.rag.indice import IndiceEnMemoria, huella_del_indice
from orquestador.rag.recuperador import RecuperadorHibrido, fusionar_rrf
from orquestador.tools.busqueda import BuscarEnFichasInput, ResultadoBusqueda, buscar_en_fichas
from tests.test_api import REDIS_DE_PRUEBA, con_redis


def correr(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def sin_recuperador_global():
    servicio.usar_recuperador(None)
    yield
    servicio.usar_recuperador(None)


class EmbeddingsCaidos(Embeddings):
    def embed_documents(self, texts):
        raise ConnectionError("servicio de embeddings caído")

    def embed_query(self, text):
        raise ConnectionError("servicio de embeddings caído")


async def recuperador_hibrido(k: int = 5) -> RecuperadorHibrido:
    fragmentos, embeddings, indice = cargar_fragmentos(), DeterministicFakeEmbedding(size=64), IndiceEnMemoria()
    await indice.asegurar(fragmentos, embeddings, "fake")
    return RecuperadorHibrido(fragmentos=fragmentos, k=k, indice=indice, embeddings=embeddings)


def test_las_fichas_se_parten_por_seccion_con_metadatos():
    fragmentos = cargar_fragmentos()
    assert len({f.id for f in fragmentos}) == len(fragmentos) > 60  # IDs estables y únicos
    assert {f.modelo for f in fragmentos} == {"tera", "polo", "virtus", "nivus", "taos", "vento", "saveiro"}
    for f in fragmentos:
        assert f.fuente == f"{f.modelo}.md" and f.seccion
        assert f.texto.splitlines()[0].endswith(f.seccion)  # cada fragmento lleva su título: no queda "huérfano"
        assert len(f.texto) <= MAX_CARACTERES + 200


def test_rrf_premia_lo_que_aparece_en_varios_rankings():
    puntos = fusionar_rrf({"bm25": ["a", "b", "c"], "vectorial": ["c", "a", "d"]})
    assert max(puntos, key=puntos.get) == "a"  # 1.º y 2.º le gana a 3.º y 1.º
    assert puntos["c"] > puntos["b"] > puntos["d"]


def test_hibrido_suma_el_ranking_filtrado_por_el_modelo_nombrado():
    async def caso():
        documentos = await (await recuperador_hibrido()).ainvoke("Latin NCAP protección ocupante adulto Nivus")
        assert len(documentos) == 5 and {d.metadata["modo"] for d in documentos} == {"hibrido"}
        rankings = set().union(*(d.metadata["posiciones"] for d in documentos))
        assert {"bm25", "vectorial", "vectorial_nivus"} <= rankings
        # con embeddings falsos (ruido) el que encuentra la sección es BM25, y el refuerzo la mantiene arriba
        assert documentos[0].metadata["fuente"] == "nivus.md"
        assert any(d.metadata["fuente"] == "nivus.md" and d.metadata["seccion"].startswith("Seguridad")
                   for d in documentos)
    correr(caso())


def test_si_los_embeddings_fallan_degrada_a_bm25_y_sigue_respondiendo():
    async def caso():
        indice = IndiceEnMemoria()
        await indice.asegurar(cargar_fragmentos(), DeterministicFakeEmbedding(size=8), "fake")
        r = RecuperadorHibrido(fragmentos=cargar_fragmentos(), indice=indice, embeddings=EmbeddingsCaidos())
        documentos = await r.ainvoke("airbags de la Saveiro")
        assert documentos and documentos[0].metadata["modo"] == "solo_bm25 (búsqueda vectorial caída)"
        assert documentos[0].metadata["fuente"] == "saveiro.md"
    correr(caso())


def test_preparar_rag_sin_embeddings_arranca_en_modo_bm25(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ninguno")
    recuperador = correr(servicio.preparar_rag())
    assert recuperador.indice is None and servicio.modo_rag() == "solo BM25"


def test_la_herramienta_usa_el_recuperador_preparado_y_conserva_el_formato_de_fuente():
    async def caso():
        servicio.usar_recuperador(await recuperador_hibrido())
        texto = await buscar_en_fichas.ainvoke({"consulta": "¿Qué tan seguro es el Tera en un choque?"})
        assert texto.startswith("[Fuente: ") and texto.count("[Fuente: ") == 5  # lo que lee el Validador
    correr(caso())


def test_entrada_y_salida_de_la_busqueda_validadas():
    with pytest.raises(ValidationError):
        BuscarEnFichasInput(consulta="ab")
    with pytest.raises(ValidationError):  # una fuente que no es un archivo .md de las fichas
        ResultadoBusqueda(modo="hibrido", fragmentos=[{"fuente": "http://x", "seccion": "s", "texto": "t"}])
    assert ResultadoBusqueda(modo="hibrido", fragmentos=[]).como_texto().startswith("ERROR")


def test_la_huella_del_indice_cambia_con_el_modelo_o_el_contenido():
    fragmentos = cargar_fragmentos()
    base = huella_del_indice(fragmentos, "text-embedding-3-small")
    assert base == huella_del_indice(fragmentos, "text-embedding-3-small")
    assert base != huella_del_indice(fragmentos, "otro-modelo")
    assert huella_del_corpus(fragmentos) != huella_del_corpus(fragmentos[:-1])


def test_sin_variable_obligatoria_el_error_dice_que_agregar(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_INDICE", "")
    with pytest.raises(RuntimeError, match="Falta RAG_INDICE en el .env.*\\.env\\.example"):
        config.rag_indice()


@con_redis
def test_redis_real_indexa_una_sola_vez_y_filtra_por_modelo():
    from redis.asyncio import from_url

    from orquestador.rag.indice import IndiceRedis

    async def caso():
        nombre = f"test-rag-{uuid.uuid4().hex[:8]}"
        fragmentos, embeddings = cargar_fragmentos(), DeterministicFakeEmbedding(size=32)
        indices = [IndiceRedis(REDIS_DE_PRUEBA, nombre) for _ in range(2)]
        try:
            # dos procesos arrancan a la vez: el lock hace que indexe uno solo y el otro reutilice
            await asyncio.gather(*(i.asegurar(fragmentos, embeddings, "fake") for i in indices))
            vector = await embeddings.aembed_query("seguridad del Taos")
            ids = await indices[1].buscar(vector, 5, modelo="taos")
            por_id = {f.id: f for f in fragmentos}
            assert len(ids) == 5 and {por_id[i].modelo for i in ids} == {"taos"}
            assert len(await indices[0].buscar(vector, 5)) == 5
        finally:
            r = from_url(REDIS_DE_PRUEBA)
            await r.execute_command("FT.DROPINDEX", nombre, "DD")
            await r.delete(f"{nombre}:huella", f"{nombre}:dimensiones", f"{nombre}:lock")
            await r.aclose()
    correr(caso())
