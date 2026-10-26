"""
Evaluación de la recuperación: BM25 solo contra el recuperador híbrido.

    python scripts/evaluar_rag.py

Sobre 20 preguntas con la sección correcta conocida, mide:
- acierto@1: la sección correcta salió primera;
- acierto@k: la sección correcta está entre los k fragmentos que recibe el LLM;
- MRR: promedio de 1/posición de la sección correcta (1 = siempre primera).

Mezcla preguntas con las palabras exactas de la ficha (donde BM25 rinde bien)
y paráfrasis ("¿qué tan seguro es en choques?", "¿dónde se fabrica?") donde
las palabras no coinciden y la búsqueda vectorial tiene que aportar. Incluye
la trampa de la Pre-Entrega 4: "el sedán del Polo" está en la ficha del Virtus.

Usa los embeddings y el índice configurados en el .env (los mismos de la API);
el costo de embeber 20 preguntas es despreciable. Guarda el resultado en
evidencia/evaluacion_rag_<fecha>.json.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from orquestador import config  # noqa: E402
from orquestador.rag.documentos import cargar_fragmentos  # noqa: E402
from orquestador.rag.indice import IndiceRedis  # noqa: E402
from orquestador.rag.recuperador import RecuperadorHibrido  # noqa: E402
from orquestador.rag.servicio import crear_embeddings  # noqa: E402

# (pregunta, ficha esperada, comienzo del título de la sección esperada)
PREGUNTAS = [
    ("Latin NCAP protección ocupante adulto Nivus", "nivus.md", "Seguridad"),
    ("¿Qué tan seguro es el Tera en un choque?", "tera.md", "Seguridad"),
    ("¿Dónde se fabrica el Taos?", "taos.md", "Plataforma"),
    ("capacidad de baúl de la versión sedán del Polo", "virtus.md", "Dimensiones"),
    ("¿Qué autos compiten con el Nivus?", "nivus.md", "Rivales"),
    ("¿Cuántos años de garantía tiene la Saveiro?", "saveiro.md", "Garantía"),
    ("¿En qué colores viene el Tera?", "tera.md", "Colores"),
    ("¿Por qué dejó de venderse el Vento?", "vento.md", "Historia"),
    ("¿Cuántas unidades del Tera se fabricaron y vendieron?", "tera.md", "Producción"),
    ("pantalla multimedia y Apple CarPlay del Nivus", "nivus.md", "Infotenimiento"),
    ("largo y distancia entre ejes del Virtus", "virtus.md", "Dimensiones"),
    ("¿Sobre qué plataforma está construido el Polo?", "polo.md", "Plataforma"),
    ("estrellas en las pruebas de choque del Taos", "taos.md", "Seguridad"),
    ("¿Cuándo salió la primera generación de la Saveiro?", "saveiro.md", "Historia"),
    ("¿Qué lugar ocupa el Virtus dentro de la gama Volkswagen?", "virtus.md", "Posicionamiento"),
    ("¿Qué versiones del Taos se venden?", "taos.md", "Versiones"),
    ("airbags de la Saveiro", "saveiro.md", "Seguridad"),
    ("¿Qué motores tenían los Vento de generaciones anteriores?", "vento.md", "Versiones mecánicas"),
    ("conectividad del Tera con el celular", "tera.md", "Infotenimiento"),
    ("¿Cómo le fue al Polo en las pruebas de choque?", "polo.md", "Seguridad"),
]


def posicion_correcta(documentos, fuente: str, seccion: str) -> int | None:
    for i, d in enumerate(documentos, start=1):
        if d.metadata["fuente"] == fuente and d.metadata["seccion"].startswith(seccion):
            return i
    return None


async def evaluar(recuperador: RecuperadorHibrido) -> dict:
    filas = []
    for pregunta, fuente, seccion in PREGUNTAS:
        documentos = await recuperador.ainvoke(pregunta)
        filas.append({"pregunta": pregunta, "esperado": f"{fuente} · {seccion}",
                      "posicion": posicion_correcta(documentos, fuente, seccion),
                      "primero": f"{documentos[0].metadata['fuente']} · {documentos[0].metadata['seccion']}"
                      if documentos else None})
    n = len(filas)
    return {
        "acierto_1": sum(f["posicion"] == 1 for f in filas) / n,
        "acierto_k": sum(f["posicion"] is not None for f in filas) / n,
        "mrr": sum(1 / f["posicion"] for f in filas if f["posicion"]) / n,
        "filas": filas,
    }


async def main(embeddings=None, modelo: str | None = None) -> dict:
    fragmentos = cargar_fragmentos()
    k = config.rag_top_k()
    embeddings = embeddings or crear_embeddings()
    if embeddings is None:
        raise SystemExit("EMBEDDINGS_PROVIDER=ninguno: no hay búsqueda vectorial para comparar")
    indice = IndiceRedis(config.redis_url(), config.rag_indice())
    await indice.asegurar(fragmentos, embeddings, modelo or config.embeddings_modelo())

    resultados = {
        "solo BM25": await evaluar(RecuperadorHibrido(fragmentos=fragmentos, k=k)),
        "híbrido": await evaluar(RecuperadorHibrido(fragmentos=fragmentos, k=k, indice=indice, embeddings=embeddings)),
    }
    print(f"\nRecuperación sobre {len(PREGUNTAS)} preguntas (k = {k} fragmentos)\n")
    print(f"{'':<12}{'acierto@1':>11}{f'acierto@{k}':>11}{'MRR':>8}")
    for nombre, r in resultados.items():
        print(f"{nombre:<12}{r['acierto_1']:>11.0%}{r['acierto_k']:>11.0%}{r['mrr']:>8.2f}")
    print("\nPreguntas donde los dos difieren:")
    for bm25, hibrido in zip(resultados["solo BM25"]["filas"], resultados["híbrido"]["filas"]):
        if bm25["posicion"] != hibrido["posicion"]:
            print(f"  · {bm25['pregunta']}\n      BM25: {bm25['posicion'] or 'no aparece'} · "
                  f"híbrido: {hibrido['posicion'] or 'no aparece'}  (esperado: {bm25['esperado']})")
    destino = Path("evidencia") / f"evaluacion_rag_{datetime.now():%Y%m%d_%H%M%S}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"k": k, "modelo_embeddings": modelo or config.embeddings_modelo(),
                                   "resultados": resultados}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {destino}")
    return resultados


if __name__ == "__main__":
    asyncio.run(main())
