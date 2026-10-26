"""
Carga y chunking de las fichas técnicas (data/*.md), con metadatos.

Estrategia (heredada de las Pre-Entregas 3 y 4, medida sobre este dataset):
- La unidad natural es la SECCIÓN ("## Seguridad y calificación Latin NCAP"):
  63 secciones de ~660 caracteres en promedio. Una sección responde una
  pregunta; partirla por un tamaño fijo cortaría tablas y listas por la mitad.
- Las secciones largas (más de MAX_CARACTERES) se subdividen por párrafos,
  con un párrafo de solapamiento para no perder el contexto en el borde.
- A cada fragmento se le antepone "Modelo — Sección" (lección de la
  Pre-Entrega 3: un fragmento que no nombra su modelo se confunde con otro;
  "el Tera obtuvo 89,88 %" y "el Nivus obtuvo 92,47 %" se parecen mucho).

Cada fragmento es un modelo Pydantic con sus metadatos (modelo, sección,
fuente), que usa el índice vectorial para filtrar y el retriever para citar.
"""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from orquestador import config

MAX_CARACTERES = 1500
_SECCION = re.compile(r"\n(?=## )")


class Fragmento(BaseModel):
    id: str = Field(description="Hash del contenido: estable entre corridas")
    fuente: str = Field(description="Archivo de la ficha, ej: 'tera.md'")
    modelo: str = Field(description="Modelo en minúsculas, ej: 'tera'")
    seccion: str = Field(description="Título de la sección, sin '## '")
    orden: int = Field(ge=0, description="Posición del fragmento dentro de la ficha")
    texto: str = Field(min_length=1, description="Texto con 'Modelo — Sección' antepuesto")


def _partir(seccion: str) -> list[str]:
    """Una sección larga -> bloques de párrafos de hasta MAX_CARACTERES, con un párrafo de solapamiento."""
    if len(seccion) <= MAX_CARACTERES:
        return [seccion]
    parrafos = [p for p in re.split(r"\n\s*\n", seccion) if p.strip()]
    bloques, actual = [], []
    for parrafo in parrafos:
        if actual and len("\n\n".join([*actual, parrafo])) > MAX_CARACTERES:
            bloques.append("\n\n".join(actual))
            actual = actual[-1:]  # solapamiento: el último párrafo pasa al bloque siguiente
        actual.append(parrafo)
    if actual:
        bloques.append("\n\n".join(actual))
    return bloques


def fragmentar(ruta: Path) -> list[Fragmento]:
    texto = ruta.read_text(encoding="utf-8")
    titulo = texto.splitlines()[0].lstrip("# ").strip()
    fragmentos = []
    for seccion in _SECCION.split(texto):
        seccion = seccion.strip()
        if not seccion or seccion.startswith("# "):
            continue
        nombre_seccion = seccion.splitlines()[0].lstrip("# ").strip()
        cuerpo = "\n".join(seccion.splitlines()[1:]).strip()
        for bloque in _partir(cuerpo) if cuerpo else [""]:
            contenido = f"{titulo} — {nombre_seccion}\n{bloque}".strip()
            fragmentos.append(Fragmento(
                id=hashlib.sha256(f"{ruta.name}|{contenido}".encode()).hexdigest()[:16],
                fuente=ruta.name, modelo=ruta.stem.lower(), seccion=nombre_seccion,
                orden=len(fragmentos), texto=contenido,
            ))
    return fragmentos


@lru_cache(maxsize=1)
def cargar_fragmentos() -> tuple[Fragmento, ...]:
    rutas = sorted(Path(config.DATA_DIR).glob("*.md"))
    if not rutas:
        raise RuntimeError(f"No hay fichas .md en '{config.DATA_DIR}'")
    return tuple(f for ruta in rutas for f in fragmentar(ruta))


def huella_del_corpus(fragmentos: tuple[Fragmento, ...]) -> str:
    """Cambia si cambia cualquier fragmento: así el índice vectorial sabe cuándo reconstruirse."""
    return hashlib.sha256("\n".join(f.id for f in fragmentos).encode()).hexdigest()[:16]
