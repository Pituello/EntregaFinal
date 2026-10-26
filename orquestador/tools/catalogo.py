"""
Herramienta del Investigador: consulta la "base de datos" de la concesionaria
de la Pre-Entrega 5 (catalogo_datos.py): versiones con specs reales y stock
por sucursal (ficticio).

Devuelve JSON con la clave "fuente", así el Validador sabe de dónde sale cada
número que el Investigador cite. Entrada y salida son modelos Pydantic: la
salida se arma como ResultadoCatalogo (stock >= 0, fuente fija) antes de
serializarse, así un error en los datos no llega al LLM como un JSON "válido".
"""
import json
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .catalogo_datos import STOCK, VERSIONES, Version, normalizar

FUENTE = "catalogo_concesionaria"


class ConsultarCatalogoInput(BaseModel):
    modelo: str = Field(
        description="Modelo de Volkswagen sin la marca ni la versión, ej: 'Tera', 'Virtus', 'Saveiro'. "
                    "Si te piden una versión (ej: 'Virtus Exclusive'), pasá solo el modelo ('Virtus') "
                    "y elegí la versión dentro del resultado."
    )


class VersionConStock(Version):
    stock: dict[str, int] = Field(description="Unidades disponibles por sucursal")

    def model_post_init(self, _contexto) -> None:
        if any(unidades < 0 for unidades in self.stock.values()):
            raise ValueError(f"stock negativo en {self.version_id}")


class ResultadoCatalogo(BaseModel):
    fuente: Literal["catalogo_concesionaria"] = FUENTE
    modelo: str
    versiones: list[VersionConStock] = Field(min_length=1)


def _resolver_modelo(texto: str) -> str | None:
    """
    'Virtus' -> 'Virtus'; 'Virtus Exclusive' -> 'Virtus'; 'VW Vento GLI' -> 'Vento'.
    En la corrida real con gpt-4o-mini, el Investigador pasaba el modelo con la
    versión pegada ('Virtus Exclusive'), la búsqueda exacta fallaba y el
    supervisor lo reenviaba una y otra vez hasta MAX_PASOS. Ahora se busca el
    nombre del modelo como palabra dentro del texto.
    """
    palabras = normalizar(texto).replace("-", " ").split()
    for modelo in sorted({v.modelo for v in VERSIONES}):
        if normalizar(modelo) in palabras:
            return modelo
    return None


@tool(args_schema=ConsultarCatalogoInput)
async def consultar_catalogo(modelo: str) -> str:
    """Consulta el catálogo de la concesionaria: todas las versiones de un modelo con
    motor, potencia (CV), torque (Nm), transmisión, y el stock de unidades en cada
    sucursal (Palermo, Belgrano, Córdoba, Rosario). Usala para datos de versiones,
    potencia, torque, cajas o stock. Para datos que NO están acá (seguridad, Latin NCAP,
    dimensiones, historia, fabricación) usá buscar_en_fichas.
    Devuelve un texto que empieza con ERROR si el modelo no está en el catálogo."""
    encontrado = _resolver_modelo(modelo)
    if encontrado is None:
        modelos = ", ".join(sorted({v.modelo for v in VERSIONES}))
        return f"ERROR: el modelo '{modelo}' no está en el catálogo. Modelos disponibles: {modelos}."
    resultado = ResultadoCatalogo(
        modelo=encontrado,
        versiones=[VersionConStock(**v.model_dump(), stock=STOCK[v.version_id])
                   for v in VERSIONES if v.modelo == encontrado],
    )
    return json.dumps(resultado.model_dump(), ensure_ascii=False)  # mismo formato que lee el Validador
