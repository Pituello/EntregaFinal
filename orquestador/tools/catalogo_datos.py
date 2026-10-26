"""
"Base de datos" simulada de una concesionaria Volkswagen, en dos tablas:

- VERSIONES: versiones de cada modelo con sus specs. Los datos técnicos son
  REALES: salen del dataset verificado de las Pre-Entregas 3 y 4 (motor,
  potencia, torque, caja).
- STOCK: unidades disponibles por versión y sucursal. Es FICTICIO: no
  existe una concesionaria real detrás, son números de ejemplo.

Por qué dos tablas separadas: el
stock se consulta por version_id, y el usuario nunca lo sabe. Así el agente
tiene que razonar en dos pasos por su cuenta: primero buscar las versiones
del modelo para obtener el ID, después consultar el stock.
"""
import unicodedata

from pydantic import BaseModel


class Version(BaseModel):
    version_id: str
    modelo: str
    version: str
    motor: str
    potencia_cv: int
    torque_nm: int
    transmision: str
    automatica: bool


def _v(version_id, modelo, version, motor, cv, nm, transmision, automatica) -> Version:
    return Version(
        version_id=version_id, modelo=modelo, version=version, motor=motor,
        potencia_cv=cv, torque_nm=nm, transmision=transmision, automatica=automatica,
    )


VERSIONES: list[Version] = [
    _v("TERA-TREND-MT5", "Tera", "Trend MSI MT5", "1.6 MSI atmosférico, 4 cilindros", 110, 155, "manual de 5 velocidades", False),
    _v("TERA-COMFORT-AT6", "Tera", "Comfort 170 TSI AT6", "1.0 TSI turbo, 3 cilindros", 101, 170, "automática Tiptronic de 6 velocidades", True),
    _v("TERA-HIGH-AT6", "Tera", "High 170 TSI AT6", "1.0 TSI turbo, 3 cilindros", 101, 170, "automática Tiptronic de 6 velocidades", True),
    _v("TERA-OUTFIT-AT6", "Tera", "Outfit 170 TSI AT6", "1.0 TSI turbo, 3 cilindros", 101, 170, "automática Tiptronic de 6 velocidades", True),
    _v("POLO-TRACK-MT5", "Polo", "Track", "1.6 atmosférico, 4 cilindros", 110, 155, "manual de 5 velocidades", False),
    _v("POLO-COMFORTLINE-AT6", "Polo", "Comfortline", "1.0 TSI turbo, 3 cilindros", 99, 165, "automática Tiptronic de 6 velocidades", True),
    _v("POLO-HIGHLINE-AT6", "Polo", "Highline", "1.0 TSI turbo, 3 cilindros", 99, 165, "automática Tiptronic de 6 velocidades", True),
    _v("VIRTUS-TRENDLINE-MT", "Virtus", "Trendline", "1.6 MSI atmosférico, 4 cilindros", 110, 155, "manual", False),
    _v("VIRTUS-EXCLUSIVE-AT6", "Virtus", "Exclusive", "1.4 TSI turbo, 4 cilindros", 150, 250, "automática de 6 velocidades", True),
    _v("NIVUS-SENSE-MT5", "Nivus", "Sense", "1.0 TSI turbo, 3 cilindros", 101, 170, "manual de 5 velocidades", False),
    _v("TAOS-14TSI-AT8", "Taos", "1.4 TSI AT8", "1.4 TSI turbo, 4 cilindros", 150, 250, "automática Tiptronic de 8 velocidades", True),
    _v("VENTO-GLI-DSG7", "Vento", "GLI", "2.0 TSI turbo", 230, 350, "automática DSG de doble embrague, 7 velocidades", True),
    _v("SAVEIRO-TRENDLINE-CS", "Saveiro", "Trendline cabina simple", "1.6 MSI, 4 cilindros", 110, 155, "manual de 5 velocidades", False),
    _v("SAVEIRO-COMFORTLINE-CD", "Saveiro", "Comfortline cabina doble", "1.6 MSI, 4 cilindros", 110, 155, "manual de 5 velocidades", False),
    _v("SAVEIRO-EXTREME-CD", "Saveiro", "Extreme cabina doble", "1.6 MSI, 4 cilindros", 110, 155, "manual de 5 velocidades", False),
]

SUCURSALES = ["Palermo", "Belgrano", "Córdoba", "Rosario"]

# FICTICIO: unidades disponibles por versión y sucursal (0 = sin stock).
STOCK: dict[str, dict[str, int]] = {
    "TERA-TREND-MT5":         {"Palermo": 4, "Belgrano": 2, "Córdoba": 3, "Rosario": 1},
    "TERA-COMFORT-AT6":       {"Palermo": 0, "Belgrano": 3, "Córdoba": 1, "Rosario": 0},
    "TERA-HIGH-AT6":          {"Palermo": 2, "Belgrano": 0, "Córdoba": 0, "Rosario": 1},
    "TERA-OUTFIT-AT6":        {"Palermo": 1, "Belgrano": 1, "Córdoba": 0, "Rosario": 0},
    "POLO-TRACK-MT5":         {"Palermo": 5, "Belgrano": 4, "Córdoba": 2, "Rosario": 3},
    "POLO-COMFORTLINE-AT6":   {"Palermo": 2, "Belgrano": 0, "Córdoba": 1, "Rosario": 2},
    "POLO-HIGHLINE-AT6":      {"Palermo": 1, "Belgrano": 1, "Córdoba": 0, "Rosario": 0},
    "VIRTUS-TRENDLINE-MT":    {"Palermo": 3, "Belgrano": 2, "Córdoba": 2, "Rosario": 1},
    "VIRTUS-EXCLUSIVE-AT6":   {"Palermo": 1, "Belgrano": 0, "Córdoba": 1, "Rosario": 0},
    "NIVUS-SENSE-MT5":        {"Palermo": 2, "Belgrano": 3, "Córdoba": 0, "Rosario": 2},
    "TAOS-14TSI-AT8":         {"Palermo": 1, "Belgrano": 2, "Córdoba": 1, "Rosario": 0},
    "VENTO-GLI-DSG7":         {"Palermo": 1, "Belgrano": 0, "Córdoba": 0, "Rosario": 0},
    "SAVEIRO-TRENDLINE-CS":   {"Palermo": 2, "Belgrano": 1, "Córdoba": 4, "Rosario": 3},
    "SAVEIRO-COMFORTLINE-CD": {"Palermo": 1, "Belgrano": 0, "Córdoba": 2, "Rosario": 1},
    "SAVEIRO-EXTREME-CD":     {"Palermo": 0, "Belgrano": 1, "Córdoba": 2, "Rosario": 0},
}


def normalizar(texto: str) -> str:
    """'Córdoba ' -> 'cordoba': para comparar sin depender de tildes ni mayúsculas."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return sin_tildes.strip().lower()
