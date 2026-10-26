"""
Herramienta con EFECTO SECUNDARIO: reservar una unidad del stock.

Es la acción "crítica" de la Pre-Entrega 7. A diferencia de consultar_catalogo
(solo lee), reservar_unidad compromete una unidad real: si se ejecuta por
error, otro cliente se queda sin el auto. Por eso ningún agente la tiene entre
sus herramientas: la ejecuta el nodo ejecutar_reserva del grafo, y a ese nodo
solo se llega pasando por la aprobación humana (app/hitl.py).

La reserva se guarda en Redis con un script Lua, que corre de forma atómica:
- controla el stock: el catálogo dice cuántas unidades hay en la sucursal y
  Redis lleva la cuenta de cuántas ya se reservaron. Con 5 pedidos en
  paralelo, dos no pueden quedarse con la última unidad;
- es idempotente: la clave de idempotencia es el ID del job. Si el worker se
  reinicia y el nodo se ejecuta otra vez, devuelve la reserva que ya existía
  en lugar de descontar stock de nuevo.
"""
import json
from datetime import datetime, timezone
from typing import Literal, Protocol

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .catalogo_datos import STOCK, SUCURSALES, VERSIONES, normalizar

PREFIJO = "vw:reserva"


class RegistroReservas(Protocol):
    async def reservar(self, clave_idempotencia: str, version_id: str, sucursal: str, stock: int,
                       reserva: dict) -> dict: ...


# KEYS[1] = registro de la reserva (idempotencia) · KEYS[2] = contador de la versión en la sucursal
# KEYS[3] = lista con todas las reservas · ARGV[1] = stock del catálogo · ARGV[2] = la reserva en JSON
_LUA_RESERVAR = """
local existente = redis.call('GET', KEYS[1])
if existente then return {'existente', existente} end
local reservadas = tonumber(redis.call('GET', KEYS[2]) or '0')
if reservadas >= tonumber(ARGV[1]) then return {'sin_stock', tostring(reservadas)} end
redis.call('INCR', KEYS[2])
redis.call('SET', KEYS[1], ARGV[2])
redis.call('RPUSH', KEYS[3], KEYS[1])
return {'nueva', ARGV[2]}
"""


class RegistroRedis:
    def __init__(self, redis_url: str, prefijo: str = PREFIJO):
        from redis.asyncio import from_url

        self._redis = from_url(redis_url, decode_responses=True)
        self._script = self._redis.register_script(_LUA_RESERVAR)
        self._prefijo = prefijo

    async def reservar(self, clave_idempotencia: str, version_id: str, sucursal: str, stock: int,
                       reserva: dict) -> dict:
        claves = [
            f"{self._prefijo}:{clave_idempotencia}",                               # la reserva de este job
            f"{self._prefijo}s:reservadas:{version_id}:{normalizar(sucursal)}",    # unidades ya reservadas
            f"{self._prefijo}s",                                                   # lista de todas las reservas
        ]
        estado, valor = await self._script(keys=claves, args=[stock, json.dumps(reserva, ensure_ascii=False)])
        if estado == "sin_stock":
            return {"estado": "sin_stock", "reservadas": int(valor)}
        return {"estado": estado, "reserva": json.loads(valor)}


class RegistroEnMemoria:
    """La misma lógica sin Redis: para los tests."""

    def __init__(self):
        self.reservas: dict[str, dict] = {}
        self.contadores: dict[str, int] = {}

    async def reservar(self, clave_idempotencia: str, version_id: str, sucursal: str, stock: int,
                       reserva: dict) -> dict:
        clave = f"{PREFIJO}:{clave_idempotencia}"
        contador = f"{version_id}:{sucursal}"
        if clave in self.reservas:
            return {"estado": "existente", "reserva": self.reservas[clave]}
        if self.contadores.get(contador, 0) >= stock:
            return {"estado": "sin_stock", "reservadas": self.contadores.get(contador, 0)}
        self.contadores[contador] = self.contadores.get(contador, 0) + 1
        self.reservas[clave] = reserva
        return {"estado": "nueva", "reserva": reserva}


_registro: RegistroReservas | None = None


def usar_registro(registro: RegistroReservas | None) -> None:
    """Permite inyectar el registro (los tests usan RegistroEnMemoria)."""
    global _registro
    _registro = registro


def _obtener_registro() -> RegistroReservas:
    global _registro
    if _registro is None:
        from orquestador import config

        _registro = RegistroRedis(config.redis_url())
    return _registro


def buscar_version(version_id: str):
    return next((v for v in VERSIONES if v.version_id == version_id.strip().upper()), None)


def resolver_sucursal(texto: str) -> str | None:
    return next((s for s in SUCURSALES if normalizar(s) == normalizar(texto)), None)


class ReservarUnidadInput(BaseModel):
    version_id: str = Field(min_length=3, max_length=40, description="ID exacto de la versión, ej: 'SAVEIRO-EXTREME-CD'")
    sucursal: str = Field(min_length=3, max_length=40, description="Palermo, Belgrano, Córdoba o Rosario")
    cliente: str = Field(min_length=1, max_length=80, description="Nombre del cliente")
    clave_idempotencia: str = Field(min_length=3, max_length=80, description="ID único de la operación (el ID del job)")


class ReservaConfirmada(BaseModel):
    """Salida de reservar_unidad: lo que queda registrado en Redis y lo que lee el nodo ejecutar_reserva."""

    reserva_id: str = Field(pattern=r"^R-[A-Z0-9\-]{1,8}$")
    version_id: str
    modelo: str
    version: str
    sucursal: Literal[tuple(SUCURSALES)]  # type: ignore[valid-type]
    cliente: str = Field(min_length=1)
    creada_en: str
    estado: Literal["nueva", "existente"] = "nueva"


@tool(args_schema=ReservarUnidadInput)
async def reservar_unidad(version_id: str, sucursal: str, cliente: str, clave_idempotencia: str) -> str:
    """Reserva UNA unidad de una versión en una sucursal. ACCIÓN CON EFECTO SECUNDARIO:
    compromete stock real, así que solo se ejecuta después de una aprobación humana.
    Devuelve un JSON con la reserva, o un texto que empieza con ERROR si no se pudo."""
    version = buscar_version(version_id)
    if version is None:
        return f"ERROR: la versión '{version_id}' no existe en el catálogo."
    nombre_sucursal = resolver_sucursal(sucursal)
    if nombre_sucursal is None:
        return f"ERROR: la sucursal '{sucursal}' no existe. Sucursales: {', '.join(SUCURSALES)}."

    stock = STOCK[version.version_id][nombre_sucursal]
    reserva = ReservaConfirmada(
        reserva_id=f"R-{clave_idempotencia[:8].upper()}",
        version_id=version.version_id,
        modelo=version.modelo,
        version=version.version,
        sucursal=nombre_sucursal,
        cliente=cliente.strip(),
        creada_en=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    resultado = await _obtener_registro().reservar(
        clave_idempotencia=clave_idempotencia, version_id=version.version_id, sucursal=nombre_sucursal,
        stock=stock, reserva=reserva.model_dump(exclude={"estado"}),
    )
    if resultado["estado"] == "sin_stock":
        return (f"ERROR: no quedan unidades de {version.modelo} {version.version} en {nombre_sucursal} "
                f"(stock del catálogo: {stock}, ya reservadas: {resultado['reservadas']}).")
    # Lo que vuelve de Redis también se valida: si el registro guardado estuviera corrupto, falla acá.
    confirmada = ReservaConfirmada(**resultado["reserva"], estado=resultado["estado"])
    return json.dumps(confirmada.model_dump(), ensure_ascii=False)
