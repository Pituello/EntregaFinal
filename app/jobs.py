"""
Estado de los jobs en Redis: el "buzón" entre la API y los workers.

Claves:
- vw:job:{id}                     hash con el estado del job (lo que devuelve GET /tasks/{id})
- vw:job:{id}:historial           lista con cada cambio de estado y cuándo pasó
- vw:job:{id}:progreso            lista con lo que hizo cada nodo del grafo (GET /tasks/{id}/eventos)
- vw:job:{id}:lease               "contrato" del worker que lo está corriendo (vence si el worker muere)
- vw:cola                         lista que funciona como cola (RPUSH para encolar, BLPOP para tomar)
- vw:jobs                         sorted set con todos los jobs por fecha (para listarlos)
- vw:en_curso                     set de jobs en RUNNING (lo revisa el vigilante de leases)
- vw:conversacion:{id}:jobs       los turnos (jobs) de una conversación, en orden
- vw:conversacion:{id}:activo     el job que la conversación tiene en curso (uno a la vez)
- vw:worker:{nombre}              latido de cada proceso worker (GET /health los lista)

Ciclo de vida:
    PENDING -> RUNNING -> DONE
                       -> FAILED            (cualquier excepción o timeout)
                       -> WAITING_APPROVAL  (pausa HITL) -> PENDING -> RUNNING -> ...
                       -> PENDING           (el worker murió: su lease venció y otro lo retoma)

Entrega final: workers independientes y escalables (devolución de la
Pre-Entrega 7: "escalar horizontalmente con workers independientes"). Con
varios procesos worker leyendo la misma cola, "quién tiene cada job" ya no
puede suponerse: cada transición que importa es un script Lua (atómico en
Redis), y cada job en RUNNING tiene un LEASE con vencimiento que su worker
renueva mientras trabaja. Si el worker muere (se apaga el contenedor, se
corta la luz), el lease vence y el vigilante de cualquier otro worker lo
devuelve a la cola para retomarlo desde su último checkpoint. Un worker que
se reinicia ya no "roba" los jobs de los demás: solo se recuperan leases
vencidos.

Errores comunes que esto evita:
- "No persistir el estado inicial": crear() guarda el job y lo encola en un
  solo script: el job existe en Redis antes de que un worker pueda tomarlo.
- "Ignorar los errores en background": el worker marca FAILED ante cualquier
  excepción (ver worker.py), así el cliente nunca queda en un polling infinito.
- Dos turnos de la misma conversación a la vez: correrían sobre el mismo hilo
  del checkpointer y se pisarían. crear() lo rechaza (409 en la API).
"""
import json
import time
import uuid
from datetime import datetime, timezone
from enum import StrEnum

from redis.asyncio import Redis, from_url

PREFIJO_JOB = "vw:job:"
COLA = "vw:cola"
INDICE = "vw:jobs"
EN_CURSO = "vw:en_curso"
PREFIJO_CONVERSACION = "vw:conversacion:"
PREFIJO_WORKER = "vw:worker:"
CAMPOS_JSON = ("simulacion", "resultado", "aprobacion", "decision")


class Estado(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINALES = {Estado.DONE, Estado.FAILED}


class ConversacionOcupada(Exception):
    """La conversación ya tiene un turno en curso (un job que no terminó)."""

    def __init__(self, conversacion_id: str, job_activo: str):
        super().__init__(f"La conversación '{conversacion_id}' ya tiene el job {job_activo} en curso")
        self.conversacion_id, self.job_activo = conversacion_id, job_activo


# --- Scripts Lua: cada uno corre de una sola vez en Redis (nadie se mete en el medio) -------------
# Crear: si la conversación tiene un job vivo, no crea nada y devuelve su id.
# KEYS: job, historial, índice, cola, activo, turnos · ARGV: job_id, score, evento, mensaje, prefijo, campos...
_LUA_CREAR = """
local activo = redis.call('GET', KEYS[5])
if activo then
  local estado = redis.call('HGET', ARGV[5] .. activo, 'estado')
  if estado and estado ~= 'DONE' and estado ~= 'FAILED' then return activo end
end
redis.call('HSET', KEYS[1], unpack(ARGV, 6))
redis.call('RPUSH', KEYS[2], ARGV[3])
redis.call('ZADD', KEYS[3], ARGV[2], ARGV[1])
redis.call('SET', KEYS[5], ARGV[1])
redis.call('RPUSH', KEYS[6], ARGV[1])
redis.call('RPUSH', KEYS[4], ARGV[4])
return ''
"""

# Reclamar: solo un job PENDING pasa a RUNNING, con el lease de quien lo tomó. Un mensaje duplicado
# en la cola (o el mismo job tomado por dos workers) no lo ejecuta dos veces.
# KEYS: job, historial, lease, en_curso · ARGV: worker, ahora, evento, lease_s, job_id
_LUA_RECLAMAR = """
if redis.call('HGET', KEYS[1], 'estado') ~= 'PENDING' then return 0 end
redis.call('HSET', KEYS[1], 'estado', 'RUNNING', 'worker', ARGV[1], 'actualizado_en', ARGV[2])
redis.call('RPUSH', KEYS[2], ARGV[3])
redis.call('SET', KEYS[3], ARGV[1], 'EX', ARGV[4])
redis.call('SADD', KEYS[4], ARGV[5])
return 1
"""

# Devolver a la cola un job RUNNING. Con ARGV[5] = '1' solo si su lease venció (el vigilante);
# con '0', siempre (apagado ordenado del propio worker).
# KEYS: job, historial, lease, en_curso, cola · ARGV: ahora, evento, mensaje, job_id, exigir_vencido
_LUA_DEVOLVER = """
if redis.call('HGET', KEYS[1], 'estado') ~= 'RUNNING' then
  redis.call('SREM', KEYS[4], ARGV[4])
  return 0
end
if ARGV[5] == '1' and redis.call('EXISTS', KEYS[3]) == 1 then return 0 end
redis.call('HSET', KEYS[1], 'estado', 'PENDING', 'actualizado_en', ARGV[1])
redis.call('RPUSH', KEYS[2], ARGV[2])
redis.call('DEL', KEYS[3])
redis.call('SREM', KEYS[4], ARGV[4])
redis.call('RPUSH', KEYS[5], ARGV[3])
return 1
"""

# Aprobar es "leer el estado y, si está esperando, cambiarlo y encolar": tiene que ser atómico,
# si no dos aprobaciones simultáneas reanudarían el grafo dos veces.
_LUA_REGISTRAR_DECISION = """
if redis.call('HGET', KEYS[1], 'estado') ~= 'WAITING_APPROVAL' then return 0 end
redis.call('HSET', KEYS[1], 'estado', 'PENDING', 'decision', ARGV[1], 'actualizado_en', ARGV[2])
redis.call('RPUSH', KEYS[2], ARGV[3])
redis.call('RPUSH', KEYS[3], ARGV[4])
return 1
"""

# Liberar la conversación solo si el job activo sigue siendo este (no pisar a un turno posterior).
_LUA_LIBERAR = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


ESPERA_COLA_S = 5


def conectar(redis_url: str) -> Redis:
    """
    Cliente async para la API y los workers. socket_timeout explícito: desde
    redis-py 8 el default es 5 s, igual que la espera del BLPOP de la cola, y
    cada worker cortaba su conexión con "Timeout reading from redis" (visto al
    probar el docker compose). El timeout del socket tiene que ser mayor.
    """
    return from_url(redis_url, decode_responses=True, socket_timeout=ESPERA_COLA_S * 6,
                    socket_connect_timeout=5, health_check_interval=30)


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clave(job_id: str) -> str:
    return f"{PREFIJO_JOB}{job_id}"


def _conversacion(conversacion_id: str, sufijo: str) -> str:
    return f"{PREFIJO_CONVERSACION}{conversacion_id}:{sufijo}"


def _mensaje(job_id: str, accion: str) -> str:
    return json.dumps({"job_id": job_id, "accion": accion})


def _evento(estado: str, detalle: str = "") -> str:
    return json.dumps({"estado": estado, "en": ahora(), "detalle": detalle}, ensure_ascii=False)


class RepositorioJobs:
    def __init__(self, redis: Redis):
        self.redis = redis
        self._crear = redis.register_script(_LUA_CREAR)
        self._reclamar = redis.register_script(_LUA_RECLAMAR)
        self._devolver = redis.register_script(_LUA_DEVOLVER)
        self._registrar_decision = redis.register_script(_LUA_REGISTRAR_DECISION)
        self._liberar = redis.register_script(_LUA_LIBERAR)

    # --- API ------------------------------------------------------------------------------
    async def crear(self, tarea: str, simulacion: dict | None = None, conversacion_id: str | None = None) -> dict:
        """
        Guarda el job (PENDING) y lo encola. Sin conversacion_id empieza una conversación nueva.
        Lanza ConversacionOcupada si la conversación tiene otro turno sin terminar.
        """
        job_id = str(uuid.uuid4())
        conversacion_id = conversacion_id or job_id
        marca = ahora()
        job = {
            "job_id": job_id,
            "conversacion_id": conversacion_id,
            "estado": Estado.PENDING.value,
            "tarea": tarea,
            "simulacion": json.dumps(simulacion or {}),
            "creado_en": marca,
            "creado_ts": f"{time.time():.3f}",
            "actualizado_en": marca,
            "duracion_ejecucion_s": "0",
        }
        campos = [x for par in job.items() for x in par]
        activo = await self._crear(
            keys=[_clave(job_id), f"{_clave(job_id)}:historial", INDICE, COLA,
                  _conversacion(conversacion_id, "activo"), _conversacion(conversacion_id, "jobs")],
            args=[job_id, time.time(), _evento(Estado.PENDING, "job creado y encolado"),
                  _mensaje(job_id, "iniciar"), PREFIJO_JOB, *campos],
        )
        if activo:
            raise ConversacionOcupada(conversacion_id, activo)
        return await self.obtener(job_id)

    async def obtener(self, job_id: str) -> dict | None:
        crudo = await self.redis.hgetall(_clave(job_id))
        if not crudo:
            return None
        job: dict = dict(crudo)
        for campo in CAMPOS_JSON:
            if campo in job:
                job[campo] = json.loads(job[campo])
        job["duracion_ejecucion_s"] = float(job.get("duracion_ejecucion_s", 0))
        if "latencia_total_s" in job:
            job["latencia_total_s"] = float(job["latencia_total_s"])
        job.setdefault("conversacion_id", job_id)  # jobs creados antes de las conversaciones
        job["historial"] = [json.loads(e) for e in await self.redis.lrange(f"{_clave(job_id)}:historial", 0, -1)]
        return job

    async def listar(self, cantidad: int = 20) -> list[dict]:
        ids = await self.redis.zrevrange(INDICE, 0, cantidad - 1)
        jobs = [await self.obtener(job_id) for job_id in ids]
        return [j for j in jobs if j is not None]

    async def turnos(self, conversacion_id: str) -> tuple[list[dict], str | None]:
        """Los jobs de una conversación, en orden, y el que está en curso (si hay)."""
        ids = await self.redis.lrange(_conversacion(conversacion_id, "jobs"), 0, -1)
        jobs = [j for j in [await self.obtener(i) for i in ids] if j is not None]
        activo = await self.redis.get(_conversacion(conversacion_id, "activo"))
        if activo and not any(j["job_id"] == activo and j["estado"] not in TERMINALES for j in jobs):
            activo = None
        return jobs, activo

    async def progreso(self, job_id: str, desde: int = 0) -> list[dict]:
        return [json.loads(e) for e in await self.redis.lrange(f"{_clave(job_id)}:progreso", desde, -1)]

    async def largo_de_cola(self) -> int:
        return await self.redis.llen(COLA)

    async def registrar_decision(self, job_id: str, decision: dict) -> bool:
        detalle = f"{'aprobado' if decision['aprobado'] else 'rechazado'} por {decision.get('revisor', 'anónimo')}"
        resultado = await self._registrar_decision(
            keys=[_clave(job_id), COLA, f"{_clave(job_id)}:historial"],
            args=[json.dumps(decision, ensure_ascii=False), ahora(), _mensaje(job_id, "reanudar"),
                  _evento(Estado.PENDING, f"decisión humana: {detalle}; reanudación encolada")],
        )
        return bool(resultado)

    # --- Worker ---------------------------------------------------------------------------
    async def tomar(self, timeout_s: int = ESPERA_COLA_S) -> dict | None:
        """BLPOP: espera un mensaje sin consumir CPU. El timeout permite cortar el bucle al apagar."""
        item = await self.redis.blpop([COLA], timeout=timeout_s)
        return json.loads(item[1]) if item else None

    async def reclamar(self, job_id: str, worker: str, accion: str, lease_s: int) -> bool:
        """PENDING -> RUNNING con lease. False si el job no estaba PENDING (duplicado, ya tomado, terminado)."""
        return bool(await self._reclamar(
            keys=[_clave(job_id), f"{_clave(job_id)}:historial", f"{_clave(job_id)}:lease", EN_CURSO],
            args=[worker, ahora(), _evento(Estado.RUNNING, f"{accion} ({worker})"), lease_s, job_id],
        ))

    async def renovar_lease(self, job_id: str, worker: str, lease_s: int) -> None:
        await self.redis.set(f"{_clave(job_id)}:lease", worker, ex=lease_s)

    async def actualizar(self, job_id: str, estado: Estado | None = None, detalle: str = "", **campos) -> None:
        valores = {"actualizado_en": ahora()}
        for campo, valor in campos.items():
            valores[campo] = json.dumps(valor, ensure_ascii=False) if campo in CAMPOS_JSON else str(valor)
        async with self.redis.pipeline(transaction=True) as pipe:
            if estado is not None:
                valores["estado"] = estado.value
                pipe.rpush(f"{_clave(job_id)}:historial", _evento(estado, detalle))
                if estado != Estado.RUNNING:  # dejó de correr: ya no necesita lease
                    pipe.delete(f"{_clave(job_id)}:lease")
                    pipe.srem(EN_CURSO, job_id)
            pipe.hset(_clave(job_id), mapping=valores)
            await pipe.execute()
        if estado in TERMINALES:  # el turno terminó: la conversación acepta el siguiente
            conversacion_id = await self.redis.hget(_clave(job_id), "conversacion_id") or job_id
            await self._liberar(keys=[_conversacion(conversacion_id, "activo")], args=[job_id])

    async def registrar_progreso(self, job_id: str, nodo: str, detalle: str) -> None:
        evento = json.dumps({"nodo": nodo, "detalle": detalle, "en": ahora()}, ensure_ascii=False)
        await self.redis.rpush(f"{_clave(job_id)}:progreso", evento)

    async def sumar_duracion(self, job_id: str, segundos: float) -> None:
        await self.redis.hincrbyfloat(_clave(job_id), "duracion_ejecucion_s", round(segundos, 3))

    async def devolver_a_la_cola(self, job_id: str, detalle: str, solo_si_vencio: bool) -> bool:
        return bool(await self._devolver(
            keys=[_clave(job_id), f"{_clave(job_id)}:historial", f"{_clave(job_id)}:lease", EN_CURSO, COLA],
            args=[ahora(), _evento(Estado.PENDING, detalle), _mensaje(job_id, "continuar"), job_id,
                  "1" if solo_si_vencio else "0"],
        ))

    async def recuperar_interrumpidos(self) -> list[str]:
        """
        Jobs en RUNNING cuyo lease venció: el worker que los tenía murió a mitad
        de camino. Se vuelven a encolar con la acción "continuar" y el grafo
        retoma desde su último checkpoint en Redis, sin repetir los pasos ya
        hechos. Lo corre el vigilante de cada worker; el script Lua garantiza
        que, aunque dos vigilantes lo vean a la vez, se encola una sola vez.
        """
        recuperados = []
        for job_id in await self.redis.smembers(EN_CURSO):
            if await self.devolver_a_la_cola(job_id, "el lease del worker venció: se retoma desde el último "
                                                     "checkpoint", solo_si_vencio=True):
                recuperados.append(job_id)
        return recuperados

    async def latido_de_worker(self, nombre: str, datos: dict, ttl_s: int) -> None:
        await self.redis.set(f"{PREFIJO_WORKER}{nombre}", json.dumps(datos, ensure_ascii=False), ex=ttl_s)

    async def baja_de_worker(self, nombre: str) -> None:
        await self.redis.delete(f"{PREFIJO_WORKER}{nombre}")

    async def workers_vivos(self) -> dict[str, dict]:
        vivos = {}
        async for clave in self.redis.scan_iter(match=f"{PREFIJO_WORKER}*", count=100):
            valor = await self.redis.get(clave)
            if valor:
                vivos[clave.removeprefix(PREFIJO_WORKER)] = json.loads(valor)
        return vivos
