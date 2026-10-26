"""
Contratos de datos de la API (entrega final: "Pydantic en todas las fronteras").

Cada endpoint declara su modelo de entrada y de salida: FastAPI valida lo que
entra (422 si no cumple) y también lo que SALE (si un cambio en el worker
dejara un job con un campo inválido, la API falla en vez de devolver un JSON
roto al cliente). Los mismos modelos generan la documentación de /docs.

El worker arma el resultado de cada job con ResultadoJob antes de guardarlo
en Redis, y el pedido de aprobación humana con PedidoAprobacion: los datos se
validan al escribirse, no recién al leerse.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.jobs import Estado
from orquestador.state import DecisionHumana, PropuestaReserva, ResultadoReserva

# conversacion_id: lo genera la API (un UUID) o lo elige el cliente. Va dentro de claves de Redis,
# así que solo se aceptan caracteres seguros.
PATRON_ID = r"^[A-Za-z0-9_\-]{1,64}$"


# --- Entrada --------------------------------------------------------------------------------
class PedidoTarea(BaseModel):
    tarea: str = Field(
        min_length=5, max_length=1000,
        examples=["Cuantos CV y cuantos Nm mas tiene el Virtus Exclusive que el Tera High?"],
    )
    conversacion_id: str | None = Field(
        default=None, pattern=PATRON_ID,
        description="Para seguir una conversación: el conversacion_id que devolvió el primer POST. "
                    "Sin él, empieza una conversación nueva.",
        examples=[None],
    )
    simulacion: dict[str, Any] | None = Field(
        default=None,
        description="Solo para probar caminos de error: romper_grafo, demora_s, fallar_investigador, corromper_analisis",
        examples=[None],
    )


# --- Salida ---------------------------------------------------------------------------------
class TareaCreada(BaseModel):
    job_id: str
    conversacion_id: str
    estado: Estado
    consultar: str = Field(description="Endpoint para hacer polling del estado")
    eventos: str = Field(description="Endpoint con el progreso en vivo (Server-Sent Events)")


class EventoJob(BaseModel):
    estado: Estado
    en: str
    detalle: str = ""


class DecisionResumida(BaseModel):
    paso: int = Field(ge=0)
    origen: Literal["llm", "regla"]
    siguiente: str
    razon: str


class ResultadoJob(BaseModel):
    respuesta: str
    turno: int = Field(ge=1, description="Número de turno dentro de la conversación")
    fuentes: list[str] = Field(description="Fuentes de los datos validados del turno")
    intervenciones_de_especialistas: int = Field(ge=0)
    escalado_humano: bool
    validaciones_fallidas: int = Field(ge=0)
    decisiones: list[DecisionResumida]
    decision_humana: DecisionHumana | None = None
    reserva: ResultadoReserva | None = None
    prompts_version: str


class PedidoAprobacion(BaseModel):
    """Lo que ve el revisor humano antes de una acción crítica."""

    accion: str
    motivo: str
    pregunta: str
    propuesta: PropuestaReserva
    stock_en_catalogo: int = Field(ge=0)
    tarea_original: str
    como_responder: str


class EstadoJob(BaseModel):
    job_id: str
    conversacion_id: str
    estado: Estado
    tarea: str
    simulacion: dict[str, Any] = Field(default_factory=dict)
    creado_en: str
    actualizado_en: str
    duracion_ejecucion_s: float = Field(ge=0, description="Tiempo de ejecución del grafo (sin la espera en cola)")
    latencia_total_s: float | None = Field(default=None, ge=0, description="De la creación al final (con la cola)")
    finalizado_en: str | None = None
    worker: str | None = Field(default=None, description="Último worker que lo ejecutó")
    error: str | None = None
    aprobacion: PedidoAprobacion | None = None
    decision: DecisionHumana | None = None
    resultado: ResultadoJob | None = None
    historial: list[EventoJob]
    siguiente_paso: str | None = None


class ResumenJob(BaseModel):
    job_id: str
    conversacion_id: str
    estado: Estado
    tarea: str
    creado_en: str
    latencia_total_s: float | None = None
    duracion_ejecucion_s: float


class TurnoConversacion(BaseModel):
    job_id: str
    estado: Estado
    pregunta: str
    respuesta: str | None = None
    creado_en: str


class Conversacion(BaseModel):
    conversacion_id: str
    turnos: list[TurnoConversacion]
    job_activo: str | None = Field(description="Turno en curso; mientras exista, no se acepta otro")
    como_seguir: str


class RespuestaAprobacion(BaseModel):
    job_id: str
    estado: Estado
    decision: DecisionHumana
    detalle: str


class WorkerVivo(BaseModel):
    nombre: str
    jobs_en_paralelo: int = Field(ge=1)
    rag: str
    llm: str
    desde: str


class Salud(BaseModel):
    estado: Literal["ok"]
    redis: Literal["ok"]
    en_cola: int = Field(ge=0)
    workers_en_proceso: int = Field(ge=0, description="Workers dentro de la API (WORKER_EN_PROCESO=true)")
    workers_vivos: list[WorkerVivo] = Field(description="Procesos worker con latido reciente (de cualquier contenedor)")
