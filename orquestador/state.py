"""
Estado compartido del orquestador: el "contrato de interfaz" entre agentes.

Tres ideas de la clase, aplicadas acá:

1. Esquemas validados, no diccionarios sueltos: lo que produce cada
   especialista es un modelo Pydantic (AporteInvestigacion, AporteAnalisis).
   Si el LLM devuelve un score fuera de rango o un campo faltante, falla al
   construirse, antes de entrar al estado.
2. Aislamiento de escritura: cada nodo es dueño de una "parcela" del estado
   (ver PARCELAS) y graph.py verifica en cada paso que no escriba fuera de
   ella. Así un agente nunca pisa lo que calculó otro.
3. Trazabilidad: decisiones, contribuciones y validaciones se ACUMULAN
   en vez de sobrescribirse: el historial completo de quién decidió qué, por
   qué y con qué resultado queda en el estado.

Entrega final, conversaciones: el thread_id del checkpointer es la
CONVERSACIÓN, no el job. Cada pregunta del usuario es un "turno" que corre
sobre el mismo hilo, así el grafo recuerda los turnos anteriores (historial)
aunque se reinicien la API y los workers. Lo que es de un solo turno
(decisiones, validaciones, contribuciones del turno) se reinicia al empezar el
siguiente: por eso esos campos usan el reducer acumular(), que suma como
operator.add pero vuelve a vacío si recibe None (ver estado_del_turno()).
"""
import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, TypeVar

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Especialista = Literal["investigador", "analista", "reservas"]
T = TypeVar("T")
TURNOS_DE_CONTEXTO = 3  # cuántos turnos anteriores ven el supervisor y la síntesis


def acumular(actual: list[T], nuevo: list[T] | None) -> list[T]:
    """Reducer: agrega como operator.add, pero None reinicia la lista (empieza un turno nuevo)."""
    return [] if nuevo is None else actual + nuevo
Siguiente = Literal["investigador", "analista", "reservas", "FINALIZAR"]


# --- Lo que produce el Investigador ------------------------------------------
class DatoCitado(BaseModel):
    """Un dato numérico con su fuente: la unidad mínima de evidencia."""

    descripcion: str = Field(description="Qué es el dato, ej: 'potencia del Tera High'")
    valor: float = Field(description="El número tal cual aparece en la fuente, sin unidad")
    unidad: str = Field(description="Unidad del valor, ej: 'CV', 'Nm', '%', 'unidades'")
    fuente: str = Field(description="De dónde salió: 'catalogo_concesionaria' o el archivo de ficha, ej: 'tera.md'")


class AporteInvestigacion(BaseModel):
    """Salida estructurada del Investigador (su parcela: 'investigacion')."""

    resumen: str = Field(description="Resumen breve de lo encontrado")
    datos: list[DatoCitado] = Field(
        default_factory=list, description="Todos los datos numéricos necesarios para la tarea, cada uno con su fuente"
    )


# --- Lo que produce el Analista ----------------------------------------------
class Calculo(BaseModel):
    descripcion: str = Field(description="Qué se calcula, ej: 'diferencia de potencia'")
    expresion: str = Field(description="Expresión aritmética usada, solo con números de la investigación, ej: '150 - 101'")
    resultado: float = Field(description="Resultado de la expresión")
    unidad: str = Field(description="Unidad del resultado")


class AporteAnalisis(BaseModel):
    """Salida estructurada del Analista (su parcela: 'analisis')."""

    calculos: list[Calculo] = Field(min_length=1, description="Cada cálculo hecho, con su expresión y resultado")
    conclusion: str = Field(description="Conclusión breve basada en los cálculos")


# --- Lo que produce el Gestor de reservas (Pre-Entrega 7) ---------------------
class PropuestaReserva(BaseModel):
    """
    Salida estructurada del Gestor de reservas. PREPARA la reserva, no la
    ejecuta: reservar compromete stock real (efecto secundario), así que la
    ejecución recién ocurre después de la aprobación humana (nodo HITL).
    """

    version_id: str = Field(description="ID exacto de la versión según consultar_catalogo, ej: 'SAVEIRO-EXTREME-CD'")
    sucursal: str = Field(description="Sucursal donde se reserva: Palermo, Belgrano, Córdoba o Rosario")
    cliente: str = Field(description="Nombre del cliente para el que se reserva la unidad")
    resumen: str = Field(description="Descripción breve de la reserva propuesta, ej: '1 Saveiro Extreme en Córdoba'")


class DecisionHumana(BaseModel):
    """Lo que responde el revisor humano en POST /tasks/{id}/approve."""

    aprobado: bool
    revisor: str = Field(default="anónimo", max_length=80)
    comentario: str = Field(default="", max_length=500)


class ResultadoReserva(BaseModel):
    """Resultado de ejecutar la herramienta con efecto secundario (reservar_unidad)."""

    ok: bool
    reserva_id: str | None = None
    detalle: str


# --- Lo que produce el Validador (determinístico, sin LLM) -------------------
class ResultadoValidacion(BaseModel):
    agente: Especialista
    ok: bool
    errores: list[str] = Field(default_factory=list)
    fallo_de_ejecucion: bool = False  # el especialista lanzó una excepción (no llegó a producir un aporte)
    # Números del análisis que no están en la investigación validada de ESTE turno. No es un error
    # de cuenta: falta un dato (por ejemplo, uno de un turno anterior), y lo tiene que buscar el investigador.
    numeros_sin_respaldo: list[float] = Field(default_factory=list)


# --- Lo que decide el Supervisor ---------------------------------------------
class DecisionSupervisor(BaseModel):
    """Salida estructurada del LLM supervisor (with_structured_output)."""

    siguiente: Siguiente = Field(description="Próximo especialista, o FINALIZAR si la tarea está completa")
    instruccion: str = Field(
        default="",
        description="Instrucción concreta y autocontenida para el especialista elegido (vacía si FINALIZAR)",
    )
    razon: str = Field(description="Justificación breve de la decisión")


class DecisionRegistrada(DecisionSupervisor):
    paso: int
    origen: Literal["llm", "regla"]  # "regla" = decisión determinística (límites, reparación, guardas)


# --- Logs ---------------------------------------------------------------------
class Contribucion(BaseModel):
    """Registro de cada intervención de un especialista (provenance)."""

    paso: int
    agente: Especialista
    ok: bool
    resumen: str
    error: str | None = None
    duracion_s: float = Field(ge=0)


# --- Memoria de la conversación (entrega final) -------------------------------
class Turno(BaseModel):
    """Una pregunta del usuario y la respuesta que recibió. Lo escribe la síntesis."""

    job_id: str
    pregunta: str
    respuesta: str
    en: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


# --- El estado ----------------------------------------------------------------
class EstadoOrquestador(MessagesState):
    """
    Hereda 'messages' (reducer add_messages) de MessagesState y suma:
    - campos de control, que solo escribe el supervisor;
    - una parcela por agente para sus resultados;
    - logs acumulativos (acumular) para la trazabilidad, que se reinician en cada turno;
    - el historial de la conversación (operator.add), que nunca se reinicia.
    """

    tarea: str
    job_id: str  # el turno en curso: un job de la API
    historial: Annotated[list[Turno], operator.add]

    # Control (dueño: supervisor)
    siguiente: Siguiente | None
    instruccion: str
    pasos: int
    reparaciones: dict[str, int]
    escalado_humano: bool
    decisiones: Annotated[list[DecisionRegistrada], acumular]

    # Parcela del investigador
    investigacion: AporteInvestigacion | None
    evidencia: list[str]  # salidas crudas de sus herramientas: contra esto se validan sus datos

    # Parcela del analista
    analisis: AporteAnalisis | None

    # Parcela del gestor de reservas (Pre-Entrega 7)
    propuesta_reserva: PropuestaReserva | None

    # Parcela del nodo de aprobación humana (HITL)
    decision_humana: DecisionHumana | None

    # Parcela del nodo que ejecuta la reserva (solo después de la aprobación)
    resultado_reserva: ResultadoReserva | None

    # Parcela del validador
    validacion: ResultadoValidacion | None
    validaciones: Annotated[list[ResultadoValidacion], acumular]

    # Parcela de la síntesis
    respuesta_final: str

    # Log compartido de solo-agregar (cada especialista agrega SU contribución)
    contribuciones: Annotated[list[Contribucion], acumular]


# Qué claves del estado puede escribir cada nodo. graph.py lo hace cumplir.
PARCELAS: dict[str, set[str]] = {
    "supervisor": {"siguiente", "instruccion", "pasos", "reparaciones", "escalado_humano", "decisiones"},
    "investigador": {"investigacion", "evidencia", "contribuciones", "messages"},
    "analista": {"analisis", "contribuciones", "messages"},
    "reservas": {"propuesta_reserva", "contribuciones", "messages"},
    "aprobacion_humana": {"decision_humana"},
    "ejecutar_reserva": {"resultado_reserva", "messages"},
    "validador": {"validacion", "validaciones"},
    "sintesis": {"respuesta_final", "messages", "historial"},
}


def analisis_rechazado(estado: dict) -> bool:
    """El último análisis de este turno no pasó la validación (y nadie lo rehízo después)."""
    ultima = next((v for v in reversed(estado.get("validaciones") or []) if v.agente == "analista"), None)
    return ultima is not None and not ultima.ok


def estado_del_turno(tarea: str, job_id: str = "") -> dict:
    """
    La entrada de cada turno. Sirve igual para el primero y para los
    siguientes de una conversación: pisa los campos del turno anterior, manda
    None a los logs (acumular los vacía) y NO toca historial, así la memoria
    de la conversación se conserva.
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(tarea)],
        "tarea": tarea,
        "job_id": job_id,
        "siguiente": None,
        "instruccion": "",
        "pasos": 0,
        "reparaciones": {},
        "escalado_humano": False,
        "decisiones": None,
        "investigacion": None,
        "evidencia": [],
        "analisis": None,
        "propuesta_reserva": None,
        "decision_humana": None,
        "resultado_reserva": None,
        "validacion": None,
        "validaciones": None,
        "respuesta_final": "",
        "contribuciones": None,
    }


def estado_inicial(tarea: str) -> dict:
    """Compatibilidad con la Pre-Entrega 6 (una tarea suelta, sin conversación)."""
    return estado_del_turno(tarea)
