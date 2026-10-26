"""
Ejecución de una tarea con traza: "construí un sistema donde puedas ver, en un
log, exactamente quién decidió qué y por qué".

correr_tarea() usa astream() para registrar cada paso del grafo en el orden
en que ocurre, y devuelve los eventos (para el .json), las líneas legibles
(para el .log y la consola) y el estado final.
"""
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.graph import config_de_ejecucion
from orquestador.state import estado_inicial

ICONOS = {"supervisor": "🧭", "investigador": "🔎", "analista": "🧮", "validador": "🛡️", "sintesis": "📝"}


def a_json(valor: Any) -> Any:
    """Convierte modelos Pydantic y mensajes a tipos serializables."""
    if isinstance(valor, BaseModel):
        return valor.model_dump()
    if isinstance(valor, BaseMessage):
        return {"tipo": valor.type, "nombre": valor.name, "contenido": valor.content}
    if isinstance(valor, dict):
        return {k: a_json(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [a_json(v) for v in valor]
    return valor


def _recortar(texto: str, limite: int = 500) -> str:
    texto = " ".join(str(texto).split())
    return texto if len(texto) <= limite else texto[:limite] + "…"


def formatear_evento(nodo: str, actualizacion: dict) -> list[str]:
    icono = ICONOS.get(nodo, "•")
    if nodo == "supervisor":
        d = actualizacion["decisiones"][-1]
        lineas = [f"{icono} Supervisor [{d.origen}] → {d.siguiente}", f"     por qué: {d.razon}"]
        if d.instruccion:
            lineas.append(f"     instrucción: {_recortar(d.instruccion, 300)}")
        if actualizacion.get("escalado_humano"):
            lineas.append("     ⚠️ ESCALADO A REVISIÓN HUMANA")
        return lineas
    if nodo in ("investigador", "analista"):
        c = actualizacion["contribuciones"][-1]
        if not c.ok:
            return [f"{icono} {nodo.capitalize()} ❌ falló ({c.duracion_s:.1f}s): {c.error}"]
        lineas = [f"{icono} {nodo.capitalize()} ({c.duracion_s:.1f}s): {_recortar(c.resumen, 300)}"]
        if nodo == "investigador":
            lineas += [f"     · {d.descripcion} = {d.valor:g} {d.unidad}  [fuente: {d.fuente}]"
                       for d in actualizacion["investigacion"].datos]
        else:
            lineas += [f"     · {k.descripcion}: {k.expresion} = {k.resultado:g} {k.unidad}"
                       for k in actualizacion["analisis"].calculos]
        return lineas
    if nodo == "validador":
        v = actualizacion["validacion"]
        if v.ok:
            return [f"{icono} Validador: ✅ el aporte del {v.agente} pasó la validación"]
        return [f"{icono} Validador: ❌ el aporte del {v.agente} NO pasó la validación"] + [
            f"     - {_recortar(e, 300)}" for e in v.errores
        ]
    if nodo == "sintesis":
        return [f"{icono} Síntesis:", *[f"     {l}" for l in actualizacion["respuesta_final"].splitlines()]]
    return [f"{icono} {nodo}: {list(actualizacion)}"]


@dataclass
class Corrida:
    tarea: str
    eventos: list[dict] = field(default_factory=list)
    lineas: list[str] = field(default_factory=list)
    estado_final: dict = field(default_factory=dict)

    @property
    def respuesta(self) -> str:
        return self.estado_final.get("respuesta_final", "")


async def correr_tarea(app: CompiledStateGraph, tarea: str, simulacion: dict | None = None,
                       imprimir: bool = True) -> Corrida:
    corrida = Corrida(tarea=tarea)
    corrida.lineas.append(f'👤 Usuario: "{tarea}"')
    if simulacion:
        corrida.lineas.append(f"   (simulación de fallos activada: {simulacion})")
    if imprimir:
        print("\n".join(corrida.lineas))

    async for modo, datos in app.astream(estado_inicial(tarea), config=config_de_ejecucion(simulacion),
                                         stream_mode=["updates", "values"]):
        if modo == "values":
            corrida.estado_final = datos
            continue
        for nodo, actualizacion in datos.items():
            corrida.eventos.append({"nodo": nodo, "actualizacion": a_json(actualizacion)})
            nuevas = formatear_evento(nodo, actualizacion)
            corrida.lineas += nuevas
            if imprimir:
                print("\n".join(nuevas))
    return corrida
