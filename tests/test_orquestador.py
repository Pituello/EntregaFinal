"""
Tests sin API key. Los cuatro componentes del grafo (supervisor, investigador,
analista y síntesis) se reemplazan por dobles con respuestas fijas, pero la
evidencia es REAL: la salida de las herramientas sobre el catálogo y las
fichas. Así se verifica la mecánica completa de forma reproducible y gratis.

Uso:
    python -m pytest -v
"""
import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.graph import ParcelaInvadida, con_parcela, construir_grafo
from orquestador.state import AporteAnalisis, AporteInvestigacion, Calculo, DatoCitado, DecisionSupervisor
from orquestador.tools.busqueda import buscar_en_fichas
from orquestador.tools.calculadora import calculadora
from orquestador.tools.catalogo import consultar_catalogo
from app.traza import correr_tarea
from orquestador.validation import validar_analisis, validar_investigacion

TAREA = "¿Cuánta más potencia tiene el Virtus Exclusive que el Tera High?"


def correr(coro):
    return asyncio.run(coro)


EVIDENCIA_CATALOGO = [correr(consultar_catalogo.ainvoke({"modelo": m})) for m in ("Tera", "Virtus")]


def dato(descripcion: str, valor: float, unidad: str = "CV", fuente: str = "catalogo_concesionaria") -> DatoCitado:
    return DatoCitado(descripcion=descripcion, valor=valor, unidad=unidad, fuente=fuente)


INVESTIGACION_OK = AporteInvestigacion(
    resumen="Potencias encontradas en el catálogo",
    datos=[dato("potencia Tera High", 101), dato("potencia Virtus Exclusive", 150)],
)
ANALISIS_OK = AporteAnalisis(
    calculos=[Calculo(descripcion="diferencia de potencia", expresion="150 - 101", resultado=49, unidad="CV")],
    conclusion="El Virtus Exclusive tiene 49 CV más.",
)


def decision(siguiente: str, instruccion: str = "", razon: str = "test") -> DecisionSupervisor:
    return DecisionSupervisor(siguiente=siguiente, instruccion=instruccion, razon=razon)


# --- Dobles de prueba ---------------------------------------------------------
class SupervisorFalso:
    def __init__(self, decisiones: list[DecisionSupervisor]):
        self.decisiones, self.llamadas = list(decisiones), 0

    async def decidir(self, estado: dict) -> DecisionSupervisor:
        self.llamadas += 1
        return self.decisiones.pop(0) if self.decisiones else decision("FINALIZAR")


class InvestigadorFalso:
    def __init__(self, respuestas: list):
        self.respuestas, self.instrucciones = list(respuestas), []

    async def ejecutar(self, instruccion: str):
        self.instrucciones.append(instruccion)
        respuesta = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta, EVIDENCIA_CATALOGO


class AnalistaFalso:
    def __init__(self, respuestas: list):
        self.respuestas, self.instrucciones = list(respuestas), []

    async def ejecutar(self, instruccion: str, investigacion):
        self.instrucciones.append(instruccion)
        return self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]


class SintetizadorFalso:
    def __init__(self):
        self.estado_recibido: dict | None = None

    async def redactar(self, estado: dict) -> str:
        self.estado_recibido = estado
        return "Respuesta final de prueba."


def orquestar(supervisor, investigador, analista, simulacion=None):
    sintetizador = SintetizadorFalso()
    app = construir_grafo(supervisor, investigador, analista, sintetizador).compile()
    corrida = correr(correr_tarea(app, TAREA, simulacion=simulacion, imprimir=False))
    return corrida, sintetizador


def nodos(corrida) -> list[str]:
    return [e["nodo"] for e in corrida.eventos]


# --- Flujo de delegación --------------------------------------------------------
def test_delegacion_completa_investigador_analista_sintesis():
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia"), decision("analista", "Calculá la diferencia"),
                           decision("FINALIZAR")])
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]))

    assert nodos(corrida) == ["supervisor", "investigador", "validador", "supervisor", "analista", "validador",
                              "supervisor", "sintesis"]
    final = corrida.estado_final
    assert [d.origen for d in final["decisiones"]] == ["llm", "llm", "llm"]
    assert all(v.ok for v in final["validaciones"])
    assert [c.agente for c in final["contribuciones"]] == ["investigador", "analista"]
    assert final["pasos"] == 2 and not final["escalado_humano"]
    assert corrida.respuesta == "Respuesta final de prueba."


# --- Validación y bucle de reparación ------------------------------------------
def test_dato_inventado_se_repara_con_el_error_especifico():
    inventado = AporteInvestigacion(resumen="x", datos=[dato("potencia Tera High", 105), dato("potencia Virtus Exclusive", 150)])
    inv = InvestigadorFalso([inventado, INVESTIGACION_OK])
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, inv, AnalistaFalso([ANALISIS_OK]))

    final = corrida.estado_final
    assert not final["validaciones"][0].ok and "105" in final["validaciones"][0].errores[0]
    assert final["validaciones"][1].ok
    reparacion = final["decisiones"][1]
    assert reparacion.origen == "regla" and reparacion.siguiente == "investigador"
    assert "no aparece en el texto" in inv.instrucciones[1] and "Buscá la potencia" in inv.instrucciones[1]
    assert sup.llamadas == 2  # la reparación no gastó una llamada al LLM supervisor


def test_fuente_inventada_no_pasa():
    aporte = AporteInvestigacion(resumen="x", datos=[dato("potencia Tera High", 101, fuente="wikipedia")])
    resultado = validar_investigacion(aporte, EVIDENCIA_CATALOGO)
    assert not resultado.ok and "wikipedia" in resultado.errores[0]


def test_cuenta_mal_hecha_se_repara():
    mal = AporteAnalisis(calculos=[Calculo(descripcion="diferencia", expresion="150 - 101", resultado=50, unidad="CV")],
                         conclusion="x")
    ana = AnalistaFalso([mal, ANALISIS_OK])
    sup = SupervisorFalso([decision("investigador", "Buscá"), decision("analista", "Calculá"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), ana)

    errores = corrida.estado_final["validaciones"][1].errores
    assert "no coincide con el recálculo" in errores[0]
    assert "recálculo de '150 - 101' (49)" in ana.instrucciones[1]
    assert corrida.estado_final["validaciones"][-1].ok


def test_numero_que_no_vino_de_la_investigacion_no_pasa():
    ajeno = AporteAnalisis(calculos=[Calculo(descripcion="diferencia", expresion="160 - 101", resultado=59, unidad="CV")],
                           conclusion="x")
    resultado = validar_analisis(ajeno, INVESTIGACION_OK)
    assert not resultado.ok and "160 no provienen de la investigación" in resultado.errores[0]


def test_si_al_analisis_le_falta_un_dato_lo_busca_el_investigador_en_vez_de_reparar_al_analista():
    """
    Regresión de la corrida real (turno 2 de una conversación): el investigador trajo solo uno
    de los dos modelos y el analista usó un número del turno anterior. Antes, dos reparaciones
    del analista con los mismos datos terminaban en escalado a revisión humana.
    """
    parcial = AporteInvestigacion(resumen="solo el Virtus", datos=[dato("potencia Virtus Exclusive", 150)])
    inv = InvestigadorFalso([parcial, INVESTIGACION_OK])
    ana = AnalistaFalso([ANALISIS_OK, ANALISIS_OK])  # las dos veces: 150 - 101
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia del Virtus Exclusive"),
                           decision("analista", "Calculá cuántos CV más tiene el Virtus que el Tera High"),
                           decision("analista", "Calculá cuántos CV más tiene el Virtus que el Tera High"),
                           decision("FINALIZAR")])
    corrida, _ = orquestar(sup, inv, ana)

    final = corrida.estado_final
    assert final["validaciones"][1].numeros_sin_respaldo == [101.0]
    regla = final["decisiones"][2]
    assert regla.origen == "regla" and regla.siguiente == "investigador" and "101" in regla.razon
    assert "101" in inv.instrucciones[1] and "potencia Virtus Exclusive = 150 CV" in inv.instrucciones[1]
    assert "Calculá cuántos CV más" in inv.instrucciones[1]
    assert not final["escalado_humano"] and final["validaciones"][-1].ok
    assert [d.siguiente for d in final["decisiones"]] == ["investigador", "analista", "investigador", "analista",
                                                          "FINALIZAR"]


def test_no_se_finaliza_con_un_analisis_rechazado():
    """
    Regresión de la segunda corrida real: el analista calculó "150 - 0" (le faltaba el dato del
    Nivus), la regla mandó al investigador a buscarlo, y después el LLM eligió FINALIZAR sin
    rehacer el análisis: la síntesis respondió "150 CV más" con la cuenta rechazada.
    """
    sin_dato = AporteAnalisis(calculos=[Calculo(descripcion="diferencia", expresion="150 - 0", resultado=150,
                                                unidad="CV")], conclusion="150 CV más")
    parcial = AporteInvestigacion(resumen="solo el Virtus", datos=[dato("potencia Virtus Exclusive", 150)])
    ana = AnalistaFalso([sin_dato, ANALISIS_OK])
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia del Virtus"),
                           decision("analista", "Calculá la diferencia de potencia"),
                           decision("FINALIZAR"),   # el error del LLM: finalizar con el análisis rechazado
                           decision("FINALIZAR")])
    corrida, sintetizador = orquestar(sup, InvestigadorFalso([parcial, INVESTIGACION_OK]), ana)

    final = corrida.estado_final
    guarda = final["decisiones"][3]
    assert guarda.origen == "regla" and guarda.siguiente == "analista" and "rechazado" in guarda.razon
    assert ana.instrucciones[1] == "Calculá la diferencia de potencia"
    assert final["analisis"] == ANALISIS_OK and final["validaciones"][-1].ok
    assert [d.siguiente for d in final["decisiones"]] == ["investigador", "analista", "investigador", "analista",
                                                          "FINALIZAR"]


def test_el_resumen_marca_un_analisis_rechazado():
    from orquestador.agents.supervisor import resumir_estado
    from orquestador.state import ResultadoValidacion, estado_del_turno

    estado = {**estado_del_turno(TAREA), "investigacion": INVESTIGACION_OK, "analisis": ANALISIS_OK,
              "validaciones": [ResultadoValidacion(agente="analista", ok=False, errores=["x"])]}
    assert "ANÁLISIS RECHAZADO POR EL VALIDADOR" in resumir_estado(estado)
    estado["validaciones"].append(ResultadoValidacion(agente="analista", ok=True))
    assert "ANÁLISIS: El Virtus Exclusive" in resumir_estado(estado)


def test_la_busqueda_de_datos_faltantes_corre_una_sola_vez_por_turno(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_REPARACIONES", "1")
    parcial = AporteInvestigacion(resumen="solo el Virtus", datos=[dato("potencia Virtus Exclusive", 150)])
    inv = InvestigadorFalso([parcial])  # nunca trae el dato que falta
    sup = SupervisorFalso([decision("investigador", "Buscá"), decision("analista", "Calculá"),
                           decision("analista", "Calculá")])
    corrida, _ = orquestar(sup, inv, AnalistaFalso([ANALISIS_OK]))

    origenes = [(d.origen, d.siguiente) for d in corrida.estado_final["decisiones"]]
    assert origenes.count(("regla", "investigador")) == 1  # la segunda vez: bucle de reparación normal
    assert corrida.estado_final["escalado_humano"]


def test_escala_a_revision_humana_tras_las_reparaciones(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_REPARACIONES", "2")
    mal = AporteAnalisis(calculos=[Calculo(descripcion="diferencia", expresion="150 - 101", resultado=50, unidad="CV")],
                         conclusion="x")
    ana = AnalistaFalso([mal])  # siempre se equivoca
    sup = SupervisorFalso([decision("investigador", "Buscá"), decision("analista", "Calculá")])
    corrida, sintetizador = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), ana)

    final = corrida.estado_final
    assert len(ana.instrucciones) == 3  # 1 intento + 2 reparaciones
    assert final["escalado_humano"] and final["decisiones"][-1].siguiente == "FINALIZAR"
    assert "revisión humana" in final["decisiones"][-1].razon
    assert sintetizador.estado_recibido["escalado_humano"]


# --- Camino de fallo --------------------------------------------------------------
def test_especialista_que_falla_no_rompe_al_supervisor():
    inv = InvestigadorFalso([RuntimeError("se cayó la base"), INVESTIGACION_OK])
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, inv, AnalistaFalso([ANALISIS_OK]))

    final = corrida.estado_final
    assert not final["contribuciones"][0].ok and "se cayó la base" in final["contribuciones"][0].error
    assert final["validaciones"][0].fallo_de_ejecucion
    assert "error técnico" in inv.instrucciones[1]
    assert final["contribuciones"][1].ok and corrida.respuesta


def test_simulacion_de_fallo_por_configuracion():
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]),
                           simulacion={"fallar_investigador": 1})
    contribuciones = corrida.estado_final["contribuciones"]
    assert "Fallo simulado" in contribuciones[0].error and contribuciones[1].ok


def test_simulacion_de_analisis_corrupto_la_detecta_el_validador():
    sup = SupervisorFalso([decision("investigador", "Buscá"), decision("analista", "Calculá"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]),
                           simulacion={"corromper_analisis": 1})
    validaciones = corrida.estado_final["validaciones"]
    assert not validaciones[1].ok and validaciones[2].ok  # detectado y reparado


# --- Límites y guardas -----------------------------------------------------------
def test_limite_de_pasos_corta_un_supervisor_infinito(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_PASOS", "3")
    sup = SupervisorFalso([decision("investigador", "Buscá otra vez")] * 50)
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]))

    final = corrida.estado_final
    assert final["pasos"] == 3
    assert final["decisiones"][-1].origen == "regla" and "MAX_PASOS" in final["decisiones"][-1].razon
    assert corrida.respuesta


def test_guarda_no_deja_analizar_sin_investigacion():
    sup = SupervisorFalso([decision("analista", "Calculá algo"), decision("FINALIZAR")])
    corrida, _ = orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]))

    primera = corrida.estado_final["decisiones"][0]
    assert primera.siguiente == "investigador" and primera.origen == "regla"
    assert nodos(corrida)[1] == "investigador"


# --- Contexto acotado y aislamiento ---------------------------------------------
def test_cada_especialista_recibe_solo_su_instruccion():
    inv, ana = InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK])
    sup = SupervisorFalso([decision("investigador", "Buscá la potencia del Tera High y del Virtus Exclusive"),
                           decision("analista", "Restá las dos potencias"), decision("FINALIZAR")])
    orquestar(sup, inv, ana)
    assert inv.instrucciones == ["Buscá la potencia del Tera High y del Virtus Exclusive"]  # ni la tarea ni el historial
    assert ana.instrucciones == ["Restá las dos potencias"]


def test_el_analista_ve_los_datos_validados_y_no_la_evidencia_cruda():
    from orquestador.agents.analyst_agent import AgenteAnalista

    mensaje = AgenteAnalista.armar_mensaje("Restá las potencias", INVESTIGACION_OK)
    assert "potencia Tera High: 101 CV" in mensaje
    assert "version_id" not in mensaje  # nada del JSON crudo del catálogo


def test_un_nodo_no_puede_escribir_fuera_de_su_parcela():
    async def analista_entrometido(state, config):
        return {"analisis": ANALISIS_OK, "investigacion": None}  # pisa la parcela del investigador

    with pytest.raises(ParcelaInvadida, match="investigacion"):
        correr(con_parcela("analista", analista_entrometido)({}, {}))


# --- Herramientas y evidencia ----------------------------------------------------
def test_validacion_lee_los_numeros_con_formato_argentino():
    evidencia = [correr(buscar_en_fichas.ainvoke({"consulta": "Latin NCAP ocupante adulto Tera"}))]
    bien = AporteInvestigacion(resumen="x", datos=[dato("adulto Tera", 89.88, "%", "tera.md")])
    mal = AporteInvestigacion(resumen="x", datos=[dato("adulto Tera", 90, "%", "tera.md")])
    assert validar_investigacion(bien, evidencia).ok
    assert not validar_investigacion(mal, evidencia).ok


def test_busqueda_refuerza_la_ficha_del_modelo_nombrado():
    resultado = correr(buscar_en_fichas.ainvoke({"consulta": "protección ocupante adulto Nivus Latin NCAP"}))
    assert resultado.startswith("[Fuente: nivus.md]") and "92,47" in resultado
    # el refuerzo es suave: no rompe la trampa de la Pre-Entrega 4
    resultado = correr(buscar_en_fichas.ainvoke({"consulta": "capacidad de baúl versión sedán del Polo"}))
    assert "[Fuente: virtus.md]" in resultado and "521" in resultado


def test_calculadora_segura():
    assert correr(calculadora.ainvoke({"expresion": "(150 - 101) / 101 * 100"})).endswith("= 48.5149")
    for maliciosa in ["__import__('os').system('ls')", "9**99999", "1/0"]:
        assert correr(calculadora.ainvoke({"expresion": maliciosa})).startswith("ERROR")


def test_agente_real_con_create_agent_devuelve_aporte_y_evidencia():
    """El cableado real (create_agent + response_format) con un LLM simulado."""
    from orquestador.agents.research_agent import AgenteInvestigador

    class LLMGuionado(BaseChatModel):
        respuestas: list[AIMessage]

        @property
        def _llm_type(self) -> str:
            return "guionado"

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LLMGuionado":
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=self.respuestas.pop(0))])

    llm = LLMGuionado(respuestas=[
        AIMessage("", tool_calls=[{"name": "consultar_catalogo", "args": {"modelo": "Tera"}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "AporteInvestigacion", "id": "2",
                                   "args": INVESTIGACION_OK.model_dump()}]),
    ])
    aporte, evidencia = correr(AgenteInvestigador(llm).ejecutar("Buscá la potencia del Tera High"))
    assert aporte == INVESTIGACION_OK
    assert len(evidencia) == 1 and '"fuente": "catalogo_concesionaria"' in evidencia[0]


def test_la_simulacion_no_altera_el_aporte_original():
    """Regresión: la simulación de análisis corrupto mutaba el objeto en el lugar (estado mutable compartido)."""
    original = ANALISIS_OK.calculos[0].resultado
    sup = SupervisorFalso([decision("investigador", "Buscá"), decision("analista", "Calculá"), decision("FINALIZAR")])
    orquestar(sup, InvestigadorFalso([INVESTIGACION_OK]), AnalistaFalso([ANALISIS_OK]),
              simulacion={"corromper_analisis": 1})
    assert ANALISIS_OK.calculos[0].resultado == original


# --- Regresiones de la corrida real con gpt-4o-mini (escenarios 1 y 2 del demo de la Pre-Entrega 6) ---
def test_catalogo_acepta_el_modelo_con_la_version_pegada():
    """El Investigador pedía 'Virtus Exclusive' y 'Vento GLI': la búsqueda exacta devolvía ERROR."""
    for pedido, modelo in [("Virtus Exclusive", "Virtus"), ("Vento GLI", "Vento"), ("VW Tera High", "Tera")]:
        resultado = correr(consultar_catalogo.ainvoke({"modelo": pedido}))
        assert not resultado.startswith("ERROR") and f'"modelo": "{modelo}"' in resultado
    assert correr(consultar_catalogo.ainvoke({"modelo": "Amarok"})).startswith("ERROR")


def test_investigacion_sin_datos_no_pasa_y_termina_escalada():
    """Una investigación vacía pasaba "sin errores" y el supervisor la reenviaba hasta MAX_PASOS."""
    vacia = AporteInvestigacion(resumen="El modelo no está en el catálogo.", datos=[])
    resultado = validar_investigacion(vacia, EVIDENCIA_CATALOGO)
    assert not resultado.ok and "No devolviste ningún dato" in resultado.errores[0]

    sup = SupervisorFalso([decision("investigador", "Buscá")] * 50)
    corrida, _ = orquestar(sup, InvestigadorFalso([vacia]), AnalistaFalso([ANALISIS_OK]))
    final = corrida.estado_final
    assert final["escalado_humano"] and final["pasos"] == 3  # 1 intento + 2 reparaciones, no 8


def test_analisis_puede_encadenar_el_resultado_de_un_calculo_anterior():
    """El Analista usaba la diferencia (2.59) para sacar un porcentaje y el Validador lo rechazaba."""
    investigacion = AporteInvestigacion(resumen="x", datos=[
        dato("adulto Tera", 89.88, "%", "tera.md"), dato("adulto Nivus", 92.47, "%", "nivus.md")])
    encadenado = AporteAnalisis(conclusion="x", calculos=[
        Calculo(descripcion="diferencia", expresion="92.47 - 89.88", resultado=2.59, unidad="pp"),
        Calculo(descripcion="mejora relativa", expresion="(2.59 / 89.88) * 100", resultado=2.88162, unidad="%"),
    ])
    assert validar_analisis(encadenado, investigacion).ok

    # el resultado de una cuenta MAL hecha no habilita a usarlo después
    mal = encadenado.model_copy(update={"calculos": [
        Calculo(descripcion="diferencia", expresion="92.47 - 89.88", resultado=3.59, unidad="pp"),
        Calculo(descripcion="mejora relativa", expresion="(3.59 / 89.88) * 100", resultado=3.99422, unidad="%"),
    ]})
    errores = validar_analisis(mal, investigacion).errores
    assert len(errores) == 2 and "3.59 no provienen de la investigación" in errores[1]
