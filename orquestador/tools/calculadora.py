"""
Calculadora segura para el Analista (y para que el Validador recalcule).

Nunca eval(): ejecutar texto generado por un LLM es una puerta abierta a
ejecución de código arbitrario. Se parsea con ast y solo se aceptan números
y operaciones aritméticas (misma idea que la notebook de la clase), con un
límite para potencias gigantes que podrían colgar el proceso.
"""
import ast
import math
import operator as op

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_OPERADORES = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos,
}
MAX_EXPONENTE = 10


def _evaluar(nodo: ast.AST) -> float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return float(nodo.value)
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        izquierda, derecha = _evaluar(nodo.left), _evaluar(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > MAX_EXPONENTE:
            raise ValueError(f"exponente demasiado grande ({derecha})")
        return _OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError("solo se permiten números y operaciones + - * / ** con paréntesis")


def evaluar_expresion(expresion: str) -> float:
    """Evalúa una expresión aritmética de forma segura (lanza ValueError si no es válida)."""
    try:
        arbol = ast.parse(expresion.replace(",", "."), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"expresión mal escrita: {e.msg}") from e
    try:
        return _evaluar(arbol.body)
    except ZeroDivisionError as e:
        raise ValueError("división por cero") from e


def numeros_de_expresion(expresion: str) -> list[float]:
    """Los números literales de una expresión ('150 - 101' -> [150.0, 101.0])."""
    arbol = ast.parse(expresion.replace(",", "."), mode="eval")
    return [float(n.value) for n in ast.walk(arbol) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]


class CalculadoraInput(BaseModel):
    expresion: str = Field(min_length=1, max_length=200,
                           description="Expresión aritmética con + - * / ** y paréntesis, ej: '(150 - 101) / 101 * 100'")


class ResultadoCalculo(BaseModel):
    expresion: str
    resultado: float

    def model_post_init(self, _contexto) -> None:
        if not math.isfinite(self.resultado):
            raise ValueError("el resultado no es un número finito")

    def como_texto(self) -> str:
        return f"{self.expresion} = {round(self.resultado, 6):g}"


@tool(args_schema=CalculadoraInput)
async def calculadora(expresion: str) -> str:
    """Calcula una expresión aritmética de forma exacta. Usala para TODA cuenta (restas,
    sumas, porcentajes, promedios): no hagas cálculos de memoria. Acepta + - * / ** y
    paréntesis. Devuelve el resultado, o un texto que empieza con ERROR si la expresión
    no es válida."""
    try:
        resultado = ResultadoCalculo(expresion=expresion, resultado=evaluar_expresion(expresion))
    except (ValueError, OverflowError) as e:  # la ValidationError de Pydantic también es un ValueError
        return f"ERROR: {e}"
    return resultado.como_texto()
