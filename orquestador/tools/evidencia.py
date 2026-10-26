"""
Lectura determinística de la evidencia que devuelven las herramientas, para
que el Validador pueda chequear de dónde sale cada número (provenance).

El punto delicado es el formato de los números en los documentos, que usan la
convención argentina: '89,88 %' (coma decimal), '2.566 mm' y '$29.900.000'
(punto de miles). extraer_numeros() devuelve todas las lecturas posibles de
cada número, así un dato se considera respaldado si coincide con alguna.
"""
import json
import re

_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
_MARCA_FUENTE = re.compile(r"\[Fuente: ([^\]]+)\]")


def _lecturas(texto_numero: str) -> set[float]:
    lecturas: set[float] = set()
    candidatos = {texto_numero}
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", texto_numero):  # 2.566 / 29.900.000 -> miles
        candidatos.add(texto_numero.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", texto_numero):  # 1,000 -> miles (formato inglés)
        candidatos.add(texto_numero.replace(",", ""))
    if texto_numero.count(",") == 1 and "." not in texto_numero:  # 89,88 -> decimal
        candidatos.add(texto_numero.replace(",", "."))
    for c in candidatos:
        try:
            lecturas.add(float(c))
        except ValueError:
            pass
    return lecturas


def extraer_numeros(texto: str) -> set[float]:
    numeros: set[float] = set()
    for m in _NUMERO.finditer(texto):
        numeros |= _lecturas(m.group())
    return numeros


def coincide(valor: float, numeros: set[float]) -> bool:
    return any(abs(valor - n) <= 1e-6 * max(1.0, abs(n)) for n in numeros)


def evidencia_por_fuente(salidas_de_herramientas: list[str]) -> dict[str, str]:
    """
    Agrupa el texto de las herramientas por fuente:
    - consultar_catalogo devuelve JSON con la clave "fuente";
    - buscar_en_fichas devuelve bloques que empiezan con "[Fuente: archivo.md]".
    """
    por_fuente: dict[str, str] = {}
    for salida in salidas_de_herramientas:
        try:
            datos = json.loads(salida)
            if isinstance(datos, dict) and "fuente" in datos:
                por_fuente[datos["fuente"]] = por_fuente.get(datos["fuente"], "") + "\n" + salida
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        partes = _MARCA_FUENTE.split(salida)  # ['', fuente1, texto1, fuente2, texto2, ...]
        for fuente, texto in zip(partes[1::2], partes[2::2]):
            por_fuente[fuente.strip()] = por_fuente.get(fuente.strip(), "") + "\n" + texto
    return por_fuente
