"""Normalización de los valores chilenos para comparar lo leído con el dato oficial (JSON)."""

import re
import unicodedata
from datetime import date

from rapidfuzz import fuzz, process

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
ABR = [m[:3] for m in MESES]
GUIONES = str.maketrans({"\xad": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "~": "-",
                         ";": ";"})
CONF_NUM = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "i": "1", "S": "5", "B": "8"})


def limpio(t):
    """Texto NFC con los guiones unificados y los espacios colapsados."""
    return " ".join(unicodedata.normalize("NFC", t or "").translate(GUIONES).split())


def sin_tildes(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t or "") if unicodedata.category(c) != "Mn")


def clave_texto(t):
    """Mayúsculas, sin tildes ni espacios ni puntuación de borde (para nombres y textos)."""
    t = sin_tildes(limpio(t)).upper()
    return re.sub(r"[\s.,;:]+", " ", t).strip()


def _mes(token):
    """Número de mes (1-12) de un nombre o abreviatura de mes, tolerando errores de OCR: un nombre mal leído
    ("Novienbre") o una abreviatura con una letra mal leída ("maP" -> "mar"). None si no se reconoce."""
    t = sin_tildes(token).lower().strip(".")
    if t in MESES:
        return MESES.index(t) + 1
    if t in ABR:
        return ABR.index(t) + 1
    if len(t) > 4:
        m = process.extractOne(t, MESES, scorer=fuzz.ratio)
        if m and m[1] >= 75:
            return MESES.index(m[0]) + 1
    elif len(t) >= 3:
        m = process.extractOne(t[:3], ABR, scorer=fuzz.ratio)
        if m and m[1] >= 66:
            return ABR.index(m[0]) + 1
    return None


def fecha(texto):
    """Fecha en cualquiera de los formatos de los documentos chilenos -> date (o None).

    Con mes escrito, el separador puede faltar en una lectura OCR ("29-dic2025") y el OCR puede dejar un punto o dos
    puntos tras el día ("8. de", "2: de") o leer "del".
    """
    t = limpio(texto)
    m = re.search(r"(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{4})", t)
    if m:
        d, mo, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"(?<!\d)(\d{1,2})[.:,]?\s*[-/ ]?\s*(?:del?\s+)?([A-Za-zÁÉÍÓÚáéíóúñÑ.]{3,12})\s*[-/ ]?\s*"
                      r"(?:del?\s+)?(\d{4})", t)
        if not m:
            return None
        d, mo, a = int(m.group(1)), _mes(m.group(2)), int(m.group(3))
        if not mo:
            return None
    try:
        return date(a, mo, d)
    except ValueError:
        return None


def rut(texto):
    """'12.345.678-5', '12345678-5', '12.345678-5' -> '12345678-5' (dígito verificador en mayúscula)."""
    t = limpio(texto).upper().replace(" ", "")
    m = re.search(r"(\d[\d.]{5,11})-?([\dK])\b", t)
    if not m:
        return None
    return f"{m.group(1).replace('.', '')}-{m.group(2)}"


def rut_ocr(texto):
    """Como `rut`, igualando antes las letras que el OCR confunde con dígitos (O->0, l->1, S->5...)."""
    return rut(limpio(texto).translate(CONF_NUM))


def rut_valido(r):
    """Un RUT completo con dígito verificador correcto (una lectura parcial del OCR no lo es)."""
    if not r or "-" not in r:
        return False
    num, dv = r.split("-")
    if not num.isdigit() or len(num) < 7:
        return False
    s, m = 0, 2
    for c in reversed(num):
        s += int(c) * m
        m = 2 if m == 7 else m + 1
    esperado = 11 - s % 11
    return dv == ("0" if esperado == 11 else "K" if esperado == 10 else str(esperado))


def numero(texto):
    """Primer número del texto (admite "%" y coma decimal); None si no hay."""
    t = limpio(texto).replace("%", "").replace(",", ".").strip()
    m = re.search(r"\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


def clases(texto):
    """Solo los códigos de clase válidos (Ley 18.290): restos de lectura (":" leído como "de") se ignoran."""
    t = sin_tildes(limpio(texto)).upper()
    return frozenset(re.findall(r"(?<![A-Z0-9])(A[1-5]|[BCDEF])(?![A-Z0-9])", t.replace("-", " ").replace("/", " ")))


def codigo(texto):
    """Código sin espacios y en mayúsculas."""
    return re.sub(r"[\s]", "", limpio(texto)).upper()


def codigo_ocr(texto):
    """Como `codigo`, igualando O/0 e I/L/|/1 (confusiones del OCR)."""
    t = codigo(texto)
    return t.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "|": "1"}))


def quitar_prefijo(texto, prefijo):
    """Texto sin `prefijo` al inicio (comparado sin tildes ni mayúsculas); sin cambios si no empieza con él."""
    t = limpio(texto)
    return t[len(prefijo):].strip(" ,:") if clave_texto(t).startswith(clave_texto(prefijo)) else t
