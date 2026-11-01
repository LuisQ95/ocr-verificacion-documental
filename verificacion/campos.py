"""Piezas comunes de la comparación: estados de campo y de documento, asignación de palabras a las zonas de la
plantilla, normalización de lecturas OCR, negaciones y señales de objetos superpuestos en un PDF nativo.

Las reglas de cada país (qué se compara contra qué y cómo se dictamina) están en su paquete (p. ej. chile/).
"""

import re
import statistics
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

from .cajas import centro, dentro, intersecta

CONFUNDIBLES = str.maketrans({"l": "I", "1": "I", "|": "I", "i": "I", "0": "O", "o": "O"})
"""Caracteres que el OCR confunde por su forma: en una lectura OCR se igualan antes de comparar textos."""
PUNTUACION_BORDE = ".,:;-_°'\"`´()[]{}|"
NEGACIONES = {"NO", "SIN", "NI", "NUNCA", "NINGUN", "NINGUNA", "NINGUNO"}
PREFIJOS_NEGACION = ("IN", "IM", "DES", "NO")

IGUAL, TOLERADO, DIFERENTE, DUDOSO = "IGUAL", "IGUAL_TOLERADO_OCR", "DIFERENTE", "DUDOSO"
DESACTUALIZADO_C, NO_APLICA = "DESACTUALIZADO", "NO_APLICA"
AUTENTICO, OBSERVADO, FALSIFICADO, DESACTUALIZADO = "AUTÉNTICO", "OBSERVADO", "FALSIFICADO", "DESACTUALIZADO"


def _sin_tildes(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def tokens(texto):
    """Palabras normalizadas para OCR (sin tildes, confundibles igualados, sin puntuación de borde)."""
    t = _sin_tildes((texto or "").translate(CONFUNDIBLES)).upper()
    return [x for x in (tok.strip(PUNTUACION_BORDE).replace(".", "").replace(",", "") for tok in t.split()) if x]


def clave_ocr(texto):
    """Forma canónica de un texto leído por OCR: sus `tokens` unidos, sin espacios."""
    return "".join(tokens(texto))


def _neg(t):
    """'NOJ' (ruido pegado a 'NO') cuenta como 'NO' y 'SUN' (una letra mal leída) como 'SIN';
    'NOREGISTRA' no (es otra palabra)."""
    for n in NEGACIONES:
        if t.startswith(n) and len(t) - len(n) <= 1:
            return n
        if len(n) >= 3 and len(t) == len(n) and sum(a != b for a, b in zip(t, n)) == 1:
            return n
    return t


def cambia_negacion(a, b):
    """True si entre `a` y `b` aparece o desaparece una negación ("NO", "SIN"...) o un prefijo negativo
    ("COMPLETA" vs "INCOMPLETA")."""
    ta, tb = [_neg(t) for t in tokens(a)], [_neg(t) for t in tokens(b)]
    if Counter(t for t in ta if t in NEGACIONES) != Counter(t for t in tb if t in NEGACIONES):
        return True
    solo_a, solo_b = set(ta) - set(tb), set(tb) - set(ta)
    return any(x == p + y or y == p + x for x in solo_a for y in solo_b for p in PREFIJOS_NEGACION)


def fuente_base(nombre):
    """Quita el prefijo de subconjunto que los PDF reales agregan a las fuentes ("ABCDEF+Arial")."""
    return re.sub(r"^[A-Z]{6}\+", "", nombre or "")


def _orden_lectura(palabras):
    """Agrupa por línea real (centros verticales cercanos) y ordena de izquierda a derecha.
    Evita que una leve inclinación del escaneo separe palabras de la misma línea."""
    if not palabras:
        return []
    altura = statistics.median(p["bbox"][3] - p["bbox"][1] for p in palabras)
    lineas = []
    for p in sorted(palabras, key=lambda p: centro(p["bbox"])[1]):
        cy = centro(p["bbox"])[1]
        if lineas and abs(cy - lineas[-1]["cy"]) <= altura * 0.5:
            lineas[-1]["palabras"].append(p)
            lineas[-1]["cy"] = statistics.mean(centro(q["bbox"])[1] for q in lineas[-1]["palabras"])
        else:
            lineas.append({"cy": cy, "palabras": [p]})
    return [p for linea in lineas for p in sorted(linea["palabras"], key=lambda p: p["bbox"][0])]


def asignar_palabras(palabras, plantilla):
    """Separa las palabras fijas de la plantilla (etiquetas) del contenido y asigna el contenido a las zonas.

    Devuelve el texto de cada zona (en orden de lectura, con su confianza mínima y sus fuentes) y la fracción de
    etiquetas encontradas en su lugar, que sirve de control de calidad de la lectura.
    La marca de agua se elimina en la imagen, no aquí: un filtro por similitud de texto con sus palabras descartaba
    datos reales ("POLICIALES" ~ "OFICIAL").
    """
    fijas = [(centro(w[:4]), w[4].upper()) for w in plantilla["palabras_fijas"]]
    encontradas = set()
    contenido = []
    for p in palabras:
        c = centro(p["bbox"])
        txt = p["texto"].upper()
        es_fija = False
        for i, (cf, tf) in enumerate(fijas):
            d = ((c[0] - cf[0]) ** 2 + (c[1] - cf[1]) ** 2) ** 0.5
            if d < 6 and (txt == tf or fuzz.ratio(txt, tf) >= 70):
                es_fija = True
                if fuzz.ratio(txt, tf) >= 85:
                    encontradas.add(i)
                break
        if not es_fija:
            contenido.append(p)

    valores = {}
    zonas = plantilla["zonas"]
    for p in contenido:
        c = centro(p["bbox"])
        zona = next((n for n, z in zonas.items() if dentro(c, z["rect"])), None)
        if zona is not None:
            valores.setdefault(zona, []).append(p)

    salida = {}
    for zona, ps in valores.items():
        ps = _orden_lectura(ps)
        salida[zona] = {
            "texto": " ".join(p["texto"] for p in ps),
            "conf": min(p.get("conf", 1.0) for p in ps),
            "fuentes": sorted({fuente_base(p["font"]) for p in ps if p.get("font")}),
        }
    return {"valores": salida, "cobertura_etiquetas": len(encontradas) / max(1, len(fijas))}


def senal(tipo, severidad, detalle):
    """Señal técnica: severidad "fraude", "observacion" o "info"."""
    return {"tipo": tipo, "severidad": severidad, "detalle": detalle}


def _zonas_tocadas(rect, plantilla):
    """Zonas de la plantilla que el rectángulo toca."""
    return [n for n, z in plantilla["zonas"].items() if intersecta(rect, z["rect"])]


def _dibujo_de_plantilla(d, dibujos_tpl, tol=1.5):
    """True si el dibujo `d` es uno de la plantilla (misma posición ±`tol` pt y mismo relleno)."""
    for t in dibujos_tpl:
        if all(abs(d[i] - t[i]) <= tol for i in range(4)):
            if (d[4] is None) == (t[4] is None) and (d[4] is None or max(abs(a - b) for a, b in zip(d[4], t[4])) < 0.03):
                return True
    return False


def senales_objetos(post, plantilla):
    """PDF nativo: objetos que tapan o alteran lo que se ve sin cambiar la capa de texto."""
    s = []
    obj = post.get("objetos") or {}
    for r in obj.get("imagenes", []):
        tocadas = _zonas_tocadas(r, plantilla)
        if tocadas:
            s.append(senal("objeto_superpuesto", "fraude",
                           f"imagen superpuesta sobre campo(s) de datos: {', '.join(tocadas[:5])}"))
    for tipo, *r in obj.get("anotaciones", []):
        tocadas = _zonas_tocadas(r, plantilla)
        s.append(senal("anotacion", "fraude" if tocadas else "observacion",
                       f"anotación '{tipo}' agregada al PDF" + (f" sobre: {', '.join(tocadas[:5])}" if tocadas else "")))
    ajenos = [d for d in obj.get("dibujos", []) if not _dibujo_de_plantilla(d, plantilla.get("dibujos", []))]
    tocadas = sorted({n for d in ajenos for n in _zonas_tocadas(d[:4], plantilla)})
    if tocadas:
        s.append(senal("objeto_superpuesto", "fraude",
                       f"{len(ajenos)} objeto(s) gráfico(s) que no son de la plantilla sobre: {', '.join(tocadas[:6])}"))
    ocultas = post.get("texto_oculto") or []
    en_zonas = sorted({n for t in ocultas for n in _zonas_tocadas(t[1:5], plantilla)})
    if en_zonas:
        s.append(senal("texto_oculto", "fraude",
                       f"texto invisible o tapado dentro de campo(s) de datos: {', '.join(en_zonas[:6])}"))
    elif ocultas:
        s.append(senal("texto_oculto", "observacion", f"{len(ocultas)} palabra(s) de texto invisible en el PDF"))
    return s
