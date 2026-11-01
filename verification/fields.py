"""Piezas comunes de la comparación: estados de campo y de documento, asignación de palabras a las zonas de la
plantilla, normalización de lecturas OCR, negaciones y señales de objetos superpuestos en un PDF nativo.

Las reglas de cada conjunto de documentos (qué se compara contra qué y cómo se dictamina) están en su perfil
(verification/profiles/).
"""

import re
import statistics
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

from .boxes import center, inside, intersects

CONFUSABLE = str.maketrans({"l": "I", "1": "I", "|": "I", "i": "I", "0": "O", "o": "O"})
"""Caracteres que el OCR confunde por su forma: en una lectura OCR se igualan antes de comparar textos."""
EDGE_PUNCTUATION = ".,:;-_°'\"`´()[]{}|"
NEGATIONS = {"NO", "SIN", "NI", "NUNCA", "NINGUN", "NINGUNA", "NINGUNO"}
NEGATION_PREFIXES = ("IN", "IM", "DES", "NO")

EQUAL, TOLERATED, DIFFERENT, DOUBTFUL = "IGUAL", "IGUAL_TOLERADO_OCR", "DIFERENTE", "DUDOSO"
OUTDATED_FIELD, NOT_APPLICABLE = "DESACTUALIZADO", "NO_APLICA"
AUTHENTIC, FLAGGED, FORGED, OUTDATED = "AUTÉNTICO", "OBSERVADO", "FALSIFICADO", "DESACTUALIZADO"
ILLEGIBLE, NOT_VALID, NOT_RECOGNIZED = "NO LEGIBLE", "NO VÁLIDO", "NO RECONOCIDO"
NO_OFFICIAL_DATA = "SIN DATO OFICIAL"
"""No se entregó su JSON oficial: no se verifica, solo se extraen sus datos."""
FIT, NOT_FIT, INCOMPLETE, NOT_EVALUABLE = "APTO", "NO APTO", "INCOMPLETO", "NO EVALUABLE"
"""Aptitud del postulante para el cargo."""


def _strip_accents(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def tokens(text):
    """Palabras normalizadas para OCR (sin tildes, confundibles igualados, sin puntuación de borde)."""
    t = _strip_accents((text or "").translate(CONFUSABLE)).upper()
    return [x for x in (tok.strip(EDGE_PUNCTUATION).replace(".", "").replace(",", "") for tok in t.split()) if x]


def ocr_key(text):
    """Forma canónica de un texto leído por OCR: sus `tokens` unidos, sin espacios."""
    return "".join(tokens(text))


def _neg(t):
    """'NOJ' (ruido pegado a 'NO') cuenta como 'NO' y 'SUN' (una letra mal leída) como 'SIN';
    'NOREGISTRA' no (es otra palabra)."""
    for n in NEGATIONS:
        if t.startswith(n) and len(t) - len(n) <= 1:
            return n
        if len(n) >= 3 and len(t) == len(n) and sum(a != b for a, b in zip(t, n)) == 1:
            return n
    return t


def negation_changes(a, b):
    """True si entre `a` y `b` aparece o desaparece una negación ("NO", "SIN"...) o un prefijo negativo
    ("COMPLETA" vs "INCOMPLETA")."""
    ta, tb = [_neg(t) for t in tokens(a)], [_neg(t) for t in tokens(b)]
    if Counter(t for t in ta if t in NEGATIONS) != Counter(t for t in tb if t in NEGATIONS):
        return True
    only_a, only_b = set(ta) - set(tb), set(tb) - set(ta)
    return any(x == p + y or y == p + x for x in only_a for y in only_b for p in NEGATION_PREFIXES)


def base_font(name):
    """Quita el prefijo de subconjunto que los PDF reales agregan a las fuentes ("ABCDEF+Arial")."""
    return re.sub(r"^[A-Z]{6}\+", "", name or "")


def _reading_order(words):
    """Agrupa por línea real (centros verticales cercanos) y ordena de izquierda a derecha.
    Evita que una leve inclinación del escaneo separe palabras de la misma línea."""
    if not words:
        return []
    level_height = statistics.median(p["bbox"][3] - p["bbox"][1] for p in words)
    lines = []
    for p in sorted(words, key=lambda p: center(p["bbox"])[1]):
        cy = center(p["bbox"])[1]
        if lines and abs(cy - lines[-1]["cy"]) <= level_height * 0.5:
            lines[-1]["palabras"].append(p)
            lines[-1]["cy"] = statistics.mean(center(q["bbox"])[1] for q in lines[-1]["palabras"])
        else:
            lines.append({"cy": cy, "palabras": [p]})
    return [p for line in lines for p in sorted(line["palabras"], key=lambda p: p["bbox"][0])]


def assign_words(words, template):
    """Separa las palabras fijas de la plantilla (etiquetas) del contenido y asigna el contenido a las zonas.

    Devuelve el texto de cada zona (en orden de lectura, con su confianza mínima y sus fuentes) y la fracción de
    etiquetas encontradas en su lugar, que sirve de control de calidad de la lectura.
    La marca de agua se elimina en la imagen, no aquí: un filtro por similitud de texto con sus palabras descartaba
    datos reales ("POLICIALES" ~ "OFICIAL").
    """
    fixed = [(center(w[:4]), w[4].upper()) for w in template["palabras_fijas"]]
    found = set()
    content = []
    for p in words:
        c = center(p["bbox"])
        txt = p["texto"].upper()
        is_fixed = False
        for i, (cf, tf) in enumerate(fixed):
            d = ((c[0] - cf[0]) ** 2 + (c[1] - cf[1]) ** 2) ** 0.5
            if d < 6 and (txt == tf or fuzz.ratio(txt, tf) >= 70):
                is_fixed = True
                if fuzz.ratio(txt, tf) >= 85:
                    found.add(i)
                break
        if not is_fixed:
            content.append(p)

    values = {}
    zones = template["zonas"]
    for p in content:
        c = center(p["bbox"])
        zone = next((n for n, z in zones.items() if inside(c, z["rect"])), None)
        if zone is not None:
            values.setdefault(zone, []).append(p)

    output = {}
    for zone, ps in values.items():
        ps = _reading_order(ps)
        output[zone] = {
            "texto": " ".join(p["texto"] for p in ps),
            "conf": min(p.get("conf", 1.0) for p in ps),
            "fuentes": sorted({base_font(p["font"]) for p in ps if p.get("font")}),
        }
    return {"valores": output, "cobertura_etiquetas": len(found) / max(1, len(fixed))}


def signal(kind, severity, detail):
    """Señal técnica: severidad "fraude", "observacion" o "info"."""
    return {"tipo": kind, "severidad": severity, "detalle": detail}


def _touched_zones(rect, template):
    """Zonas de la plantilla que el rectángulo toca."""
    return [n for n, z in template["zonas"].items() if intersects(rect, z["rect"])]


def _template_drawing(d, tpl_drawings, tol=1.5):
    """True si el dibujo `d` es uno de la plantilla (misma posición ±`tol` pt y mismo relleno)."""
    for t in tpl_drawings:
        if all(abs(d[i] - t[i]) <= tol for i in range(4)):
            if (d[4] is None) == (t[4] is None) and (d[4] is None or max(abs(a - b) for a, b in zip(d[4], t[4])) < 0.03):
                return True
    return False


def object_signals(post, template):
    """PDF nativo: objetos que tapan o alteran lo que se ve sin cambiar la capa de texto."""
    s = []
    obj = post.get("objetos") or {}
    for r in obj.get("imagenes", []):
        touched = _touched_zones(r, template)
        if touched:
            s.append(signal("objeto_superpuesto", "fraude",
                           f"imagen superpuesta sobre campo(s) de datos: {', '.join(touched[:5])}"))
    for kind, *r in obj.get("anotaciones", []):
        touched = _touched_zones(r, template)
        s.append(signal("anotacion", "fraude" if touched else "observacion",
                       f"anotación '{kind}' agregada al PDF" + (f" sobre: {', '.join(touched[:5])}" if touched else "")))
    foreign_items = [d for d in obj.get("dibujos", []) if not _template_drawing(d, template.get("dibujos", []))]
    touched = sorted({n for d in foreign_items for n in _touched_zones(d[:4], template)})
    if touched:
        s.append(signal("objeto_superpuesto", "fraude",
                        f"{len(foreign_items)} objeto(s) gráfico(s) que no son de la plantilla sobre: "
                        f"{', '.join(touched[:6])}"))
    hidden_words = post.get("texto_oculto") or []
    in_zones = sorted({n for t in hidden_words for n in _touched_zones(t[1:5], template)})
    if in_zones:
        s.append(signal("texto_oculto", "fraude",
                       f"texto invisible o tapado dentro de campo(s) de datos: {', '.join(in_zones[:6])}"))
    elif hidden_words:
        s.append(signal("texto_oculto", "observacion", f"{len(hidden_words)} palabra(s) de texto invisible en el PDF"))
    return s
