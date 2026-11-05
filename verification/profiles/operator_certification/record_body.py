"""Sección variable de la Hoja de Vida (licencias, anotaciones y antecedentes): se lee como texto línea a línea
(no por zonas fijas, porque la cantidad de filas cambia por persona) y se convierte en filas."""

import re
import statistics

from rapidfuzz import fuzz

from .normalize import MONTH_ABBR, MONTHS, text_key, date_value

VALUES_COLUMN_X = 195
"""Límite (pt): los encabezados de anotación empiezan a la izquierda de esto; su texto, desde ~211."""


def lines(words):
    """Agrupa palabras por línea (centro vertical) y las ordena de izquierda a derecha."""
    if not words:
        return []
    height = statistics.median(p["bbox"][3] - p["bbox"][1] for p in words)
    out = []
    for p in sorted(words, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2):
        cy = (p["bbox"][1] + p["bbox"][3]) / 2
        if out and abs(cy - out[-1]["cy"]) <= height * 0.55:
            out[-1]["p"].append(p)
            out[-1]["cy"] = statistics.mean((q["bbox"][1] + q["bbox"][3]) / 2 for q in out[-1]["p"])
        else:
            out.append({"cy": cy, "p": [p]})
    res = []
    for l in out:
        ps = sorted(l["p"], key=lambda p: p["bbox"][0])
        res.append({"x0": ps[0]["bbox"][0], "texto": " ".join(p["texto"] for p in ps),
                    "conf": min(p.get("conf", 1.0) for p in ps)})
    return res


def _starts(t, word, threshold=80):
    """True si la primera palabra de `t` se parece a `palabra` (tolerancia a errores de OCR)."""
    first = t.split(" ", 1)[0] if t else ""
    return fuzz.ratio(first, word) >= threshold


def _index(tokens, word, threshold=75):
    """Posición del primer token parecido a `palabra` (None si ninguno)."""
    return next((i for i, t in enumerate(tokens) if fuzz.ratio(t, word) >= threshold), None)


def _license(t):
    """'PRIMERA CLASE B FECHA OTORGAMIENTO 5 MARZO 2019' con tolerancia a errores de OCR en las palabras fijas.

    Si la clase (una letra) no se detectó o se leyó mal, queda vacía: la fila se empareja por registro y fecha.
    """
    tok = t.split()
    if len(tok) < 4:
        return None
    if _starts(tok[0], "PRIMERA", 75):
        registry = "PRIMERA"
    elif _starts(tok[0], "ULTIMA", 75):
        registry = "ULTIMA"
    else:
        return None
    i_class = _index(tok[:3], "CLASE")
    i_granted = _index(tok, "OTORGAMIENTO", 70)
    if i_class is None or i_granted is None or i_granted <= i_class:
        return None
    lclass = tok[i_class + 1]
    if i_granted == i_class + 1 or fuzz.ratio(lclass, "FECHA") >= 70 or not re.fullmatch(r"A[1-5]|[BCDEF]", lclass):
        lclass = ""
    rest = " ".join(tok[i_granted + 1:])
    return {"Registro": registry, "Clase": lclass, "fecha_txt": rest, "fecha": date_value(rest), "Municipalidad": ""}


def parse(lines_by_page):
    """Filas de licencias, anotaciones y antecedentes (en orden de página).

    - Municipalidad: los restos de la fecha de la fila anterior (mes, día, año) que el OCR deja al inicio no son
      parte del nombre.
    - Antecedentes: la fila empieza con el RUN sin puntos y sigue con SIN/CON (Courier grande; el OCR puede leer mal
      "ANTECEDENTES"), así que se reconoce por su estructura. El registro se decide solo por la palabra siguiente
      ("PRN CONDUCTOR" juntado contiene "RNC": "pRNConductor"); si no se lee, se asigna en orden (PRN, luego RNC).
    - Se omiten los encabezados de la tabla de antecedentes; una línea a la izquierda de VALUES_COLUMN_X es el
      encabezado de una anotación.
    """
    out = {"licencias": [], "anotaciones": [], "antecedentes": {}, "antecedentes_conf": {}, "conf_min": 1.0}
    current = None
    for line in (l for pg_ in lines_by_page for l in pg_):
        t = text_key(line["texto"])
        out["conf_min"] = min(out["conf_min"], line["conf"])
        lic = _license(t)
        if lic:
            out["licencias"].append(lic | {"conf": line["conf"]})
            current = None
            continue
        if _starts(t, "MUNICIPALIDAD", 75) and out["licencias"] and not out["licencias"][-1]["Municipalidad"]:
            rest = t.split()[1:]
            while len(rest) > 1 and (re.fullmatch(r"[\d.,\-/]+", rest[0])
                                     or rest[0].lower().strip(".") in MONTHS + MONTH_ABBR):
                rest = rest[1:]
            out["licencias"][-1]["Municipalidad"] = " ".join(rest)
            out["licencias"][-1]["conf"] = min(out["licencias"][-1]["conf"], line["conf"])
            continue
        tok = t.split()
        i_prev = _index(tok, "ANTECEDENTES", 75)
        if i_prev is None and len(tok) >= 2 and re.fullmatch(r"\d{7,9}-?[\dK]", tok[0]) \
                and (fuzz.ratio(tok[1], "SIN") >= 67 or fuzz.ratio(tok[1], "CON") >= 67):
            i_prev = 2
        if i_prev is not None:
            sig = tok[i_prev + 1][:3] if i_prev + 1 < len(tok) else ""
            reg = "PRN" if fuzz.ratio(sig, "PRN") >= 67 else "RNC" if fuzz.ratio(sig, "RNC") >= 67 else None
            if reg is None:
                reg = "PRN" if "PRN" not in out["antecedentes"] else "RNC"
            prior = tok[i_prev - 1] if i_prev else ""
            out["antecedentes"][reg] = "CON ANTECEDENTES" if fuzz.ratio(prior, "CON") >= 67 and prior != "SIN" \
                else "SIN ANTECEDENTES"
            out["antecedentes_conf"][reg] = line["conf"]
            continue
        compact = t.replace(" ", "")
        if compact.startswith("RUN") or "HOJADEVIDA" in compact or fuzz.partial_ratio("HOJADEVIDA", compact) >= 85 \
                or compact in ("CONDUCTOR", "RUN"):
            continue
        if line["x0"] < VALUES_COLUMN_X:
            current = {"Tipo": t, "detalle": [], "Fecha anotación": "", "fecha": None, "conf": line["conf"]}
            out["anotaciones"].append(current)
            continue
        if current is not None:
            if t.startswith("FECHA ANOTACION") or _starts(t, "FECHA") and "ANOTACION" in t:
                current["Fecha anotación"] = t.split(":", 1)[-1].strip()
                current["fecha"] = date_value(current["Fecha anotación"])
            else:
                current["detalle"].append(t)
            current["conf"] = min(current["conf"], line["conf"])
    for a in out["anotaciones"]:
        a["Detalle"] = " ".join(a.pop("detalle"))
    return out


def from_json(js):
    """Las mismas filas desde el JSON oficial (para comparar en igualdad de condiciones)."""
    lic = [{"Registro": l["Registro"], "Clase": l["Clase"], "fecha": date_value(l["Fecha otorgamiento"]),
            "Municipalidad": text_key(l["Municipalidad"])} for l in js.get("Licencias registradas", [])]
    note = [{"Tipo": text_key(a["Tipo"]), "Detalle": text_key(a["Detalle"]), "fecha": date_value(a["Fecha anotación"])}
            for a in js.get("Anotaciones", [])]
    return {"licencias": lic, "anotaciones": note, "antecedentes": dict(js.get("Antecedentes", {}))}
