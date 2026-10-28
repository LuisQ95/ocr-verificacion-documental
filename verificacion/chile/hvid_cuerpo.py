"""Sección variable de la Hoja de Vida (licencias, anotaciones y antecedentes): se lee como texto línea a línea
(no por zonas fijas, porque la cantidad de filas cambia por persona) y se convierte en filas."""

import re
import statistics

from rapidfuzz import fuzz

from .normalizar_cl import ABR, MESES, clave_texto, fecha, limpio

X_COLUMNA_VALORES = 195
"""Límite (pt): los encabezados de anotación empiezan a la izquierda de esto; su texto, desde ~211."""


def lineas(palabras):
    """Agrupa palabras por línea (centro vertical) y las ordena de izquierda a derecha."""
    if not palabras:
        return []
    alto = statistics.median(p["bbox"][3] - p["bbox"][1] for p in palabras)
    out = []
    for p in sorted(palabras, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2):
        cy = (p["bbox"][1] + p["bbox"][3]) / 2
        if out and abs(cy - out[-1]["cy"]) <= alto * 0.55:
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


def _empieza(t, palabra, umbral=80):
    """True si la primera palabra de `t` se parece a `palabra` (tolerancia a errores de OCR)."""
    primera = t.split(" ", 1)[0] if t else ""
    return fuzz.ratio(primera, palabra) >= umbral


def _indice(tokens, palabra, umbral=75):
    """Posición del primer token parecido a `palabra` (None si ninguno)."""
    return next((i for i, t in enumerate(tokens) if fuzz.ratio(t, palabra) >= umbral), None)


def _licencia(t):
    """'PRIMERA CLASE B FECHA OTORGAMIENTO 5 MARZO 2019' con tolerancia a errores de OCR en las palabras fijas.

    Si la clase (una letra) no se detectó o se leyó mal, queda vacía: la fila se empareja por registro y fecha.
    """
    tok = t.split()
    if len(tok) < 4:
        return None
    if _empieza(tok[0], "PRIMERA", 75):
        registro = "PRIMERA"
    elif _empieza(tok[0], "ULTIMA", 75):
        registro = "ULTIMA"
    else:
        return None
    i_clase = _indice(tok[:3], "CLASE")
    i_otorg = _indice(tok, "OTORGAMIENTO", 70)
    if i_clase is None or i_otorg is None or i_otorg <= i_clase:
        return None
    clase = tok[i_clase + 1]
    if i_otorg == i_clase + 1 or fuzz.ratio(clase, "FECHA") >= 70 or not re.fullmatch(r"A[1-5]|[BCDEF]", clase):
        clase = ""
    resto = " ".join(tok[i_otorg + 1:])
    return {"Registro": registro, "Clase": clase, "fecha_txt": resto, "fecha": fecha(resto), "Municipalidad": ""}


def parsear(lineas_por_pagina):
    """Filas de licencias, anotaciones y antecedentes (en orden de página).

    - Municipalidad: los restos de la fecha de la fila anterior (mes, día, año) que el OCR deja al inicio no son
      parte del nombre.
    - Antecedentes: la fila empieza con el RUN sin puntos y sigue con SIN/CON (Courier grande; el OCR puede leer mal
      "ANTECEDENTES"), así que se reconoce por su estructura. El registro se decide solo por la palabra siguiente
      ("PRN CONDUCTOR" juntado contiene "RNC": "pRNConductor"); si no se lee, se asigna en orden (PRN, luego RNC).
    - Se omiten los encabezados de la tabla de antecedentes; una línea a la izquierda de X_COLUMNA_VALORES es el
      encabezado de una anotación.
    """
    out = {"licencias": [], "anotaciones": [], "antecedentes": {}, "antecedentes_conf": {}, "conf_min": 1.0}
    actual = None
    for linea in (l for pag in lineas_por_pagina for l in pag):
        t = clave_texto(linea["texto"])
        out["conf_min"] = min(out["conf_min"], linea["conf"])
        lic = _licencia(t)
        if lic:
            out["licencias"].append(lic | {"conf": linea["conf"]})
            actual = None
            continue
        if _empieza(t, "MUNICIPALIDAD", 75) and out["licencias"] and not out["licencias"][-1]["Municipalidad"]:
            resto = t.split()[1:]
            while len(resto) > 1 and (re.fullmatch(r"[\d.,\-/]+", resto[0]) or resto[0].lower().strip(".") in MESES + ABR):
                resto = resto[1:]
            out["licencias"][-1]["Municipalidad"] = " ".join(resto)
            out["licencias"][-1]["conf"] = min(out["licencias"][-1]["conf"], linea["conf"])
            continue
        tok = t.split()
        i_ant = _indice(tok, "ANTECEDENTES", 75)
        if i_ant is None and len(tok) >= 2 and re.fullmatch(r"\d{7,9}-?[\dK]", tok[0]) \
                and (fuzz.ratio(tok[1], "SIN") >= 67 or fuzz.ratio(tok[1], "CON") >= 67):
            i_ant = 2
        if i_ant is not None:
            sig = tok[i_ant + 1][:3] if i_ant + 1 < len(tok) else ""
            reg = "PRN" if fuzz.ratio(sig, "PRN") >= 67 else "RNC" if fuzz.ratio(sig, "RNC") >= 67 else None
            if reg is None:
                reg = "PRN" if "PRN" not in out["antecedentes"] else "RNC"
            previo = tok[i_ant - 1] if i_ant else ""
            out["antecedentes"][reg] = "CON ANTECEDENTES" if fuzz.ratio(previo, "CON") >= 67 and previo != "SIN" \
                else "SIN ANTECEDENTES"
            out["antecedentes_conf"][reg] = linea["conf"]
            continue
        compacto = t.replace(" ", "")
        if compacto.startswith("RUN") or "HOJADEVIDA" in compacto or fuzz.partial_ratio("HOJADEVIDA", compacto) >= 85 \
                or compacto in ("CONDUCTOR", "RUN"):
            continue
        if linea["x0"] < X_COLUMNA_VALORES:
            actual = {"Tipo": t, "detalle": [], "Fecha anotación": "", "fecha": None, "conf": linea["conf"]}
            out["anotaciones"].append(actual)
            continue
        if actual is not None:
            if t.startswith("FECHA ANOTACION") or _empieza(t, "FECHA") and "ANOTACION" in t:
                actual["Fecha anotación"] = t.split(":", 1)[-1].strip()
                actual["fecha"] = fecha(actual["Fecha anotación"])
            else:
                actual["detalle"].append(t)
            actual["conf"] = min(actual["conf"], linea["conf"])
    for a in out["anotaciones"]:
        a["Detalle"] = " ".join(a.pop("detalle"))
    return out


def desde_json(js):
    """Las mismas filas desde el JSON oficial (para comparar en igualdad de condiciones)."""
    lic = [{"Registro": l["Registro"], "Clase": l["Clase"], "fecha": fecha(l["Fecha otorgamiento"]),
            "Municipalidad": clave_texto(l["Municipalidad"])} for l in js.get("Licencias registradas", [])]
    anot = [{"Tipo": clave_texto(a["Tipo"]), "Detalle": clave_texto(a["Detalle"]), "fecha": fecha(a["Fecha anotación"])}
            for a in js.get("Anotaciones", [])]
    return {"licencias": lic, "anotaciones": anot, "antecedentes": dict(js.get("Antecedentes", {}))}
