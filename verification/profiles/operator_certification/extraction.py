"""Datos leídos de un documento del postulante, en JSON con la MISMA estructura y claves que el JSON oficial
(webscraping), para poder compararlos o revisarlos después. Incluye un bloque "_control" con el método de lectura y
la confianza de cada campo: lo leído de una foto (OCR) puede tener errores y debe revisarse."""

import re

from ... import config as C
from . import normalize as N

NO_KEY = {"folio": "Folio", "codigo_verificacion": "Código Verificación", "fecha_emision": "Fecha emisión"}
FORMAT = {"ameco_rev02": "RR-DA-09 Rev.02", "ameco_rev03": "RR-DA-09 Rev.03"}
SECONDARY = ("pie_", "evaluador_de", "folio_bajo_barra")
"""Zonas que repiten un dato que ya se lee en otra parte: no se extraen."""
CONF_MEDIUM = 0.50
"""Niveles de la confianza calibrada del OCR (probabilidad estimada de que la lectura sea correcta): ALTA usa el
mismo umbral con que el sistema decide si una diferencia es real (C.OCR_HIGH_CONF); MEDIA desde este valor."""
LEVELS = ("ALTA", "MEDIA", "BAJA", "SIN LECTURA")
SCALE = (f"confianza = probabilidad estimada de que la lectura sea correcta. ALTA >= {C.OCR_HIGH_CONF:.2f}; "
          f"MEDIA >= {CONF_MEDIUM:.2f}; BAJA < {CONF_MEDIUM:.2f}; SIN LECTURA = campo vacío. Es una estimación: un "
          "valor ALTA también puede estar mal leído (por ejemplo, un apellido poco común).")


def _put(js, key, value):
    """'Exámenes/Señaléticas', 'Evaluadores[1]', 'Nombre' -> lugar en el JSON."""
    parts = key.split("/")
    d = js
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    m = re.fullmatch(r"(.+)\[(\d+)\]", parts[-1])
    if m:
        entries = d.setdefault(m.group(1), [])
        i = int(m.group(2))
        entries.extend([""] * (i + 1 - len(entries)))
        entries[i] = value
    else:
        d[parts[-1]] = value


def _record_date(date_value, text, dd):
    """Fecha de la sección de licencias/anotaciones con el formato del JSON oficial ("5 Marzo 2019", "01 Febrero
    2013"); si no se pudo interpretar, el texto leído tal cual."""
    if not date_value:
        return text
    d = f"{date_value.day:02d}" if dd == "%02d" else str(date_value.day)
    return f"{d} {N.MONTHS[date_value.month - 1].capitalize()} {date_value.year}"


def extract(doc):
    """doc: documento agrupado (pipeline.assemble_document). Devuelve el JSON con lo que dice el documento.

    - `_confianza` tiene la misma estructura que el JSON: {"confianza": 0-1, "nivel": ALTA/MEDIA/BAJA/SIN LECTURA}.
    - En códigos, RUT, clases y fechas se quitan los espacios alrededor de los guiones ("AMCH - 12345",
      "B - D - A2", como los deja un PDF de Word).
    - La sección variable de la Hoja de Vida se lee por filas: cada licencia y cada anotación lleva la confianza de
      su fila.
    """
    js = {"Fuente": "Extraído del documento del postulante (NO es un dato oficial)"}
    if doc["tipo_documento"] == "ameco":
        js["Formato"] = FORMAT.get(doc["plantilla"], "")
    conf = {}
    for zone, info in doc["zonas"].items():
        if "(pág. 2)" in zone or zone.startswith(SECONDARY) or info["tipo"] == "cuerpo":
            continue
        key = info["clave"] or NO_KEY.get(zone)
        if not key:
            continue
        v = doc["valores"].get(zone) or {}
        text = N.clean_text(v.get("texto", ""))
        if info["tipo"] in ("codigo", "rut", "clases", "fecha"):
            text = re.sub(r"\s*-\s*", "-", text)
        if zone == "tipo_operador":
            text = N.strip_prefix(text, "Operador Equipo,")
        if zone == "fecha_emision":
            text = re.split(r"\s*[-.]?\s*Contin[uú]a en la p[aá]gina", text)[0].rstrip(" .-")
        _put(js, key, text)
        _put(conf, key, _level(v.get("conf", 1.0), text, doc["es_ocr"]))
    if doc["tipo_documento"] == "hoja_vida" and doc.get("filas"):
        f = doc["filas"]
        js["Licencias registradas"] = [{"Registro": x["Registro"], "Clase": x["Clase"],
                                        "Fecha otorgamiento": _record_date(x["fecha"], x["fecha_txt"], "%-d"),
                                        "Municipalidad": x["Municipalidad"]} for x in f["licencias"]]
        js["Anotaciones"] = [{"Tipo": x["Tipo"], "Detalle": x["Detalle"],
                              "Fecha anotación": _record_date(x["fecha"], re.sub(r"^FECHA\s*ANOTACION\s*:?\s*", "",
                                                                             x["Fecha anotación"]), "%02d")}
                             for x in f["anotaciones"]]
        js["Antecedentes"] = dict(f["antecedentes"])
        conf["Licencias registradas"] = [_level(x.get("conf", 1.0), "x", doc["es_ocr"]) for x in f["licencias"]]
        conf["Anotaciones"] = [_level(x.get("conf", 1.0), "x", doc["es_ocr"]) for x in f["anotaciones"]]
        conf["Antecedentes"] = {k: _level(f.get("antecedentes_conf", {}).get(k, f.get("conf_min", 1.0)), v, doc["es_ocr"])
                                for k, v in f["antecedentes"].items()}
    rows = flat(js, conf)
    js["_control"] = {
        "archivo": doc["archivo"],
        "metodo": "OCR (imagen o PDF escaneado): revisar los campos de confianza MEDIA o BAJA" if doc["es_ocr"]
        else "lectura directa del texto del PDF (sin OCR): confianza ALTA en todos los campos leídos",
        "escala_confianza": SCALE,
        "resumen_confianza": {n: sum(1 for x in rows if x["nivel"] == n) for n in LEVELS},
        "campos_dudosos": [x["campo"] for x in rows if x["nivel"] in ("MEDIA", "BAJA", "SIN LECTURA")],
    }
    js["_confianza"] = conf
    return js


def _level(c, text, is_ocr):
    """Confianza y nivel de un campo; sin texto es SIN LECTURA y un PDF original (sin OCR) es siempre ALTA."""
    if not text:
        return {"confianza": 0.0, "nivel": "SIN LECTURA"}
    c = round(float(c if is_ocr else 1.0), 3)
    return {"confianza": c, "nivel": "ALTA" if c >= C.OCR_HIGH_CONF else "MEDIA" if c >= CONF_MEDIUM else "BAJA"}


def flat(js, conf, prefix=""):
    """Lista campo / valor / confianza / nivel (para mostrar o revisar), recorriendo el JSON y su _confianza.
    En una fila (licencia, anotación) todos sus campos llevan la confianza de la fila."""
    out = []
    for k, v in js.items():
        if k.startswith("_") or k in ("Fuente", "Formato"):
            continue
        c = conf.get(k) if isinstance(conf, dict) else None
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out += flat(v, c or {}, name + " / ")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                ci = c[i] if isinstance(c, list) and i < len(c) else {}
                if isinstance(x, dict):
                    out += [{"campo": f"{name} {i + 1} / {kk}", "valor": vv, **ci} for kk, vv in x.items()]
                else:
                    out.append({"campo": f"{name} {i + 1}", "valor": x, **ci})
        else:
            out.append({"campo": name, "valor": v, **(c or {})})
    return out
