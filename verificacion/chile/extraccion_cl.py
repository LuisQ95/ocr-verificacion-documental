"""Datos leídos de un documento del postulante, en JSON con la MISMA estructura y claves que el JSON oficial
(webscraping), para poder compararlos o revisarlos después. Incluye un bloque "_control" con el método de lectura y
la confianza de cada campo: lo leído de una foto (OCR) puede tener errores y debe revisarse."""

import re

from .. import config as C
from . import normalizar_cl as N

SIN_CLAVE = {"folio": "Folio", "codigo_verificacion": "Código Verificación", "fecha_emision": "Fecha emisión"}
FORMATO = {"ameco_rev02": "RR-DA-09 Rev.02", "ameco_rev03": "RR-DA-09 Rev.03"}
SECUNDARIAS = ("pie_", "evaluador_de", "folio_bajo_barra")
"""Zonas que repiten un dato que ya se lee en otra parte: no se extraen."""
CONF_MEDIA = 0.50
"""Niveles de la confianza calibrada del OCR (probabilidad estimada de que la lectura sea correcta): ALTA usa el
mismo umbral con que el sistema decide si una diferencia es real (C.CONF_OCR_ALTA); MEDIA desde este valor."""
NIVELES = ("ALTA", "MEDIA", "BAJA", "SIN LECTURA")
ESCALA = (f"confianza = probabilidad estimada de que la lectura sea correcta. ALTA >= {C.CONF_OCR_ALTA:.2f}; "
          f"MEDIA >= {CONF_MEDIA:.2f}; BAJA < {CONF_MEDIA:.2f}; SIN LECTURA = campo vacío. Es una estimación: un "
          "valor ALTA también puede estar mal leído (por ejemplo, un apellido poco común).")


def _poner(js, clave, valor):
    """'Exámenes/Señaléticas', 'Evaluadores[1]', 'Nombre' -> lugar en el JSON."""
    partes = clave.split("/")
    d = js
    for p in partes[:-1]:
        d = d.setdefault(p, {})
    m = re.fullmatch(r"(.+)\[(\d+)\]", partes[-1])
    if m:
        lista = d.setdefault(m.group(1), [])
        i = int(m.group(2))
        lista.extend([""] * (i + 1 - len(lista)))
        lista[i] = valor
    else:
        d[partes[-1]] = valor


def _fecha_hv(fecha, texto, dia):
    """Fecha de la sección de licencias/anotaciones con el formato del JSON oficial ("5 Marzo 2019", "01 Febrero
    2013"); si no se pudo interpretar, el texto leído tal cual."""
    if not fecha:
        return texto
    d = f"{fecha.day:02d}" if dia == "%02d" else str(fecha.day)
    return f"{d} {N.MESES[fecha.month - 1].capitalize()} {fecha.year}"


def extraer(doc):
    """doc: documento agrupado (pipeline_cl.unir_documento). Devuelve el JSON con lo que dice el documento.

    - `_confianza` tiene la misma estructura que el JSON: {"confianza": 0-1, "nivel": ALTA/MEDIA/BAJA/SIN LECTURA}.
    - En códigos, RUT, clases y fechas se quitan los espacios alrededor de los guiones ("AMCH - 12345",
      "B - D - A2", como los deja un PDF de Word).
    - La sección variable de la Hoja de Vida se lee por filas: cada licencia y cada anotación lleva la confianza de
      su fila.
    """
    js = {"Fuente": "Extraído del documento del postulante (NO es un dato oficial)"}
    if doc["tipo_documento"] == "ameco":
        js["Formato"] = FORMATO.get(doc["plantilla"], "")
    conf = {}
    for zona, info in doc["zonas"].items():
        if "(pág. 2)" in zona or zona.startswith(SECUNDARIAS) or info["tipo"] == "cuerpo":
            continue
        clave = info["clave"] or SIN_CLAVE.get(zona)
        if not clave:
            continue
        v = doc["valores"].get(zona) or {}
        texto = N.limpio(v.get("texto", ""))
        if info["tipo"] in ("codigo", "rut", "clases", "fecha"):
            texto = re.sub(r"\s*-\s*", "-", texto)
        if zona == "tipo_operador":
            texto = N.quitar_prefijo(texto, "Operador Equipo,")
        if zona == "fecha_emision":
            texto = re.split(r"\s*[-.]?\s*Contin[uú]a en la p[aá]gina", texto)[0].rstrip(" .-")
        _poner(js, clave, texto)
        _poner(conf, clave, _nivel(v.get("conf", 1.0), texto, doc["es_ocr"]))
    if doc["tipo_documento"] == "hoja_vida" and doc.get("filas"):
        f = doc["filas"]
        js["Licencias registradas"] = [{"Registro": x["Registro"], "Clase": x["Clase"],
                                        "Fecha otorgamiento": _fecha_hv(x["fecha"], x["fecha_txt"], "%-d"),
                                        "Municipalidad": x["Municipalidad"]} for x in f["licencias"]]
        js["Anotaciones"] = [{"Tipo": x["Tipo"], "Detalle": x["Detalle"],
                              "Fecha anotación": _fecha_hv(x["fecha"], re.sub(r"^FECHA\s*ANOTACION\s*:?\s*", "",
                                                                             x["Fecha anotación"]), "%02d")}
                             for x in f["anotaciones"]]
        js["Antecedentes"] = dict(f["antecedentes"])
        conf["Licencias registradas"] = [_nivel(x.get("conf", 1.0), "x", doc["es_ocr"]) for x in f["licencias"]]
        conf["Anotaciones"] = [_nivel(x.get("conf", 1.0), "x", doc["es_ocr"]) for x in f["anotaciones"]]
        conf["Antecedentes"] = {k: _nivel(f.get("antecedentes_conf", {}).get(k, f.get("conf_min", 1.0)), v, doc["es_ocr"])
                                for k, v in f["antecedentes"].items()}
    filas = plano(js, conf)
    js["_control"] = {
        "archivo": doc["archivo"],
        "metodo": "OCR (imagen o PDF escaneado): revisar los campos de confianza MEDIA o BAJA" if doc["es_ocr"]
        else "lectura directa del texto del PDF (sin OCR): confianza ALTA en todos los campos leídos",
        "escala_confianza": ESCALA,
        "resumen_confianza": {n: sum(1 for x in filas if x["nivel"] == n) for n in NIVELES},
        "campos_dudosos": [x["campo"] for x in filas if x["nivel"] in ("MEDIA", "BAJA", "SIN LECTURA")],
    }
    js["_confianza"] = conf
    return js


def _nivel(c, texto, es_ocr):
    """Confianza y nivel de un campo; sin texto es SIN LECTURA y un PDF original (sin OCR) es siempre ALTA."""
    if not texto:
        return {"confianza": 0.0, "nivel": "SIN LECTURA"}
    c = round(float(c if es_ocr else 1.0), 3)
    return {"confianza": c, "nivel": "ALTA" if c >= C.CONF_OCR_ALTA else "MEDIA" if c >= CONF_MEDIA else "BAJA"}


def plano(js, conf, prefijo=""):
    """Lista campo / valor / confianza / nivel (para mostrar o revisar), recorriendo el JSON y su _confianza.
    En una fila (licencia, anotación) todos sus campos llevan la confianza de la fila."""
    out = []
    for k, v in js.items():
        if k.startswith("_") or k in ("Fuente", "Formato"):
            continue
        c = conf.get(k) if isinstance(conf, dict) else None
        nombre = f"{prefijo}{k}"
        if isinstance(v, dict):
            out += plano(v, c or {}, nombre + " / ")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                ci = c[i] if isinstance(c, list) and i < len(c) else {}
                if isinstance(x, dict):
                    out += [{"campo": f"{nombre} {i + 1} / {kk}", "valor": vv, **ci} for kk, vv in x.items()]
                else:
                    out.append({"campo": f"{nombre} {i + 1}", "valor": x, **ci})
        else:
            out.append({"campo": nombre, "valor": v, **(c or {})})
    return out
