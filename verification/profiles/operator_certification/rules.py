"""Comparación de cada documento con su dato oficial (JSON del webscraping) y dictamen por documento.

Estados por campo: IGUAL, IGUAL_TOLERADO_OCR, DIFERENTE, DUDOSO (lectura poco confiable: revisión humana),
DESACTUALIZADO (dato oficial posterior a la emisión del documento), NO_APLICA (dato propio de la emisión).

Dictamen por documento (en este orden de prioridad):
  NO LEGIBLE      no se pudo ubicar/leer el documento (no se acusa)
  FALSIFICADO     un dato no coincide con el oficial o hay evidencia técnica de alteración
  NO VÁLIDO       versión "COPIA, no válida como certificado" (no sirve aunque sea genuina)
  OBSERVADO       lecturas dudosas o indicios que requieren revisión humana
  DESACTUALIZADO  genuino pero emitido hace más de N días (Hoja de Vida) o con datos que cambiaron después
  AUTÉNTICO       todo coincide
"""

import re
from datetime import date

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ... import fields as K
from ... import config as C
from ...boxes import iou
from . import record_body as HV
from . import normalize as N
from .normalize import parse_seal

EQUAL, TOLERATED, DIFFERENT, DOUBTFUL = K.EQUAL, K.TOLERATED, K.DIFFERENT, K.DOUBTFUL
OUTDATED_C, NOT_APPLICABLE = K.OUTDATED_FIELD, K.NOT_APPLICABLE
AUTHENTIC, FLAGGED, FORGED, OUTDATED = K.AUTHENTIC, K.FLAGGED, K.FORGED, K.OUTDATED
ILLEGIBLE, NOT_VALID, NOT_RECOGNIZED, NO_OFFICIAL_DATA = K.ILLEGIBLE, K.NOT_VALID, K.NOT_RECOGNIZED, K.NO_OFFICIAL_DATA
RECORD_VALIDITY_DAYS = 30
"""Una Hoja de Vida más antigua que esto queda DESACTUALIZADA (ver requirements.yaml)."""

VALID_SOURCES = {
    "hoja_vida": {"Courier", "Helvetica", "Helvetica-Bold"},
    "ameco_rev03": {"ArialMT", "Arial-BoldMT", "Arial", "Arial-Bold", "Arial Bold", "Arial Regular"},
    "ameco_rev02": {"CIDFont+F1", "CIDFont+F2", "CIDFont+F3", "CIDFont+F4", "Verdana", "Verdana-Bold",
                    "Verdana-Italic", "TimesNewRoman", "TimesNewRomanPSMT"},
}
"""Fuentes de los documentos genuinos; las de un documento rehecho en Word o Canva son otras."""


def json_value(js, key):
    """'Exámenes/Señaléticas', 'Evaluadores[1]', 'Nombre' -> valor del JSON (las claves pueden tener puntos)."""
    v = js
    for part in key.split("/"):
        m = re.fullmatch(r"(.+)\[(\d+)\]", part)
        if m:
            v = (v or {}).get(m.group(1), [])
            v = v[int(m.group(2))] if len(v) > int(m.group(2)) else None
        else:
            v = (v or {}).get(part) if isinstance(v, dict) else None
    return "" if v is None else str(v)


VALID_VALUES = {"estado_evaluacion": {"APROBADO", "REPROBADO"}, "evaluacion_practica": {"APROBADO", "REPROBADO"},
           "nota_aprobacion_final": {"APROBADO", "REPROBADO"}, "licencia_ley": {"18290"}}
"""Valores posibles de los campos cerrados: leer otro valor es un error de lectura, no una falsificación."""
OTHER_FIELD_VALUE = ("APROBAD", "REPROBAD")
"""Raíz de los valores de los campos cerrados. Si aparece dentro de un nombre, un domicilio o un código, el OCR leyó
algo ajeno al campo (una firma, un timbre o la interfaz de una captura): es una lectura contaminada, no un dato
cambiado."""
CONNECTORS = {"DE", "DEL", "N", "NO", "NRO"}
"""Palabras de enlace del formato ("de TALCAHUANO", "N° MEJILLONES"): leídas junto al valor oficial completo no lo
cambian."""
MIN_RENEWAL_DAYS = 365
"""Las licencias se renuevan cada varios años: dos otorgamientos de la misma licencia a menos de esto son sospechosos."""
PAGE2_SUFFIX = " (pág. 2)"
"""Sufijo de las zonas de la página 2 de la Hoja de Vida (ver pipeline.assemble_document)."""
EVALUATOR_ZONES = ("evaluador1_nombre", "evaluador2_nombre", "evaluador_de_competencia")
"""Nombres de los evaluadores. Quien falsifica pone el nombre de otro evaluador; una lectura casi idéntica al oficial
(1-2 caracteres) es un error de lectura de la firma o del timbre vecino, no un cambio."""


def _ocr_similar(a, b, max_dist=2):
    """Diferencia del tamaño de un error de lectura (1-2 caracteres): en OCR no basta para acusar."""
    return Levenshtein.distance(a, b) <= max_dist


def _tolerable_reading(kind, official, read_value, zone):
    """Motivo si la lectura OCR es el valor oficial con ruido que no lo cambia; "" si no.

    Solo para textos (no fechas, RUT, números ni clases, que tienen sus propias reglas). Las palabras que se quitan
    nunca forman parte de un valor alternativo, así que un falsificador no gana nada con ellas."""
    if kind not in ("identidad", "descriptivo", "categorico") or zone in VALID_VALUES:
        return ""
    ko = N.text_key(official)
    kl = N.text_key(N.strip_prefix(N.clean_text(read_value), "Operador Equipo,") if zone == "tipo_operador" else read_value)
    if len(ko) < 3:
        return ""
    tokens = kl.split()
    no_link = [t for t in tokens if re.sub(r"[^A-Z0-9]", "", t) not in CONNECTORS]
    if no_link != tokens and " ".join(no_link) == ko:
        return "solo se leyeron palabras de enlace vecinas junto al valor oficial"
    without_foreign = [t for t in tokens if not ("ROBAD" in t or "PROBAD" in t or len(t) >= 4 and t in "REPROBADO")]
    if without_foreign != tokens and " ".join(without_foreign) == ko and not any(v in ko for v in OTHER_FIELD_VALUE):
        return "se ignoró texto de otro campo (APROBADO...) leído dentro de la zona"
    if zone in EVALUATOR_ZONES and ko != kl and _ocr_similar(ko, kl):
        return "nombre del evaluador casi idéntico al oficial (error de lectura)"
    return ""


def compare_value(kind, official, read_value, is_ocr, conf, zone=""):
    """Estado y motivo de un campo según su tipo.

    Regla para documentos leídos por OCR: quien falsifica reemplaza un valor por OTRO VALOR VÁLIDO (otro nombre,
    un RUT con dígito verificador correcto, APROBADO en vez de REPROBADO, una clase que existe, una fecha válida).
    Un error de lectura produce algo inválido o casi igual: eso es DUDOSO (revisión humana), no FALSIFICADO.

    Se toleran (IGUAL_TOLERADO_OCR) las lecturas que solo agregan palabras de enlace al valor oficial completo, las
    que coinciden al quitar el texto de otro campo (APROBADO...) y, en los nombres de evaluador, las casi idénticas.
    """
    state, reason = _compare_value(kind, official, read_value, is_ocr, conf, zone)
    if state not in (DIFFERENT, DOUBTFUL) or not is_ocr:
        return state, reason
    tolerated = _tolerable_reading(kind, official, read_value, zone)
    if tolerated:
        return TOLERATED, tolerated
    if state != DIFFERENT:
        return state, reason
    le = N.clean_text(read_value)
    if zone == "tipo_operador":
        le = N.strip_prefix(le, "Operador Equipo,")
    if kind == "rut":
        r = N.national_id_ocr(le)
        if not N.national_id_valid(r):
            return DOUBTFUL, "RUT leído con dígito verificador inválido (error de lectura)"
    elif kind == "clases":
        valid_items = N.classes(le)
        junk = [t for t in re.split(r"[\s\-/]+", N.strip_accents(le).upper()) if t and t not in valid_items]
        if junk:
            return DOUBTFUL, f"clases con códigos inexistentes ({', '.join(junk[:3])}): error de lectura"
    elif zone in VALID_VALUES:
        if N.text_key(le) not in VALID_VALUES[zone]:
            return DOUBTFUL, f"'{le}' no es un valor posible del campo (error de lectura)"
    elif kind in ("identidad", "codigo"):
        if _ocr_similar(N.text_key(official), N.text_key(le)):
            return DOUBTFUL, "difiere en 1-2 caracteres: posible error de lectura"
    elif kind in ("descriptivo", "categorico"):
        if _ocr_similar(N.text_key(official), N.text_key(le), 2 if len(official) < 30 else 3):
            return DOUBTFUL, "difiere en pocos caracteres: posible error de lectura"
    if kind in ("identidad", "codigo", "descriptivo", "categorico"):
        ko, kl = N.text_key(official), N.text_key(le)
        if len(kl) >= 3 and (ko.startswith(kl) or ko.endswith(kl)):
            return DOUBTFUL, "lectura incompleta (texto cortado): el resto del valor no se leyó"
        if len(ko) >= 3 and ko in kl:
            return DOUBTFUL, "el valor oficial se leyó junto a texto vecino (lectura corrida)"
        if zone not in VALID_VALUES:
            foreign = [v for v in OTHER_FIELD_VALUE if v in kl.replace(" ", "") and v not in ko.replace(" ", "")]
            if foreign:
                return DOUBTFUL, f"la lectura incluye texto de otro campo ({foreign[0]}...): lectura contaminada"
    return state, reason


def _compare_value(kind, official, read_value, is_ocr, conf, zone=""):
    """Comparación base por tipo (fecha, rut, numero, clases, codigo; el resto como texto), antes de las reglas de
    error de lectura de `compare_value`. Con OCR y confianza baja, una diferencia queda DUDOSA.

    En clases, un código con forma de clase pero inexistente ("A44") puede ser una clase agregada mal leída: se
    revisa en vez de darse por igual.
    """
    of, le = N.clean_text(official), N.clean_text(read_value)
    if zone == "tipo_operador":
        le = N.strip_prefix(le, "Operador Equipo,")
    if not le:
        if is_ocr:
            return DOUBTFUL, "el OCR no leyó texto en la zona"
        return (EQUAL, "") if not of else (DIFFERENT, "el campo está vacío en el documento")
    if of == le:
        return EQUAL, ""
    low = is_ocr and conf < C.OCR_HIGH_CONF

    def different(reason=""):
        """DIFERENTE, o DUDOSO si la lectura tiene confianza baja."""
        return (DOUBTFUL, f"lectura con confianza baja ({conf:.2f}){': ' + reason if reason else ''}") if low \
            else (DIFFERENT, reason)

    if kind == "fecha":
        fo, fl = N.date_value(of), N.date_value(le)
        if fl is None:
            return (DOUBTFUL, "fecha ilegible") if is_ocr else (DIFFERENT, "no es una fecha válida")
        return (TOLERATED, "misma fecha en otro formato") if fo == fl else different(f"{fl} vs oficial {fo}")
    if kind == "rut":
        ro, rl = N.national_id(of), (N.national_id_ocr(le) if is_ocr else N.national_id(le))
        if rl is None:
            return (DOUBTFUL, "RUT ilegible") if is_ocr else (DIFFERENT, "no es un RUT")
        return (TOLERATED, "mismo RUT con otro formato") if ro == rl else different()
    if kind == "numero":
        no, nl = N.number(of), N.number(le.translate(N.DIGIT_CONFUSIONS) if is_ocr else le)
        if no is None:
            return different("el dato oficial no es numérico")
        if is_ocr and not any(ch.isdigit() for ch in le):
            return DOUBTFUL, "número ilegible (el OCR no leyó ningún dígito)"
        if nl is None:
            return (DOUBTFUL, "número ilegible") if is_ocr else (DIFFERENT, "")
        return (TOLERATED, "mismo valor") if abs(no - nl) < 0.05 else different()
    if kind == "clases":
        if N.classes(of) != N.classes(le):
            return different()
        rare = [t for t in re.split(r"[^A-Z0-9]+", N.strip_accents(le).upper()) if re.fullmatch(r"[A-F]\d+", t)
                 and not re.fullmatch(r"A[1-5]", t)]
        if is_ocr and rare:
            return DOUBTFUL, f"código de clase inexistente leído ({', '.join(rare[:3])}): revisar si hay una clase agregada"
        return TOLERATED, "mismas clases"
    if kind == "codigo":
        if N.code(of) == N.code(le) or (is_ocr and N.code_ocr(of) == N.code_ocr(le)):
            return TOLERATED, "solo difieren caracteres confundibles"
        return different()
    ko, kl = N.text_key(of), N.text_key(le)
    if ko == kl:
        return TOLERATED, "solo difieren tildes, mayúsculas o puntuación"
    if K.negation_changes(of, le):
        return different("cambia una negación")
    if is_ocr and K.ocr_key(of) == K.ocr_key(le):
        return TOLERATED, "solo difieren caracteres confundibles del OCR"
    if is_ocr and kind == "descriptivo" and len(ko) >= 14 and fuzz.ratio(ko, kl) >= 92:
        return TOLERATED, "texto casi idéntico (ruido de OCR)"
    return different()


def _plausible_renewal(doc_date, official_date):
    """True si la fecha impresa puede ser la de la licencia antes de su renovación oficial: anterior y con al menos
    MIN_RENEWAL_DAYS de diferencia. Una fecha impresa poco antes de la renovación es la técnica de quien "renueva"
    una licencia vencida en el documento."""
    return bool(doc_date and official_date and (official_date - doc_date).days >= MIN_RENEWAL_DAYS)


def _single_row_without_class(lo, official_data, rows, used_rows):
    """[índice] de la fila del documento que solo puede ser la licencia oficial `lo`, o [] si no hay una sola.

    Caso: la clase de la fila no se leyó (una letra suelta que el detector no ve o lee mal) y su fecha no es la oficial
    ni la de otra licencia. Si en el documento hay UNA sola fila así del mismo registro (PRIMERA/ULTIMA) y municipalidad,
    y entre las licencias oficiales de ese registro `lo` es la ÚNICA que no aparece con su clase ni con su fecha, son la
    misma licencia: su fecha se compara (es la técnica de quien cambia la fecha de una licencia)."""
    def appears(o):
        """True si la licencia oficial `o` figura en el documento con su clase o, sin clase, con su fecha."""
        return any(ld["Registro"] == o["Registro"] and (N.code_ocr(ld["Clase"]) == N.code_ocr(o["Clase"])
                                                        or not ld["Clase"] and ld["fecha"] == o["fecha"]) for ld in rows)

    missing_n = [o for o in official_data if o["Registro"] == lo["Registro"] and not appears(o)]
    dates = {o["fecha"] for o in official_data if o["Registro"] == lo["Registro"]}
    cand = [j for j, ld in enumerate(rows) if j not in used_rows and not ld["Clase"] and ld["Registro"] == lo["Registro"]
            and ld["fecha"] and ld["fecha"] not in dates
            and N.text_key(ld["Municipalidad"]) == N.text_key(lo["Municipalidad"])]
    return cand if len(cand) == 1 and missing_n == [lo] else []


def _compare_rows(doc_rows, js, issue, is_ocr, has_p2=True):
    """Licencias, anotaciones y antecedentes: conjuntos de filas. Un registro oficial anterior a la emisión
    que no figura en el documento = ocultado; uno posterior = documento desactualizado.

    - Licencias: clave (registro, clase); se comparan fecha y municipalidad. Con OCR, si no se detectó la clase, es la
      misma fila cuando coinciden registro y fecha, o cuando la licencia oficial se renovó después de la emisión y la
      fila muestra una fecha anterior que no es la de ninguna otra licencia (la fila antes de renovarse), o cuando es
      la única fila sin clase que puede corresponderle (_fila_unica_sin_clase). Una fila
      "nueva" solo acusa si está bien formada (clase existente y fecha válida).
    - Anotaciones: se emparejan por fecha de anotación o por tipo y detalle. Sin la página 2, una anotación cortada
      al final de la página 1 (su fecha quedó en la página 2) se compara solo en lo que se ve. Un encabezado suelto
      mal leído (sin texto ni fecha) no es una anotación.
    - Sin la página 2, las licencias, anotaciones y antecedentes que faltan pueden estar en ella: NO_APLICA.
    - Antecedentes: si el oficial tiene anotaciones posteriores a la emisión, un "SIN ANTECEDENTES" impreso es un
      estado que cambió después (DESACTUALIZADO), no una falsificación.
    """
    of = HV.from_json(js)
    fields = []

    def missing_state(row_date, what):
        """Estado de un registro oficial que no figura en el documento."""
        if row_date and issue and row_date > issue:
            return OUTDATED_C, f"{what} posterior a la emisión del documento"
        if is_ocr and doc_rows["conf_min"] < C.OCR_HIGH_CONF:
            return DOUBTFUL, f"{what} oficial no encontrado en el documento (lectura con confianza baja)"
        return DIFFERENT, f"{what} oficial que no figura en el documento"

    used_rows = set()
    for lo in of["licencias"]:
        cand = [j for j, ld in enumerate(doc_rows["licencias"]) if j not in used_rows
                and ld["Registro"] == lo["Registro"] and N.code_ocr(ld["Clase"]) == N.code_ocr(lo["Clase"])]
        if not cand and is_ocr:
            cand = [j for j, ld in enumerate(doc_rows["licencias"]) if j not in used_rows and not ld["Clase"]
                    and ld["Registro"] == lo["Registro"] and ld["fecha"] == lo["fecha"]]
        if not cand and is_ocr and issue and lo["fecha"] and lo["fecha"] > issue:
            cand = [j for j, ld in enumerate(doc_rows["licencias"]) if j not in used_rows and not ld["Clase"]
                    and ld["Registro"] == lo["Registro"] and _plausible_renewal(ld["fecha"], lo["fecha"])
                    and not any(o["Registro"] == ld["Registro"] and o["fecha"] == ld["fecha"] for o in of["licencias"])]
        if not cand and is_ocr:
            cand = _single_row_without_class(lo, of["licencias"], doc_rows["licencias"], used_rows)
        name = f"licencia {lo['Registro'].lower()} clase {lo['Clase']}"
        if not cand:
            e, m = missing_state(lo["fecha"], "licencia")
            if e == DIFFERENT and not has_p2:
                e, m = NOT_APPLICABLE, "puede figurar en la página 2, que no fue entregada"
            fields.append({"zona": name, "oficial": f"{lo['fecha']} {lo['Municipalidad']}", "postulante": "",
                           "estado": e, "motivo": m, "conf": 1.0})
            continue
        j = min(cand, key=lambda j: abs(((doc_rows["licencias"][j]["fecha"] or date.min) - (lo["fecha"] or date.min)).days))
        used_rows.add(j)
        ld = doc_rows["licencias"][j]
        if ld["fecha"] == lo["fecha"]:
            e, m = EQUAL, ""
        elif ld["fecha"] is None:
            e, m = (DOUBTFUL, "fecha ilegible") if is_ocr else (DIFFERENT, "fecha inválida")
        elif issue and lo["fecha"] and lo["fecha"] > issue and ld["fecha"] < lo["fecha"]:
            e, m = (OUTDATED_C, "la licencia se renovó después de la emisión del documento") \
                if _plausible_renewal(ld["fecha"], lo["fecha"]) else \
                (DOUBTFUL, f"la fecha impresa ({ld['fecha']}) es de menos de un año antes de la renovación oficial: "
                         "revisar si fue alterada")
        else:
            e, m = compare_value("fecha", str(lo["fecha"]), ld["fecha_txt"], is_ocr, ld["conf"])
        fields.append({"zona": f"{name} - fecha", "oficial": str(lo["fecha"]), "postulante": ld["fecha_txt"],
                       "estado": e, "motivo": m, "conf": round(ld["conf"], 3)})
        e2, m2 = compare_value("categorico", lo["Municipalidad"], ld["Municipalidad"], is_ocr, ld["conf"])
        if e2 == DIFFERENT and issue and lo["fecha"] and lo["fecha"] > issue:
            e2, m2 = OUTDATED_C, "cambió después de la emisión"
        fields.append({"zona": f"{name} - municipalidad", "oficial": lo["Municipalidad"], "postulante": ld["Municipalidad"],
                       "estado": e2, "motivo": m2, "conf": round(ld["conf"], 3)})
    for j, ld in enumerate(doc_rows["licencias"]):
        if j not in used_rows:
            good = bool(N.classes(ld["Clase"])) and ld["fecha"] is not None
            e = DOUBTFUL if is_ocr and (ld["conf"] < C.OCR_HIGH_CONF or not good) else DIFFERENT
            fields.append({"zona": f"licencia {ld['Registro'].lower()} clase {ld['Clase']}", "oficial": "",
                           "postulante": f"{ld['fecha_txt']} {ld['Municipalidad']}", "estado": e,
                           "motivo": "licencia que no existe en el registro oficial", "conf": round(ld["conf"], 3)})

    used_rows = set()
    for ao in of["anotaciones"]:
        cand = [j for j, ad in enumerate(doc_rows["anotaciones"]) if j not in used_rows and
                (ad["fecha"] == ao["fecha"] or fuzz.ratio(ad["Tipo"], ao["Tipo"]) >= 80
                 and fuzz.ratio(ad["Detalle"], ao["Detalle"]) >= 80)]
        if not cand and not has_p2:
            cand = [j for j, ad in enumerate(doc_rows["anotaciones"]) if j not in used_rows and ad["fecha"] is None
                    and fuzz.ratio(ad["Tipo"], ao["Tipo"]) >= 80 and len(ad["Detalle"]) >= 10
                    and fuzz.ratio(ad["Detalle"], ao["Detalle"][:len(ad["Detalle"])]) >= 88]
        name = f"anotación {ao['Tipo'][:40]} ({ao['fecha']})"
        if not cand:
            e, m = missing_state(ao["fecha"], "anotación")
            if e == DIFFERENT and not has_p2:
                e, m = NOT_APPLICABLE, "puede figurar en la página 2, que no fue entregada"
            fields.append({"zona": name, "oficial": ao["Detalle"][:120], "postulante": "", "estado": e, "motivo": m,
                           "conf": 1.0})
            continue
        j = cand[0]
        used_rows.add(j)
        ad = doc_rows["anotaciones"][j]
        sim = fuzz.ratio(ad["Detalle"], ao["Detalle"])
        cut_at = not has_p2 and 10 <= len(ad["Detalle"]) < len(ao["Detalle"]) and \
            fuzz.ratio(ad["Detalle"], ao["Detalle"][:len(ad["Detalle"])]) >= (88 if is_ocr else 97)
        if sim >= 97 or (is_ocr and sim >= 88):
            e, m = (EQUAL if sim == 100 else TOLERATED), ("" if sim == 100 else "texto casi idéntico")
        elif cut_at:
            e, m = TOLERATED, "el texto sigue en la página 2 (no entregada): coincide lo visible"
        elif is_ocr and ad["conf"] < C.OCR_HIGH_CONF:
            e, m = DOUBTFUL, f"texto de la anotación leído con confianza baja ({ad['conf']:.2f})"
        else:
            e, m = DIFFERENT, "el texto de la anotación no coincide"
        fields.append({"zona": name, "oficial": ao["Detalle"][:120], "postulante": ad["Detalle"][:120], "estado": e,
                       "motivo": m, "conf": round(ad["conf"], 3)})
    for j, ad in enumerate(doc_rows["anotaciones"]):
        if j not in used_rows:
            if not ad["Detalle"] and ad["fecha"] is None:
                continue
            good = ad["fecha"] is not None and len(ad["Detalle"]) >= 20
            e = DOUBTFUL if is_ocr and (ad["conf"] < C.OCR_HIGH_CONF or not good) else DIFFERENT
            fields.append({"zona": f"anotación {ad['Tipo'][:40]}", "oficial": "", "postulante": ad["Detalle"][:120],
                           "estado": e, "motivo": "anotación que no existe en el registro oficial",
                           "conf": round(ad["conf"], 3)})

    for reg in ("PRN", "RNC"):
        vo, vd = of["antecedentes"].get(reg, ""), doc_rows["antecedentes"].get(reg, "")
        if not vd and not has_p2:
            e, m = NOT_APPLICABLE, "la página 2 (antecedentes) no fue entregada"
        elif not vd:
            e, m = (DOUBTFUL, "no se leyó") if is_ocr else (DIFFERENT, "no figura en el documento")
        elif vo == vd:
            e, m = EQUAL, ""
        elif vd == "SIN ANTECEDENTES" and any(a["fecha"] and issue and a["fecha"] > issue for a in of["anotaciones"]):
            e, m = OUTDATED_C, "el estado de antecedentes cambió después de la emisión del documento"
        else:
            e, m = DIFFERENT, "cambia el estado de antecedentes"
        fields.append({"zona": f"antecedentes {reg}", "oficial": vo, "postulante": vd, "estado": e, "motivo": m, "conf": 1.0})
    return fields


def code_signals(doc, values):
    """Timbre electrónico y código de barras vs lo impreso (un dato editado no cambia el timbre)."""
    s = []
    cc = doc.get("codigos") or {}
    folio = N.code(values.get("folio", {}).get("texto", ""))
    for text in cc.get("timbre", []):
        t = parse_seal(text)
        if not t:
            continue
        delta = []
        if t.get("FOLIO") and folio and t["FOLIO"] != folio and not doc.get("es_ocr"):
            delta.append(f"folio {t['FOLIO']} vs impreso {folio}")
        run = N.national_id(values.get("run", {}).get("texto", ""))
        if t.get("RUN") and run and N.national_id(t["RUN"]) != run:
            if not doc.get("es_ocr") or values.get("run", {}).get("conf", 1) >= C.OCR_HIGH_CONF:
                delta.append(f"RUN {t['RUN']} vs impreso {run}")
        nm = N.text_key(values.get("nombre", {}).get("texto", ""))
        if t.get("NOM") and nm and fuzz.ratio(N.text_key(t["NOM"]), nm) < (95 if not doc.get("es_ocr") else 80):
            delta.append(f"nombre '{t['NOM']}' vs impreso '{nm}'")
        fe = N.date_value(values.get("fecha_emision", {}).get("texto", ""))
        if t.get("FE") and fe and t["FE"][:10] != fe.isoformat():
            if not doc.get("es_ocr") or values.get("fecha_emision", {}).get("conf", 1) >= C.OCR_HIGH_CONF:
                delta.append(f"fecha de emisión {t['FE'][:10]} vs impresa {fe}")
        if delta:
            s.append(K.signal("timbre", "fraude", "el timbre electrónico no coincide con lo impreso: " + "; ".join(delta)))
        else:
            s.append(K.signal("timbre", "info", "timbre electrónico legible y coherente con lo impreso"))
        break
    for b in cc.get("barras", []):
        if folio and N.code(b) != folio and re.fullmatch(r"\d{12}", N.code(b)) and not doc.get("es_ocr"):
            s.append(K.signal("codigo_barras", "fraude",
                             f"el código de barras ({b}) no coincide con el folio impreso ({folio})"))
    return s


def native_signals(doc, template):
    """Señales de un PDF nativo: objetos superpuestos, anotaciones, metadatos, revisiones y tipografías ajenas.

    - Las imágenes propias del documento genuino (logos, sellos, timbre, firmas, QR) no son superposiciones.
    - Un resaltado o subrayado es transparente: no oculta ni cambia datos (lo agrega quien revisa el PDF), así que
      solo se observa.
    - La versión COPIA tiene marca de agua y otra disposición: sus imágenes y gráficos no son los de la plantilla.
      Se observan (la copia se rechaza igual como NO VÁLIDO y un dato distinto sigue acusando).
    """
    objects = dict(doc.get("objetos") or {})
    objects["imagenes"] = [r for r in objects.get("imagenes", [])
                           if not any(iou(r, e) > 0.6 for e in template.get("imagenes_esperadas", []))]
    s = K.object_signals(dict(doc, objetos=objects), template)
    for x in s:
        if x["tipo"] == "anotacion" and re.search(r"'(Highlight|Underline|Squiggly)'", x["detalle"]):
            x.update(severidad="observacion", detalle=x["detalle"].replace(
                "agregada al PDF", "agregada al PDF después de su emisión (marca transparente, no oculta datos)"))
        elif doc.get("copia") and x["tipo"] == "objeto_superpuesto":
            x.update(severidad="observacion", detalle=x["detalle"] + " (formato COPIA: marca de agua propia)")
    meta = doc.get("meta") or {}
    if meta.get("producer"):
        s.append(K.signal("metadatos", "info", f"PDF producido con '{meta.get('producer')}'"))
    if (doc.get("revisiones_pdf") or 1) > 1:
        s.append(K.signal("metadatos", "info",
                             f"el PDF tiene {doc['revisiones_pdf']} versiones guardadas (edición incremental)"))
    valid_items = VALID_SOURCES["hoja_vida" if template["tipo_documento"] == "hoja_vida" else template["id"]]
    foreign = sorted({K.base_font(p["font"]) for p in doc.get("palabras_valor", []) if p.get("font")}
                    - {K.base_font(f) for f in valid_items} - valid_items)
    if foreign:
        s.append(K.signal("fuentes", "observacion", f"tipografía que no usa el emisor en los datos: {', '.join(foreign[:4])}"))
    return s


FAILED_READ_REASONS = ("fecha ilegible", "RUT ilegible", "número ilegible", "el OCR no leyó", "lectura incompleta",
                           "difiere en 1-2", "difiere en pocos", "no es un valor posible", "clases con códigos")
"""Motivos de DUDOSO que describen una lectura fallida (algo inválido o casi igual al oficial), no un valor distinto."""
HIGH_PRIORITY_CONF = 0.5
"""Confianza desde la cual un valor distinto (pero no concluyente) leído en un documento OBSERVADO lo pone primero
en la revisión. Medido en el subconjunto de ajuste."""
HIGH, LOW = "ALTA", "BAJA"


def review_priority(fields, signals):
    """Prioridad de un documento OBSERVADO en la revisión humana.

    ALTA: un indicio técnico (objeto superpuesto, tipografía ajena, timbre...) o un valor distinto al oficial leído con
    confianza de al menos HIGH_PRIORITY_CONF. BAJA: solo lecturas fallidas o la página que falta. No cambia el
    dictamen: ordena la revisión para que las falsificaciones probables se vean primero."""
    if any(s["severidad"] == "observacion" and s["tipo"] != "paginas" for s in signals):
        return HIGH
    if any(c["estado"] == DOUBTFUL and c["conf"] >= HIGH_PRIORITY_CONF and c["postulante"]
           and not c["motivo"].startswith(FAILED_READ_REASONS) for c in fields):
        return HIGH
    return LOW


def verdict(fields, signals, illegible=False, copy_doc=False, outdated=False):
    """(dictamen, motivos sin repetir) con la prioridad del módulo. Una COPIA no sirve como certificado: se rechaza
    aunque tenga lecturas u observaciones por revisar, pero un dato distinto la marca como FALSIFICADA."""
    delta = [c for c in fields if c["estado"] == DIFFERENT]
    doubtful = [c for c in fields if c["estado"] == DOUBTFUL]
    des = [c for c in fields if c["estado"] == OUTDATED_C]
    if illegible:
        v = ILLEGIBLE
    elif delta or any(x["severidad"] == "fraude" for x in signals):
        v = FORGED
    elif copy_doc:
        v = NOT_VALID
    elif doubtful or any(x["severidad"] == "observacion" for x in signals):
        v = FLAGGED
    elif des or outdated:
        v = OUTDATED
    else:
        v = AUTHENTIC
    reasons = [f"{c['zona']}: oficial '{c['oficial']}' vs documento '{c['postulante']}'"
               + (f" ({c['motivo']})" if c["motivo"] else "") for c in delta]
    reasons += [x["detalle"] for x in signals if x["severidad"] in ("fraude", "observacion")]
    reasons += [f"{c['zona']}: {c['motivo']}" for c in doubtful]
    reasons += [f"{c['zona']}: {c['motivo']}" for c in des[:5]]
    return v, list(dict.fromkeys(reasons))


def compare_document(doc, js, doc_templates, declared, today):
    """doc: lectura agrupada de un documento (valores por zona, filas de la Hoja de Vida, señales técnicas).
    js: dato oficial (o None). Devuelve campos, señales y dictamen.

    - Lo que no se pudo leer no se compara: acusaría por datos "faltantes" que solo son ilegibles.
    - Una diferencia en una zona que contiene texto impreso del formato vecino es una lectura desalineada (DUDOSO).
    - Identidad: el documento debe estar a nombre del RUT declarado. Solo un RUT completo y válido (dígito
      verificador) puede ser "de otra persona"; una lectura parcial no.
    - La cabecera de la página 2 repite nombre, RUN y fecha de nacimiento: si la página 1 ya confirmó el valor, una
      lectura dudosa de la repetición no manda a revisión (un valor distinto leído con confianza sí acusa).
    """
    is_ocr = doc["es_ocr"]
    values = doc["valores"]
    signals = list(doc.get("senales", []))
    fields = []
    doc_type = doc["tipo_documento"]
    issue = N.date_value(values.get("fecha_emision", {}).get("texto", "")) if doc_type == "hoja_vida" else None

    if doc.get("ilegible"):
        pass
    elif js is None:
        signals.append(K.signal("fuente_oficial", "observacion", "no hay dato oficial (JSON) para comparar este documento"))
    else:
        for zone, info in doc_templates.items():
            if not info["clave"]:
                continue
            v = values.get(zone)
            if v is None and zone.startswith(("pie_", "evaluador_de")):
                continue
            read_value, conf = (v or {}).get("texto", ""), (v or {}).get("conf", 0.0 if is_ocr else 1.0)
            official = json_value(js, info["clave"])
            e, m = compare_value(info["tipo"], official, read_value, is_ocr, conf, zone)
            if e == DIFFERENT and is_ocr:
                owned = N.strip_prefix(N.clean_text(read_value), "Operador Equipo,") if zone == "tipo_operador" else read_value
                foreign = (set(N.text_key(owned).split()) & doc.get("formato_ajeno", {}).get(zone, set())) \
                    - set(N.text_key(official).split())
                if foreign:
                    e, m = DOUBTFUL, (f"la zona contiene texto impreso del formato ({', '.join(sorted(foreign)[:3])}): "
                                    "lectura desalineada")
            fields.append({"zona": zone, "oficial": official, "postulante": read_value, "estado": e, "motivo": m,
                           "conf": round(conf, 3)})
        states = {c["zona"]: c["estado"] for c in fields}
        for c in fields:
            if c["zona"].endswith(PAGE2_SUFFIX) and c["estado"] == DOUBTFUL \
                    and states.get(c["zona"][:-len(PAGE2_SUFFIX)]) in (EQUAL, TOLERATED):
                c["estado"], c["motivo"] = TOLERATED, "la página 1 confirma el valor; esta repetición se leyó con ruido"
        if doc_type == "hoja_vida" and doc.get("filas") is not None:
            fields += _compare_rows(doc["filas"], js, issue, is_ocr,
                                      has_p2="página 2" not in (doc.get("pagina_faltante") or ""))
        for zone in ("folio", "codigo_verificacion", "fecha_emision"):
            if zone in values:
                fields.append({"zona": zone, "oficial": "", "postulante": values[zone]["texto"], "estado": NOT_APPLICABLE,
                               "motivo": "dato propio de esta emisión", "conf": round(values[zone].get("conf", 1.0), 3)})

    run_doc = N.national_id(values.get("run", values.get("rut", {})).get("texto", "")) if not is_ocr else \
        N.national_id_ocr(values.get("run", values.get("rut", {})).get("texto", ""))
    conf_id = values.get("run", values.get("rut", {})).get("conf", 1.0)
    if declared.get("rut") and run_doc and N.national_id_valid(run_doc) and run_doc != N.national_id(declared["rut"]):
        signals.append(K.signal("identidad", "fraude" if (not is_ocr or conf_id >= C.OCR_HIGH_CONF) else "observacion",
                              f"el documento está a nombre de otro RUT ({run_doc}); "
                              f"el postulante declaró {declared['rut']}"))

    outdated = False
    if doc_type == "hoja_vida" and issue:
        n_days = (today - issue).days
        if n_days > RECORD_VALIDITY_DAYS:
            outdated = True
            signals.append(K.signal("vigencia", "info",
                                  f"Hoja de Vida emitida hace {n_days} días (máximo {RECORD_VALIDITY_DAYS})"))
    if doc.get("copia"):
        signals.append(K.signal("copia", "info", "versión 'COPIA, no válida como certificado': se requiere el original"))
    if doc.get("pagina_faltante"):
        signals.append(K.signal("paginas", "observacion", doc["pagina_faltante"]))
    v, reasons = verdict(fields, signals, illegible=doc.get("ilegible", False), copy_doc=doc.get("copia", False),
                           outdated=outdated)
    if doc.get("ilegible"):
        reasons = [doc.get("motivo_ilegible", "no se pudo leer el documento")] + reasons
    return {"veredicto": v, "campos": fields, "senales": signals, "motivos": reasons, "emision": str(issue or ""),
            "prioridad_revision": review_priority(fields, signals) if v == FLAGGED else ""}
