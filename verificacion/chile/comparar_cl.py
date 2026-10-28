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

from .. import config as C
from .. import comparacion as K
from . import hvid_cuerpo as HV
from . import normalizar_cl as N

IGUAL, TOLERADO, DIFERENTE, DUDOSO = K.IGUAL, K.TOLERADO, K.DIFERENTE, K.DUDOSO
DESACT_C, NO_APLICA = K.DESACTUALIZADO_C, K.NO_APLICA
AUTENTICO, OBSERVADO, FALSIFICADO, DESACTUALIZADO = K.AUTENTICO, K.OBSERVADO, K.FALSIFICADO, K.DESACTUALIZADO
NO_LEGIBLE, NO_VALIDO, NO_RECONOCIDO = "NO LEGIBLE", "NO VÁLIDO", "NO RECONOCIDO"
SIN_DATO_OFICIAL = "SIN DATO OFICIAL"
"""No se entregó su JSON oficial: no se verifica, solo se extraen sus datos."""

FUENTES_VALIDAS = {
    "hoja_vida": {"Courier", "Helvetica", "Helvetica-Bold"},
    "ameco_rev03": {"ArialMT", "Arial-BoldMT", "Arial", "Arial-Bold", "Arial Bold", "Arial Regular"},
    "ameco_rev02": {"CIDFont+F1", "CIDFont+F2", "CIDFont+F3", "CIDFont+F4", "Verdana", "Verdana-Bold",
                    "Verdana-Italic", "TimesNewRoman", "TimesNewRomanPSMT"},
}
"""Fuentes de los documentos genuinos; las de un documento rehecho en Word o Canva son otras."""


def _senal(tipo, severidad, detalle):
    """Señal técnica: severidad "fraude", "observacion" o "info"."""
    return {"tipo": tipo, "severidad": severidad, "detalle": detalle}


def valor_json(js, clave):
    """'Exámenes/Señaléticas', 'Evaluadores[1]', 'Nombre' -> valor del JSON (las claves pueden tener puntos)."""
    v = js
    for parte in clave.split("/"):
        m = re.fullmatch(r"(.+)\[(\d+)\]", parte)
        if m:
            v = (v or {}).get(m.group(1), [])
            v = v[int(m.group(2))] if len(v) > int(m.group(2)) else None
        else:
            v = (v or {}).get(parte) if isinstance(v, dict) else None
    return "" if v is None else str(v)


VALIDOS = {"estado_evaluacion": {"APROBADO", "REPROBADO"}, "evaluacion_practica": {"APROBADO", "REPROBADO"},
           "nota_aprobacion_final": {"APROBADO", "REPROBADO"}, "licencia_ley": {"18290"}}
"""Valores posibles de los campos cerrados: leer otro valor es un error de lectura, no una falsificación."""
VALOR_DE_OTRO_CAMPO = ("APROBAD", "REPROBAD")
"""Raíz de los valores de los campos cerrados. Si aparece dentro de un nombre, un domicilio o un código, el OCR leyó
algo ajeno al campo (una firma, un timbre o la interfaz de una captura): es una lectura contaminada, no un dato
cambiado."""


def _parecido_ocr(a, b, max_dist=2):
    """Diferencia del tamaño de un error de lectura (1-2 caracteres): en OCR no basta para acusar."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b) <= max_dist


def comparar_valor(tipo, oficial, leido, es_ocr, conf, zona=""):
    """Estado y motivo de un campo según su tipo.

    Regla para documentos leídos por OCR: quien falsifica reemplaza un valor por OTRO VALOR VÁLIDO (otro nombre,
    un RUT con dígito verificador correcto, APROBADO en vez de REPROBADO, una clase que existe, una fecha válida).
    Un error de lectura produce algo inválido o casi igual: eso es DUDOSO (revisión humana), no FALSIFICADO."""
    estado, motivo = _comparar_valor(tipo, oficial, leido, es_ocr, conf, zona)
    if estado != DIFERENTE or not es_ocr:
        return estado, motivo
    le = N.limpio(leido)
    if zona == "tipo_operador":
        le = N.quitar_prefijo(le, "Operador Equipo,")
    if tipo == "rut":
        r = N.rut_ocr(le)
        if not N.rut_valido(r):
            return DUDOSO, "RUT leído con dígito verificador inválido (error de lectura)"
    elif tipo == "clases":
        validas = N.clases(le)
        basura = [t for t in re.split(r"[\s\-/]+", N.sin_tildes(le).upper()) if t and t not in validas]
        if basura:
            return DUDOSO, f"clases con códigos inexistentes ({', '.join(basura[:3])}): error de lectura"
    elif zona in VALIDOS:
        if N.clave_texto(le) not in VALIDOS[zona]:
            return DUDOSO, f"'{le}' no es un valor posible del campo (error de lectura)"
    elif tipo in ("identidad", "codigo"):
        if _parecido_ocr(N.clave_texto(oficial), N.clave_texto(le)):
            return DUDOSO, "difiere en 1-2 caracteres: posible error de lectura"
    elif tipo in ("descriptivo", "categorico"):
        if _parecido_ocr(N.clave_texto(oficial), N.clave_texto(le), 2 if len(oficial) < 30 else 3):
            return DUDOSO, "difiere en pocos caracteres: posible error de lectura"
    if tipo in ("identidad", "codigo", "descriptivo", "categorico"):
        ko, kl = N.clave_texto(oficial), N.clave_texto(le)
        if len(kl) >= 3 and (ko.startswith(kl) or ko.endswith(kl)):
            return DUDOSO, "lectura incompleta (texto cortado): el resto del valor no se leyó"
        if len(ko) >= 3 and ko in kl:
            return DUDOSO, "el valor oficial se leyó junto a texto vecino (lectura corrida)"
        if zona not in VALIDOS:
            ajenas = [v for v in VALOR_DE_OTRO_CAMPO if v in kl.replace(" ", "") and v not in ko.replace(" ", "")]
            if ajenas:
                return DUDOSO, f"la lectura incluye texto de otro campo ({ajenas[0]}...): lectura contaminada"
    return estado, motivo


def _comparar_valor(tipo, oficial, leido, es_ocr, conf, zona=""):
    """Comparación base por tipo (fecha, rut, numero, clases, codigo; el resto como texto), antes de las reglas de
    error de lectura de `comparar_valor`. Con OCR y confianza baja, una diferencia queda DUDOSA.

    En clases, un código con forma de clase pero inexistente ("A44") puede ser una clase agregada mal leída: se
    revisa en vez de darse por igual.
    """
    of, le = N.limpio(oficial), N.limpio(leido)
    if zona == "tipo_operador":
        le = N.quitar_prefijo(le, "Operador Equipo,")
    if not le:
        if es_ocr:
            return DUDOSO, "el OCR no leyó texto en la zona"
        return (IGUAL, "") if not of else (DIFERENTE, "el campo está vacío en el documento")
    if of == le:
        return IGUAL, ""
    baja = es_ocr and conf < C.CONF_OCR_ALTA

    def diferente(motivo=""):
        """DIFERENTE, o DUDOSO si la lectura tiene confianza baja."""
        return (DUDOSO, f"lectura con confianza baja ({conf:.2f}){': ' + motivo if motivo else ''}") if baja \
            else (DIFERENTE, motivo)

    if tipo == "fecha":
        fo, fl = N.fecha(of), N.fecha(le)
        if fl is None:
            return (DUDOSO, "fecha ilegible") if es_ocr else (DIFERENTE, "no es una fecha válida")
        return (TOLERADO, "misma fecha en otro formato") if fo == fl else diferente(f"{fl} vs oficial {fo}")
    if tipo == "rut":
        ro, rl = N.rut(of), (N.rut_ocr(le) if es_ocr else N.rut(le))
        if rl is None:
            return (DUDOSO, "RUT ilegible") if es_ocr else (DIFERENTE, "no es un RUT")
        return (TOLERADO, "mismo RUT con otro formato") if ro == rl else diferente()
    if tipo == "numero":
        no, nl = N.numero(of), N.numero(le.translate(N.CONF_NUM) if es_ocr else le)
        if no is None:
            return diferente("el dato oficial no es numérico")
        if es_ocr and not any(ch.isdigit() for ch in le):
            return DUDOSO, "número ilegible (el OCR no leyó ningún dígito)"
        if nl is None:
            return (DUDOSO, "número ilegible") if es_ocr else (DIFERENTE, "")
        return (TOLERADO, "mismo valor") if abs(no - nl) < 0.05 else diferente()
    if tipo == "clases":
        if N.clases(of) != N.clases(le):
            return diferente()
        raros = [t for t in re.split(r"[^A-Z0-9]+", N.sin_tildes(le).upper()) if re.fullmatch(r"[A-F]\d+", t)
                 and not re.fullmatch(r"A[1-5]", t)]
        if es_ocr and raros:
            return DUDOSO, f"código de clase inexistente leído ({', '.join(raros[:3])}): revisar si hay una clase agregada"
        return TOLERADO, "mismas clases"
    if tipo == "codigo":
        if N.codigo(of) == N.codigo(le) or (es_ocr and N.codigo_ocr(of) == N.codigo_ocr(le)):
            return TOLERADO, "solo difieren caracteres confundibles"
        return diferente()
    ko, kl = N.clave_texto(of), N.clave_texto(le)
    if ko == kl:
        return TOLERADO, "solo difieren tildes, mayúsculas o puntuación"
    if K.cambia_negacion(of, le):
        return diferente("cambia una negación")
    if es_ocr and K.normalizar(of, "descriptivo", True) == K.normalizar(le, "descriptivo", True):
        return TOLERADO, "solo difieren caracteres confundibles del OCR"
    if es_ocr and tipo == "descriptivo" and len(ko) >= 14 and fuzz.ratio(ko, kl) >= 92:
        return TOLERADO, "texto casi idéntico (ruido de OCR)"
    return diferente()


def _comparar_filas(doc_filas, js, emision, es_ocr, tiene_p2=True):
    """Licencias, anotaciones y antecedentes: conjuntos de filas. Un registro oficial anterior a la emisión
    que no figura en el documento = ocultado; uno posterior = documento desactualizado.

    - Licencias: clave (registro, clase); se comparan fecha y municipalidad. Con OCR, si no se detectó la clase, es la
      misma fila cuando coinciden registro y fecha. Una fila "nueva" solo acusa si está bien formada (clase
      existente y fecha válida).
    - Anotaciones: se emparejan por fecha de anotación o por tipo y detalle. Sin la página 2, una anotación cortada
      al final de la página 1 (su fecha quedó en la página 2) se compara solo en lo que se ve. Un encabezado suelto
      mal leído (sin texto ni fecha) no es una anotación.
    - Sin la página 2, las licencias, anotaciones y antecedentes que faltan pueden estar en ella: NO_APLICA.
    - Antecedentes: si el oficial tiene anotaciones posteriores a la emisión, un "SIN ANTECEDENTES" impreso es un
      estado que cambió después (DESACTUALIZADO), no una falsificación.
    """
    of = HV.desde_json(js)
    campos = []

    def estado_falta(fecha_reg, que):
        """Estado de un registro oficial que no figura en el documento."""
        if fecha_reg and emision and fecha_reg > emision:
            return DESACT_C, f"{que} posterior a la emisión del documento"
        if es_ocr and doc_filas["conf_min"] < C.CONF_OCR_ALTA:
            return DUDOSO, f"{que} oficial no encontrado en el documento (lectura con confianza baja)"
        return DIFERENTE, f"{que} oficial que no figura en el documento"

    usadas = set()
    for i, lo in enumerate(of["licencias"]):
        cand = [j for j, ld in enumerate(doc_filas["licencias"]) if j not in usadas
                and ld["Registro"] == lo["Registro"] and N.codigo_ocr(ld["Clase"]) == N.codigo_ocr(lo["Clase"])]
        if not cand and es_ocr:
            cand = [j for j, ld in enumerate(doc_filas["licencias"]) if j not in usadas and not ld["Clase"]
                    and ld["Registro"] == lo["Registro"] and ld["fecha"] == lo["fecha"]]
        nombre = f"licencia {lo['Registro'].lower()} clase {lo['Clase']}"
        if not cand:
            e, m = estado_falta(lo["fecha"], "licencia")
            if e == DIFERENTE and not tiene_p2:
                e, m = NO_APLICA, "puede figurar en la página 2, que no fue entregada"
            campos.append({"zona": nombre, "oficial": f"{lo['fecha']} {lo['Municipalidad']}", "postulante": "",
                           "estado": e, "motivo": m, "conf": 1.0})
            continue
        j = min(cand, key=lambda j: abs(((doc_filas["licencias"][j]["fecha"] or date.min) - (lo["fecha"] or date.min)).days))
        usadas.add(j)
        ld = doc_filas["licencias"][j]
        if ld["fecha"] == lo["fecha"]:
            e, m = IGUAL, ""
        elif ld["fecha"] is None:
            e, m = (DUDOSO, "fecha ilegible") if es_ocr else (DIFERENTE, "fecha inválida")
        elif emision and lo["fecha"] and lo["fecha"] > emision and ld["fecha"] < lo["fecha"]:
            e, m = DESACT_C, "la licencia se renovó después de la emisión del documento"
        else:
            e, m = comparar_valor("fecha", str(lo["fecha"]), ld["fecha_txt"], es_ocr, ld["conf"])
        campos.append({"zona": f"{nombre} - fecha", "oficial": str(lo["fecha"]), "postulante": ld["fecha_txt"],
                       "estado": e, "motivo": m, "conf": round(ld["conf"], 3)})
        e2, m2 = comparar_valor("categorico", lo["Municipalidad"], ld["Municipalidad"], es_ocr, ld["conf"])
        if e2 == DIFERENTE and emision and lo["fecha"] and lo["fecha"] > emision:
            e2, m2 = DESACT_C, "cambió después de la emisión"
        campos.append({"zona": f"{nombre} - municipalidad", "oficial": lo["Municipalidad"], "postulante": ld["Municipalidad"],
                       "estado": e2, "motivo": m2, "conf": round(ld["conf"], 3)})
    for j, ld in enumerate(doc_filas["licencias"]):
        if j not in usadas:
            bien = bool(N.clases(ld["Clase"])) and ld["fecha"] is not None
            e = DUDOSO if es_ocr and (ld["conf"] < C.CONF_OCR_ALTA or not bien) else DIFERENTE
            campos.append({"zona": f"licencia {ld['Registro'].lower()} clase {ld['Clase']}", "oficial": "",
                           "postulante": f"{ld['fecha_txt']} {ld['Municipalidad']}", "estado": e,
                           "motivo": "licencia que no existe en el registro oficial", "conf": round(ld["conf"], 3)})

    usadas = set()
    for ao in of["anotaciones"]:
        cand = [j for j, ad in enumerate(doc_filas["anotaciones"]) if j not in usadas and
                (ad["fecha"] == ao["fecha"] or fuzz.ratio(ad["Tipo"], ao["Tipo"]) >= 80 and fuzz.ratio(ad["Detalle"], ao["Detalle"]) >= 80)]
        if not cand and not tiene_p2:
            cand = [j for j, ad in enumerate(doc_filas["anotaciones"]) if j not in usadas and ad["fecha"] is None
                    and fuzz.ratio(ad["Tipo"], ao["Tipo"]) >= 80 and len(ad["Detalle"]) >= 10
                    and fuzz.ratio(ad["Detalle"], ao["Detalle"][:len(ad["Detalle"])]) >= 88]
        nombre = f"anotación {ao['Tipo'][:40]} ({ao['fecha']})"
        if not cand:
            e, m = estado_falta(ao["fecha"], "anotación")
            if e == DIFERENTE and not tiene_p2:
                e, m = NO_APLICA, "puede figurar en la página 2, que no fue entregada"
            campos.append({"zona": nombre, "oficial": ao["Detalle"][:120], "postulante": "", "estado": e, "motivo": m,
                           "conf": 1.0})
            continue
        j = cand[0]
        usadas.add(j)
        ad = doc_filas["anotaciones"][j]
        sim = fuzz.ratio(ad["Detalle"], ao["Detalle"])
        corte = not tiene_p2 and 10 <= len(ad["Detalle"]) < len(ao["Detalle"]) and \
            fuzz.ratio(ad["Detalle"], ao["Detalle"][:len(ad["Detalle"])]) >= (88 if es_ocr else 97)
        if sim >= 97 or (es_ocr and sim >= 88):
            e, m = (IGUAL if sim == 100 else TOLERADO), ("" if sim == 100 else "texto casi idéntico")
        elif corte:
            e, m = TOLERADO, "el texto sigue en la página 2 (no entregada): coincide lo visible"
        elif es_ocr and ad["conf"] < C.CONF_OCR_ALTA:
            e, m = DUDOSO, f"texto de la anotación leído con confianza baja ({ad['conf']:.2f})"
        else:
            e, m = DIFERENTE, "el texto de la anotación no coincide"
        campos.append({"zona": nombre, "oficial": ao["Detalle"][:120], "postulante": ad["Detalle"][:120], "estado": e,
                       "motivo": m, "conf": round(ad["conf"], 3)})
    for j, ad in enumerate(doc_filas["anotaciones"]):
        if j not in usadas:
            if not ad["Detalle"] and ad["fecha"] is None:
                continue
            bien = ad["fecha"] is not None and len(ad["Detalle"]) >= 20
            e = DUDOSO if es_ocr and (ad["conf"] < C.CONF_OCR_ALTA or not bien) else DIFERENTE
            campos.append({"zona": f"anotación {ad['Tipo'][:40]}", "oficial": "", "postulante": ad["Detalle"][:120],
                           "estado": e, "motivo": "anotación que no existe en el registro oficial", "conf": round(ad["conf"], 3)})

    for reg in ("PRN", "RNC"):
        vo, vd = of["antecedentes"].get(reg, ""), doc_filas["antecedentes"].get(reg, "")
        if not vd and not tiene_p2:
            e, m = NO_APLICA, "la página 2 (antecedentes) no fue entregada"
        elif not vd:
            e, m = (DUDOSO, "no se leyó") if es_ocr else (DIFERENTE, "no figura en el documento")
        elif vo == vd:
            e, m = IGUAL, ""
        elif vd == "SIN ANTECEDENTES" and any(a["fecha"] and emision and a["fecha"] > emision for a in of["anotaciones"]):
            e, m = DESACT_C, "el estado de antecedentes cambió después de la emisión del documento"
        else:
            e, m = DIFERENTE, "cambia el estado de antecedentes"
        campos.append({"zona": f"antecedentes {reg}", "oficial": vo, "postulante": vd, "estado": e, "motivo": m, "conf": 1.0})
    return campos


def senales_codigos(doc, valores):
    """Timbre electrónico y código de barras vs lo impreso (un dato editado no cambia el timbre)."""
    s = []
    cod = doc.get("codigos") or {}
    folio = N.codigo(valores.get("folio", {}).get("texto", ""))
    for texto in cod.get("timbre", []):
        from .lectura_cl import parsear_timbre
        t = parsear_timbre(texto)
        if not t:
            continue
        dif = []
        if t.get("FOLIO") and folio and t["FOLIO"] != folio and not doc.get("es_ocr"):
            dif.append(f"folio {t['FOLIO']} vs impreso {folio}")
        run = N.rut(valores.get("run", {}).get("texto", ""))
        if t.get("RUN") and run and N.rut(t["RUN"]) != run:
            if not doc.get("es_ocr") or valores.get("run", {}).get("conf", 1) >= C.CONF_OCR_ALTA:
                dif.append(f"RUN {t['RUN']} vs impreso {run}")
        nom = N.clave_texto(valores.get("nombre", {}).get("texto", ""))
        if t.get("NOM") and nom and fuzz.ratio(N.clave_texto(t["NOM"]), nom) < (95 if not doc.get("es_ocr") else 80):
            dif.append(f"nombre '{t['NOM']}' vs impreso '{nom}'")
        fe = N.fecha(valores.get("fecha_emision", {}).get("texto", ""))
        if t.get("FE") and fe and t["FE"][:10] != fe.isoformat():
            if not doc.get("es_ocr") or valores.get("fecha_emision", {}).get("conf", 1) >= C.CONF_OCR_ALTA:
                dif.append(f"fecha de emisión {t['FE'][:10]} vs impresa {fe}")
        if dif:
            s.append(_senal("timbre", "fraude", "el timbre electrónico no coincide con lo impreso: " + "; ".join(dif)))
        else:
            s.append(_senal("timbre", "info", "timbre electrónico legible y coherente con lo impreso"))
        break
    for b in cod.get("barras", []):
        if folio and N.codigo(b) != folio and re.fullmatch(r"\d{12}", N.codigo(b)) and not doc.get("es_ocr"):
            s.append(_senal("codigo_barras", "fraude", f"el código de barras ({b}) no coincide con el folio impreso ({folio})"))
    return s


def _iou(a, b):
    """Intersección sobre unión de dos rectángulos."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def senales_nativo(doc, plantilla):
    """Señales de un PDF nativo: objetos superpuestos, anotaciones, metadatos, revisiones y tipografías ajenas.

    - Las imágenes propias del documento genuino (logos, sellos, timbre, firmas, QR) no son superposiciones.
    - Un resaltado o subrayado es transparente: no oculta ni cambia datos (lo agrega quien revisa el PDF), así que
      solo se observa.
    - La versión COPIA tiene marca de agua y otra disposición: sus imágenes y gráficos no son los de la plantilla.
      Se observan (la copia se rechaza igual como NO VÁLIDO y un dato distinto sigue acusando).
    """
    objetos = dict(doc.get("objetos") or {})
    objetos["imagenes"] = [r for r in objetos.get("imagenes", [])
                           if not any(_iou(r, e) > 0.6 for e in plantilla.get("imagenes_esperadas", []))]
    s = K.senales_objetos(dict(doc, objetos=objetos), plantilla)
    for x in s:
        if x["tipo"] == "anotacion" and re.search(r"'(Highlight|Underline|Squiggly)'", x["detalle"]):
            x.update(severidad="observacion", detalle=x["detalle"].replace("agregada al PDF", "agregada al PDF después de "
                                                                            "su emisión (marca transparente, no oculta datos)"))
        elif doc.get("copia") and x["tipo"] == "objeto_superpuesto":
            x.update(severidad="observacion", detalle=x["detalle"] + " (formato COPIA: marca de agua propia)")
    meta = doc.get("meta") or {}
    if meta.get("producer"):
        s.append(_senal("metadatos", "info", f"PDF producido con '{meta.get('producer')}'"))
    if (doc.get("revisiones_pdf") or 1) > 1:
        s.append(_senal("metadatos", "info", f"el PDF tiene {doc['revisiones_pdf']} versiones guardadas (edición incremental)"))
    validas = FUENTES_VALIDAS["hoja_vida" if plantilla["tipo_documento"] == "hoja_vida" else plantilla["id"]]
    ajenas = sorted({K.fuente_base(p["font"]) for p in doc.get("palabras_valor", []) if p.get("font")}
                    - {K.fuente_base(f) for f in validas} - validas)
    if ajenas:
        s.append(_senal("fuentes", "observacion", f"tipografía que no usa el emisor en los datos: {', '.join(ajenas[:4])}"))
    return s


def veredicto(campos, senales, ilegible=False, copia=False, desactualizado=False):
    """(dictamen, motivos sin repetir) con la prioridad del módulo. Una COPIA no sirve como certificado: se rechaza
    aunque tenga lecturas u observaciones por revisar, pero un dato distinto la marca como FALSIFICADA."""
    dif = [c for c in campos if c["estado"] == DIFERENTE]
    dud = [c for c in campos if c["estado"] == DUDOSO]
    des = [c for c in campos if c["estado"] == DESACT_C]
    if ilegible:
        v = NO_LEGIBLE
    elif dif or any(x["severidad"] == "fraude" for x in senales):
        v = FALSIFICADO
    elif copia:
        v = NO_VALIDO
    elif dud or any(x["severidad"] == "observacion" for x in senales):
        v = OBSERVADO
    elif des or desactualizado:
        v = DESACTUALIZADO
    else:
        v = AUTENTICO
    motivos = [f"{c['zona']}: oficial '{c['oficial']}' vs documento '{c['postulante']}'"
               + (f" ({c['motivo']})" if c["motivo"] else "") for c in dif]
    motivos += [x["detalle"] for x in senales if x["severidad"] in ("fraude", "observacion")]
    motivos += [f"{c['zona']}: {c['motivo']}" for c in dud]
    motivos += [f"{c['zona']}: {c['motivo']}" for c in des[:5]]
    return v, list(dict.fromkeys(motivos))


def comparar_documento(doc, js, plantillas_doc, declarado, hoy):
    """doc: lectura agrupada de un documento (valores por zona, filas de la Hoja de Vida, señales técnicas).
    js: dato oficial (o None). Devuelve campos, señales y dictamen.

    - Lo que no se pudo leer no se compara: acusaría por datos "faltantes" que solo son ilegibles.
    - Una diferencia en una zona que contiene texto impreso del formato vecino es una lectura desalineada (DUDOSO).
    - Identidad: el documento debe estar a nombre del RUT declarado. Solo un RUT completo y válido (dígito
      verificador) puede ser "de otra persona"; una lectura parcial no.
    """
    es_ocr = doc["es_ocr"]
    valores = doc["valores"]
    senales = list(doc.get("senales", []))
    campos = []
    tipo_doc = doc["tipo_documento"]
    emision = N.fecha(valores.get("fecha_emision", {}).get("texto", "")) if tipo_doc == "hoja_vida" else None

    if doc.get("ilegible"):
        pass
    elif js is None:
        senales.append(_senal("fuente_oficial", "observacion", "no hay dato oficial (JSON) para comparar este documento"))
    else:
        for zona, info in plantillas_doc.items():
            if not info["clave"]:
                continue
            v = valores.get(zona)
            if v is None and zona.startswith(("pie_", "evaluador_de")):
                continue
            leido, conf = (v or {}).get("texto", ""), (v or {}).get("conf", 0.0 if es_ocr else 1.0)
            oficial = valor_json(js, info["clave"])
            e, m = comparar_valor(info["tipo"], oficial, leido, es_ocr, conf, zona)
            if e == DIFERENTE and es_ocr:
                propio = N.quitar_prefijo(N.limpio(leido), "Operador Equipo,") if zona == "tipo_operador" else leido
                ajenas = (set(N.clave_texto(propio).split()) & doc.get("formato_ajeno", {}).get(zona, set())) \
                    - set(N.clave_texto(oficial).split())
                if ajenas:
                    e, m = DUDOSO, (f"la zona contiene texto impreso del formato ({', '.join(sorted(ajenas)[:3])}): "
                                    "lectura desalineada")
            campos.append({"zona": zona, "oficial": oficial, "postulante": leido, "estado": e, "motivo": m,
                           "conf": round(conf, 3)})
        if tipo_doc == "hoja_vida" and doc.get("filas") is not None:
            campos += _comparar_filas(doc["filas"], js, emision, es_ocr,
                                      tiene_p2="página 2" not in (doc.get("pagina_faltante") or ""))
        for zona in ("folio", "codigo_verificacion", "fecha_emision"):
            if zona in valores:
                campos.append({"zona": zona, "oficial": "", "postulante": valores[zona]["texto"], "estado": NO_APLICA,
                               "motivo": "dato propio de esta emisión", "conf": round(valores[zona].get("conf", 1.0), 3)})

    run_doc = N.rut(valores.get("run", valores.get("rut", {})).get("texto", "")) if not es_ocr else \
        N.rut_ocr(valores.get("run", valores.get("rut", {})).get("texto", ""))
    conf_id = valores.get("run", valores.get("rut", {})).get("conf", 1.0)
    if declarado.get("rut") and run_doc and N.rut_valido(run_doc) and run_doc != N.rut(declarado["rut"]):
        senales.append(_senal("identidad", "fraude" if (not es_ocr or conf_id >= C.CONF_OCR_ALTA) else "observacion",
                              f"el documento está a nombre de otro RUT ({run_doc}); el postulante declaró {declarado['rut']}"))

    desactualizado = False
    if tipo_doc == "hoja_vida" and emision:
        dias = (hoy - emision).days
        if dias > C.HVID_DIAS_VIGENCIA:
            desactualizado = True
            senales.append(_senal("vigencia", "info", f"Hoja de Vida emitida hace {dias} días (máximo {C.HVID_DIAS_VIGENCIA})"))
    if doc.get("copia"):
        senales.append(_senal("copia", "info", "versión 'COPIA, no válida como certificado': se requiere el original"))
    if doc.get("pagina_faltante"):
        senales.append(_senal("paginas", "observacion", doc["pagina_faltante"]))
    v, motivos = veredicto(campos, senales, ilegible=doc.get("ilegible", False), copia=doc.get("copia", False),
                           desactualizado=desactualizado)
    if doc.get("ilegible"):
        motivos = [doc.get("motivo_ilegible", "no se pudo leer el documento")] + motivos
    return {"veredicto": v, "campos": campos, "senales": senales, "motivos": motivos, "emision": str(emision or "")}
