"""Conocimiento del perfil sobre sus documentos que usa el núcleo (lectura, agrupación, dato oficial e identidad).

Documentos del perfil:
  hoja_vida (páginas hoja_vida_p1 y hoja_vida_p2)  registro de conductor de 2 páginas, con timbre electrónico
  ameco_rev02, ameco_rev03                          certificado de competencia de operador (dos versiones)
"""

import json
from collections import defaultdict

from rapidfuzz import fuzz

from . import normalize as N
from . import record_body as RB
from . import rules as K
from .templates import DOCUMENT_TYPE

RECORD = "hoja_vida"
"""Tipo del registro de conductor (2 páginas, con timbre)."""
CERTIFICATE = "ameco"
"""Tipo del certificado de competencia."""
RECORD_PAGES = ("hoja_vida_p1", "hoja_vida_p2")
"""Páginas del registro, en orden."""
RECORD_JSON = "hoja_vida_conductor.json"
"""Nombre del JSON oficial del registro en la carpeta de datos oficiales."""
CERTIFICATE_JSON_GLOB = "ameco_*.json"
"""Patrón de los JSON oficiales de los certificados."""

SEALED_TYPES = {RECORD}
"""Tipos de documento que llevan timbre electrónico (PDF417): se insiste en leerlo."""
SIBLING_PAGES = {RECORD: RECORD_PAGES}
"""Páginas de un mismo documento que comparten la cabecera: al alinear, entre ellas se decide por las etiquetas del
cuerpo."""
ONE_PER_IMAGE = {CERTIFICATE}
"""Tipos que no se repiten en una misma foto: tras encontrar uno, el segundo documento se busca entre las demás
plantillas."""
RECORD_HEADER_END_Y = 160
"""Altura (pt) donde termina la cabecera del registro: debajo de "No válido para obtener ni renovar LICENCIA
CONDUCIR" empieza el contenido."""

UNRECOGNIZED_HINT = "la Hoja de Vida ni a un certificado AMECO"
"""Complemento del motivo de una imagen nítida que no es ninguno de los documentos del perfil."""
EXPECTED_DOCUMENTS = [(RECORD, "Hoja de Vida del Conductor"), ("ameco_rev02", "Certificado AMECO (Rev.02)"),
                      ("ameco_rev03", "Certificado AMECO (Rev.03)")]
"""Documentos que el perfil sabe verificar: si no se entrega uno, figura como DOCUMENTO FALTANTE."""
REGISTRY_COLUMNS = {"rut": ("rut declarado", "rut"), "nombre": ("nombre completo", "nombre"),
                    "puesto": ("cargo postulado", "puesto", "cargo")}
"""Encabezados aceptados de cada columna del registro de postulantes (candidatos.xlsx)."""
ID_LABEL = "RUT"
"""Nombre del número de identidad en los mensajes."""
ID_EXAMPLE = "12.345.678-5"
"""Ejemplo de número de identidad para los mensajes."""

RECORD_ISSUE_FIELDS = ("folio", "codigo_verificacion", "fecha_emision")
"""Datos de la emisión que se imprimen en las dos páginas del registro: identifican a qué emisión pertenece cada
página."""


def is_copy_native(page_text):
    """True si el texto de una página PDF es el de la versión "COPIA, no válida como certificado"."""
    t = page_text.upper()
    return "COPIA NO VALIDO COMO CERTIFICADO" in t or "COPIA NO VÁLIDA COMO CERTIFICADO" in t


def is_copy_ocr(text):
    """True si el texto leído por OCR (en mayúsculas) tiene la marca de agua o el pie de la versión COPIA."""
    return "COPIA" in text and ("VALID" in text or "CERTIFICADO" in text.split("COPIA", 1)[1][:80])


def copy_template(tpl):
    """Plantilla para ubicar la versión COPIA de un PDF nativo. La COPIA del registro agrega la línea "COPIA DE" bajo
    la cabecera: el contenido baja ~12 pt pero la cabecera (logo, marco) no, así que la geometría se calcula solo con
    las etiquetas del contenido."""
    if tpl["tipo_documento"] != RECORD:
        return tpl
    return dict(tpl, palabras_fijas=[w for w in tpl["palabras_fijas"] if w[1] > RECORD_HEADER_END_Y])


def body_lines(words):
    """Líneas de la sección de filas variables (zona de tipo "cuerpo")."""
    return RB.lines(words)


def _issue_matches(p1, p2):
    """Cuántos datos de la emisión (folio, código de verificación, fecha de emisión) leídos en ambas páginas
    coinciden."""
    v1, v2 = p1.get("valores") or {}, p2.get("valores") or {}
    return sum(1 for z in RECORD_ISSUE_FIELDS
               if N.code_ocr(v1.get(z, {}).get("texto", "")) and
               N.code_ocr(v1.get(z, {}).get("texto", "")) == N.code_ocr(v2.get(z, {}).get("texto", "")))


def group_pages(pages):
    """Páginas -> documentos. El registro junta su página 1 y 2: las del mismo archivo, en orden; las que vienen en
    archivos separados, por los datos de la emisión que ambas páginas repiten (folio, código, fecha de emisión) y, a
    igualdad, en orden de archivo. Así, dos registros de distintas fechas entregados en fotos sueltas no se
    mezclan."""
    docs, rec1, rec2 = [], [], []
    by_file = defaultdict(list)
    for p in pages:
        kind = DOCUMENT_TYPE.get(p["plantilla"]) if p.get("plantilla") else None
        if kind == RECORD and not p.get("ilegible"):
            by_file[p["archivo"]].append(p)
        else:
            docs.append([p])
    for ps in by_file.values():
        p1 = [p for p in ps if p["plantilla"] == RECORD_PAGES[0]]
        p2 = [p for p in ps if p["plantilla"] == RECORD_PAGES[1]]
        while p1 and p2:
            docs.append([p1.pop(0), p2.pop(0)])
        rec1 += p1
        rec2 += p2
    while rec1 and rec2:
        a = rec1.pop(0)
        b = max(rec2, key=lambda q: (_issue_matches(a, q), -rec2.index(q)))
        rec2.remove(b)
        docs.append([a, b])
    docs += [[p] for p in rec1 + rec2]
    return docs


def page_suffix(template_id):
    """Sufijo de las zonas de una página en el documento unido: la página 2 del registro lleva K.PAGE2_SUFFIX."""
    return K.PAGE2_SUFFIX if template_id == RECORD_PAGES[1] else ""


def order_pages(kind, ps, doc):
    """Páginas de un documento en orden; anota en `doc` la página faltante del registro."""
    if kind != RECORD:
        return ps
    ps = sorted(ps, key=lambda p: p["plantilla"])
    if len(ps) == 1:
        missing = 2 if ps[0]["plantilla"] == RECORD_PAGES[0] else 1
        doc["pagina_faltante"] = f"falta la página {missing} de la Hoja de Vida"
    return ps


def finish_document(kind, ps, doc):
    """Completa el documento unido: filas de la sección variable y señales del timbre del registro (con los valores
    de cabecera de la página 1; los de la página 2 solo si falta la 1)."""
    if kind != RECORD:
        return
    doc["filas"] = RB.parse([p.get("lineas_cuerpo", []) for p in ps if not p.get("ilegible")])
    doc["senales"] += K.code_signals(doc, {z: v for z, v in doc["valores"].items() if not z.endswith(K.PAGE2_SUFFIX)})


def row_details(doc):
    """Datos adicionales de la fila de resultado: las filas leídas de la sección variable del registro."""
    if not doc.get("filas"):
        return {}
    fields = ("Registro", "Clase", "fecha_txt", "Municipalidad", "Tipo", "Fecha anotación", "Detalle")
    rows = {k: [{c: str(x.get(c, "")) for c in fields if c in x} for x in doc["filas"][k]]
            for k in ("licencias", "anotaciones")}
    return {"filas_hoja_vida": rows | {"antecedentes": doc["filas"]["antecedentes"]}}


def official_data_of(folder, official_dir):
    """JSON oficiales de una carpeta: el registro y los certificados (por N°, con el nombre de su archivo)."""
    d = official_dir / folder
    rec = d / RECORD_JSON
    out = {RECORD: json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else None, CERTIFICATE: {},
           "archivo_ameco": {}}
    for f in sorted(d.glob(CERTIFICATE_JSON_GLOB)) if d.is_dir() else []:
        js = json.loads(f.read_text(encoding="utf-8"))
        key = N.code(js.get("N°", f.stem))
        out[CERTIFICATE][key] = js
        out["archivo_ameco"][key] = f.name
    return out


def read_official_bytes(jsons):
    """[(nombre, bytes)] -> (oficiales como los usa el flujo, advertencias). El tipo se reconoce por el contenido."""
    of = {RECORD: None, CERTIFICATE: {}, "archivo_ameco": {}, "archivo_hoja_vida": ""}
    warns = []
    for name, data in jsons:
        try:
            js = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            warns.append(f"'{name}' no es un JSON válido: no se usó.")
            continue
        if not isinstance(js, dict):
            warns.append(f"'{name}' no tiene el formato esperado: no se usó.")
        elif "Licencias registradas" in js or "R.U.N." in js:
            if of[RECORD] is not None:
                warns.append(f"Se subió más de un JSON de Hoja de Vida: se usó '{of['archivo_hoja_vida']}' y no '{name}'.")
                continue
            of[RECORD], of["archivo_hoja_vida"] = js, name
        elif "N°" in js or "Exámenes" in js:
            key = N.code(js.get("N°", name))
            of[CERTIFICATE][key] = js
            of["archivo_ameco"][key] = name
        else:
            warns.append(f"'{name}' no es un JSON de Hoja de Vida ni de certificado AMECO: no se usó.")
    return of, warns


def has_official(official_data):
    """True si se entregó al menos un dato oficial."""
    return official_data[RECORD] is not None or bool(official_data[CERTIFICATE])


def _certificate_json(doc, official_data):
    """El JSON oficial del certificado: por N° leído; si no se encuentra, por versión (Rev.02 / Rev.03)."""
    certificates = official_data.get(CERTIFICATE, {})
    read_value = N.code(doc["valores"].get("numero_certificado", {}).get("texto", ""))
    if read_value in certificates:
        return read_value, certificates[read_value]
    if doc["es_ocr"]:
        for k in certificates:
            if N.code_ocr(k) == N.code_ocr(read_value):
                return k, certificates[k]
    version = "Rev.02" if doc["plantilla"] == "ameco_rev02" else "Rev.03"
    cand = [k for k, js in certificates.items() if version in js.get("Formato", "")]
    return (cand[0], certificates[cand[0]]) if len(cand) == 1 else (None, None)


def official_for(doc, official_data):
    """(clave, JSON oficial, nombre del archivo oficial) del documento; JSON None si no se entregó."""
    if doc["tipo_documento"] == RECORD:
        key, js = RECORD_JSON, official_data[RECORD]
        name = official_data.get("archivo_hoja_vida") or key
    else:
        key, js = _certificate_json(doc, official_data)
        name = official_data["archivo_ameco"].get(key, key or "")
    return key, js, name


def identity_crosschecks(identity, official_data):
    """Advertencias si los JSON oficiales parecen de otra persona que la ingresada. Si no se ingresó la fecha de
    nacimiento, se toma la del registro."""
    warns = []
    name = N.text_key(N.strip_accents(identity["nombre_completo"]))
    sources = []
    if official_data[RECORD]:
        js = official_data[RECORD]
        sources.append((f"JSON de Hoja de Vida ({official_data['archivo_hoja_vida']})",
                        js.get("Nombre", ""), js.get("R.U.N.", "")))
        fn = N.date_value(js.get("Fecha nacimiento", ""))
        if fn and identity["fecha_nacimiento"] and fn != identity["fecha_nacimiento"]:
            warns.append(f"La fecha de nacimiento ingresada ({identity['fecha_nacimiento']:%d-%m-%Y}) no coincide con la "
                         f"del JSON de Hoja de Vida ({fn:%d-%m-%Y}).")
        if fn and not identity["fecha_nacimiento"]:
            identity["fecha_nacimiento"], identity["fecha_nacimiento_fuente"] = fn, "JSON de Hoja de Vida"
    for key, js in official_data[CERTIFICATE].items():
        sources.append((f"JSON del certificado {key}", js.get("Nombre", ""), js.get("RUT", "")))
    for what, nm, id_number in sources:
        if nm and fuzz.token_set_ratio(name, N.text_key(N.strip_accents(nm))) < 90:
            warns.append(f"El nombre ingresado no coincide con el del {what}: '{nm}'.")
        if id_number and N.national_id(id_number) and N.national_id(id_number) != N.national_id(identity["rut"]):
            warns.append(f"El {ID_LABEL} ingresado ({identity['rut']}) no coincide con el del {what}: {id_number}. "
                         "La comparación de ese documento no es confiable.")
    return warns


def missing_document_warnings(kind, official_data):
    """Advertencias cuando se subió el dato oficial de un documento esperado (`kind`) pero no el documento."""
    warns = []
    if kind == RECORD and official_data[RECORD]:
        warns.append("Se subió el JSON de la Hoja de Vida, pero no la Hoja de Vida del postulante.")
    version = "Rev.02" if kind == "ameco_rev02" else "Rev.03"
    for key, js in official_data[CERTIFICATE].items():
        if kind.startswith(CERTIFICATE) and version in js.get("Formato", ""):
            warns.append(f"Se subió el JSON del certificado {key} ({version}), pero no el certificado.")
    return warns


def is_profile_document(kind):
    """True si el tipo de una fila de resultado es uno de los documentos del perfil."""
    return kind.startswith((RECORD, CERTIFICATE))


def parse_id(text):
    """Número de identidad normalizado (o None)."""
    return N.national_id(text)


def id_valid(id_number):
    """True si el número de identidad normalizado es válido (dígito verificador)."""
    return N.national_id_valid(id_number)


def format_id(id_number):
    """Número de identidad normalizado ("12345678-5") con puntos de miles ("12.345.678-5")."""
    num, dv = id_number.split("-")
    return f"{int(num):,}".replace(",", ".") + f"-{dv}"


def run_meta():
    """Datos del perfil que se guardan con cada ejecución."""
    return {"dias_vigencia_hoja_vida": K.RECORD_VALIDITY_DAYS}
