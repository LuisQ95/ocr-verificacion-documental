"""Verificación de UN postulante con archivos recibidos en memoria (visor de pruebas).

Nada se escribe en disco: los documentos del postulante y los JSON oficiales llegan como bytes, se leen desde memoria
(PyMuPDF / PIL), se comparan y se descartan. El resultado (dictámenes, motivos, valores leídos) es lo único que queda,
y solo en la memoria del servidor de pruebas mientras dure la sesión.
"""

import io
import json
import re
import threading
import unicodedata
from collections import Counter
from datetime import date

import cv2

from . import fields as K
from . import pipeline as PL
from . import reading as L
from . import report
from .profiles import ACTIVE as PF

D, N, Q = PF.documents, PF.normalize, PF.requirements

MISSING = "DOCUMENTO FALTANTE"
NOT_IDENTIFIED = "NO IDENTIFICADO"
"""El documento no aparece entre los archivos legibles, pero hay archivos ilegibles (puede ser uno de ellos)."""
PARTICLES = {"DE", "DEL", "LA", "LAS", "LOS", "Y", "DA", "DI", "VAN", "VON", "SAN", "SANTA"}
MAX_PAGES_PER_FILE = 20
"""Páginas máximas de un archivo (PDF o TIFF): los documentos pedidos tienen 1 o 2; un archivo de cientos de páginas
ocuparía el equipo por horas."""
MAX_PAGES_PER_VERIFICATION = 60
"""Páginas máximas de una verificación (todos los archivos del postulante)."""
READER_IDLE_S = 5 * 60
"""Tras este tiempo sin verificaciones, el lector OCR se descarga y libera la GPU para otros trabajos."""
_started = False
_readers = {}
"""Lector OCR cargado por dispositivo (True = GPU, False = CPU): se reutiliza entre verificaciones seguidas."""
_reader_lock = threading.Lock()
_timer = None


def _words(text):
    """Palabras del texto, en mayúsculas y normalizadas (NFC)."""
    t = unicodedata.normalize("NFC", text or "").upper()
    return [w for w in re.split(r"\s+", t.strip()) if w]


def _units(words):
    """Nombres o apellidos completos: las partículas se unen a la palabra siguiente ("DE LA CRUZ" = 1 apellido).
    Las partículas sueltas al final se pegan a la última unidad."""
    out, pending = [], []
    for w in words:
        pending.append(w)
        if w not in PARTICLES:
            out.append(" ".join(pending))
            pending = []
    if pending:
        out = out[:-1] + [" ".join(out[-1:] + pending)] if out else [" ".join(pending)]
    return out


def validate_identity(names, surnames, national_id, birth_date="", job_title=""):
    """Devuelve (identidad, errores, advertencias). Errores impiden verificar; advertencias se muestran.

    El cargo es texto libre y se guarda en mayúsculas.
    """
    errors, warns = [], []
    pn, pa = _words(names), _words(surnames)
    for label, ps in (("nombres", pn), ("apellidos", pa)):
        bad = [w for w in ps if not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ'\-]+", w)]
        if bad:
            errors.append(f"Los {label} solo pueden tener letras (revise: {', '.join(bad)}).")
    un, ua = _units(pn), _units(pa)
    if not 1 <= len(un) <= 3:
        errors.append("Ingrese entre 1 y 3 nombres.")
    if not 1 <= len(ua) <= 2:
        errors.append("Ingrese 1 o 2 apellidos.")
    elif len(ua) == 1:
        warns.append("Se ingresó un solo apellido: puede haber otra persona con el mismo nombre y apellido. "
                     f"Para distinguirla se usan el {D.ID_LABEL} y la fecha de nacimiento.")
    r = D.parse_id(national_id or "")
    if not r:
        errors.append(f"Ingrese el {D.ID_LABEL} (por ejemplo {D.ID_EXAMPLE}).")
    elif not D.id_valid(r):
        errors.append(f"El {D.ID_LABEL} {national_id} no es válido (el dígito verificador no corresponde).")
    fn = None
    if (birth_date or "").strip():
        try:
            fn = date.fromisoformat(birth_date.strip())
        except ValueError:
            fn = N.date_value(birth_date)
        if fn is None:
            errors.append("La fecha de nacimiento no es válida.")
        elif fn >= date.today():
            errors.append("La fecha de nacimiento no puede ser futura.")
    job_title = " ".join(_words(job_title))
    if len(job_title) > 80 or (job_title and not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ0-9 .,()/\-]+", job_title)):
        errors.append("El cargo solo puede tener letras, números y signos simples (máximo 80 caracteres).")
    identity = {"nombres": " ".join(pn), "apellidos": " ".join(pa), "nombre_completo": " ".join(pn + pa),
             "un_apellido": len(ua) == 1, "rut": D.format_id(r) if r else "", "fecha_nacimiento": fn,
             "fecha_nacimiento_fuente": "ingresada" if fn else "", "cargo": job_title}
    return identity, errors, warns


def _start():
    """Carga plantillas y puntos ORB una vez por proceso. init_worker limita OpenCV a 1 hilo (pensado para varios
    procesos); aquí hay uno solo, así que se le devuelven todos los hilos."""
    global _started
    if not _started:
        L.init_worker()
        cv2.setNumThreads(-1)
        _started = True


def _release_readers():
    """Descarga los lectores OCR cargados (al cumplirse READER_IDLE_S sin uso)."""
    with _reader_lock:
        while _readers:
            PL.release_reader(_readers.popitem()[1])


def _with_reader(use_gpu, action):
    """Ejecuta accion(lector) con el lector OCR del dispositivo, cargándolo solo si no está en memoria. Al terminar
    se programa su descarga tras READER_IDLE_S sin uso."""
    global _timer
    with _reader_lock:
        if _timer:
            _timer.cancel()
        if use_gpu not in _readers:
            _readers[use_gpu] = PL.create_reader(use_gpu)
        try:
            return action(_readers[use_gpu])
        finally:
            _timer = threading.Timer(READER_IDLE_S, _release_readers)
            _timer.daemon = True
            _timer.start()


def verify_in_memory(identity, files, jsons, today=None, use_gpu=True, progress=None, log=lambda m: None):
    """ident: de validate_identity; archivos / jsons: [(nombre, bytes)]. Devuelve el resultado completo.

    Si hay archivos que no se pudieron leer, un documento que no aparece queda NO IDENTIFICADO en vez de FALTANTE
    (puede ser uno de ellos). Sin ningún dato oficial no se analiza nada: solo se entregan los datos extraídos.
    """
    today = today or date.today()
    progress = progress or (lambda *a: None)
    _start()
    pl, req = L.templates(), Q.load()
    folder = identity["nombre_completo"]
    official_data, warns = D.read_official_bytes(jsons)
    warns += D.identity_crosschecks(identity, official_data)

    pages, rejected, seen_counts = [], [], Counter()
    units = []
    for name, data in files:
        seen_counts[name] += 1
        name = name if seen_counts[name] == 1 else f"{name} ({seen_counts[name]})"
        if not any(name.lower().endswith(e) for e in L.ACCEPTED_EXT):
            rejected.append({"carpeta": folder, "archivo": name, "origen": "postulante",
                           "motivo": "formato no admitido (se aceptan PDF y fotos JPG, PNG, HEIC, WEBP, TIFF, BMP)"})
            continue
        a = L.InMemoryFile(name, data, folder)
        try:
            us = L.units(a)
        except Exception as e:
            rejected.append({"carpeta": folder, "archivo": name, "origen": "postulante",
                           "motivo": f"no se pudo abrir el archivo ({type(e).__name__})"})
            continue
        if len(us) > MAX_PAGES_PER_FILE:
            reason = f"tiene {len(us)} páginas (máximo {MAX_PAGES_PER_FILE} por archivo)"
        elif len(units) + len(us) > MAX_PAGES_PER_VERIFICATION:
            reason = f"se superó el máximo de {MAX_PAGES_PER_VERIFICATION} páginas por verificación"
        else:
            units += us
            continue
        rejected.append({"carpeta": folder, "archivo": name, "origen": "postulante", "motivo": reason})
    for i, u in enumerate(units, 1):
        progress("leyendo documentos", i, len(units))
        pages += PL.pages_of_unit(L.analyze(u))
    def ocr_progress(h, t):
        """Progreso de la lectura OCR."""
        progress("leyendo imágenes con OCR", h, t)

    if any("imagen_ocr" in p for p in pages):
        model = _with_reader(use_gpu, lambda reader: PL.read_ocr(pages, use_gpu, log, ocr_progress, reader))
    else:
        model = "no se usó OCR"
    for p in pages:
        if p.get("plantilla") and not p.get("ilegible"):
            PL.page_values(p, pl)
    progress("comparando con el dato oficial", 0, 0)
    registry = {folder: {"rut": identity["rut"], "nombre": folder, "puesto": identity["cargo"]}}
    person, rows = PL.assess_folder(folder, pages, registry, None, req, today, pl, official_data=official_data)
    PL.unique_names(rows)

    expected_docs = []
    illegibles = [f["archivo"] for f in rows if f["veredicto"] == K.ILLEGIBLE and not D.is_profile_document(
        f["tipo_documento"])]
    for kind, label in D.EXPECTED_DOCUMENTS:
        docs = [f for f in rows if f["tipo_documento"].startswith(kind)]
        state = "ENTREGADO" if docs else (NOT_IDENTIFIED if illegibles else MISSING)
        expected_docs.append({"tipo": kind, "documento": label, "estado": state,
                          "veredictos": [f["veredicto"] for f in docs], "archivos": [f["archivo"] for f in docs],
                          "nota": (f"no se encontró entre los archivos legibles; puede ser uno de los ilegibles "
                                   f"({', '.join(illegibles)})") if state == NOT_IDENTIFIED else ""})
        if state == NOT_IDENTIFIED:
            continue
        if not docs:
            warns += D.missing_document_warnings(kind, official_data)
    person["faltantes"] = [e["documento"] for e in expected_docs if e["estado"] == MISSING]
    person["no_identificados"] = [e["documento"] for e in expected_docs if e["estado"] == NOT_IDENTIFIED]
    no_official = not D.has_official(official_data)
    if no_official:
        person.update(veredicto=K.NO_OFFICIAL_DATA, aptitud=K.NO_OFFICIAL_DATA,
                       motivos_aptitud=["No se subieron datos oficiales (JSON): no se verificó la autenticidad ni la "
                                        "aptitud. Se entregan los datos extraídos de los documentos."])
    meta = {"fecha": date.today().isoformat(), "fecha_referencia": str(today), "modelo_ocr": model,
            "modo": "visor de pruebas (en memoria, sin guardar archivos)", "sin_datos_oficiales": no_official}
    return {"identidad": identity, "persona": person, "documentos": rows, "esperados": expected_docs,
            "advertencias": list(dict.fromkeys(warns)), "no_admitidos": rejected, "meta": meta}


def excel_in_memory(result):
    """El mismo reporte Excel del flujo normal, generado en memoria (bytes)."""
    buf = io.BytesIO()
    person = dict(result["persona"], carpeta=result["identidad"]["nombre_completo"])
    meta = dict(result["meta"], documentos_faltantes=", ".join(person.get("faltantes", [])) or "ninguno",
                documentos_no_identificados=", ".join(person.get("no_identificados", [])) or "ninguno",
                advertencias=" | ".join(result["advertencias"]) or "ninguna")
    report.excel(buf, [person], result["documentos"], meta, result["no_admitidos"])
    return buf.getvalue()


def _file_name(text):
    """Nombre de archivo seguro (ASCII, sin espacios ni símbolos, máximo 120 caracteres)."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", t).strip("_")[:120] or "documento"


def _json(data):
    """JSON legible (UTF-8, con sangría) en bytes."""
    return json.dumps(data, ensure_ascii=False, indent=1, default=str).encode("utf-8")


def zip_in_memory(result, label):
    """Todo el resultado en un .zip (en memoria): resumen, datos extraídos de cada documento (JSON con las claves del
    JSON oficial), diferencias con el dato oficial y el reporte Excel."""
    import zipfile
    buf = io.BytesIO()
    p, i = result["persona"], result["identidad"]
    docs = result["documentos"]
    compared = [d for d in docs if d["campos"]]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LEAME.txt", (
            f"Verificación de {label} ({result.get('fecha', '')})\n\n"
            "resumen.json                 identidad, dictamen por documento, documentos faltantes, aptitud y advertencias\n"
            "datos_extraidos/*.json       lo que dice cada documento del postulante, con las mismas claves que el JSON\n"
            "                             oficial. '_confianza' repite la estructura y da, para cada dato, la confianza\n"
            "                             (0 a 1) y su nivel ALTA / MEDIA / BAJA / SIN LECTURA; '_control' indica el\n"
            "                             método de lectura, la escala y los campos a revisar\n"
            + ("diferencias.json             campo por campo: dato oficial vs. leído, solo donde no coinciden\n"
               if compared else "")
            + "reporte_verificacion.xlsx    el mismo reporte en Excel\n"
            + ("\nNo se subieron datos oficiales (JSON): no se verificó la autenticidad ni la aptitud.\n"
               if result["meta"].get("sin_datos_oficiales") else "")).encode("utf-8"))
        z.writestr("resumen.json", _json({
            "postulante": {"nombres": i["nombres"], "apellidos": i["apellidos"], "rut": i["rut"],
                           "fecha_nacimiento": f"{i['fecha_nacimiento']:%d-%m-%Y}" if i["fecha_nacimiento"] else "",
                           "cargo": p["puesto"]},
            "autenticidad": p["veredicto"], "aptitud": p["aptitud"], "motivos_aptitud": p["motivos_aptitud"],
            "advertencias": result["advertencias"],
            "documentos_esperados": [{k: e[k] for k in ("documento", "estado", "veredictos", "archivos")}
                                     for e in result["esperados"]],
            "documentos": [{"archivo": d["archivo"], "tipo": d["tipo_documento"], "dictamen": d["veredicto"],
                            "motivos": d["motivos"], "dato_oficial": d["archivo_oficial"]} for d in docs],
            "no_admitidos": [{"archivo": x["archivo"], "motivo": x["motivo"]} for x in result["no_admitidos"]],
            "fecha_referencia": result["meta"]["fecha_referencia"]}))
        used_counts = Counter()
        for d in docs:
            if not d.get("json_extraido"):
                continue
            base = _file_name(f"{d['tipo_documento']}_{d['archivo']}")
            used_counts[base] += 1
            name = base if used_counts[base] == 1 else f"{base}_{used_counts[base]}"
            z.writestr(f"datos_extraidos/{name}.json", _json(d["json_extraido"]))
        if compared:
            z.writestr("diferencias.json", _json([
                {"archivo": d["archivo"], "dato_oficial": d["archivo_oficial"], "dictamen": d["veredicto"],
                 "campos": [{"campo": c["zona"], "oficial": c["oficial"], "leido": c["postulante"], "estado": c["estado"],
                             "motivo": c["motivo"], "confianza": c["conf"]}
                            for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")]} for d in compared]))
        z.writestr("reporte_verificacion.xlsx", excel_in_memory(result))
    return buf.getvalue()
