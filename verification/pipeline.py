"""Pipeline de verificación de documentos contra sus datos oficiales en JSON.

  1. Plantillas del perfil (official_file/) y zonas de cada campo (perfil: templates)
  2. Unidades de entrada: cada página de cada PDF, cada imagen y cada página de un TIFF de cada carpeta
  3. Lectura en CPU, en paralelo (reading.py): PDF nativo -> texto visible directo; PDF de imágenes e imágenes ->
     alineación con la plantilla (4 giros, 2 documentos por foto); timbre y código de barras
  4. OCR en GPU de las páginas alineadas (modelo adoptado en config.py)
  5. Agrupación de páginas por documento (perfil: documents.group_pages)
  6. Comparación campo a campo con el JSON (webscrap_files/<carpeta>/*.json) + señales técnicas + identidad
     (perfil: rules)
  7. Aptitud según los requisitos del perfil (perfil: requirements)
  8. Reporte: <salida>/verification.db y <salida>/verification_report.xlsx

Uso: python -u -m verification.pipeline [--limit 50] [--output results/run]

El lector OCR (PyTorch) se importa solo dentro de `read_ocr`, nunca a nivel de módulo: cada proceso de lectura
re-importa este archivo y no debe cargar PyTorch.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path

from openpyxl import load_workbook

from . import config as C
from . import fields as KC
from . import reading as L
from . import resources as R
from .boxes import center, inside
from .profiles import ACTIVE as PF
from .report import write_report

D, EX, N, P, Q, K = PF.documents, PF.extraction, PF.normalize, PF.templates, PF.requirements, PF.rules


class StoppedForResources(RuntimeError):
    """La verificación se detuvo por falta de memoria del sistema o porque se canceló."""


def read_registry(path):
    """candidatos.xlsx: carpeta -> {identidad, nombre, puesto, n} (columnas del perfil)."""
    if not path or not Path(path).exists():
        return {}
    rows = list(load_workbook(path, read_only=True).active.iter_rows(values_only=True))
    header = [str(c or "").strip().lower() for c in rows[0]]

    def col(*names):
        """Índice de la primera columna cuyo encabezado es uno de `nombres` (None si no hay)."""
        return next((header.index(n) for n in names if n in header), None)
    ic = col("carpeta")
    cols = {k: col(*names) for k, names in D.REGISTRY_COLUMNS.items()} | {"n": col("n")}
    return {str(f[ic]): {k: f[i] if i is not None else None for k, i in cols.items()}
            for f in rows[1:] if f[ic] is not None}


def tasks(folders, applicants_dir):
    """Unidades por leer de cada carpeta y archivos no admitidos (formato no soportado o que no se pueden abrir)."""
    entries, rejected_files = [], []
    for c in folders:
        d = applicants_dir / c
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if not f.is_file():
                continue
            if f.suffix.lower() not in L.ACCEPTED_EXT:
                rejected_files.append({"carpeta": c, "archivo": f.name, "origen": "postulante",
                                     "motivo": f"formato no admitido ({f.suffix or 'sin extensión'})"})
                continue
            try:
                entries += L.units(f)
            except Exception as e:
                rejected_files.append({"carpeta": c, "archivo": f.name, "origen": "postulante",
                                     "motivo": f"no se pudo abrir: {type(e).__name__}"})
    return entries, rejected_files


def pages_of_unit(r):
    """Una unidad (página/imagen) puede contener 0, 1 o 2 documentos: devuelve una 'página' por cada uno.

    Sin alineación, el tipo es solo una conjetura (la plantilla con más coincidencias): no se compara contra ningún
    dato oficial ni se afirma qué documento es. Si la imagen es borrosa, es ilegible; si es nítida, es otro
    documento (cédula, licencia...) o una foto que no muestra el documento completo (NO RECONOCIDO).
    """
    base = {k: r.get(k) for k in ("ruta", "carpeta", "archivo", "unidad", "pagina", "entrada", "meta", "error",
                                  "revisiones_pdf", "capa_texto_ignorada")}
    if r.get("error"):
        return [base | {"plantilla": None, "ilegible": True, "motivo_ilegible": f"no se pudo leer el archivo: {r['error']}"}]
    if r.get("nativo"):
        if not r.get("plantilla"):
            return [base | {"plantilla": None, "no_reconocido": True, "motivo": r.get("motivo")}]
        return [base | {k: r.get(k) for k in ("plantilla", "palabras", "geometria", "alineacion_ok", "texto_oculto",
                                              "objetos", "copia", "codigos", "clasificacion")} | {"es_ocr": False}]
    out = []
    for i, d in enumerate(r.get("documentos") or []):
        p = base | {"plantilla": d["plantilla"], "es_ocr": True, "alineacion_ok": d["alineacion_ok"],
                    "alineacion": d.get("alineacion"), "rotacion_aplicada": d.get("rotacion_aplicada"),
                    "codigos": r.get("codigos"), "nitidez": r.get("nitidez"), "doc_en_imagen": i}
        if not d["alineacion_ok"]:
            sharp = r.get("nitidez") or 0
            p.update(plantilla=None, plantilla_probable=d["plantilla"])
            if sharp < C.MIN_SHARPNESS:
                p.update(ilegible=True, motivo_ilegible=f"imagen ilegible (nitidez {sharp:.0f}): no se pudo identificar "
                                                        "ni leer el documento")
            else:
                p.update(no_reconocido=True, motivo=f"imagen nítida que no corresponde a {D.UNRECOGNIZED_HINT}, "
                                                    f"o no muestra el documento completo "
                                                    f"({d.get('alineacion_motivo')})")
        else:
            p["imagen_ocr"] = d["imagen_ocr"]
        out.append(p)
    if not out:
        out.append(base | {"plantilla": None, "ilegible": True, "motivo_ilegible": "imagen sin documento reconocible"})
    return out


def create_reader(use_gpu):
    """Lector OCR (detector + reconocedor) en GPU, con su límite de VRAM, o en CPU.

    El modelo adoptado lee palabras completas (hasta 30 caracteres): no se parten los recortes largos.
    """
    from .ocr import OCRReader
    if use_gpu:
        R.limit_gpu(C.OCR_VRAM_BUDGET_GB)
    reader = OCRReader(device=None if use_gpu else "cpu")
    reader.max_aspect = 15.0
    return reader


def release_reader(reader):
    """Descarga un lector OCR y devuelve su memoria de GPU."""
    del reader
    import torch
    torch.cuda.empty_cache()


def read_ocr(pages, use_gpu, log=print, on_read=None, reader=None):
    """OCR (GPU) de las páginas alineadas, por lotes. Devuelve el nombre del modelo usado.

    `lector`: uno ya cargado (se reutiliza y no se descarga); sin él se carga uno y se descarga al terminar. Las
    páginas llegan alineadas a la resolución de la plantilla: las cajas leídas quedan en puntos de la plantilla.
    """
    pending = [p for p in pages if "imagen_ocr" in p]
    if not pending:
        return "no se usó OCR"
    owned = reader is None
    reader = create_reader(use_gpu) if owned else reader
    log(f"OCR: {reader.model_name} en {reader.device} | {len(pending)} páginas")
    pl = P.load()
    for i in range(0, len(pending), C.OCR_BATCH_PAGES):
        batch = pending[i:i + C.OCR_BATCH_PAGES]
        zones = [[{"rect": z["rect"], "nombre": n, "tipo": z["tipo"]}
                  for n, z in pl[p["plantilla"]]["zonas"].items() if z["tipo"] != "cuerpo"] for p in batch]
        free_zones = [[z["rect"] for z in pl[p["plantilla"]]["zonas"].values() if z["tipo"] == "cuerpo"] for p in batch]
        words = reader.read([p["imagen_ocr"] for p in batch], zones, free_zones)
        for p, pw in zip(batch, words):
            p["palabras"] = pw
            del p["imagen_ocr"]
        if on_read:
            on_read(i + len(batch), len(pending))
    name = reader.model_name
    if owned:
        release_reader(reader)
    return name


def page_values(p, pl):
    """Palabras -> valor de cada zona; filas de la sección variable; palabras de valores (fuentes).

    En lecturas OCR también se detecta la versión COPIA (marca de agua y pie "Copia no válida como certificado") y
    se marca como ilegible la página donde casi no se leen las etiquetas del formato.
    """
    tpl = pl[p["plantilla"]]
    words = p.get("palabras") or []
    assigned = KC.assign_words(words, tpl)
    p["valores"] = assigned["valores"]
    p["cobertura_etiquetas"] = assigned["cobertura_etiquetas"]
    rects = [z["rect"] for z in tpl["zonas"].values() if z["tipo"] != "cuerpo"]
    p["palabras_valor"] = [w for w in words if any(inside(center(w["bbox"]), r) for r in rects)]
    if "cuerpo" in tpl["zonas"]:
        r = tpl["zonas"]["cuerpo"]["rect"]
        p["lineas_cuerpo"] = D.body_lines([w for w in words if inside(center(w["bbox"]), r)])
    if p.get("es_ocr"):
        text = " ".join(w["texto"] for w in words).upper()
        p["copia"] = D.is_copy_ocr(text)
    if p.get("es_ocr") and assigned["cobertura_etiquetas"] < C.MIN_OCR_LABEL_COVERAGE and not p.get("ilegible"):
        p.update(ilegible=True, motivo_ilegible=f"casi no se leen las etiquetas del formato "
                                                f"({assigned['cobertura_etiquetas']:.0%}): imagen ilegible o muy pequeña")


def assemble_document(ps, pl):
    """Junta las páginas de un documento: valores y zonas (las de la pág. 2 con sufijo), códigos, señales del PDF
    nativo, datos que agrega el perfil (p. ej. filas variables) y etiquetas vecinas de cada zona.

    - Dos documentos en una misma foto se distinguen por su posición en la imagen ("[documento 2]").
    - Los valores de cabecera de la pág. 2 se usan para el timbre solo si falta la pág. 1.
    - `formato_ajeno`: etiquetas impresas del formato FUERA de cada zona pero en las líneas vecinas (hasta 30 pt
      arriba o abajo). Si aparecen dentro del valor leído, la zona se leyó corrida una línea (desalineación), no es
      un dato cambiado.
    """
    pid0 = ps[0].get("plantilla")
    kind = P.DOCUMENT_TYPE.get(pid0) if pid0 else None
    files = list(dict.fromkeys(p["archivo"] for p in ps))
    units = list(dict.fromkeys(p["unidad"] + (f" [documento {p['doc_en_imagen'] + 1}]" if p.get("doc_en_imagen") else "")
                                  for p in ps))
    doc = {"carpeta": ps[0]["carpeta"], "archivo": " + ".join(units), "archivos": files,
           "tipo_documento": kind or "desconocido", "plantilla": pid0, "es_ocr": any(p.get("es_ocr") for p in ps),
           "ilegible": any(p.get("ilegible") for p in ps), "copia": any(p.get("copia") for p in ps),
           "motivo_ilegible": next((p.get("motivo_ilegible") for p in ps if p.get("motivo_ilegible")), ""),
           "no_reconocido": any(p.get("no_reconocido") for p in ps),
           "motivo_no_reconocido": next((p["motivo"] for p in ps if p.get("no_reconocido") and p.get("motivo")), ""),
           "valores": {}, "senales": [],
           "codigos": {"timbre": [], "barras": []}, "meta": ps[0].get("meta") or {}, "entrada": ps[0].get("entrada")}
    doc_zones = {}
    ps = D.order_pages(kind, ps, doc)
    for p in ps:
        if not p.get("plantilla") or p.get("ilegible"):
            continue
        suffix = D.page_suffix(p["plantilla"])
        for z, v in (p.get("valores") or {}).items():
            doc["valores"][z + suffix] = v
        for z, info in pl[p["plantilla"]]["zonas"].items():
            if info["tipo"] != "cuerpo":
                doc_zones[z + suffix] = info
        for k in ("timbre", "barras"):
            doc["codigos"][k] += (p.get("codigos") or {}).get(k, [])
        if not p.get("es_ocr"):
            pn = dict(p)
            doc["senales"] += K.native_signals(pn, pl[p["plantilla"]])
            doc["revisiones_pdf"] = p.get("revisiones_pdf")
    D.finish_document(kind, ps, doc)
    doc["zonas"] = doc_zones
    doc["formato_ajeno"] = {}
    for p in ps:
        if not p.get("plantilla") or p.get("ilegible"):
            continue
        suffix = D.page_suffix(p["plantilla"])
        fixed = pl[p["plantilla"]]["palabras_fijas"]
        for z, info in pl[p["plantilla"]]["zonas"].items():
            r = info["rect"]
            doc["formato_ajeno"][z + suffix] = {t for x0, y0, x1, y1, txt in fixed
                                             if (x1 < r[0] or x0 > r[2] or y1 < r[1] or y0 > r[3])
                                             and y1 >= r[1] - 30 and y0 <= r[3] + 30
                                             for t in N.text_key(txt).split() if len(t) >= 4 and t.isalpha()}
    doc["formato"] = "PDF_NATIVO" if not doc["es_ocr"] else ("IMAGEN" if doc["entrada"] == "imagen" else "ESCANEADO")
    return doc


def assess_folder(folder, pages, registry, official_dir, req, today, pl, official_data=None):
    """Dictamen de cada documento de una carpeta y resultado de la persona (autenticidad general y aptitud).

    oficiales: los JSON ya cargados en memoria; si no se dan, se leen de official_dir/<carpeta>.
    Sin dato oficial no hay con qué verificar: el documento no se analiza, solo se extraen sus datos.
    En cada fila, `archivo_oficial` es el nombre del JSON y `archivo_oficial_clave` el N° del certificado;
    El perfil puede agregar datos a la fila (documents.row_details), p. ej. las filas leídas de una sección variable.
    """
    declared = registry.get(folder, {})
    official_data = official_data if official_data is not None else D.official_data_of(folder, official_dir)
    rows = []
    for group in D.group_pages(pages):
        doc = assemble_document(group, pl)
        if doc["no_reconocido"] or (doc["plantilla"] is None and not doc["ilegible"]):
            rows.append(_row(doc, {"veredicto": KC.NOT_RECOGNIZED, "campos": [], "senales": [],
                                     "motivos": [doc["motivo_no_reconocido"] or
                                                 "el archivo no corresponde a ninguno de los documentos solicitados"]}, ""))
            continue
        if doc["plantilla"] is None:
            r = {"veredicto": KC.ILLEGIBLE, "campos": [], "senales": [], "motivos": [doc["motivo_ilegible"]]}
            rows.append(_row(doc, r, ""))
            continue
        json_key, js, official_name = D.official_for(doc, official_data)
        if js is None:
            r = {"veredicto": KC.NO_OFFICIAL_DATA, "campos": [], "senales": [], "emision": "",
                 "motivos": ["no se entregó el dato oficial (JSON) de este documento: no se verificó, solo se "
                             "extrajeron sus datos"]}
        else:
            r = K.compare_document(doc, js, doc["zonas"], declared, today)
        row = _row(doc, r, official_name if js is not None else "")
        row["json_extraido"] = EX.extract(doc)
        row.update(D.row_details(doc))
        row["archivo_oficial_clave"] = json_key
        rows.append(row)
    position = declared.get("puesto") or req["puesto_por_defecto"]
    assessable = [f for f in rows if f["veredicto"] != KC.NOT_RECOGNIZED]
    fitness, fit_reasons = Q.evaluate_person(position, assessable, official_data, req, today)
    v = Counter(f["veredicto"] for f in assessable)
    overall = next((x for x in (KC.FORGED, KC.FLAGGED, KC.ILLEGIBLE, KC.NOT_VALID, KC.OUTDATED,
                                KC.NO_OFFICIAL_DATA, KC.AUTHENTIC) if v[x]), KC.FLAGGED)
    person = {"carpeta": folder, "dni": str(declared.get("rut") or ""), "declarado": bool(declared),
               "documentos": len(assessable), "falsificados": v[KC.FORGED], "observados": v[KC.FLAGGED] + v[KC.ILLEGIBLE],
               "desactualizados": v[KC.OUTDATED],
               "tipos_falsificados": [f["tipo_documento"] for f in assessable if f["veredicto"] == KC.FORGED],
               "veredicto": overall, "puesto": position, "aptitud": fitness, "motivos_aptitud": fit_reasons}
    return person, rows


def _row(doc, r, official_file):
    """Fila de resultado de un documento, en el formato que usa el reporte."""
    return {"carpeta": doc["carpeta"], "archivo": doc["archivo"], "archivos": doc["archivos"],
            "tipo_documento": doc["plantilla"] or doc["tipo_documento"], "formato": doc.get("formato", ""),
            "veredicto": r["veredicto"], "motivos": r["motivos"], "campos": r["campos"], "senales": r["senales"],
            "misma_instancia": None, "archivo_oficial": official_file, "alineacion_ok": not doc.get("ilegible"),
            "alineacion": {}, "residual_etiquetas_pt": None, "producer": (doc.get("meta") or {}).get("producer", ""),
            "emision": r.get("emision", ""), "prioridad_revision": r.get("prioridad_revision", "")}


def unique_names(rows):
    """Agrega " (2)", " (3)"... al nombre de los documentos repetidos de una carpeta: el reporte usa el nombre como
    clave."""
    seen_counts = Counter()
    for f in rows:
        seen_counts[f["archivo"]] += 1
        if seen_counts[f["archivo"]] > 1:
            f["archivo"] = f"{f['archivo']} ({seen_counts[f['archivo']]})"
    return rows


def version_fingerprint():
    """Huella de 16 caracteres del código, la configuración, las plantillas y el modelo OCR usados."""
    h = hashlib.sha256()
    for f in sorted((C.ROOT / "verification").rglob("*.py")) + sorted((C.ROOT / "verification").rglob("*.yaml")) \
            + sorted(C.TEMPLATES_DIR.glob("*.pdf")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    if C.PARSEQ_MODEL.exists():
        st = C.PARSEQ_MODEL.stat()
        h.update(f"{st.st_size}-{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def verify(folders, registry, applicants_dir=C.APPLICANTS_DIR, official_dir=C.OFFICIAL_DATA_DIR, workers=C.READ_WORKERS,
              use_gpu=True, log=print, progress=None, cancelled=lambda: False, today=None):
    """Verifica las carpetas dadas: lectura en paralelo, OCR, dictamen y aptitud.

    Devuelve (personas, documentos, no_admitidos, meta). Cada documento lleva un nombre único dentro de su carpeta
    (el reporte lo usa como clave). Levanta DetenidoPorRecursos si la memoria se vuelve crítica o se cancela.
    """
    t0 = time.time()
    today = today or date.today()
    pl = P.load()
    req = Q.load()
    entries, rejected_files = tasks(folders, Path(applicants_dir))
    log(f"{len(folders)} postulantes | {len(entries)} páginas/imágenes por leer | {len(rejected_files)} no admitidos")
    pages = []
    watchdog = R.Watchdog().start()
    with Pool(workers, initializer=L.init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(L.analyze, entries, chunksize=2), 1):
            if watchdog.alert or cancelled():
                pool.terminate()
                watchdog.stop()
                raise StoppedForResources(watchdog.alert or "cancelado")
            pages += pages_of_unit(r)
            if progress:
                progress("lectura", i, len(entries))
            if i % 200 == 0:
                log(f"  {i}/{len(entries)} leídas ({time.time() - t0:.0f}s)")
    watchdog.stop()
    model = read_ocr(pages, use_gpu, log, (lambda h, t: progress("ocr", h, t)) if progress else None)
    for p in pages:
        if p.get("plantilla") and not p.get("ilegible"):
            page_values(p, pl)
    by_folder = defaultdict(list)
    for p in pages:
        by_folder[p["carpeta"]].append(p)
    people, documents = [], []
    for c in folders:
        person, rows = assess_folder(c, by_folder.get(c, []), registry, Path(official_dir), req, today, pl)
        unique_names(rows)
        people.append(person)
        documents += rows
    meta = {"fecha": datetime.now().isoformat(timespec="seconds"), "huella_version": version_fingerprint(),
            "postulantes": len(people), "documentos_postulante": len(documents), "archivos_no_admitidos": len(rejected_files),
            "modelo_ocr": model, "duracion_s": round(time.time() - t0, 1), "fecha_referencia": str(today),
            "perfil": C.PROFILE} | D.run_meta()
    return people, documents, rejected_files, meta


def main():
    """Línea de comandos: verifica las carpetas de postulantes y escribe el reporte en --salida."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folders", nargs="*", default=None, help="solo estas carpetas de postulantes")
    ap.add_argument("--workers", type=int, default=C.READ_WORKERS)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--applicants", default=str(C.APPLICANTS_DIR))
    ap.add_argument("--official-data", default=str(C.OFFICIAL_DATA_DIR))
    ap.add_argument("--output", default=str(C.RESULTS_DIR / "run"))
    ap.add_argument("--today", default=None, help="fecha de referencia AAAA-MM-DD (por defecto, hoy)")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--subset", choices=["ajuste", "reservado"], default=None,
                    help="medición: ajuste = postulantes con N impar del registro; reservado = N par")
    a = ap.parse_args()
    applicants_dir = Path(a.applicants)
    registry = read_registry(applicants_dir / "candidatos.xlsx")
    folders = a.folders or sorted(d.name for d in applicants_dir.iterdir() if d.is_dir())
    if a.subset:
        odd = a.subset == "ajuste"
        folders = [c for c in folders if (registry.get(c, {}).get("n") or 0) % 2 == (1 if odd else 0)]
    if a.limit:
        folders = folders[:a.limit]
    today = date.fromisoformat(a.today) if a.today else None
    try:
        with R.RunLock("pipeline"):
            R.verify(vram_gb=0 if a.cpu else C.OCR_VRAM_BUDGET_GB, ram_gb=C.RAM_BUDGET_GB, window_s=a.window)
            people, documents, rejected, meta = verify(folders, registry, applicants_dir, a.official_data, a.workers,
                                                           not a.cpu, today=today)
    except (R.InsufficientResources, StoppedForResources) as e:
        print(f"NO SE COMPLETÓ: {e}", flush=True)
        raise SystemExit(2) from e
    _, xlsx = write_report(people, documents, meta, Path(a.output), rejected)
    (Path(a.output) / "results.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False, default=str)
                                                                for d in documents), encoding="utf-8")
    print("\nPostulantes:", dict(Counter(p["veredicto"] for p in people)))
    print("Aptitud:    ", dict(Counter(p["aptitud"] for p in people)))
    print("Documentos: ", dict(Counter(d["veredicto"] for d in documents)))
    print(f"Reporte: {xlsx} | Duración: {meta['duracion_s']}s", flush=True)


if __name__ == "__main__":
    main()
