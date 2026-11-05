"""Visor local de verificación de postulantes (solo accesible desde esta PC: 127.0.0.1).

  - Buscar postulantes (nombre o número de identidad) y revisar uno, varios o todos.
  - Revisiones en segundo plano, una a la vez, con chequeo de GPU/RAM (si no hay recursos, espera)
    y el mismo candado que el pipeline de consola (nunca corren dos verificaciones juntas).
  - Caché: un postulante ya revisado cuyos archivos no cambiaron no se vuelve a procesar.
  - Resumen por postulante, por tipo de documento y por formato; detalle con campos y señales.
  - Excel por revisión y Excel acumulado.

Uso: python -m viewer.server

Los datos del visor (base y Excel) van a results/viewer, o a la carpeta de la variable de entorno VIEWER_DATA
(para pruebas). Se registra el lector HEIC para mostrar miniaturas de fotos de iPhone.
"""

import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
import webbrowser
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import re

import pillow_heif
import pymupdf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from verification import config as C
from verification import resources as R
from verification import reading as L
from verification import pipeline as PL
from verification.profiles import ACTIVE as PF
from verification.report import write_report

pillow_heif.register_heif_opener()

VIEWER_DIR = Path(os.environ.get("VIEWER_DATA", C.RESULTS_DIR / "viewer"))
DB = VIEWER_DIR / "viewer.db"
PAGE = Path(__file__).parent / "static" / "index.html"
ALLOWED_ROOTS = tuple(os.path.realpath(d) + os.sep for d in (C.APPLICANTS_DIR, C.OFFICIAL_DATA_DIR))

@asynccontextmanager
async def _lifespan(_app):
    """Al arrancar: prepara la base y lanza el hilo que atiende la cola de revisiones."""
    _init_db()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="Verificación de postulantes", lifespan=_lifespan)
_queue = queue.Queue()
_jobs = {}
"""id -> estado en memoria (para el progreso en vivo)"""
_lock = threading.Lock()


def _db():
    """Conexión a la base del visor (filas accesibles por nombre de columna)."""
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Crea las tablas si faltan y marca como interrumpidas las revisiones que quedaron a medias al cerrarse el
    visor."""
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trabajos (id TEXT PRIMARY KEY, fecha TEXT, tipo TEXT, carpetas TEXT,
                estado TEXT, mensaje TEXT, n_postulantes INTEGER, procesados INTEGER, desde_cache INTEGER,
                segundo_lector INTEGER, excel TEXT, duracion_s REAL);
            CREATE TABLE IF NOT EXISTS ultimas (carpeta TEXT PRIMARY KEY, trabajo_id TEXT, fecha TEXT, firma TEXT,
                persona TEXT, documentos TEXT);
        """)
        conn.execute("UPDATE trabajos SET estado='interrumpido', mensaje='el visor se cerró durante la revisión' "
                    "WHERE estado IN ('en_cola','esperando_recursos','procesando')")


def _save_job(t):
    """Guarda (o actualiza) el registro de una revisión."""
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO trabajos VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (t["id"], t["fecha"], t["tipo"], json.dumps(t["carpetas"], ensure_ascii=False), t["estado"],
                     t.get("mensaje", ""), len(t["carpetas"]), t.get("procesados", 0), t.get("desde_cache", 0),
                     int(t["segundo_lector"]), t.get("excel", ""), t.get("duracion_s")))


def _registry():
    """Registro de postulantes (identidad declarada y cargo por carpeta)."""
    return PL.read_registry(C.APPLICANT_REGISTRY)


def _applicant_folders():
    """Carpetas de postulant_files, en orden."""
    return sorted(d.name for d in C.APPLICANTS_DIR.iterdir() if d.is_dir())


def _signature(folder, segundo_lector, fingerprint):
    """Si cambian los archivos (del postulante o los oficiales), el modelo o las reglas, cambia la firma."""
    h = hashlib.sha256(f"{fingerprint}|{segundo_lector}".encode())
    for base in (C.APPLICANTS_DIR, C.OFFICIAL_DATA_DIR):
        d = base / folder
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if f.is_file():
                st = f.stat()
                h.update(f"{base.name}/{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def _without_values(doc):
    """Documento sin los valores completos leídos y oficiales (no se guardan en la base del visor)."""
    return {k: v for k, v in doc.items() if k not in ("valores_oficiales", "valores_postulante")}


TYPES = {pid: label for kind, label in PF.documents.EXPECTED_DOCUMENTS for pid in PF.templates.FILES
         if pid.startswith(kind)} | {"desconocido": "No identificado"}
"""Nombre visible de cada tipo; "No identificado" es un archivo ajeno (NO RECONOCIDO) o ilegible (NO LEGIBLE)."""


def _type_label(kind):
    """Nombre visible de un tipo de documento."""
    return TYPES.get(kind or "desconocido", (kind or "desconocido").replace("_", " ").capitalize())


FORMAT_LABEL = {"PDF_NATIVO": "PDF nativo", "ESCANEADO": "Escaneado", "IMAGEN": "Foto / imagen"}


def _latest(carpetas=None):
    """Última revisión de cada postulante (de todos, o solo de `carpetas`)."""
    with _db() as conn:
        rows = conn.execute("SELECT * FROM ultimas").fetchall()
    output = {f["carpeta"]: {"persona": json.loads(f["persona"]), "documentos": json.loads(f["documentos"]),
                             "fecha": f["fecha"], "trabajo_id": f["trabajo_id"]} for f in rows}
    return {c: output[c] for c in carpetas if c in output} if carpetas is not None else output


def _excel(destination, latest, meta):
    """Escribe el reporte Excel de esas revisiones en `destino` y devuelve su ruta."""
    people = [u["persona"] for u in latest.values()]
    documents = [d for u in latest.values() for d in u["documentos"]]
    _, xlsx = write_report(people, documents, meta, destination)
    return xlsx


def _worker():
    """Hilo que atiende las revisiones de a una; el visor sigue funcionando aunque falle una revisión."""
    while True:
        t = _queue.get()
        try:
            _process(t)
        except InterruptedError as e:
            t.update(estado="cancelado", mensaje=str(e))
        except Exception as e:
            t.update(estado="error", mensaje=f"{type(e).__name__}: {e}")
        finally:
            t["duracion_s"] = round(time.time() - t["_t0"], 1)
            _save_job(t)


def _process(t):
    """Revisa los postulantes cuya firma cambió (o todos si se fuerza).

    Espera recursos y el candado de otra verificación (de consola o del visor), reintentando cada 30 s. El Excel de
    la revisión se genera al descargarlo: así la página se entera del fin sin esperar.
    """
    t["_t0"] = time.time()
    fingerprint = PL.version_fingerprint()
    signatures = {c: _signature(c, t["segundo_lector"], fingerprint) for c in t["carpetas"]}
    with _db() as conn:
        saved_items = {f["carpeta"]: f["firma"] for f in conn.execute("SELECT carpeta, firma FROM ultimas")}
    pending_items = [c for c in t["carpetas"] if t["forzar"] or saved_items.get(c) != signatures[c]]
    t["desde_cache"] = len(t["carpetas"]) - len(pending_items)

    if pending_items:
        while True:
            if t.get("cancelar"):
                raise InterruptedError("revisión cancelada")
            try:
                t.update(estado="esperando_recursos", fase="revisando GPU y memoria")
                with R.RunLock("pipeline"):
                    R.verify(vram_gb=C.OCR_VRAM_BUDGET_GB, ram_gb=C.RAM_BUDGET_GB, window_s=10,
                                log=lambda m: None)
                    t.update(estado="procesando", mensaje="", _t_proceso=time.time())
                    _execute(t, pending_items, signatures)
                break
            except R.InsufficientResources as e:
                t.update(mensaje=f"{e}. Se reintenta en 30 s.")
                for _ in range(30):
                    if t.get("cancelar"):
                        raise InterruptedError("revisión cancelada") from e
                    time.sleep(1)

    t.update(estado="terminado", fase="terminado", procesados=len(pending_items), mensaje="", huella=fingerprint)


def _execute(t, pending_items, signatures):
    """Verifica los postulantes pendientes y guarda su última revisión con su firma."""
    from verification.batch_run import verify_batch

    def progress(phase, done, total):
        """Actualiza el progreso visible de la revisión."""
        t.update(fase=phase, hechos=done, total=total)

    people, documents, rejected_files, _ = verify_batch(
        pending_items, _registry(), second_reader=t["segundo_lector"], progress=progress,
        log=lambda m: t.update(mensaje=m), cancelled=lambda: t.get("cancelar", False))
    by_folder = defaultdict(list)
    for d in documents:
        by_folder[d["carpeta"]].append(_without_values(d))
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as conn:
        for p in people:
            conn.execute("INSERT OR REPLACE INTO ultimas VALUES (?,?,?,?,?,?)",
                        (p["carpeta"], t["id"], now, signatures.get(p["carpeta"], ""),
                         json.dumps(p, ensure_ascii=False, default=str),
                         json.dumps(by_folder[p["carpeta"]], ensure_ascii=False, default=str)))
    t["no_admitidos"] = rejected_files


class ReviewRequest(BaseModel):
    """Pedido de revisión: carpetas elegidas o todas; `forzar` ignora la caché."""

    carpetas: list[str] = []
    todos: bool = False
    segundo_lector: bool = True
    forzar: bool = False


@app.get("/", response_class=HTMLResponse)
def home():
    """Página del visor."""
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/postulantes")
def applicants():
    """Postulantes con su identidad declarada y el resultado de su última revisión (si la hay)."""
    registry = _registry()
    reviewed = {c: u["persona"] for c, u in _latest().items()}
    return [{"carpeta": c, "dni": str((registry.get(c) or {}).get("rut") or ""), "registrado": c in registry,
             "veredicto": reviewed.get(c, {}).get("veredicto"), "aptitud": reviewed.get(c, {}).get("aptitud")}
            for c in _applicant_folders()]


@app.post("/api/revisar")
def review(review_request: ReviewRequest):
    """Encola una revisión de las carpetas pedidas que existen."""
    existing = set(_applicant_folders())
    carpetas = sorted(existing) if review_request.todos else [c for c in review_request.carpetas if c in existing]
    if not carpetas:
        raise HTTPException(400, "No hay postulantes válidos para revisar")
    t = {"id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
         "fecha": datetime.now().isoformat(timespec="seconds"), "tipo": "todos" if review_request.todos else "seleccion",
         "carpetas": carpetas, "segundo_lector": review_request.segundo_lector, "forzar": review_request.forzar,
         "estado": "en_cola", "fase": "en cola", "hechos": 0, "total": 0}
    with _lock:
        _jobs[t["id"]] = t
    _save_job(t)
    _queue.put(t)
    return {"id": t["id"], "postulantes": len(carpetas)}


@app.post("/api/trabajos/{tid}/cancelar")
def cancel(tid: str):
    """Pide cancelar una revisión activa (se detiene en el próximo punto seguro)."""
    t = _jobs.get(tid)
    if not t or t["estado"] in ("terminado", "error", "cancelado"):
        raise HTTPException(404, "No hay una revisión activa con ese id")
    t["cancelar"] = True
    return {"ok": True}


@app.get("/api/estado")
def state():
    """GPU, memoria, revisión en curso con su progreso, cuántas esperan y la última terminada."""
    running = [t for t in _jobs.values() if t["estado"] in ("en_cola", "esperando_recursos", "procesando")]
    current = next((t for t in running if t["estado"] != "en_cola"), running[0] if running else None)
    return {
        "gpu": R.gpu(), "memoria": R.memory(),
        "actual": None if not current else {k: current.get(k) for k in (
            "id", "tipo", "estado", "fase", "hechos", "total", "mensaje", "desde_cache")}
        | {"postulantes": len(current["carpetas"]),
           "transcurrido_s": round(time.time() - current["_t0"], 1) if current.get("_t0") else 0,
           "inicio_proceso_s": round(time.time() - current["_t_proceso"], 1) if current.get("_t_proceso") else None},
        "en_cola": max(0, len(running) - 1),
        "ultimo_terminado": max((t["id"] for t in _jobs.values() if t["estado"] == "terminado"), default=None),
    }


@app.get("/api/trabajos")
def jobs():
    """Últimas 50 revisiones (sin la lista de carpetas)."""
    with _db() as conn:
        rows = conn.execute("SELECT * FROM trabajos ORDER BY fecha DESC LIMIT 50").fetchall()
    return [dict(f) | {"carpetas": None} for f in rows]


@app.get("/api/resumen")
def summary(trabajo: str = "acumulado"):
    """Resumen acumulado o de una revisión: dictámenes, aptitud, conteos por tipo y formato, y problemas por
    persona."""
    if trabajo == "acumulado":
        latest = _latest()
    else:
        with _db() as conn:
            row = conn.execute("SELECT carpetas FROM trabajos WHERE id=?", (trabajo,)).fetchone()
        if not row:
            raise HTTPException(404, "Revisión no encontrada")
        latest = _latest(json.loads(row["carpetas"]))

    people = [u["persona"] for u in latest.values()]
    by_type, by_format = defaultdict(Counter), defaultdict(Counter)
    rows = []
    for c, u in sorted(latest.items()):
        p = u["persona"]
        problems = []
        for d in u["documentos"]:
            kind, fmt = _type_label(d["tipo_documento"]), FORMAT_LABEL.get(d["formato"], d["formato"])
            by_type[kind][d["veredicto"]] += 1
            by_type[kind]["total"] += 1
            by_format[fmt][d["veredicto"]] += 1
            by_format[fmt]["total"] += 1
            if d["veredicto"] != "AUTÉNTICO":
                problems.append({"archivo": d["archivo"], "tipo": kind, "formato": fmt, "veredicto": d["veredicto"]})
        rows.append({"carpeta": c, "dni": p["dni"], "veredicto": p["veredicto"], "aptitud": p["aptitud"],
                      "documentos": p["documentos"], "falsificados": p["falsificados"], "observados": p["observados"],
                      "desactualizados": p.get("desactualizados", 0), "problemas": problems, "fecha": u["fecha"]})
    return {
        "alcance": trabajo, "revisados": len(people),
        "veredictos": Counter(p["veredicto"] for p in people),
        "aptitud": Counter(p["aptitud"] for p in people),
        "por_tipo": by_type, "por_formato": by_format, "personas": rows,
    }


@app.get("/api/persona/{folder}")
def person(folder: str):
    """Detalle de la última revisión de un postulante. La miniatura de cada documento es su primera página, sin
    las marcas "(2)" ni "[documento 2]" del nombre."""
    u = _latest([folder]).get(folder)
    if not u:
        raise HTTPException(404, "Ese postulante aún no fue revisado")
    docs = []
    for d in u["documentos"]:
        show = re.sub(r" \(\d+\)$", "", d["archivo"].split(" + ")[0]).split(" [")[0]
        docs.append({
            "archivo": d["archivo"], "archivo_ver": show, "tipo": _type_label(d["tipo_documento"]),
            "formato": FORMAT_LABEL.get(d["formato"], d["formato"]), "veredicto": d["veredicto"],
            "motivos": d["motivos"], "misma_instancia": d.get("misma_instancia"),
            "archivo_oficial": d.get("archivo_oficial", ""), "emision": d.get("emision", ""),
            "filas_hoja_vida": d.get("filas_hoja_vida"),
            "campos": [c for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")],
            "senales": [s for s in d["senales"] if s["severidad"] != "info"],
        })
    return {"persona": u["persona"], "fecha": u["fecha"], "documentos": docs}


def _safe_path(ruta):
    """Ruta resuelta solo si es un archivo dentro de postulant_files o webscrap_files; si no, 404."""
    p = os.path.realpath(ruta)
    if not p.startswith(ALLOWED_ROOTS) or not os.path.isfile(p):
        raise HTTPException(404, "Archivo no disponible")
    return Path(p)


@app.get("/api/miniatura")
def thumbnail(ruta: str, pagina: int = 0):
    """PNG de una página de PDF (70 DPI) o de una imagen reducida."""
    p = _safe_path(ruta)
    if p.suffix.lower() == ".pdf":
        with pymupdf.open(p) as doc:
            png = doc[min(pagina, doc.page_count - 1)].get_pixmap(dpi=70).tobytes("png")
    else:
        im = Image.open(p, formats=L.IMAGE_FORMATS).convert("RGB")
        im.thumbnail((700, 900))
        import io
        buf = io.BytesIO()
        im.save(buf, "PNG")
        png = buf.getvalue()
    return Response(png, media_type="image/png")


@app.get("/api/archivo_postulante/{folder}/{file}")
def applicant_file(folder: str, file: str):
    """Miniatura de un archivo del postulante; "#pN" en el nombre indica la página."""
    name, _, pg_ = file.partition("#p")
    p = _safe_path(C.APPLICANTS_DIR / folder / name)
    return thumbnail(str(p), int(pg_) - 1 if pg_ else 0)


@app.get("/api/oficial/{folder}/{file}")
def official(folder: str, file: str):
    """Dato oficial (JSON del webscraping) con que se comparó el documento."""
    p = _safe_path(C.OFFICIAL_DATA_DIR / folder / file)
    if p.suffix.lower() != ".json":
        raise HTTPException(404, "Archivo no disponible")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/excel/{trabajo}")
def excel(trabajo: str):
    """Excel acumulado, o el de una revisión terminada (se genera la primera vez que se descarga)."""
    if trabajo == "acumulado":
        latest = _latest()
        if not latest:
            raise HTTPException(404, "Aún no hay revisiones")
        ruta = _excel(VIEWER_DIR / "acumulado", latest, {"alcance": "acumulado", "postulantes": len(latest),
                                                        "fecha": datetime.now().isoformat(timespec="seconds")})
        return FileResponse(ruta, filename=f"verificacion_acumulado_{datetime.now():%Y%m%d-%H%M}.xlsx")
    with _db() as conn:
        row = conn.execute("SELECT * FROM trabajos WHERE id=?", (trabajo,)).fetchone()
    if not row or row["estado"] != "terminado":
        raise HTTPException(404, "Esa revisión no terminó: no tiene Excel")
    job_id = row["id"]
    ruta = Path(row["excel"]) if row["excel"] else None
    if not ruta or not ruta.exists():
        latest = _latest(json.loads(row["carpetas"]))
        ruta = _excel(VIEWER_DIR / "jobs" / job_id, latest,
                      {"revision": job_id, "fecha": row["fecha"], "tipo": row["tipo"], "postulantes": len(latest),
                       "desde_cache": row["desde_cache"]})
        with _db() as conn:
            conn.execute("UPDATE trabajos SET excel=? WHERE id=?", (str(ruta), job_id))
    return FileResponse(ruta, filename=f"verificacion_{job_id}.xlsx")


def main(port=8000, open_browser=True):
    """Arranca el visor en 127.0.0.1 y, si `abrir`, abre el navegador."""
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="no abrir el navegador automáticamente")
    a = ap.parse_args()
    main(a.port, open_browser=not a.no_browser)
