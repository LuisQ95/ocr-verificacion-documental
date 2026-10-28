"""Visor local de verificación de postulantes (solo accesible desde esta PC: 127.0.0.1).

  - Buscar postulantes (nombre o RUT) y revisar uno, varios o todos.
  - Revisiones en segundo plano, una a la vez, con chequeo de GPU/RAM (si no hay recursos, espera)
    y el mismo candado que el pipeline de consola (nunca corren dos verificaciones juntas).
  - Caché: un postulante ya revisado cuyos archivos no cambiaron no se vuelve a procesar.
  - Resumen por postulante, por tipo de documento y por formato; detalle con campos y señales.
  - Excel por revisión y Excel acumulado.

Uso: python -m visor.servidor

Los datos del visor (base y Excel) van a resultados/visor, o a la carpeta de la variable de entorno VISOR_DATOS
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

from verificacion import config as C
from verificacion import recursos as R
from verificacion.chile import pipeline_cl as PL
from verificacion.reporte import escribir_reporte

pillow_heif.register_heif_opener()

DIR_VISOR = Path(os.environ.get("VISOR_DATOS", C.DIR_RESULTADOS / "visor"))
DB = DIR_VISOR / "visor.db"
PAGINA = Path(__file__).parent / "static" / "index.html"
RAICES_PERMITIDAS = [C.DIR_POSTULANTES.resolve(), C.DIR_WEB.resolve()]

@asynccontextmanager
async def _ciclo_de_vida(_app):
    """Al arrancar: prepara la base y lanza el hilo que atiende la cola de revisiones."""
    _iniciar_db()
    threading.Thread(target=_trabajador, daemon=True).start()
    yield


app = FastAPI(title="Verificación de postulantes", lifespan=_ciclo_de_vida)
_cola = queue.Queue()
_trabajos = {}
"""id -> estado en memoria (para el progreso en vivo)"""
_bloqueo = threading.Lock()


def _db():
    """Conexión a la base del visor (filas accesibles por nombre de columna)."""
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def _iniciar_db():
    """Crea las tablas si faltan y marca como interrumpidas las revisiones que quedaron a medias al cerrarse el
    visor."""
    DIR_VISOR.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS trabajos (id TEXT PRIMARY KEY, fecha TEXT, tipo TEXT, carpetas TEXT,
                estado TEXT, mensaje TEXT, n_postulantes INTEGER, procesados INTEGER, desde_cache INTEGER,
                segundo_lector INTEGER, excel TEXT, duracion_s REAL);
            CREATE TABLE IF NOT EXISTS ultimas (carpeta TEXT PRIMARY KEY, trabajo_id TEXT, fecha TEXT, firma TEXT,
                persona TEXT, documentos TEXT);
        """)
        con.execute("UPDATE trabajos SET estado='interrumpido', mensaje='el visor se cerró durante la revisión' "
                    "WHERE estado IN ('en_cola','esperando_recursos','procesando')")


def _guardar_trabajo(t):
    """Guarda (o actualiza) el registro de una revisión."""
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO trabajos VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (t["id"], t["fecha"], t["tipo"], json.dumps(t["carpetas"], ensure_ascii=False), t["estado"],
                     t.get("mensaje", ""), len(t["carpetas"]), t.get("procesados", 0), t.get("desde_cache", 0),
                     int(t["segundo_lector"]), t.get("excel", ""), t.get("duracion_s")))


def _registro():
    """Registro de postulantes (RUT declarado y cargo por carpeta)."""
    return PL.leer_registro(C.REGISTRO_POSTULANTES)


def _carpetas_postulantes():
    """Carpetas de postulant_files, en orden."""
    return sorted(d.name for d in C.DIR_POSTULANTES.iterdir() if d.is_dir())


def _firma(carpeta, segundo_lector, huella):
    """Si cambian los archivos (del postulante o los oficiales), el modelo o las reglas, cambia la firma."""
    h = hashlib.sha256(f"{huella}|{segundo_lector}".encode())
    for base in (C.DIR_POSTULANTES, C.DIR_WEB):
        d = base / carpeta
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if f.is_file():
                st = f.stat()
                h.update(f"{base.name}/{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def _sin_valores(doc):
    """Documento sin los valores completos leídos y oficiales (no se guardan en la base del visor)."""
    return {k: v for k, v in doc.items() if k not in ("valores_oficiales", "valores_postulante")}


TIPOS = {"hoja_vida_p1": "Hoja de Vida del Conductor", "hoja_vida_p2": "Hoja de Vida del Conductor",
         "ameco_rev02": "Certificado AMECO (Rev.02)", "ameco_rev03": "Certificado AMECO (Rev.03)",
         "desconocido": "No identificado"}
"""Nombre visible de cada tipo; "No identificado" es un archivo ajeno (NO RECONOCIDO) o ilegible (NO LEGIBLE)."""


def _etiqueta_tipo(tipo):
    """Nombre visible de un tipo de documento."""
    return TIPOS.get(tipo or "desconocido", (tipo or "desconocido").replace("_", " ").capitalize())


ETIQUETA_FORMATO = {"PDF_NATIVO": "PDF nativo", "ESCANEADO": "Escaneado", "IMAGEN": "Foto / imagen"}


def _ultimas(carpetas=None):
    """Última revisión de cada postulante (de todos, o solo de `carpetas`)."""
    with _db() as con:
        filas = con.execute("SELECT * FROM ultimas").fetchall()
    salida = {f["carpeta"]: {"persona": json.loads(f["persona"]), "documentos": json.loads(f["documentos"]),
                             "fecha": f["fecha"], "trabajo_id": f["trabajo_id"]} for f in filas}
    return {c: salida[c] for c in carpetas if c in salida} if carpetas is not None else salida


def _excel(destino, ultimas, meta):
    """Escribe el reporte Excel de esas revisiones en `destino` y devuelve su ruta."""
    personas = [u["persona"] for u in ultimas.values()]
    documentos = [d for u in ultimas.values() for d in u["documentos"]]
    _, xlsx = escribir_reporte(personas, documentos, meta, destino)
    return xlsx


def _trabajador():
    """Hilo que atiende las revisiones de a una; el visor sigue funcionando aunque falle una revisión."""
    while True:
        t = _cola.get()
        try:
            _procesar(t)
        except InterruptedError as e:
            t.update(estado="cancelado", mensaje=str(e))
        except Exception as e:
            t.update(estado="error", mensaje=f"{type(e).__name__}: {e}")
        finally:
            t["duracion_s"] = round(time.time() - t["_t0"], 1)
            _guardar_trabajo(t)


def _procesar(t):
    """Revisa los postulantes cuya firma cambió (o todos si se fuerza).

    Espera recursos y el candado de otra verificación (de consola o del visor), reintentando cada 30 s. El Excel de
    la revisión se genera al descargarlo: así la página se entera del fin sin esperar.
    """
    t["_t0"] = time.time()
    huella = PL.huella_version()
    firmas = {c: _firma(c, t["segundo_lector"], huella) for c in t["carpetas"]}
    with _db() as con:
        guardadas = {f["carpeta"]: f["firma"] for f in con.execute("SELECT carpeta, firma FROM ultimas")}
    pendientes = [c for c in t["carpetas"] if t["forzar"] or guardadas.get(c) != firmas[c]]
    t["desde_cache"] = len(t["carpetas"]) - len(pendientes)

    if pendientes:
        while True:
            if t.get("cancelar"):
                raise InterruptedError("revisión cancelada")
            try:
                t.update(estado="esperando_recursos", fase="revisando GPU y memoria")
                with R.Candado("pipeline"):
                    R.verificar(vram_gb=C.PRESUPUESTO_VRAM_OCR_GB, ram_gb=C.PRESUPUESTO_RAM_GB, ventana_s=10,
                                log=lambda m: None)
                    t.update(estado="procesando", mensaje="", _t_proceso=time.time())
                    _ejecutar(t, pendientes, firmas)
                break
            except R.RecursosInsuficientes as e:
                t.update(mensaje=f"{e}. Se reintenta en 30 s.")
                for _ in range(30):
                    if t.get("cancelar"):
                        raise InterruptedError("revisión cancelada") from e
                    time.sleep(1)

    t.update(estado="terminado", fase="terminado", procesados=len(pendientes), mensaje="", huella=huella)


def _ejecutar(t, pendientes, firmas):
    """Verifica los postulantes pendientes y guarda su última revisión con su firma."""
    from verificacion.chile.lote_cl import verificar_lote

    def progreso(fase, hechos, total):
        """Actualiza el progreso visible de la revisión."""
        t.update(fase=fase, hechos=hechos, total=total)

    personas, documentos, no_admitidos, _ = verificar_lote(
        pendientes, _registro(), segundo_lector=t["segundo_lector"], progreso=progreso,
        log=lambda m: t.update(mensaje=m), cancelado=lambda: t.get("cancelar", False))
    por_carpeta = defaultdict(list)
    for d in documentos:
        por_carpeta[d["carpeta"]].append(_sin_valores(d))
    ahora = datetime.now().isoformat(timespec="seconds")
    with _db() as con:
        for p in personas:
            con.execute("INSERT OR REPLACE INTO ultimas VALUES (?,?,?,?,?,?)",
                        (p["carpeta"], t["id"], ahora, firmas.get(p["carpeta"], ""),
                         json.dumps(p, ensure_ascii=False, default=str),
                         json.dumps(por_carpeta[p["carpeta"]], ensure_ascii=False, default=str)))
    t["no_admitidos"] = no_admitidos


class PedidoRevision(BaseModel):
    """Pedido de revisión: carpetas elegidas o todas; `forzar` ignora la caché."""

    carpetas: list[str] = []
    todos: bool = False
    segundo_lector: bool = True
    forzar: bool = False


@app.get("/", response_class=HTMLResponse)
def inicio():
    """Página del visor."""
    return PAGINA.read_text(encoding="utf-8")


@app.get("/api/postulantes")
def postulantes():
    """Postulantes con su RUT declarado y el resultado de su última revisión (si la hay)."""
    registro = _registro()
    revisados = {c: u["persona"] for c, u in _ultimas().items()}
    return [{"carpeta": c, "dni": str((registro.get(c) or {}).get("rut") or ""), "registrado": c in registro,
             "veredicto": revisados.get(c, {}).get("veredicto"), "aptitud": revisados.get(c, {}).get("aptitud")}
            for c in _carpetas_postulantes()]


@app.post("/api/revisar")
def revisar(pedido: PedidoRevision):
    """Encola una revisión de las carpetas pedidas que existen."""
    existentes = set(_carpetas_postulantes())
    carpetas = sorted(existentes) if pedido.todos else [c for c in pedido.carpetas if c in existentes]
    if not carpetas:
        raise HTTPException(400, "No hay postulantes válidos para revisar")
    t = {"id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
         "fecha": datetime.now().isoformat(timespec="seconds"), "tipo": "todos" if pedido.todos else "seleccion",
         "carpetas": carpetas, "segundo_lector": pedido.segundo_lector, "forzar": pedido.forzar,
         "estado": "en_cola", "fase": "en cola", "hechos": 0, "total": 0}
    with _bloqueo:
        _trabajos[t["id"]] = t
    _guardar_trabajo(t)
    _cola.put(t)
    return {"id": t["id"], "postulantes": len(carpetas)}


@app.post("/api/trabajos/{tid}/cancelar")
def cancelar(tid: str):
    """Pide cancelar una revisión activa (se detiene en el próximo punto seguro)."""
    t = _trabajos.get(tid)
    if not t or t["estado"] in ("terminado", "error", "cancelado"):
        raise HTTPException(404, "No hay una revisión activa con ese id")
    t["cancelar"] = True
    return {"ok": True}


@app.get("/api/estado")
def estado():
    """GPU, memoria, revisión en curso con su progreso, cuántas esperan y la última terminada."""
    activos = [t for t in _trabajos.values() if t["estado"] in ("en_cola", "esperando_recursos", "procesando")]
    actual = next((t for t in activos if t["estado"] != "en_cola"), activos[0] if activos else None)
    return {
        "gpu": R.gpu(), "memoria": R.memoria(),
        "actual": None if not actual else {k: actual.get(k) for k in (
            "id", "tipo", "estado", "fase", "hechos", "total", "mensaje", "desde_cache")}
        | {"postulantes": len(actual["carpetas"]),
           "transcurrido_s": round(time.time() - actual["_t0"], 1) if actual.get("_t0") else 0,
           "inicio_proceso_s": round(time.time() - actual["_t_proceso"], 1) if actual.get("_t_proceso") else None},
        "en_cola": max(0, len(activos) - 1),
        "ultimo_terminado": max((t["id"] for t in _trabajos.values() if t["estado"] == "terminado"), default=None),
    }


@app.get("/api/trabajos")
def trabajos():
    """Últimas 50 revisiones (sin la lista de carpetas)."""
    with _db() as con:
        filas = con.execute("SELECT * FROM trabajos ORDER BY fecha DESC LIMIT 50").fetchall()
    return [dict(f) | {"carpetas": None} for f in filas]


@app.get("/api/resumen")
def resumen(trabajo: str = "acumulado"):
    """Resumen acumulado o de una revisión: dictámenes, aptitud, conteos por tipo y formato, y problemas por
    persona."""
    if trabajo == "acumulado":
        ultimas = _ultimas()
    else:
        with _db() as con:
            fila = con.execute("SELECT carpetas FROM trabajos WHERE id=?", (trabajo,)).fetchone()
        if not fila:
            raise HTTPException(404, "Revisión no encontrada")
        ultimas = _ultimas(json.loads(fila["carpetas"]))

    personas = [u["persona"] for u in ultimas.values()]
    por_tipo, por_formato = defaultdict(Counter), defaultdict(Counter)
    filas = []
    for c, u in sorted(ultimas.items()):
        p = u["persona"]
        problemas = []
        for d in u["documentos"]:
            tipo, formato = _etiqueta_tipo(d["tipo_documento"]), ETIQUETA_FORMATO.get(d["formato"], d["formato"])
            por_tipo[tipo][d["veredicto"]] += 1
            por_tipo[tipo]["total"] += 1
            por_formato[formato][d["veredicto"]] += 1
            por_formato[formato]["total"] += 1
            if d["veredicto"] != "AUTÉNTICO":
                problemas.append({"archivo": d["archivo"], "tipo": tipo, "formato": formato, "veredicto": d["veredicto"]})
        filas.append({"carpeta": c, "dni": p["dni"], "veredicto": p["veredicto"], "aptitud": p["aptitud"],
                      "documentos": p["documentos"], "falsificados": p["falsificados"], "observados": p["observados"],
                      "desactualizados": p.get("desactualizados", 0), "problemas": problemas, "fecha": u["fecha"]})
    return {
        "alcance": trabajo, "revisados": len(personas),
        "veredictos": Counter(p["veredicto"] for p in personas),
        "aptitud": Counter(p["aptitud"] for p in personas),
        "por_tipo": por_tipo, "por_formato": por_formato, "personas": filas,
    }


@app.get("/api/persona/{carpeta}")
def persona(carpeta: str):
    """Detalle de la última revisión de un postulante. La miniatura de cada documento es su primera página, sin
    las marcas "(2)" ni "[documento 2]" del nombre."""
    u = _ultimas([carpeta]).get(carpeta)
    if not u:
        raise HTTPException(404, "Ese postulante aún no fue revisado")
    docs = []
    for d in u["documentos"]:
        ver = re.sub(r" \(\d+\)$", "", d["archivo"].split(" + ")[0]).split(" [")[0]
        docs.append({
            "archivo": d["archivo"], "archivo_ver": ver, "tipo": _etiqueta_tipo(d["tipo_documento"]),
            "formato": ETIQUETA_FORMATO.get(d["formato"], d["formato"]), "veredicto": d["veredicto"],
            "motivos": d["motivos"], "misma_instancia": d.get("misma_instancia"),
            "archivo_oficial": d.get("archivo_oficial", ""), "emision": d.get("emision", ""),
            "filas_hoja_vida": d.get("filas_hoja_vida"),
            "campos": [c for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")],
            "senales": [s for s in d["senales"] if s["severidad"] != "info"],
        })
    return {"persona": u["persona"], "fecha": u["fecha"], "documentos": docs}


def _ruta_segura(ruta):
    """Ruta resuelta solo si es un archivo dentro de postulant_files o webscrap_files; si no, 404."""
    p = Path(ruta).resolve()
    if not any(p.is_relative_to(r) for r in RAICES_PERMITIDAS) or not p.is_file():
        raise HTTPException(404, "Archivo no disponible")
    return p


@app.get("/api/miniatura")
def miniatura(ruta: str, pagina: int = 0):
    """PNG de una página de PDF (70 DPI) o de una imagen reducida."""
    p = _ruta_segura(ruta)
    if p.suffix.lower() == ".pdf":
        with pymupdf.open(p) as doc:
            png = doc[min(pagina, doc.page_count - 1)].get_pixmap(dpi=70).tobytes("png")
    else:
        im = Image.open(p).convert("RGB")
        im.thumbnail((700, 900))
        import io
        buf = io.BytesIO()
        im.save(buf, "PNG")
        png = buf.getvalue()
    return Response(png, media_type="image/png")


@app.get("/api/archivo_postulante/{carpeta}/{archivo}")
def archivo_postulante(carpeta: str, archivo: str):
    """Miniatura de un archivo del postulante; "#pN" en el nombre indica la página."""
    nombre, _, pag = archivo.partition("#p")
    p = _ruta_segura(C.DIR_POSTULANTES / carpeta / nombre)
    return miniatura(str(p), int(pag) - 1 if pag else 0)


@app.get("/api/oficial/{carpeta}/{archivo}")
def oficial(carpeta: str, archivo: str):
    """Dato oficial (JSON del webscraping) con que se comparó el documento."""
    p = _ruta_segura(C.DIR_WEB / carpeta / archivo)
    if p.suffix.lower() != ".json":
        raise HTTPException(404, "Archivo no disponible")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/excel/{trabajo}")
def excel(trabajo: str):
    """Excel acumulado, o el de una revisión terminada (se genera la primera vez que se descarga)."""
    if trabajo == "acumulado":
        ultimas = _ultimas()
        if not ultimas:
            raise HTTPException(404, "Aún no hay revisiones")
        ruta = _excel(DIR_VISOR / "acumulado", ultimas, {"alcance": "acumulado", "postulantes": len(ultimas),
                                                        "fecha": datetime.now().isoformat(timespec="seconds")})
        return FileResponse(ruta, filename=f"verificacion_acumulado_{datetime.now():%Y%m%d-%H%M}.xlsx")
    with _db() as con:
        fila = con.execute("SELECT * FROM trabajos WHERE id=?", (trabajo,)).fetchone()
    if not fila or fila["estado"] != "terminado":
        raise HTTPException(404, "Esa revisión no terminó: no tiene Excel")
    ruta = Path(fila["excel"]) if fila["excel"] else None
    if not ruta or not ruta.exists():
        ultimas = _ultimas(json.loads(fila["carpetas"]))
        ruta = _excel(DIR_VISOR / "trabajos" / trabajo, ultimas,
                      {"revision": trabajo, "fecha": fila["fecha"], "tipo": fila["tipo"], "postulantes": len(ultimas),
                       "desde_cache": fila["desde_cache"]})
        with _db() as con:
            con.execute("UPDATE trabajos SET excel=? WHERE id=?", (str(ruta), trabajo))
    return FileResponse(ruta, filename=f"verificacion_{trabajo}.xlsx")


def main(puerto=8000, abrir=True):
    """Arranca el visor en 127.0.0.1 y, si `abrir`, abre el navegador."""
    if abrir:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{puerto}")).start()
    uvicorn.run(app, host="127.0.0.1", port=puerto, log_level="warning")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8000)
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador automáticamente")
    a = ap.parse_args()
    main(a.puerto, abrir=not a.sin_navegador)
