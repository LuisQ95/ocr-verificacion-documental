"""Visor de PRUEBAS: un evaluador externo sube los documentos de un postulante y sus JSON oficiales y recibe el
dictamen. Es una aplicación aparte del visor principal: no ve postulant_files, webscrap_files ni resultados.

Privacidad:
  - Los archivos llegan dentro del cuerpo de la petición (JSON con base64) y se procesan en memoria: no se
    escriben en disco (no hay subida multipart, que guarda en temporales los archivos de más de 1 MB).
  - Los resultados quedan solo en la memoria de este proceso, por sesión; se borran con "Borrar resultados",
    al cerrar sesión, tras 8 h sin uso o al cerrar el visor. No se registra el contenido en logs.
Acceso:
  - Solo desde esta PC o desde las IP de evaluation_viewer/access.json (la PC del evaluador en Tailscale), y con
    contraseña (python -m evaluation_viewer.configurar).

Uso: python -m evaluation_viewer.servidor
"""

import base64
import binascii
import json
import queue
import secrets
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import config as C
from verification import resources as R
from verification.profiles.operator_certification import extraction as EX
from verification import in_memory as M
from verification.profiles.operator_certification import normalize as N
from verification.profiles.operator_certification import requirements as Q

from . import access as A

PAGE = Path(__file__).parent / "static" / "index.html"
COOKIE = "sesion_pruebas"
SESSION_DURATION_S = 8 * 3600
MAX_APPLICANT_FILES, MAX_FILE_MB = 10, 25
MAX_OFFICIAL_FILES, MAX_JSON_MB = 10, 2
MAX_BODY = 300 * 1024 * 1024
"""Tamaño máximo de la petición: el base64 de todos los archivos admitidos, con margen."""
MAX_SHORT_BODY = 64 * 1024
"""Tamaño máximo de las peticiones sin archivos (inicio de sesión, validación de la identidad)."""
MAX_RESULTS_PER_SESSION = 50
MAX_ATTEMPTS, LOCK_TIMEOUT_S = 5, 15 * 60

app = FastAPI(title="Visor de pruebas", docs_url=None, redoc_url=None, openapi_url=None)
_cfg = A.load()
_cfg_version = A.version(A.CONFIG_PATH)
"""Versión (fecha de modificación) de access.json leída: si el archivo cambia, se relee sin reiniciar el visor."""
_sessions = {}
"""token -> {"ultimo": t, "resultados": {id: resultado}}"""
_jobs = {}
"""id -> estado del trabajo (sin los archivos una vez procesado)"""
_failures = {}
"""ip -> (intentos fallidos, bloqueado_hasta)"""
_queue = queue.Queue()
_lock = threading.Lock()


def _config():
    """Configuración de acceso vigente. Se relee cuando access.json cambia: agregar o quitar una IP, o cambiar la
    contraseña, rige desde la petición siguiente."""
    global _cfg, _cfg_version
    v = A.version(A.CONFIG_PATH)
    if v != _cfg_version:
        _cfg, _cfg_version = A.load(A.CONFIG_PATH), v
    return _cfg


@app.middleware("http")
async def _only_allowed_ips(request: Request, call_next):
    """Rechaza (403) toda petición de una IP no permitida y agrega cabeceras de seguridad a las respuestas."""
    ip = request.client.host if request.client else ""
    if not A.ip_allowed(ip, _config()):
        return Response("Acceso no permitido", status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _clean_sessions():
    """Borra las sesiones vencidas y los trabajos terminados que ya no pertenecen a una sesión viva."""
    now = time.time()
    for t in [t for t, s in _sessions.items() if now - s["ultimo"] > SESSION_DURATION_S]:
        del _sessions[t]
    alive = list(_sessions.values())
    for tid in [tid for tid, t in _jobs.items() if t["estado"] in ("terminado", "error")
                and (now - t["creado"] > SESSION_DURATION_S or not any(t["sesion"] is s for s in alive))]:
        del _jobs[tid]


def _session(request: Request):
    """Sesión de la cookie (renueva su último uso) o 401 si no hay sesión válida."""
    with _lock:
        _clean_sessions()
        s = _sessions.get(request.cookies.get(COOKIE, ""))
        if not s:
            raise HTTPException(401, "Inicie sesión")
        s["ultimo"] = time.time()
        return s


@app.get("/", response_class=HTMLResponse)
def home():
    """Página del visor."""
    return PAGE.read_text(encoding="utf-8")


@app.post("/api/login")
async def login(request: Request):
    """Abre sesión con la contraseña (cookie HttpOnly, SameSite estricta). Tras MAX_ATTEMPTS fallidos, la IP queda
    bloqueada LOCK_TIMEOUT_S segundos."""
    ip = request.client.host
    attempts, until = _failures.get(ip, (0, 0))
    if until > time.time():
        raise HTTPException(429, f"Demasiados intentos. Espere {int((until - time.time()) / 60) + 1} min.")
    data = await _json_body(request, MAX_SHORT_BODY)
    if not A.correct_key(str(data.get("clave", "")), _config().get("clave")):
        attempts += 1
        _failures[ip] = (0, time.time() + LOCK_TIMEOUT_S) if attempts >= MAX_ATTEMPTS else (attempts, 0)
        raise HTTPException(401, "Contraseña incorrecta")
    _failures.pop(ip, None)
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = {"ultimo": time.time(), "resultados": {}}
    r = JSONResponse({"ok": True})
    r.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=SESSION_DURATION_S)
    return r


@app.post("/api/logout")
def logout(request: Request):
    """Cierra la sesión: sus resultados se descartan."""
    with _lock:
        _sessions.pop(request.cookies.get(COOKIE, ""), None)
    r = JSONResponse({"ok": True})
    r.delete_cookie(COOKIE)
    return r


@app.get("/api/sesion")
def session(request: Request):
    """Datos para el formulario: cargos configurados, cargo por defecto y límites de carga."""
    _session(request)
    req = Q.load()
    return {"cargos": [c.upper() for c in req["puestos"]], "cargo_por_defecto": req["puesto_por_defecto"].upper(),
            "limites": {"archivos": MAX_APPLICANT_FILES, "mb_archivo": MAX_FILE_MB,
                        "jsons": MAX_OFFICIAL_FILES, "mb_json": MAX_JSON_MB}}


def _decode(entries, upper_limit, mb, what):
    """[(nombre, bytes)] desde la lista de archivos en base64, validando cantidad, tamaño y contenido."""
    if not isinstance(entries, list) or len(entries) > upper_limit:
        raise HTTPException(400, f"Se admiten como máximo {upper_limit} {what}.")
    out = []
    for x in entries:
        name = Path(str(x.get("nombre", "archivo"))).name[:150] or "archivo"
        try:
            data = base64.b64decode(x.get("base64", ""), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, f"No se pudo recibir '{name}'.") from exc
        if len(data) > mb * 1024 * 1024:
            raise HTTPException(400, f"'{name}' supera {mb} MB.")
        if not data:
            raise HTTPException(400, f"'{name}' está vacío.")
        out.append((name, data))
    return out


async def _json_body(request: Request, upper_limit=MAX_BODY):
    """Cuerpo JSON (un objeto) leído en memoria por partes, cortando con 413 apenas supera `maximo` bytes."""
    try:
        length = int(request.headers.get("content-length") or 0)
    except ValueError as exc:
        raise HTTPException(400, "Petición inválida.") from exc
    if length > upper_limit:
        raise HTTPException(413, "Los archivos superan el tamaño permitido.")
    body = bytearray()
    async for part in request.stream():
        body += part
        if len(body) > upper_limit:
            raise HTTPException(413, "Los archivos superan el tamaño permitido.")
    try:
        data = json.loads(bytes(body))
    except ValueError as exc:
        raise HTTPException(400, "Petición inválida.") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "Petición inválida.")
    return data


@app.post("/api/validar")
async def validate(request: Request):
    """Validación de la identidad mientras se escribe (sin archivos)."""
    _session(request)
    d = await _json_body(request, MAX_SHORT_BODY)
    _, errors, warns = M.validate_identity(d.get("nombres", ""), d.get("apellidos", ""), d.get("rut", ""),
                                          d.get("fecha_nacimiento", ""), d.get("cargo", ""))
    return {"errores": errors, "advertencias": warns}


@app.post("/api/verificar")
async def verify(request: Request):
    """Valida la identidad y los archivos y encola la verificación; devuelve el id del trabajo."""
    s = _session(request)
    d = await _json_body(request)
    identity, errors, warns = M.validate_identity(d.get("nombres", ""), d.get("apellidos", ""), d.get("rut", ""),
                                              d.get("fecha_nacimiento", ""), d.get("cargo", ""))
    files = _decode(d.get("postulante", []), MAX_APPLICANT_FILES, MAX_FILE_MB, "documentos del postulante")
    jsons = _decode(d.get("oficiales", []), MAX_OFFICIAL_FILES, MAX_JSON_MB, "JSON oficiales")
    if not files:
        errors.append("Suba al menos un documento del postulante.")
    if errors:
        raise HTTPException(400, " ".join(errors))
    del d
    t = {"id": uuid.uuid4().hex[:12], "sesion": s, "estado": "en_cola", "fase": "en cola", "hechos": 0, "total": 0,
         "mensaje": "", "ident": identity, "adv": warns, "archivos": files, "jsons": jsons, "creado": time.time()}
    with _lock:
        _jobs[t["id"]] = t
    _queue.put(t)
    return {"id": t["id"]}


@app.get("/api/trabajo/{tid}")
def job(tid: str, request: Request):
    """Estado y progreso de un trabajo de la sesión, con cuántos esperan antes en la cola."""
    s = _session(request)
    t = _jobs.get(tid)
    if not t or t["sesion"] is not s:
        raise HTTPException(404, "No encontrado")
    pos = sum(1 for x in list(_queue.queue) if x["creado"] < t["creado"])
    return {k: t.get(k) for k in ("estado", "fase", "hechos", "total", "mensaje", "resultado_id")} | {"en_cola_antes": pos}


def _process(t):
    """Verifica un trabajo en memoria. Hay una verificación a la vez en el equipo (también respeta las de consola y
    las del visor principal); si la GPU está ocupada, las imágenes se leen con CPU."""
    def progress(phase, done, total):
        """Actualiza el progreso visible del trabajo."""
        t.update(fase=phase, hechos=done, total=total)

    use_gpu = True
    while True:
        try:
            with R.RunLock("pipeline"):
                t.update(estado="procesando", fase="revisando GPU y memoria")
                try:
                    R.verify(vram_gb=C.OCR_VRAM_BUDGET_GB, ram_gb=C.RAM_BUDGET_GB, window_s=5,
                                log=lambda m: None)
                except R.InsufficientResources:
                    use_gpu = False
                    t["mensaje"] = "GPU ocupada: las imágenes se leen con CPU (más lento)."
                return M.verify_in_memory(t["ident"], t["archivos"], t["jsons"], use_gpu=use_gpu,
                                              progress=progress)
        except R.InsufficientResources:
            t.update(estado="en_cola", fase="esperando a otra verificación en curso en el equipo")
            time.sleep(15)


def _worker():
    """Hilo que atiende la cola. Los documentos se descartan apenas termina cada trabajo; ante un error solo se
    informa su tipo, nunca el contenido."""
    while True:
        t = _queue.get()
        try:
            res = _process(t)
            res["advertencias"] = list(dict.fromkeys(t["adv"] + res["advertencias"]))
            res["id"], res["fecha"] = t["id"], datetime.now().strftime("%d-%m-%Y %H:%M")
            with _lock:
                rs = t["sesion"]["resultados"]
                rs[t["id"]] = res
                while len(rs) > MAX_RESULTS_PER_SESSION:
                    rs.pop(next(iter(rs)))
            t.update(estado="terminado", fase="terminado", resultado_id=t["id"])
        except Exception as e:
            t.update(estado="error", mensaje=f"No se pudo completar la verificación ({type(e).__name__}).")
        finally:
            t.pop("archivos", None)
            t.pop("jsons", None)


threading.Thread(target=_worker, daemon=True).start()


def _name_key(name):
    """Nombre normalizado para detectar homónimos."""
    return N.text_key(N.strip_accents(name))


def _labels(results):
    """Nombre a mostrar: si hay homónimos (mismo nombre, distinto RUT) se agrega la fecha de nacimiento."""
    by_name = {}
    for r in results:
        by_name.setdefault(_name_key(r["identidad"]["nombre_completo"]), set()).add(r["identidad"]["rut"])
    out = {}
    for r in results:
        i = r["identidad"]
        et = i["nombre_completo"]
        if len(by_name[_name_key(et)]) > 1:
            et += f" ({i['fecha_nacimiento']:%d-%m-%Y})" if i["fecha_nacimiento"] else f" (RUT {i['rut']})"
        out[r["id"]] = et
    return out


def _view(r, label):
    """Resultado en el formato que muestra la página (sin campos iguales ni señales informativas)."""
    i = r["identidad"]
    return {
        "id": r["id"], "fecha": r["fecha"], "etiqueta": label, "nombres": i["nombres"], "apellidos": i["apellidos"],
        "rut": i["rut"], "fecha_nacimiento": f"{i['fecha_nacimiento']:%d-%m-%Y}" if i["fecha_nacimiento"] else "",
        "fecha_nacimiento_fuente": i["fecha_nacimiento_fuente"], "cargo": r["persona"]["puesto"],
        "veredicto": r["persona"]["veredicto"], "aptitud": r["persona"]["aptitud"],
        "motivos_aptitud": r["persona"]["motivos_aptitud"], "advertencias": r["advertencias"],
        "esperados": r["esperados"], "no_admitidos": r["no_admitidos"],
        "sin_datos_oficiales": r["meta"].get("sin_datos_oficiales", False),
        "documentos": [{"archivo": d["archivo"], "tipo": d["tipo_documento"], "formato": d["formato"],
                        "veredicto": d["veredicto"], "prioridad_revision": d.get("prioridad_revision", ""),
                        "motivos": d["motivos"], "archivo_oficial": d["archivo_oficial"],
                        "campos": [c for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")],
                        "senales": [x for x in d["senales"] if x["severidad"] != "info"],
                        "json_extraido": d.get("json_extraido"),
                        "campos_extraidos": EX.flat(d["json_extraido"], d["json_extraido"].get("_confianza", {}))
                        if d.get("json_extraido") else []}
                       for d in r["documentos"]],
    }


@app.get("/api/resultados")
def results(request: Request):
    """Resultados de la sesión, del más reciente al más antiguo."""
    s = _session(request)
    rs = list(s["resultados"].values())
    et = _labels(rs)
    return [_view(r, et[r["id"]]) for r in reversed(rs)]


@app.get("/api/zip/{rid}")
def download_zip(rid: str, request: Request):
    """Todo el resultado en un solo .zip (resumen, JSON extraídos, diferencias y Excel), generado en memoria."""
    s = _session(request)
    r = s["resultados"].get(rid)
    if not r:
        raise HTTPException(404, "No encontrado")
    name = _labels(list(s["resultados"].values()))[rid]
    file = "verificacion_" + "".join(c if c.isalnum() else "_" for c in N.strip_accents(name)) + ".zip"
    return Response(M.zip_in_memory(r, name), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{file}"'})


@app.post("/api/borrar")
def delete(request: Request):
    """Borra todos los resultados de la sesión."""
    s = _session(request)
    with _lock:
        s["resultados"].clear()
    return {"ok": True}


def main():
    """Arranca el visor en la dirección configurada (exige contraseña definida); sin registro de accesos."""
    cfg = _config()
    if not cfg.get("clave"):
        raise SystemExit("Defina primero la contraseña: python -m evaluation_viewer.configure --clave")
    print(f"Visor de pruebas en http://{cfg['host']}:{cfg['puerto']}  "
          f"(IP permitidas: {', '.join(cfg['ips_permitidas']) or 'solo esta PC'}; los cambios de acceso rigen sin "
          "reiniciar)", flush=True)
    uvicorn.run(app, host=cfg["host"], port=cfg["puerto"], log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
