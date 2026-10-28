"""Visor de PRUEBAS: un evaluador externo sube los documentos de un postulante y sus JSON oficiales y recibe el
dictamen. Es una aplicación aparte del visor principal: no ve postulant_files, webscrap_files ni resultados.

Privacidad:
  - Los archivos llegan dentro del cuerpo de la petición (JSON con base64) y se procesan en memoria: no se
    escriben en disco (no hay subida multipart, que guarda en temporales los archivos de más de 1 MB).
  - Los resultados quedan solo en la memoria de este proceso, por sesión; se borran con "Borrar resultados",
    al cerrar sesión, tras 8 h sin uso o al cerrar el visor. No se registra el contenido en logs.
Acceso:
  - Solo desde esta PC o desde las IP de visor_pruebas/acceso.json (la PC del evaluador en Tailscale), y con
    contraseña (python -m visor_pruebas.configurar).

Uso: python -m visor_pruebas.servidor
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

from verificacion import config as C
from verificacion import recursos as R
from verificacion.chile import extraccion_cl as EX
from verificacion.chile import memoria_cl as M
from verificacion.chile import normalizar_cl as N
from verificacion.chile import requisitos_cl as Q

from . import acceso as A

PAGINA = Path(__file__).parent / "static" / "index.html"
COOKIE = "sesion_pruebas"
DURACION_SESION_S = 8 * 3600
MAX_ARCHIVOS_POSTULANTE, MAX_MB_ARCHIVO = 10, 25
MAX_ARCHIVOS_OFICIALES, MAX_MB_JSON = 10, 2
MAX_CUERPO = 300 * 1024 * 1024
"""Tamaño máximo de la petición: el base64 de todos los archivos admitidos, con margen."""
MAX_RESULTADOS_SESION = 50
MAX_INTENTOS, BLOQUEO_S = 5, 15 * 60

app = FastAPI(title="Visor de pruebas", docs_url=None, redoc_url=None, openapi_url=None)
_cfg = A.cargar()
_sesiones = {}
"""token -> {"ultimo": t, "resultados": {id: resultado}}"""
_trabajos = {}
"""id -> estado del trabajo (sin los archivos una vez procesado)"""
_fallos = {}
"""ip -> (intentos fallidos, bloqueado_hasta)"""
_cola = queue.Queue()
_bloqueo = threading.Lock()


@app.middleware("http")
async def _solo_ips_permitidas(request: Request, call_next):
    """Rechaza (403) toda petición de una IP no permitida y agrega cabeceras de seguridad a las respuestas."""
    ip = request.client.host if request.client else ""
    if not A.ip_permitida(ip, _cfg):
        return Response("Acceso no permitido", status_code=403)
    respuesta = await call_next(request)
    respuesta.headers["Cache-Control"] = "no-store"
    respuesta.headers["X-Frame-Options"] = "DENY"
    respuesta.headers["X-Content-Type-Options"] = "nosniff"
    respuesta.headers["Referrer-Policy"] = "no-referrer"
    return respuesta


def _limpiar_sesiones():
    """Borra las sesiones vencidas y los trabajos terminados que ya no pertenecen a una sesión viva."""
    ahora = time.time()
    for t in [t for t, s in _sesiones.items() if ahora - s["ultimo"] > DURACION_SESION_S]:
        del _sesiones[t]
    vivas = [s for s in _sesiones.values()]
    for tid in [tid for tid, t in _trabajos.items() if t["estado"] in ("terminado", "error")
                and (ahora - t["creado"] > DURACION_SESION_S or not any(t["sesion"] is s for s in vivas))]:
        del _trabajos[tid]


def _sesion(request: Request):
    """Sesión de la cookie (renueva su último uso) o 401 si no hay sesión válida."""
    with _bloqueo:
        _limpiar_sesiones()
        s = _sesiones.get(request.cookies.get(COOKIE, ""))
        if not s:
            raise HTTPException(401, "Inicie sesión")
        s["ultimo"] = time.time()
        return s


@app.get("/", response_class=HTMLResponse)
def inicio():
    """Página del visor."""
    return PAGINA.read_text(encoding="utf-8")


@app.post("/api/login")
async def login(request: Request):
    """Abre sesión con la contraseña (cookie HttpOnly, SameSite estricta). Tras MAX_INTENTOS fallidos, la IP queda
    bloqueada BLOQUEO_S segundos."""
    ip = request.client.host
    intentos, hasta = _fallos.get(ip, (0, 0))
    if hasta > time.time():
        raise HTTPException(429, f"Demasiados intentos. Espere {int((hasta - time.time()) / 60) + 1} min.")
    datos = await request.json()
    if not A.clave_correcta(str(datos.get("clave", "")), _cfg.get("clave")):
        intentos += 1
        _fallos[ip] = (0, time.time() + BLOQUEO_S) if intentos >= MAX_INTENTOS else (intentos, 0)
        raise HTTPException(401, "Contraseña incorrecta")
    _fallos.pop(ip, None)
    token = secrets.token_urlsafe(32)
    with _bloqueo:
        _sesiones[token] = {"ultimo": time.time(), "resultados": {}}
    r = JSONResponse({"ok": True})
    r.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=DURACION_SESION_S)
    return r


@app.post("/api/logout")
def logout(request: Request):
    """Cierra la sesión: sus resultados se descartan."""
    with _bloqueo:
        _sesiones.pop(request.cookies.get(COOKIE, ""), None)
    r = JSONResponse({"ok": True})
    r.delete_cookie(COOKIE)
    return r


@app.get("/api/sesion")
def sesion(request: Request):
    """Datos para el formulario: cargos configurados, cargo por defecto y límites de carga."""
    _sesion(request)
    req = Q.cargar()
    return {"cargos": [c.upper() for c in req["puestos"]], "cargo_por_defecto": req["puesto_por_defecto"].upper(),
            "limites": {"archivos": MAX_ARCHIVOS_POSTULANTE, "mb_archivo": MAX_MB_ARCHIVO,
                        "jsons": MAX_ARCHIVOS_OFICIALES, "mb_json": MAX_MB_JSON}}


def _decodificar(lista, maximo, mb, que):
    """[(nombre, bytes)] desde la lista de archivos en base64, validando cantidad, tamaño y contenido."""
    if not isinstance(lista, list) or len(lista) > maximo:
        raise HTTPException(400, f"Se admiten como máximo {maximo} {que}.")
    out = []
    for x in lista:
        nombre = Path(str(x.get("nombre", "archivo"))).name[:150] or "archivo"
        try:
            datos = base64.b64decode(x.get("base64", ""), validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, f"No se pudo recibir '{nombre}'.")
        if len(datos) > mb * 1024 * 1024:
            raise HTTPException(400, f"'{nombre}' supera {mb} MB.")
        if not datos:
            raise HTTPException(400, f"'{nombre}' está vacío.")
        out.append((nombre, datos))
    return out


async def _cuerpo_json(request: Request):
    """Cuerpo JSON leído en memoria por partes, cortando con 413 apenas supera MAX_CUERPO."""
    largo = int(request.headers.get("content-length") or 0)
    if largo > MAX_CUERPO:
        raise HTTPException(413, "Los archivos superan el tamaño permitido.")
    cuerpo = bytearray()
    async for parte in request.stream():
        cuerpo += parte
        if len(cuerpo) > MAX_CUERPO:
            raise HTTPException(413, "Los archivos superan el tamaño permitido.")
    try:
        return json.loads(bytes(cuerpo))
    except ValueError:
        raise HTTPException(400, "Petición inválida.")


@app.post("/api/validar")
async def validar(request: Request):
    """Validación de la identidad mientras se escribe (sin archivos)."""
    _sesion(request)
    d = await request.json()
    _, errores, adv = M.validar_identidad(d.get("nombres", ""), d.get("apellidos", ""), d.get("rut", ""),
                                          d.get("fecha_nacimiento", ""), d.get("cargo", ""))
    return {"errores": errores, "advertencias": adv}


@app.post("/api/verificar")
async def verificar(request: Request):
    """Valida la identidad y los archivos y encola la verificación; devuelve el id del trabajo."""
    s = _sesion(request)
    d = await _cuerpo_json(request)
    ident, errores, adv = M.validar_identidad(d.get("nombres", ""), d.get("apellidos", ""), d.get("rut", ""),
                                              d.get("fecha_nacimiento", ""), d.get("cargo", ""))
    archivos = _decodificar(d.get("postulante", []), MAX_ARCHIVOS_POSTULANTE, MAX_MB_ARCHIVO, "documentos del postulante")
    jsons = _decodificar(d.get("oficiales", []), MAX_ARCHIVOS_OFICIALES, MAX_MB_JSON, "JSON oficiales")
    if not archivos:
        errores.append("Suba al menos un documento del postulante.")
    if errores:
        raise HTTPException(400, " ".join(errores))
    del d
    t = {"id": uuid.uuid4().hex[:12], "sesion": s, "estado": "en_cola", "fase": "en cola", "hechos": 0, "total": 0,
         "mensaje": "", "ident": ident, "adv": adv, "archivos": archivos, "jsons": jsons, "creado": time.time()}
    with _bloqueo:
        _trabajos[t["id"]] = t
    _cola.put(t)
    return {"id": t["id"]}


@app.get("/api/trabajo/{tid}")
def trabajo(tid: str, request: Request):
    """Estado y progreso de un trabajo de la sesión, con cuántos esperan antes en la cola."""
    s = _sesion(request)
    t = _trabajos.get(tid)
    if not t or t["sesion"] is not s:
        raise HTTPException(404, "No encontrado")
    pos = sum(1 for x in list(_cola.queue) if x["creado"] < t["creado"])
    return {k: t.get(k) for k in ("estado", "fase", "hechos", "total", "mensaje", "resultado_id")} | {"en_cola_antes": pos}


def _procesar(t):
    """Verifica un trabajo en memoria. Hay una verificación a la vez en el equipo (también respeta las de consola y
    las del visor principal); si la GPU está ocupada, las imágenes se leen con CPU."""
    def progreso(fase, hechos, total):
        """Actualiza el progreso visible del trabajo."""
        t.update(fase=fase, hechos=hechos, total=total)

    usar_gpu = True
    while True:
        try:
            with R.Candado("pipeline"):
                t.update(estado="procesando", fase="revisando GPU y memoria")
                try:
                    R.verificar(vram_gb=C.PRESUPUESTO_VRAM_OCR_GB, ram_gb=C.PRESUPUESTO_RAM_GB, ventana_s=5,
                                log=lambda m: None)
                except R.RecursosInsuficientes:
                    usar_gpu = False
                    t["mensaje"] = "GPU ocupada: las imágenes se leen con CPU (más lento)."
                return M.verificar_en_memoria(t["ident"], t["archivos"], t["jsons"], usar_gpu=usar_gpu,
                                              progreso=progreso)
        except R.RecursosInsuficientes:
            t.update(estado="en_cola", fase="esperando a otra verificación en curso en el equipo")
            time.sleep(15)


def _trabajador():
    """Hilo que atiende la cola. Los documentos se descartan apenas termina cada trabajo; ante un error solo se
    informa su tipo, nunca el contenido."""
    while True:
        t = _cola.get()
        try:
            res = _procesar(t)
            res["advertencias"] = list(dict.fromkeys(t["adv"] + res["advertencias"]))
            res["id"], res["fecha"] = t["id"], datetime.now().strftime("%d-%m-%Y %H:%M")
            with _bloqueo:
                rs = t["sesion"]["resultados"]
                rs[t["id"]] = res
                while len(rs) > MAX_RESULTADOS_SESION:
                    rs.pop(next(iter(rs)))
            t.update(estado="terminado", fase="terminado", resultado_id=t["id"])
        except Exception as e:
            t.update(estado="error", mensaje=f"No se pudo completar la verificación ({type(e).__name__}).")
        finally:
            t.pop("archivos", None)
            t.pop("jsons", None)


threading.Thread(target=_trabajador, daemon=True).start()


def _clave_nombre(nombre):
    """Nombre normalizado para detectar homónimos."""
    return N.clave_texto(N.sin_tildes(nombre))


def _etiquetas(resultados):
    """Nombre a mostrar: si hay homónimos (mismo nombre, distinto RUT) se agrega la fecha de nacimiento."""
    por_nombre = {}
    for r in resultados:
        por_nombre.setdefault(_clave_nombre(r["identidad"]["nombre_completo"]), set()).add(r["identidad"]["rut"])
    out = {}
    for r in resultados:
        i = r["identidad"]
        et = i["nombre_completo"]
        if len(por_nombre[_clave_nombre(et)]) > 1:
            et += f" ({i['fecha_nacimiento']:%d-%m-%Y})" if i["fecha_nacimiento"] else f" (RUT {i['rut']})"
        out[r["id"]] = et
    return out


def _vista(r, etiqueta):
    """Resultado en el formato que muestra la página (sin campos iguales ni señales informativas)."""
    i = r["identidad"]
    return {
        "id": r["id"], "fecha": r["fecha"], "etiqueta": etiqueta, "nombres": i["nombres"], "apellidos": i["apellidos"],
        "rut": i["rut"], "fecha_nacimiento": f"{i['fecha_nacimiento']:%d-%m-%Y}" if i["fecha_nacimiento"] else "",
        "fecha_nacimiento_fuente": i["fecha_nacimiento_fuente"], "cargo": r["persona"]["puesto"],
        "veredicto": r["persona"]["veredicto"], "aptitud": r["persona"]["aptitud"],
        "motivos_aptitud": r["persona"]["motivos_aptitud"], "advertencias": r["advertencias"],
        "esperados": r["esperados"], "no_admitidos": r["no_admitidos"],
        "sin_datos_oficiales": r["meta"].get("sin_datos_oficiales", False),
        "documentos": [{"archivo": d["archivo"], "tipo": d["tipo_documento"], "formato": d["formato"],
                        "veredicto": d["veredicto"], "motivos": d["motivos"], "archivo_oficial": d["archivo_oficial"],
                        "campos": [c for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")],
                        "senales": [x for x in d["senales"] if x["severidad"] != "info"],
                        "json_extraido": d.get("json_extraido"),
                        "campos_extraidos": EX.plano(d["json_extraido"], d["json_extraido"].get("_confianza", {}))
                        if d.get("json_extraido") else []}
                       for d in r["documentos"]],
    }


@app.get("/api/resultados")
def resultados(request: Request):
    """Resultados de la sesión, del más reciente al más antiguo."""
    s = _sesion(request)
    rs = list(s["resultados"].values())
    et = _etiquetas(rs)
    return [_vista(r, et[r["id"]]) for r in reversed(rs)]


@app.get("/api/zip/{rid}")
def descargar_zip(rid: str, request: Request):
    """Todo el resultado en un solo .zip (resumen, JSON extraídos, diferencias y Excel), generado en memoria."""
    s = _sesion(request)
    r = s["resultados"].get(rid)
    if not r:
        raise HTTPException(404, "No encontrado")
    nombre = _etiquetas(list(s["resultados"].values()))[rid]
    archivo = "verificacion_" + "".join(c if c.isalnum() else "_" for c in N.sin_tildes(nombre)) + ".zip"
    return Response(M.zip_en_memoria(r, nombre), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{archivo}"'})


@app.post("/api/borrar")
def borrar(request: Request):
    """Borra todos los resultados de la sesión."""
    s = _sesion(request)
    with _bloqueo:
        s["resultados"].clear()
    return {"ok": True}


def main():
    """Arranca el visor en la dirección configurada (exige contraseña definida); sin registro de accesos."""
    if not _cfg.get("clave"):
        raise SystemExit("Defina primero la contraseña: python -m visor_pruebas.configurar --clave")
    print(f"Visor de pruebas en http://{_cfg['host']}:{_cfg['puerto']}  "
          f"(IP permitidas: {', '.join(_cfg['ips_permitidas']) or 'solo esta PC'})", flush=True)
    uvicorn.run(app, host=_cfg["host"], port=_cfg["puerto"], log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
