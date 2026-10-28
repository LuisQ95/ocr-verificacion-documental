"""Verificación de UN postulante con archivos recibidos en memoria (visor de pruebas).

Nada se escribe en disco: los documentos del postulante y los JSON oficiales llegan como bytes, se leen desde memoria
(PyMuPDF / PIL), se comparan y se descartan. El resultado (dictámenes, motivos, valores leídos) es lo único que queda,
y solo en la memoria del servidor de pruebas mientras dure la sesión.
"""

import io
import json
import re
import unicodedata
from collections import Counter
from datetime import date

import cv2
from rapidfuzz import fuzz

from .. import reporte
from . import comparar_cl as K
from . import lectura_cl as L
from . import normalizar_cl as N
from . import pipeline_cl as PL
from . import requisitos_cl as Q

ESPERADOS = [("hoja_vida", "Hoja de Vida del Conductor"), ("ameco_rev02", "Certificado AMECO (Rev.02)"),
             ("ameco_rev03", "Certificado AMECO (Rev.03)")]
"""Documentos que el sistema sabe verificar: si no se entrega uno, figura como DOCUMENTO FALTANTE."""
FALTANTE = "DOCUMENTO FALTANTE"
NO_IDENTIFICADO = "NO IDENTIFICADO"
"""El documento no aparece entre los archivos legibles, pero hay archivos ilegibles (puede ser uno de ellos)."""
PARTICULAS = {"DE", "DEL", "LA", "LAS", "LOS", "Y", "DA", "DI", "VAN", "VON", "SAN", "SANTA"}
_iniciado = False


def _palabras(texto):
    """Palabras del texto, en mayúsculas y normalizadas (NFC)."""
    t = unicodedata.normalize("NFC", texto or "").upper()
    return [w for w in re.split(r"\s+", t.strip()) if w]


def _unidades(palabras):
    """Nombres o apellidos completos: las partículas se unen a la palabra siguiente ("DE LA CRUZ" = 1 apellido).
    Las partículas sueltas al final se pegan a la última unidad."""
    out, pend = [], []
    for w in palabras:
        pend.append(w)
        if w not in PARTICULAS:
            out.append(" ".join(pend))
            pend = []
    if pend:
        out = out[:-1] + [" ".join(out[-1:] + pend)] if out else [" ".join(pend)]
    return out


def validar_identidad(nombres, apellidos, rut, fecha_nacimiento="", cargo=""):
    """Devuelve (identidad, errores, advertencias). Errores impiden verificar; advertencias se muestran.

    El cargo es texto libre y se guarda en mayúsculas.
    """
    errores, adv = [], []
    pn, pa = _palabras(nombres), _palabras(apellidos)
    for etiqueta, ps in (("nombres", pn), ("apellidos", pa)):
        malas = [w for w in ps if not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ'\-]+", w)]
        if malas:
            errores.append(f"Los {etiqueta} solo pueden tener letras (revise: {', '.join(malas)}).")
    un, ua = _unidades(pn), _unidades(pa)
    if not 1 <= len(un) <= 3:
        errores.append("Ingrese entre 1 y 3 nombres.")
    if not 1 <= len(ua) <= 2:
        errores.append("Ingrese 1 o 2 apellidos.")
    elif len(ua) == 1:
        adv.append("Se ingresó un solo apellido: puede haber otra persona con el mismo nombre y apellido. "
                   "Para distinguirla se usan el RUT y la fecha de nacimiento.")
    r = N.rut(rut or "")
    if not r:
        errores.append("Ingrese el RUT (por ejemplo 12.345.678-5).")
    elif not N.rut_valido(r):
        errores.append(f"El RUT {rut} no es válido (el dígito verificador no corresponde).")
    fn = None
    if (fecha_nacimiento or "").strip():
        try:
            fn = date.fromisoformat(fecha_nacimiento.strip())
        except ValueError:
            fn = N.fecha(fecha_nacimiento)
        if fn is None:
            errores.append("La fecha de nacimiento no es válida.")
        elif fn >= date.today():
            errores.append("La fecha de nacimiento no puede ser futura.")
    cargo = " ".join(_palabras(cargo))
    if len(cargo) > 80 or (cargo and not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ0-9 .,()/\-]+", cargo)):
        errores.append("El cargo solo puede tener letras, números y signos simples (máximo 80 caracteres).")
    ident = {"nombres": " ".join(pn), "apellidos": " ".join(pa), "nombre_completo": " ".join(pn + pa),
             "un_apellido": len(ua) == 1, "rut": _formato_rut(r) if r else "", "fecha_nacimiento": fn,
             "fecha_nacimiento_fuente": "ingresada" if fn else "", "cargo": cargo}
    return ident, errores, adv


def _formato_rut(r):
    """RUT normalizado ("12345678-5") con puntos de miles ("12.345.678-5")."""
    num, dv = r.split("-")
    return f"{int(num):,}".replace(",", ".") + f"-{dv}"


def leer_oficiales(jsons):
    """[(nombre, bytes)] -> (oficiales como los usa el flujo, advertencias). El tipo se reconoce por el contenido."""
    of = {"hoja_vida": None, "ameco": {}, "archivo_ameco": {}, "archivo_hoja_vida": ""}
    adv = []
    for nombre, datos in jsons:
        try:
            js = json.loads(datos.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            adv.append(f"'{nombre}' no es un JSON válido: no se usó.")
            continue
        if not isinstance(js, dict):
            adv.append(f"'{nombre}' no tiene el formato esperado: no se usó.")
        elif "Licencias registradas" in js or "R.U.N." in js:
            if of["hoja_vida"] is not None:
                adv.append(f"Se subió más de un JSON de Hoja de Vida: se usó '{of['archivo_hoja_vida']}' y no '{nombre}'.")
                continue
            of["hoja_vida"], of["archivo_hoja_vida"] = js, nombre
        elif "N°" in js or "Exámenes" in js:
            clave = N.codigo(js.get("N°", nombre))
            of["ameco"][clave] = js
            of["archivo_ameco"][clave] = nombre
        else:
            adv.append(f"'{nombre}' no es un JSON de Hoja de Vida ni de certificado AMECO: no se usó.")
    return of, adv


def _cruces_identidad(ident, oficiales):
    """Advertencias si los JSON oficiales parecen de otra persona que la ingresada."""
    adv = []
    nombre = N.clave_texto(N.sin_tildes(ident["nombre_completo"]))
    fuentes = []
    if oficiales["hoja_vida"]:
        js = oficiales["hoja_vida"]
        fuentes.append((f"JSON de Hoja de Vida ({oficiales['archivo_hoja_vida']})",
                        js.get("Nombre", ""), js.get("R.U.N.", "")))
        fn = N.fecha(js.get("Fecha nacimiento", ""))
        if fn and ident["fecha_nacimiento"] and fn != ident["fecha_nacimiento"]:
            adv.append(f"La fecha de nacimiento ingresada ({ident['fecha_nacimiento']:%d-%m-%Y}) no coincide con la "
                       f"del JSON de Hoja de Vida ({fn:%d-%m-%Y}).")
        if fn and not ident["fecha_nacimiento"]:
            ident["fecha_nacimiento"], ident["fecha_nacimiento_fuente"] = fn, "JSON de Hoja de Vida"
    for clave, js in oficiales["ameco"].items():
        fuentes.append((f"JSON del certificado {clave}", js.get("Nombre", ""), js.get("RUT", "")))
    for que, nom, rut in fuentes:
        if nom and fuzz.token_set_ratio(nombre, N.clave_texto(N.sin_tildes(nom))) < 90:
            adv.append(f"El nombre ingresado no coincide con el del {que}: '{nom}'.")
        if rut and N.rut(rut) and N.rut(rut) != N.rut(ident["rut"]):
            adv.append(f"El RUT ingresado ({ident['rut']}) no coincide con el del {que}: {rut}. "
                       "La comparación de ese documento no es confiable.")
    return adv


def _iniciar():
    """Carga plantillas y puntos ORB una vez por proceso. init_worker limita OpenCV a 1 hilo (pensado para varios
    procesos); aquí hay uno solo, así que se le devuelven todos los hilos."""
    global _iniciado
    if not _iniciado:
        L.init_worker()
        cv2.setNumThreads(-1)
        _iniciado = True


def verificar_en_memoria(ident, archivos, jsons, hoy=None, usar_gpu=True, progreso=None, log=lambda m: None):
    """ident: de validar_identidad; archivos / jsons: [(nombre, bytes)]. Devuelve el resultado completo.

    Si hay archivos que no se pudieron leer, un documento que no aparece queda NO IDENTIFICADO en vez de FALTANTE
    (puede ser uno de ellos). Sin ningún dato oficial no se analiza nada: solo se entregan los datos extraídos.
    """
    hoy = hoy or date.today()
    progreso = progreso or (lambda *a: None)
    _iniciar()
    pl, req = L._PL, Q.cargar()
    carpeta = ident["nombre_completo"]
    oficiales, adv = leer_oficiales(jsons)
    adv += _cruces_identidad(ident, oficiales)

    paginas, no_adm, vistos = [], [], Counter()
    unidades = []
    for nombre, datos in archivos:
        vistos[nombre] += 1
        nombre = nombre if vistos[nombre] == 1 else f"{nombre} ({vistos[nombre]})"
        if not any(nombre.lower().endswith(e) for e in L.EXT_ADMITIDAS):
            no_adm.append({"carpeta": carpeta, "archivo": nombre, "origen": "postulante",
                           "motivo": "formato no admitido (se aceptan PDF y fotos JPG, PNG, HEIC, WEBP, TIFF, BMP)"})
            continue
        a = L.ArchivoMemoria(nombre, datos, carpeta)
        try:
            unidades += L.unidades(a)
        except Exception as e:
            no_adm.append({"carpeta": carpeta, "archivo": nombre, "origen": "postulante",
                           "motivo": f"no se pudo abrir el archivo ({type(e).__name__})"})
    for i, u in enumerate(unidades, 1):
        progreso("leyendo documentos", i, len(unidades))
        paginas += PL.paginas_de_unidad(L.analizar(u))
    modelo = PL.leer_ocr(paginas, usar_gpu, log, lambda h, t: progreso("leyendo imágenes con OCR", h, t))
    for p in paginas:
        if p.get("plantilla") and not p.get("ilegible"):
            PL.valores_de_pagina(p, pl)
    progreso("comparando con el dato oficial", 0, 0)
    registro = {carpeta: {"rut": ident["rut"], "nombre": carpeta, "puesto": ident["cargo"]}}
    persona, filas = PL.dictaminar(carpeta, paginas, registro, None, req, hoy, pl, oficiales=oficiales)
    vistos = Counter()
    for f in filas:
        vistos[f["archivo"]] += 1
        if vistos[f["archivo"]] > 1:
            f["archivo"] = f"{f['archivo']} ({vistos[f['archivo']]})"
        if f["tipo_documento"].startswith("hoja_vida") and oficiales["archivo_hoja_vida"] and f["archivo_oficial"]:
            f["archivo_oficial"] = oficiales["archivo_hoja_vida"]

    esperados = []
    ilegibles = [f["archivo"] for f in filas if f["veredicto"] == K.NO_LEGIBLE and not f["tipo_documento"].startswith(
        ("hoja_vida", "ameco"))]
    for tipo, etiqueta in ESPERADOS:
        docs = [f for f in filas if f["tipo_documento"].startswith(tipo)]
        estado = "ENTREGADO" if docs else (NO_IDENTIFICADO if ilegibles else FALTANTE)
        esperados.append({"tipo": tipo, "documento": etiqueta, "estado": estado,
                          "veredictos": [f["veredicto"] for f in docs], "archivos": [f["archivo"] for f in docs],
                          "nota": (f"no se encontró entre los archivos legibles; puede ser uno de los ilegibles "
                                   f"({', '.join(ilegibles)})") if estado == NO_IDENTIFICADO else ""})
        if estado == NO_IDENTIFICADO:
            continue
        if not docs:
            if tipo == "hoja_vida" and oficiales["hoja_vida"]:
                adv.append("Se subió el JSON de la Hoja de Vida, pero no la Hoja de Vida del postulante.")
            version = "Rev.02" if tipo == "ameco_rev02" else "Rev.03"
            for clave, js in oficiales["ameco"].items():
                if tipo.startswith("ameco") and version in js.get("Formato", ""):
                    adv.append(f"Se subió el JSON del certificado {clave} ({version}), pero no el certificado.")
    persona["faltantes"] = [e["documento"] for e in esperados if e["estado"] == FALTANTE]
    persona["no_identificados"] = [e["documento"] for e in esperados if e["estado"] == NO_IDENTIFICADO]
    sin_oficiales = oficiales["hoja_vida"] is None and not oficiales["ameco"]
    if sin_oficiales:
        persona.update(veredicto=K.SIN_DATO_OFICIAL, aptitud=K.SIN_DATO_OFICIAL,
                       motivos_aptitud=["No se subieron datos oficiales (JSON): no se verificó la autenticidad ni la "
                                        "aptitud. Se entregan los datos extraídos de los documentos."])
    meta = {"fecha": date.today().isoformat(), "fecha_referencia": str(hoy), "modelo_ocr": modelo,
            "modo": "visor de pruebas (en memoria, sin guardar archivos)", "sin_datos_oficiales": sin_oficiales}
    return {"identidad": ident, "persona": persona, "documentos": filas, "esperados": esperados,
            "advertencias": list(dict.fromkeys(adv)), "no_admitidos": no_adm, "meta": meta}


def excel_en_memoria(resultado):
    """El mismo reporte Excel del flujo normal, generado en memoria (bytes)."""
    buf = io.BytesIO()
    persona = dict(resultado["persona"], carpeta=resultado["identidad"]["nombre_completo"])
    meta = dict(resultado["meta"], documentos_faltantes=", ".join(persona.get("faltantes", [])) or "ninguno",
                documentos_no_identificados=", ".join(persona.get("no_identificados", [])) or "ninguno",
                advertencias=" | ".join(resultado["advertencias"]) or "ninguna")
    reporte._excel(buf, [persona], resultado["documentos"], meta, resultado["no_admitidos"])
    return buf.getvalue()


def _nombre_archivo(texto):
    """Nombre de archivo seguro (ASCII, sin espacios ni símbolos, máximo 120 caracteres)."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", t).strip("_")[:120] or "documento"


def _json(datos):
    """JSON legible (UTF-8, con sangría) en bytes."""
    return json.dumps(datos, ensure_ascii=False, indent=1, default=str).encode("utf-8")


def zip_en_memoria(resultado, etiqueta):
    """Todo el resultado en un .zip (en memoria): resumen, datos extraídos de cada documento (JSON con las claves del
    JSON oficial), diferencias con el dato oficial y el reporte Excel."""
    import zipfile
    buf = io.BytesIO()
    p, i = resultado["persona"], resultado["identidad"]
    docs = resultado["documentos"]
    comparados = [d for d in docs if d["campos"]]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LEAME.txt", (
            f"Verificación de {etiqueta} ({resultado.get('fecha', '')})\n\n"
            "resumen.json                 identidad, dictamen por documento, documentos faltantes, aptitud y advertencias\n"
            "datos_extraidos/*.json       lo que dice cada documento del postulante, con las mismas claves que el JSON\n"
            "                             oficial. '_confianza' repite la estructura y da, para cada dato, la confianza\n"
            "                             (0 a 1) y su nivel ALTA / MEDIA / BAJA / SIN LECTURA; '_control' indica el\n"
            "                             método de lectura, la escala y los campos a revisar\n"
            + ("diferencias.json             campo por campo: dato oficial vs. leído, solo donde no coinciden\n"
               if comparados else "")
            + "reporte_verificacion.xlsx    el mismo reporte en Excel\n"
            + ("\nNo se subieron datos oficiales (JSON): no se verificó la autenticidad ni la aptitud.\n"
               if resultado["meta"].get("sin_datos_oficiales") else "")).encode("utf-8"))
        z.writestr("resumen.json", _json({
            "postulante": {"nombres": i["nombres"], "apellidos": i["apellidos"], "rut": i["rut"],
                           "fecha_nacimiento": f"{i['fecha_nacimiento']:%d-%m-%Y}" if i["fecha_nacimiento"] else "",
                           "cargo": p["puesto"]},
            "autenticidad": p["veredicto"], "aptitud": p["aptitud"], "motivos_aptitud": p["motivos_aptitud"],
            "advertencias": resultado["advertencias"],
            "documentos_esperados": [{k: e[k] for k in ("documento", "estado", "veredictos", "archivos")}
                                     for e in resultado["esperados"]],
            "documentos": [{"archivo": d["archivo"], "tipo": d["tipo_documento"], "dictamen": d["veredicto"],
                            "motivos": d["motivos"], "dato_oficial": d["archivo_oficial"]} for d in docs],
            "no_admitidos": [{"archivo": x["archivo"], "motivo": x["motivo"]} for x in resultado["no_admitidos"]],
            "fecha_referencia": resultado["meta"]["fecha_referencia"]}))
        usados = Counter()
        for d in docs:
            if not d.get("json_extraido"):
                continue
            base = _nombre_archivo(f"{d['tipo_documento']}_{d['archivo']}")
            usados[base] += 1
            nombre = base if usados[base] == 1 else f"{base}_{usados[base]}"
            z.writestr(f"datos_extraidos/{nombre}.json", _json(d["json_extraido"]))
        if comparados:
            z.writestr("diferencias.json", _json([
                {"archivo": d["archivo"], "dato_oficial": d["archivo_oficial"], "dictamen": d["veredicto"],
                 "campos": [{"campo": c["zona"], "oficial": c["oficial"], "leido": c["postulante"], "estado": c["estado"],
                             "motivo": c["motivo"], "confianza": c["conf"]}
                            for c in d["campos"] if c["estado"] not in ("IGUAL", "NO_APLICA")]} for d in comparados]))
        z.writestr("reporte_verificacion.xlsx", excel_en_memoria(resultado))
    return buf.getvalue()
