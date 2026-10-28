"""Pipeline de verificación de los documentos chilenos contra los datos oficiales en JSON.

  1. Plantillas chilenas (official_file/) y zonas de cada campo (chile/plantillas_cl.py)
  2. Unidades de entrada: cada página de cada PDF, cada imagen y cada página de un TIFF de cada carpeta
  3. Lectura en CPU, en paralelo (chile/lectura_cl.py): PDF nativo -> texto visible directo; PDF de imágenes e
     imágenes -> alineación con la plantilla (4 giros, 2 documentos por foto); timbre y código de barras
  4. OCR en GPU de las páginas alineadas (modelo adoptado en config.py)
  5. Agrupación por documento: las 2 páginas de la Hoja de Vida (aunque vengan en archivos separados o
     desordenadas); cada certificado AMECO
  6. Comparación campo a campo con el JSON (webscrap_files/<carpeta>/*.json) + señales técnicas + identidad
  7. Aptitud según config/requisitos_cl.yaml
  8. Reporte: <salida>/verificacion.db y <salida>/reporte_verificacion.xlsx

Uso: python -u -m verificacion.chile.pipeline_cl [--limite 50] [--salida resultados/chile]

El lector OCR (PyTorch) se importa solo dentro de `leer_ocr`, nunca a nivel de módulo: cada proceso de lectura
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

from .. import comparacion as KV
from .. import config as C
from .. import recursos as R
from ..reporte import escribir_reporte
from . import comparar_cl as K
from . import extraccion_cl as EX
from . import hvid_cuerpo as HV
from . import lectura_cl as L
from . import normalizar_cl as N
from . import plantillas_cl as P
from . import requisitos_cl as Q


class DetenidoPorRecursos(RuntimeError):
    """La verificación se detuvo por falta de memoria del sistema o porque se canceló."""


def leer_registro(ruta):
    """candidatos.xlsx: carpeta -> {rut, nombre, puesto}."""
    if not ruta or not Path(ruta).exists():
        return {}
    filas = list(load_workbook(ruta, read_only=True).active.iter_rows(values_only=True))
    cab = [str(c or "").strip().lower() for c in filas[0]]

    def col(*nombres):
        """Índice de la primera columna cuyo encabezado es uno de `nombres` (None si no hay)."""
        return next((cab.index(n) for n in nombres if n in cab), None)
    ic, ir, inom, ip, i_n = col("carpeta"), col("rut declarado", "rut"), col("nombre completo", "nombre"), \
        col("cargo postulado", "puesto", "cargo"), col("n")
    return {str(f[ic]): {"rut": f[ir] if ir is not None else None, "nombre": f[inom] if inom is not None else None,
                         "puesto": f[ip] if ip is not None else None, "n": f[i_n] if i_n is not None else None}
            for f in filas[1:] if f[ic] is not None}


def tareas(carpetas, dir_post):
    """Unidades por leer de cada carpeta y archivos no admitidos (formato no soportado o que no se pueden abrir)."""
    lista, no_admitidos = [], []
    for c in carpetas:
        d = dir_post / c
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if not f.is_file():
                continue
            if f.suffix.lower() not in L.EXT_ADMITIDAS:
                no_admitidos.append({"carpeta": c, "archivo": f.name, "origen": "postulante",
                                     "motivo": f"formato no admitido ({f.suffix or 'sin extensión'})"})
                continue
            try:
                lista += L.unidades(f)
            except Exception as e:
                no_admitidos.append({"carpeta": c, "archivo": f.name, "origen": "postulante",
                                     "motivo": f"no se pudo abrir: {type(e).__name__}"})
    return lista, no_admitidos


def oficiales_de(carpeta, dir_web):
    """JSON oficiales de una carpeta: Hoja de Vida y certificados AMECO (por N°, con el nombre de su archivo)."""
    d = dir_web / carpeta
    hv = d / "hoja_vida_conductor.json"
    out = {"hoja_vida": json.loads(hv.read_text(encoding="utf-8")) if hv.exists() else None, "ameco": {},
           "archivo_ameco": {}}
    for f in sorted(d.glob("ameco_*.json")) if d.is_dir() else []:
        js = json.loads(f.read_text(encoding="utf-8"))
        clave = N.codigo(js.get("N°", f.stem))
        out["ameco"][clave] = js
        out["archivo_ameco"][clave] = f.name
    return out


def paginas_de_unidad(r):
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
            nit = r.get("nitidez") or 0
            p.update(plantilla=None, plantilla_probable=d["plantilla"])
            if nit < C.NITIDEZ_MINIMA:
                p.update(ilegible=True, motivo_ilegible=f"imagen ilegible (nitidez {nit:.0f}): no se pudo identificar "
                                                        "ni leer el documento")
            else:
                p.update(no_reconocido=True, motivo=f"imagen nítida que no corresponde a la Hoja de Vida ni a un "
                                                    f"certificado AMECO, o no muestra el documento completo "
                                                    f"({d.get('alineacion_motivo')})")
        else:
            p["imagen_ocr"] = d["imagen_ocr"]
            p["escala_ocr"] = d.get("escala_ocr", 1)
        out.append(p)
    if not out:
        out.append(base | {"plantilla": None, "ilegible": True, "motivo_ilegible": "imagen sin documento reconocible"})
    return out


def leer_ocr(paginas, usar_gpu, log=print, al_leer=None):
    """OCR (GPU) de las páginas alineadas, por lotes. Devuelve el nombre del modelo usado.

    El modelo adoptado lee palabras completas (hasta 30 caracteres): no se parten los recortes largos. Si una imagen
    se alineó a otra escala, las zonas se escalan y las cajas leídas se devuelven a puntos de la plantilla.
    """
    pend = [p for p in paginas if "imagen_ocr" in p]
    if not pend:
        return "no se usó OCR"
    from ..ocr import LectorOCR
    if usar_gpu:
        R.limitar_gpu(C.PRESUPUESTO_VRAM_OCR_GB)
    lector = LectorOCR(dispositivo=None if usar_gpu else "cpu")
    lector.max_aspecto = 15.0
    log(f"OCR: {lector.nombre_modelo} en {lector.dispositivo} | {len(pend)} páginas")
    pl = P.cargar()
    for i in range(0, len(pend), C.OCR_BATCH_PAGINAS):
        lote = pend[i:i + C.OCR_BATCH_PAGINAS]
        zonas = [[{"rect": [v * p.get("escala_ocr", 1) for v in z["rect"]], "nombre": n, "tipo": z["tipo"]}
                  for n, z in pl[p["plantilla"]]["zonas"].items() if z["tipo"] != "cuerpo"] for p in lote]
        palabras = lector.leer([p["imagen_ocr"] for p in lote], zonas)
        for p, pw in zip(lote, palabras):
            f = p.get("escala_ocr", 1)
            if f != 1:
                for w in pw:
                    w["bbox"] = [v / f for v in w["bbox"]]
            p["palabras"] = pw
            del p["imagen_ocr"]
        if al_leer:
            al_leer(i + len(lote), len(pend))
    nombre = lector.nombre_modelo
    del lector
    import torch
    torch.cuda.empty_cache()
    return nombre


def valores_de_pagina(p, pl):
    """Palabras -> valor de cada zona; filas de la sección variable; palabras de valores (fuentes).

    En lecturas OCR también se detecta la versión COPIA (marca de agua y pie "Copia no válida como certificado") y
    se marca como ilegible la página donde casi no se leen las etiquetas del formato.
    """
    tpl = pl[p["plantilla"]]
    palabras = p.get("palabras") or []
    asig = KV.asignar_palabras(palabras, tpl)
    p["valores"] = asig["valores"]
    p["cobertura_etiquetas"] = asig["cobertura_etiquetas"]
    rects = [z["rect"] for z in tpl["zonas"].values() if z["tipo"] != "cuerpo"]
    def centro(b):
        """Centro de una caja (x0, y0, x1, y1)."""
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    p["palabras_valor"] = [w for w in palabras if any(r[0] <= centro(w["bbox"])[0] <= r[2]
                                                      and r[1] <= centro(w["bbox"])[1] <= r[3] for r in rects)]
    if "cuerpo" in tpl["zonas"]:
        r = tpl["zonas"]["cuerpo"]["rect"]
        p["lineas_cuerpo"] = HV.lineas([w for w in palabras if r[0] <= centro(w["bbox"])[0] <= r[2]
                                        and r[1] <= centro(w["bbox"])[1] <= r[3]])
    if p.get("es_ocr"):
        texto = " ".join(w["texto"] for w in palabras).upper()
        p["copia"] = "COPIA" in texto and ("VALID" in texto or "CERTIFICADO" in texto.split("COPIA", 1)[1][:80])
    if p.get("es_ocr") and asig["cobertura_etiquetas"] < 0.25 and not p.get("ilegible"):
        p.update(ilegible=True, motivo_ilegible=f"casi no se leen las etiquetas del formato "
                                                f"({asig['cobertura_etiquetas']:.0%}): imagen ilegible o muy pequeña")


def agrupar(paginas):
    """Páginas -> documentos. La Hoja de Vida junta su página 1 y 2 (mismo archivo; si no, en orden de archivo)."""
    docs, hv1, hv2 = [], [], []
    por_archivo = defaultdict(list)
    for p in paginas:
        tipo = P.TIPO_DOC.get(p["plantilla"]) if p.get("plantilla") else None
        if tipo == "hoja_vida" and not p.get("ilegible"):
            por_archivo[p["archivo"]].append(p)
        else:
            docs.append([p])
    for ps in por_archivo.values():
        p1 = [p for p in ps if p["plantilla"] == "hoja_vida_p1"]
        p2 = [p for p in ps if p["plantilla"] == "hoja_vida_p2"]
        while p1 and p2:
            docs.append([p1.pop(0), p2.pop(0)])
        hv1 += p1
        hv2 += p2
    while hv1 and hv2:
        docs.append([hv1.pop(0), hv2.pop(0)])
    docs += [[p] for p in hv1 + hv2]
    return docs


def unir_documento(ps, pl):
    """Junta las páginas de un documento: valores y zonas (las de la pág. 2 con sufijo), códigos, señales del PDF
    nativo, filas de la Hoja de Vida y etiquetas vecinas de cada zona.

    - Dos documentos en una misma foto se distinguen por su posición en la imagen ("[documento 2]").
    - Los valores de cabecera de la pág. 2 se usan para el timbre solo si falta la pág. 1.
    - `formato_ajeno`: etiquetas impresas del formato FUERA de cada zona pero en las líneas vecinas (hasta 30 pt
      arriba o abajo). Si aparecen dentro del valor leído, la zona se leyó corrida una línea (desalineación), no es
      un dato cambiado.
    """
    pid0 = ps[0].get("plantilla")
    tipo = P.TIPO_DOC.get(pid0) if pid0 else None
    archivos = list(dict.fromkeys(p["archivo"] for p in ps))
    unidades = list(dict.fromkeys(p["unidad"] + (f" [documento {p['doc_en_imagen'] + 1}]" if p.get("doc_en_imagen") else "")
                                  for p in ps))
    doc = {"carpeta": ps[0]["carpeta"], "archivo": " + ".join(unidades), "archivos": archivos,
           "tipo_documento": tipo or "desconocido", "plantilla": pid0, "es_ocr": any(p.get("es_ocr") for p in ps),
           "ilegible": any(p.get("ilegible") for p in ps), "copia": any(p.get("copia") for p in ps),
           "motivo_ilegible": next((p.get("motivo_ilegible") for p in ps if p.get("motivo_ilegible")), ""),
           "no_reconocido": any(p.get("no_reconocido") for p in ps),
           "motivo_no_reconocido": next((p["motivo"] for p in ps if p.get("no_reconocido") and p.get("motivo")), ""),
           "valores": {}, "senales": [],
           "codigos": {"timbre": [], "barras": []}, "meta": ps[0].get("meta") or {}, "entrada": ps[0].get("entrada")}
    zonas_doc = {}
    if tipo == "hoja_vida":
        ps = sorted(ps, key=lambda p: p["plantilla"])
        if len(ps) == 1:
            falta = 2 if ps[0]["plantilla"] == "hoja_vida_p1" else 1
            doc["pagina_faltante"] = f"falta la página {falta} de la Hoja de Vida"
    for p in ps:
        if not p.get("plantilla") or p.get("ilegible"):
            continue
        suf = " (pág. 2)" if p["plantilla"] == "hoja_vida_p2" else ""
        for z, v in (p.get("valores") or {}).items():
            doc["valores"][z + suf] = v
        for z, info in pl[p["plantilla"]]["zonas"].items():
            if info["tipo"] != "cuerpo":
                zonas_doc[z + suf] = info
        for k in ("timbre", "barras"):
            doc["codigos"][k] += (p.get("codigos") or {}).get(k, [])
        if not p.get("es_ocr"):
            pn = dict(p)
            doc["senales"] += K.senales_nativo(pn, pl[p["plantilla"]])
            doc["revisiones_pdf"] = p.get("revisiones_pdf")
    if tipo == "hoja_vida":
        doc["filas"] = HV.parsear([p.get("lineas_cuerpo", []) for p in ps if not p.get("ilegible")])
        doc["senales"] += K.senales_codigos(doc, {z: v for z, v in doc["valores"].items() if "(pág. 2)" not in z})
    doc["zonas"] = zonas_doc
    doc["formato_ajeno"] = {}
    for p in ps:
        if not p.get("plantilla") or p.get("ilegible"):
            continue
        suf = " (pág. 2)" if p["plantilla"] == "hoja_vida_p2" else ""
        fijas = pl[p["plantilla"]]["palabras_fijas"]
        for z, info in pl[p["plantilla"]]["zonas"].items():
            r = info["rect"]
            doc["formato_ajeno"][z + suf] = {t for x0, y0, x1, y1, txt in fijas
                                             if (x1 < r[0] or x0 > r[2] or y1 < r[1] or y0 > r[3])
                                             and y1 >= r[1] - 30 and y0 <= r[3] + 30
                                             for t in N.clave_texto(txt).split() if len(t) >= 4 and t.isalpha()}
    doc["formato"] = "PDF_NATIVO" if not doc["es_ocr"] else ("IMAGEN" if doc["entrada"] == "imagen" else "ESCANEADO")
    return doc


def _json_ameco(doc, oficiales):
    """El JSON oficial del certificado: por N° leído; si no se encuentra, por versión (Rev.02 / Rev.03)."""
    amecos = oficiales.get("ameco", {})
    leido = N.codigo(doc["valores"].get("numero_certificado", {}).get("texto", ""))
    if leido in amecos:
        return leido, amecos[leido]
    if doc["es_ocr"]:
        for k in amecos:
            if N.codigo_ocr(k) == N.codigo_ocr(leido):
                return k, amecos[k]
    version = "Rev.02" if doc["plantilla"] == "ameco_rev02" else "Rev.03"
    cand = [k for k, js in amecos.items() if version in js.get("Formato", "")]
    return (cand[0], amecos[cand[0]]) if len(cand) == 1 else (None, None)


def dictaminar(carpeta, paginas, registro, dir_web, req, hoy, pl, oficiales=None):
    """Dictamen de cada documento de una carpeta y resultado de la persona (autenticidad general y aptitud).

    oficiales: los JSON ya cargados en memoria; si no se dan, se leen de dir_web/<carpeta>.
    Sin dato oficial no hay con qué verificar: el documento no se analiza, solo se extraen sus datos.
    En cada fila, `archivo_oficial` es el nombre del JSON y `archivo_oficial_clave` el N° del certificado;
    `filas_hoja_vida` son las filas leídas de la sección variable, para revisarlas.
    """
    declarado = registro.get(carpeta, {})
    oficiales = oficiales if oficiales is not None else oficiales_de(carpeta, dir_web)
    filas = []
    for grupo in agrupar(paginas):
        doc = unir_documento(grupo, pl)
        if doc["no_reconocido"] or (doc["plantilla"] is None and not doc["ilegible"]):
            filas.append(_fila(doc, {"veredicto": K.NO_RECONOCIDO, "campos": [], "senales": [],
                                     "motivos": [doc["motivo_no_reconocido"] or
                                                 "el archivo no corresponde a ninguno de los documentos solicitados"]}, ""))
            continue
        if doc["plantilla"] is None:
            r = {"veredicto": K.NO_LEGIBLE, "campos": [], "senales": [], "motivos": [doc["motivo_ilegible"]]}
            filas.append(_fila(doc, r, ""))
            continue
        if doc["tipo_documento"] == "hoja_vida":
            clave_js, js = "hoja_vida_conductor.json", oficiales["hoja_vida"]
        else:
            clave_js, js = _json_ameco(doc, oficiales)
        if js is None:
            r = {"veredicto": K.SIN_DATO_OFICIAL, "campos": [], "senales": [], "emision": "",
                 "motivos": ["no se entregó el dato oficial (JSON) de este documento: no se verificó, solo se "
                             "extrajeron sus datos"]}
        else:
            r = K.comparar_documento(doc, js, doc["zonas"], declarado, hoy)
        fila = _fila(doc, r, oficiales["archivo_ameco"].get(clave_js, clave_js or "") if js is not None else "")
        fila["json_extraido"] = EX.extraer(doc)
        if doc.get("filas"):
            fila["filas_hoja_vida"] = {k: [{c: str(x.get(c, "")) for c in ("Registro", "Clase", "fecha_txt", "Municipalidad",
                                                                             "Tipo", "Fecha anotación", "Detalle")
                                            if c in x} for x in doc["filas"][k]] for k in ("licencias", "anotaciones")} | \
                {"antecedentes": doc["filas"]["antecedentes"]}
        fila["archivo_oficial_clave"] = clave_js
        filas.append(fila)
    puesto = declarado.get("puesto") or req["puesto_por_defecto"]
    evaluables = [f for f in filas if f["veredicto"] != K.NO_RECONOCIDO]
    aptitud, motivos_apt = Q.evaluar_persona(puesto, evaluables, oficiales, req, hoy)
    v = Counter(f["veredicto"] for f in evaluables)
    general = next((x for x in (K.FALSIFICADO, K.OBSERVADO, K.NO_LEGIBLE, K.NO_VALIDO, K.DESACTUALIZADO,
                                K.SIN_DATO_OFICIAL, K.AUTENTICO) if v[x]), K.OBSERVADO)
    persona = {"carpeta": carpeta, "dni": str(declarado.get("rut") or ""), "declarado": bool(declarado),
               "documentos": len(evaluables), "falsificados": v[K.FALSIFICADO], "observados": v[K.OBSERVADO] + v[K.NO_LEGIBLE],
               "desactualizados": v[K.DESACTUALIZADO],
               "tipos_falsificados": [f["tipo_documento"] for f in evaluables if f["veredicto"] == K.FALSIFICADO],
               "veredicto": general, "puesto": puesto, "aptitud": aptitud, "motivos_aptitud": motivos_apt}
    return persona, filas


def _fila(doc, r, archivo_oficial):
    """Fila de resultado de un documento, en el formato que usa el reporte."""
    return {"carpeta": doc["carpeta"], "archivo": doc["archivo"], "archivos": doc["archivos"],
            "tipo_documento": doc["plantilla"] or doc["tipo_documento"], "formato": doc.get("formato", ""),
            "veredicto": r["veredicto"], "motivos": r["motivos"], "campos": r["campos"], "senales": r["senales"],
            "misma_instancia": None, "archivo_oficial": archivo_oficial, "alineacion_ok": not doc.get("ilegible"),
            "alineacion": {}, "residual_etiquetas_pt": None, "producer": (doc.get("meta") or {}).get("producer", ""),
            "emision": r.get("emision", "")}


def huella_version():
    """Huella de 16 caracteres del código, la configuración, las plantillas y el modelo OCR usados."""
    h = hashlib.sha256()
    for f in sorted((C.ROOT / "verificacion").rglob("*.py")) + sorted((C.ROOT / "config").glob("*.yaml")) \
            + sorted(C.DIR_OFICIAL.glob("*.pdf")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    if C.MODELO_PARSEQ.exists():
        st = C.MODELO_PARSEQ.stat()
        h.update(f"{st.st_size}-{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def verificar(carpetas, registro, dir_post=C.DIR_POSTULANTES, dir_web=C.DIR_WEB, workers=C.CHILE_WORKERS,
              usar_gpu=True, log=print, progreso=None, cancelado=lambda: False, hoy=None):
    """Verifica las carpetas dadas: lectura en paralelo, OCR, dictamen y aptitud.

    Devuelve (personas, documentos, no_admitidos, meta). Cada documento lleva un nombre único dentro de su carpeta
    (el reporte lo usa como clave). Levanta DetenidoPorRecursos si la memoria se vuelve crítica o se cancela.
    """
    t0 = time.time()
    hoy = hoy or date.today()
    pl = P.cargar()
    req = Q.cargar()
    lista, no_admitidos = tareas(carpetas, Path(dir_post))
    log(f"{len(carpetas)} postulantes | {len(lista)} páginas/imágenes por leer | {len(no_admitidos)} no admitidos")
    paginas = []
    vigia = R.Vigia().iniciar()
    with Pool(workers, initializer=L.init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(L.analizar, lista, chunksize=2), 1):
            if vigia.alerta or cancelado():
                pool.terminate()
                vigia.detener()
                raise DetenidoPorRecursos(vigia.alerta or "cancelado")
            paginas += paginas_de_unidad(r)
            if progreso:
                progreso("lectura", i, len(lista))
            if i % 200 == 0:
                log(f"  {i}/{len(lista)} leídas ({time.time() - t0:.0f}s)")
    vigia.detener()
    modelo = leer_ocr(paginas, usar_gpu, log, (lambda h, t: progreso("ocr", h, t)) if progreso else None)
    for p in paginas:
        if p.get("plantilla") and not p.get("ilegible"):
            valores_de_pagina(p, pl)
    por_carpeta = defaultdict(list)
    for p in paginas:
        por_carpeta[p["carpeta"]].append(p)
    personas, documentos = [], []
    for c in carpetas:
        persona, filas = dictaminar(c, por_carpeta.get(c, []), registro, Path(dir_web), req, hoy, pl)
        vistos = Counter()
        for f in filas:
            vistos[f["archivo"]] += 1
            if vistos[f["archivo"]] > 1:
                f["archivo"] = f"{f['archivo']} ({vistos[f['archivo']]})"
        personas.append(persona)
        documentos += filas
    meta = {"fecha": datetime.now().isoformat(timespec="seconds"), "huella_version": huella_version(),
            "postulantes": len(personas), "documentos_postulante": len(documentos), "archivos_no_admitidos": len(no_admitidos),
            "modelo_ocr": modelo, "duracion_s": round(time.time() - t0, 1), "fecha_referencia": str(hoy),
            "dias_vigencia_hoja_vida": C.HVID_DIAS_VIGENCIA}
    return personas, documentos, no_admitidos, meta


def main():
    """Línea de comandos: verifica las carpetas de postulantes y escribe el reporte en --salida."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--carpetas", nargs="*", default=None, help="solo estas carpetas de postulantes")
    ap.add_argument("--workers", type=int, default=C.CHILE_WORKERS)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--postulantes", default=str(C.DIR_POSTULANTES))
    ap.add_argument("--oficiales", default=str(C.DIR_WEB))
    ap.add_argument("--salida", default=str(C.DIR_RESULTADOS / "chile"))
    ap.add_argument("--hoy", default=None, help="fecha de referencia AAAA-MM-DD (por defecto, hoy)")
    ap.add_argument("--ventana", type=int, default=60)
    ap.add_argument("--subconjunto", choices=["ajuste", "reservado"], default=None,
                    help="medición: ajuste = postulantes con N impar del registro; reservado = N par")
    a = ap.parse_args()
    dir_post = Path(a.postulantes)
    registro = leer_registro(dir_post / "candidatos.xlsx")
    carpetas = a.carpetas or sorted(d.name for d in dir_post.iterdir() if d.is_dir())
    if a.subconjunto:
        impar = a.subconjunto == "ajuste"
        carpetas = [c for c in carpetas if (registro.get(c, {}).get("n") or 0) % 2 == (1 if impar else 0)]
    if a.limite:
        carpetas = carpetas[:a.limite]
    hoy = date.fromisoformat(a.hoy) if a.hoy else None
    try:
        with R.Candado("pipeline"):
            R.verificar(vram_gb=0 if a.cpu else C.PRESUPUESTO_VRAM_OCR_GB, ram_gb=C.PRESUPUESTO_RAM_GB, ventana_s=a.ventana)
            personas, documentos, no_adm, meta = verificar(carpetas, registro, dir_post, a.oficiales, a.workers,
                                                           not a.cpu, hoy=hoy)
    except (R.RecursosInsuficientes, DetenidoPorRecursos) as e:
        print(f"NO SE COMPLETÓ: {e}", flush=True)
        raise SystemExit(2) from e
    _, xlsx = escribir_reporte(personas, documentos, meta, Path(a.salida), no_adm)
    (Path(a.salida) / "resultados.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False, default=str)
                                                                for d in documentos), encoding="utf-8")
    print("\nPostulantes:", dict(Counter(p["veredicto"] for p in personas)))
    print("Aptitud:    ", dict(Counter(p["aptitud"] for p in personas)))
    print("Documentos: ", dict(Counter(d["veredicto"] for d in documentos)))
    print(f"Reporte: {xlsx} | Duración: {meta['duracion_s']}s", flush=True)


if __name__ == "__main__":
    main()
