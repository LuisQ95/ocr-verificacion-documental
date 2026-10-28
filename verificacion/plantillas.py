"""Registro de plantillas: deduce automáticamente las zonas de cada campo a partir de los PDF
en blanco de official_file/. No usa ningún conocimiento del generador de datos.

Reglas de deducción (sobre la plantilla en blanco):
  - Etiqueta pequeña en mayúsculas + línea guía debajo  -> campo (zona entre etiqueta y línea)
  - Etiqueta sin línea que termina en ":"               -> campo a la derecha de la etiqueta
  - Etiqueta sin línea                                  -> campo debajo de la etiqueta
  - Banda de encabezado de tabla + líneas de fila       -> una zona por celda
  - Recuadro vacío (sin relleno, alto 20-40 pt)         -> campo dentro del recuadro

El resultado se guarda en modelos/plantillas/<id>.json junto con la imagen de la plantilla usada
para alinear. Cada zona lleva además:
  - "tipo":   cómo se compara (identidad, codigo, fecha, numero, categorico, descriptivo)
  - "cambio": cómo se comporta en el tiempo (identidad, permanente, instancia, temporal)
Ambos se proponen por el nombre de la etiqueta y se pueden corregir a mano en el JSON. Un JSON con
"manual": true nunca se sobrescribe.

Formatos reales: para un formato nuevo o una versión nueva de un formato, se agrega su PDF en blanco
(o su JSON de zonas anotado a mano) con un id distinto, p. ej. "01_licencia_conducir@v2". Las versiones
anteriores siguen sirviendo para clasificar documentos antiguos.
"""

import json
import re
import unicodedata

import cv2
import numpy as np
import pymupdf

from . import config as C


def slug(texto):
    """Texto en minúsculas, sin tildes y con "_" en lugar de todo lo que no sea letra o número."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def _spans_horizontales(page):
    """Tramos de texto horizontales de la página; el texto rotado (marca de agua) no es contenido de campos."""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            if abs(l["dir"][1]) > 0.01:
                continue
            for s in l["spans"]:
                if s["text"].strip():
                    spans.append({"texto": s["text"].strip(), "bbox": pymupdf.Rect(s["bbox"]),
                                  "size": s["size"], "font": s["font"]})
    return spans


def palabras_pdf(page, incluir_ocultas=False):
    """Palabras horizontales de un PDF con su caja, fuente y color.

    Descarta (salvo incluir_ocultas=True) el texto que no se ve: modo de dibujo invisible (3),
    opacidad casi nula o fuera de la página. Un falsificador puede esconder ahí los datos oficiales
    mientras lo visible dice otra cosa. El texto rotado (marca de agua) tampoco es contenido."""
    palabras = []
    for s in page.get_texttrace():
        if abs(s["dir"][1]) > 0.01:
            continue
        oculta = s["type"] == 3 or s["opacity"] < 0.05
        if oculta and not incluir_ocultas:
            continue
        actual = None
        for uc, _glifo, _origen, bbox in s["chars"]:
            ch = chr(uc)
            r = pymupdf.Rect(bbox)
            if ch.isspace() or not r.intersects(page.rect):
                actual = None
                continue
            if actual is None:
                actual = {"texto": "", "bbox": r, "font": s["font"], "size": s["size"],
                          "color": s["color"], "oculta": oculta}
                palabras.append(actual)
            actual["texto"] += ch
            actual["bbox"] = actual["bbox"] | r
    for p in palabras:
        p["bbox"] = [p["bbox"].x0, p["bbox"].y0, p["bbox"].x1, p["bbox"].y1]
    return palabras


def dibujos_pdf(page):
    """Rectángulos de los objetos vectoriales (con su relleno y color de trazo)."""
    salida = []
    for g in page.get_drawings():
        r = g["rect"]
        salida.append([r.x0, r.y0, r.x1, r.y1, list(g["fill"]) if g.get("fill") else None,
                       list(g["color"]) if g.get("color") else None])
    return salida


REGLAS_ZONA = [
    ("dni", "identidad", "identidad"),
    ("apellido", "identidad", "identidad"),
    ("nombre", "identidad", "identidad"),
    ("fecha_de_nacimiento", "fecha", "identidad"),
    ("codigo_de_verificacion", "codigo", "instancia"),
    ("integridad", "codigo", "instancia"),
    ("fecha_de_emision", "fecha", "instancia"),
    ("valido_hasta", "fecha", "instancia"),
    ("n_de_certificado", "codigo", "instancia"),
    ("n_de_licencia", "codigo", "permanente"),
    ("n_de_registro", "codigo", "permanente"),
    ("fecha_de_expedicion", "fecha", "temporal"),
    ("fecha_de_revalidacion", "fecha", "temporal"),
    ("fecha_de_obtencion", "fecha", "temporal"),
    ("fecha", "fecha", "temporal"),
    ("monto", "numero", "temporal"),
    ("puntos", "numero", "temporal"),
    ("n_de_infracciones", "numero", "temporal"),
    ("estado", "categorico", "temporal"),
    ("categoria", "categorico", "temporal"),
    ("gravedad", "categorico", "temporal"),
    ("situacion", "categorico", "temporal"),
    ("nivel", "categorico", "temporal"),
    ("restricciones", "categorico", "temporal"),
    ("recuadro", "categorico", "temporal"),
    ("codigo", "codigo", "temporal"),
]
"""(fragmento del nombre de zona, tipo de comparación, comportamiento en el tiempo); gana el primero que coincide."""


def tipo_zona(nombre, zona):
    """(tipo, cambio) de una zona según su nombre o columna, con REGLAS_ZONA; por defecto descriptivo y temporal."""
    clave_nombre = zona.get("columna") or nombre
    if zona.get("columna") == "n":
        return "numero", "temporal"
    for fragmento, tipo, cambio in REGLAS_ZONA:
        if fragmento in clave_nombre or (fragmento == "recuadro" and nombre.startswith("recuadro")):
            return tipo, cambio
    return "descriptivo", "temporal"


def _es_etiqueta(s):
    """Etiqueta de campo: texto pequeño (≤ 8,1 pt), en mayúsculas, corto y con letras."""
    t = s["texto"]
    return s["size"] <= 8.1 and t == t.upper() and len(t) <= 40 and any(c.isalpha() for c in t)


def construir_zonas(pdf_path):
    """Zonas de la primera página de una plantilla en blanco, con las reglas del módulo, en este orden:

    1. Tablas: banda con relleno claro (ancho > 400, alto < 25) con textos dentro; una zona por celda.
    2. Campos con línea guía corta debajo de una etiqueta.
    3. Etiquetas sin línea: a la derecha si terminan en ":", si no, debajo.
    4. Recuadros vacíos sin relleno.

    Devuelve (información de la plantilla, página) para poder rasterizarla.
    """
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    spans = _spans_horizontales(page)
    dibujos = page.get_drawings()

    lineas, rects = [], []
    for g in dibujos:
        for it in g["items"]:
            if it[0] == "l" and abs(it[1].y - it[2].y) < 0.1:
                lineas.append((min(it[1].x, it[2].x), it[1].y, max(it[1].x, it[2].x)))
            elif it[0] == "re":
                rects.append((pymupdf.Rect(it[1]), g.get("fill")))

    zonas, tablas = {}, {}

    usados = set()
    for r, fill in rects:
        if not fill or not (r.width > 400 and r.height < 25) or min(fill) < 0.5:
            continue
        cabeceras = sorted([s for s in spans if r.contains(s["bbox"].tl + (1, 1)) and s["texto"].strip()],
                           key=lambda s: s["bbox"].x0)
        if not cabeceras:
            continue
        nombre_tabla = f"tabla{len(tablas) + 1}"
        filas_y = sorted({y for x0, y, x1 in lineas if y > r.y1 and x1 - x0 > r.width * 0.9})
        bordes = [r.y1] + filas_y
        cols = []
        for i, s in enumerate(cabeceras):
            x0 = s["bbox"].x0 - 3
            x1 = cabeceras[i + 1]["bbox"].x0 - 3 if i + 1 < len(cabeceras) else r.x1
            cols.append((slug(s["texto"]) or f"col{i + 1}", x0, x1))
            usados.add(id(s))
        for f in range(len(bordes) - 1):
            for col, x0, x1 in cols:
                zonas[f"{nombre_tabla}_f{f + 1}_{col}"] = {
                    "rect": [x0, bordes[f] + 0.5, x1, bordes[f + 1] - 0.5], "clase": "celda",
                    "tabla": nombre_tabla, "fila": f + 1, "columna": col}
        tablas[nombre_tabla] = {"filas": len(bordes) - 1, "columnas": [c[0] for c in cols]}

    guias = [(x0, y, x1) for x0, y, x1 in lineas if x1 - x0 < 500]
    for x0, y, x1 in guias:
        cands = [s for s in spans if _es_etiqueta(s) and id(s) not in usados
                 and abs(s["bbox"].x0 - x0) < 2 and 8 < y - s["bbox"].y1 < 25]
        if not cands:
            continue
        et = max(cands, key=lambda s: s["bbox"].y1)
        usados.add(id(et))
        zonas[_unico(zonas, slug(et["texto"]))] = {
            "rect": [x0 - 2, et["bbox"].y1 + 0.5, x1 + 2, y + 1.5], "clase": "campo", "etiqueta": et["texto"]}

    for s in spans:
        if id(s) in usados or not _es_etiqueta(s):
            continue
        b = s["bbox"]
        if s["texto"].endswith(":"):
            rect = [b.x1 + 2, b.y0 - 2, b.x1 + 220, b.y1 + 2]
        else:
            rect = [b.x0 - 2, b.y1 + 0.5, b.x0 + 180, b.y1 + 17]
        zonas[_unico(zonas, slug(s["texto"]))] = {"rect": rect, "clase": "campo", "etiqueta": s["texto"]}

    n = 0
    for r, fill in rects:
        if fill is None and 20 <= r.height <= 40 and r.width > 200:
            if not any(r.intersects(s["bbox"]) for s in spans):
                n += 1
                zonas[f"recuadro{n}"] = {"rect": [r.x0 + 1, r.y0 + 1, r.x1 - 1, r.y1 - 1], "clase": "recuadro"}

    for nombre, z in zonas.items():
        z["tipo"], z["cambio"] = tipo_zona(nombre, z)

    palabras_fijas = [[*p["bbox"], p["texto"]] for p in palabras_pdf(page)]
    return {
        "id": pdf_path.stem.removeprefix("plantilla_"),
        "archivo": pdf_path.name,
        "version": 1,
        "vigente_desde": None,
        "manual": False,
        "tamano": [page.rect.width, page.rect.height],
        "zonas": zonas,
        "tablas": tablas,
        "palabras_fijas": palabras_fijas,
        "dibujos": dibujos_pdf(page),
    }, page


def _unico(zonas, nombre):
    """`nombre`, o `nombre_2`, `nombre_3`... si ya existe en `zonas`."""
    if nombre not in zonas:
        return nombre
    i = 2
    while f"{nombre}_{i}" in zonas:
        i += 1
    return f"{nombre}_{i}"


def registrar_todas():
    """Procesa official_file/ y escribe las zonas y la imagen de cada plantilla (las editadas a mano no se pisan)."""
    C.DIR_ZONAS.mkdir(parents=True, exist_ok=True)
    registro = []
    for pdf in sorted(C.DIR_OFICIAL.glob("*.pdf")):
        info, page = construir_zonas(pdf)
        destino = C.DIR_ZONAS / f"{info['id']}.json"
        if destino.exists() and json.loads(destino.read_text(encoding="utf-8")).get("manual"):
            continue
        pix = page.get_pixmap(dpi=C.DPI, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        cv2.imwrite(str(C.DIR_ZONAS / f"{info['id']}.png"), img)
        (C.DIR_ZONAS / f"{info['id']}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        registro.append(info)
    return registro


def cargar_todas():
    """Plantillas registradas en modelos/plantillas, con su imagen en gris para alinear."""
    plantillas = {}
    for js in sorted(C.DIR_ZONAS.glob("*.json")):
        info = json.loads(js.read_text(encoding="utf-8"))
        info["imagen"] = cv2.imread(str(C.DIR_ZONAS / f"{info['id']}.png"), cv2.IMREAD_GRAYSCALE)
        plantillas[info["id"]] = info
    return plantillas


if __name__ == "__main__":
    for p in registrar_todas():
        celdas = sum(z["clase"] == "celda" for z in p["zonas"].values())
        print(f"{p['id']}: {len(p['zonas']) - celdas} campos, {celdas} celdas, tablas={list(p['tablas'])}")
        for k, z in p["zonas"].items():
            if z["clase"] != "celda":
                print(f"    {k:28s} {z['clase']:9s} {[round(v) for v in z['rect']]}")
