"""Lectura de PDF con PyMuPDF: palabras visibles y ocultas, dibujos, imágenes y anotaciones, página renderizada y
ubicación de un PDF nativo sobre su plantilla."""

import math

import cv2
import numpy as np
import pymupdf

from . import config as C
from .cajas import centro


def palabras_pdf(page, incluir_ocultas=False):
    """Palabras horizontales de un PDF con su caja, fuente, tamaño, color y si están ocultas.

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


def render_gris(page, dpi=C.DPI):
    """Página PDF rasterizada en gris (uint8) a `dpi`."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def palabras_no_visibles(palabras, img):
    """Separa las palabras de la capa de texto que no se ven en la página renderizada a C.DPI (mismo color que el
    fondo, diminutas o tapadas por algo uniforme). Devuelve (visibles, ocultas)."""
    visibles, ocultas = [], []
    for p in palabras:
        x0, y0, x1, y1 = (int(round(v * C.ESCALA)) for v in p["bbox"])
        region = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if region.size == 0 or (y1 - y0) < 4 or region.std() < 6:
            ocultas.append(p)
        else:
            visibles.append(p)
    return visibles, ocultas


def imagen_pagina_completa(page):
    """True si una imagen cubre más del 80 % de la página (un escaneo guardado como PDF)."""
    area = page.rect.width * page.rect.height
    return any(pymupdf.Rect(i["bbox"]).get_area() > 0.8 * area for i in page.get_image_info())


def transformar_rect(r, M):
    """Rectángulo (x0, y0, x1, y1) llevado a coordenadas de la plantilla con la transformación afín `M` (sin cambio
    si `M` es None)."""
    if M is None:
        return [float(v) for v in r[:4]]
    esq = cv2.transform(np.float32([[[r[0], r[1]], [r[2], r[3]]]]), np.float32(M))[0]
    return [float(esq[0, 0]), float(esq[0, 1]), float(esq[1, 0]), float(esq[1, 1])]


def transformar_palabras(palabras, M):
    """Lleva las cajas de las palabras a coordenadas de la plantilla con la transformación afín `M`."""
    for p in palabras:
        p["bbox"] = transformar_rect(p["bbox"], M)
    return palabras


def objetos_pagina(page, M):
    """Imágenes, anotaciones y dibujos de la página, en coordenadas de la plantilla."""
    return {
        "imagenes": [transformar_rect(i["bbox"], M) for i in page.get_image_info()],
        "anotaciones": [[a.type[1], *transformar_rect(tuple(a.rect), M)] for a in page.annots()],
        "dibujos": [transformar_rect(d[:4], M) + d[4:] for d in dibujos_pdf(page)],
    }


def geometria_nativa(palabras, plantilla):
    """Transformación doc -> plantilla estimada con las palabras fijas (etiquetas, títulos).

    Solo se usan palabras sin ambigüedad (una vez en la plantilla y una vez en el documento); hacen falta 4."""
    por_texto = {}
    for w in plantilla["palabras_fijas"]:
        por_texto.setdefault(w[4], []).append(centro(w))
    doc_por_texto = {}
    for p in palabras:
        doc_por_texto.setdefault(p["texto"], []).append(centro(p["bbox"]))
    src, dst = [], []
    for t, cs in por_texto.items():
        ds = doc_por_texto.get(t, [])
        if len(cs) == 1 and len(ds) == 1:
            src.append(ds[0])
            dst.append(cs[0])
    if len(src) < 4:
        return None
    M, _ = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst), method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
    if M is None:
        return None
    proy = cv2.transform(np.float32(src)[None], M)[0]
    err = np.linalg.norm(proy - np.float32(dst), axis=1)
    escala = math.hypot(M[0, 0], M[1, 0])
    return {"M": M.tolist(), "dx": float(M[0, 2]), "dy": float(M[1, 2]), "escala": escala,
            "rotacion": math.degrees(math.atan2(M[1, 0], M[0, 0])),
            "residual_pt": float(np.median(err)), "puntos": len(src)}
