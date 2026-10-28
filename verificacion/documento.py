"""Análisis de un documento en CPU (sin GPU): lectura nativa, QR, metadatos, clasificación
contra las plantillas, alineación y control de calidad de la alineación.

Se ejecuta en procesos paralelos; el OCR (GPU) ocurre después en el proceso principal.
"""

import json
import math
from collections import Counter

import cv2
import numpy as np
import pymupdf
import zxingcpp

from PIL import Image, ImageOps

from . import config as C
from .plantillas import dibujos_pdf, palabras_pdf

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
EXT_ADMITIDAS = EXT_IMAGEN | {".pdf"}
FORMATOS_IMAGEN = ("JPEG", "PNG", "TIFF", "BMP", "WEBP")
"""Formatos que Pillow puede abrir: solo los de fotos y escaneos. Pillow reconoce el formato por el contenido, no
por la extensión, y la mayoría de sus fallas de seguridad están en decodificadores de formatos poco usados (FPX, EPS,
GIF, JPEG 2000, SPIDER...): con esta lista un archivo malicioso disfrazado de foto no llega a ellos."""

_PLANTILLAS = None
_FEATURES = None
"""(keypoints, descriptores) ORB de cada plantilla, calculados una vez por proceso."""
_MASCARAS = None
"""Líneas finas de cada plantilla, para borrarlas antes del OCR."""


def init_worker(plantillas):
    """Inicializa cada proceso con las plantillas y sus descriptores ORB (se calculan una vez)."""
    global _PLANTILLAS, _FEATURES, _MASCARAS
    cv2.setNumThreads(1)
    _PLANTILLAS = plantillas
    orb = _orb()
    _FEATURES = {pid: orb.detectAndCompute(p["imagen"], None) for pid, p in plantillas.items()}
    _MASCARAS = {pid: mascara_lineas(p["imagen"]) for pid, p in plantillas.items()}


def _orb():
    """Detector ORB con los parámetros usados para documentos (muchos puntos, umbral FAST bajo)."""
    return cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10)


def render_gris(page, dpi=C.DPI):
    """Página PDF rasterizada en gris (uint8) a `dpi`."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def corregir_iluminacion(img_gris, kernel=41):
    """Normaliza la iluminación local: estima el fondo (cierre morfológico, que borra el texto) y
    divide por él. Una sombra o un lado más oscuro de una foto dejan de alterar los umbrales.
    El fondo se acota (>= 160) para no 'aclarar' bandas oscuras legítimas, como un encabezado."""
    fondo = cv2.morphologyEx(img_gris, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)))
    fondo = np.maximum(fondo, 160).astype(np.float32)
    return np.clip(img_gris.astype(np.float32) * 255.0 / fondo, 0, 255).astype(np.uint8)


def blanquear(img_gris):
    """Preprocesamiento previo al OCR: corrige la iluminación y lleva a blanco lo muy claro
    (marca de agua, fondo, ruido leve)."""
    out = corregir_iluminacion(img_gris)
    out[out > C.UMBRAL_BLANQUEO] = 255
    return out


def mascara_lineas(plantilla_img, largo_px=60):
    """Ubica las líneas finas (guías bajo los campos, rejilla de tablas, bordes de recuadros) en la
    plantilla en blanco. Se hace sobre la plantilla porque es una imagen limpia: en un escaneo
    desenfocado, una fila de píxeles de texto en negrita puede parecer una línea.

    Solo cuentan las líneas sobre fondo claro: en una banda oscura con texto blanco (encabezado), los espacios
    entre letras parecen "líneas" verticales y no deben tocarse. La máscara se dilata 2 px para absorber el
    pequeño error residual de la alineación."""
    tinta = (plantilla_img < 235).astype(np.uint8) * 255
    mascara = np.zeros_like(tinta)
    for fino, grueso in (((largo_px, 1), (largo_px, 7)), ((1, largo_px), (7, largo_px))):
        lineas = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, fino))
        solidos = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, grueso))
        mascara |= cv2.subtract(lineas, cv2.dilate(solidos, np.ones((3, 3), np.uint8)))
    fondo_claro = cv2.blur(plantilla_img, (31, 31)) > 170
    return (cv2.dilate(mascara, np.ones((5, 5), np.uint8)) > 0) & fondo_claro


def quitar_lineas(img_alineada, mascara):
    """Borra las líneas de la plantilla en la imagen ya alineada: el detector de texto las confunde
    con texto ("-----", "7H77"). Solo se blanquean píxeles claros: si un trazo oscuro de texto
    toca una línea, se conserva."""
    out = img_alineada.copy()
    out[mascara & (img_alineada > 120)] = 255
    return out


def ancho_plantilla_px():
    """(ancho, alto) en px de una página A4 a la resolución de trabajo."""
    return round(595 * C.ESCALA), round(842 * C.ESCALA)


def recortar_documento(img):
    """Para fotos y escaneos sueltos: busca el borde del documento (el cuadrilátero más grande, de
    proporción tipo A4), corrige la perspectiva y lleva la imagen al tamaño de trabajo.
    Si no encuentra un borde claro, solo reescala (la alineación con la plantilla hace el resto).

    Solo se recorta con un contorno de proporción casi exacta (±0,06): un recorte equivocado (p. ej. que deja
    fuera un encabezado oscuro) deforma la página y es peor que no recortar; la tarjeta interior de un documento
    o el borde de la imagen tampoco cuentan como página. Sin recorte, la hoja ocupa ~70-80 % de una foto típica:
    se reescala a 1,3 veces el ancho de la plantilla para mantener una resolución cercana a la suya.
    Devuelve (imagen, se_recorto)."""
    ancho, alto = ancho_plantilla_px()
    h, w = img.shape
    esc = 1000 / max(h, w)
    chica = cv2.GaussianBlur(cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA), (5, 5), 0)
    bordes = cv2.dilate(cv2.Canny(chica, 50, 150), None)
    objetivo = alto / ancho
    candidatos = []
    for mascara in (bordes,):
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.3 * chica.size or area > 0.98 * chica.size:
                continue
            aprox = None
            for eps in (0.02, 0.03, 0.05):
                a = cv2.approxPolyDP(cv2.convexHull(c), eps * cv2.arcLength(c, True), True)
                if len(a) == 4:
                    aprox = a
                    break
            if aprox is None:
                continue
            pts = aprox.reshape(4, 2).astype(np.float32) / esc
            s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
            orden = np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])
            lados = np.linalg.norm(orden - np.roll(orden, -1, axis=0), axis=1)
            proporcion = (lados[1] + lados[3]) / max(1e-6, lados[0] + lados[2])
            if abs(proporcion - objetivo) <= 0.06:
                candidatos.append((-area, orden))
    if candidatos:
        orden = min(candidatos, key=lambda x: x[0])[1]
        M = cv2.getPerspectiveTransform(orden, np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]]))
        return cv2.warpPerspective(img, M, (ancho, alto), borderValue=255), True
    nuevo_ancho = round(ancho * 1.3)
    return cv2.resize(img, (nuevo_ancho, round(h * nuevo_ancho / w)), interpolation=cv2.INTER_AREA), False


def cargar_imagen(ruta):
    """Imagen suelta (foto o escaneo) -> gris, con la rotación EXIF aplicada."""
    with Image.open(ruta, formats=FORMATOS_IMAGEN) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L")).copy()


def leer_qr(img_gris):
    """Contenido del primer código QR de la imagen (JSON si lo es, si no {"_texto": ...}); None si no hay."""
    for r in zxingcpp.read_barcodes(img_gris):
        if r.format == zxingcpp.BarcodeFormat.QRCode and r.text:
            try:
                return json.loads(r.text)
            except json.JSONDecodeError:
                return {"_texto": r.text}
    return None


def tinta_por_zona(img_alineada, plantilla):
    """Fracción de píxeles oscuros en cada zona (para distinguir 'campo vacío' de 'texto no leído').

    El umbral < 170 incluye texto fino y gris (p. ej. Courier de 8 pt), que tras el escaneo queda claro."""
    tinta = {}
    for nombre, z in plantilla["zonas"].items():
        x0, y0, x1, y1 = (int(v * C.ESCALA) for v in z["rect"])
        recorte = img_alineada[max(0, y0 + 2):max(0, y1 - 2), max(0, x0 + 2):max(0, x1 - 2)]
        tinta[nombre] = float((recorte < 170).mean()) if recorte.size else 0.0
    return tinta


def _centro(b):
    """Centro de una caja (x0, y0, x1, y1)."""
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def clasificar_nativo(palabras):
    """Plantilla de un PDF nativo: la que comparte más palabras fijas. Devuelve (id, puntaje por plantilla)."""
    textos = Counter(p["texto"] for p in palabras)
    puntajes = {}
    for pid, p in _PLANTILLAS.items():
        fijas = Counter(w[4] for w in p["palabras_fijas"])
        comunes = sum(min(n, textos[t]) for t, n in fijas.items())
        puntajes[pid] = comunes / max(1, sum(fijas.values()))
    mejor = max(puntajes, key=puntajes.get)
    return mejor, puntajes


def geometria_nativa(palabras, plantilla):
    """Transformación doc -> plantilla estimada con las palabras fijas (etiquetas, títulos).

    Solo se usan palabras sin ambigüedad (una vez en la plantilla y una vez en el documento); hacen falta 4."""
    por_texto = {}
    for w in plantilla["palabras_fijas"]:
        por_texto.setdefault(w[4], []).append(_centro(w))
    doc_por_texto = {}
    for p in palabras:
        doc_por_texto.setdefault(p["texto"], []).append(_centro(p["bbox"]))
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


def transformar_palabras(palabras, M):
    """Lleva las cajas de las palabras a coordenadas de la plantilla con la transformación afín `M`."""
    M = np.float32(M)
    for p in palabras:
        x0, y0, x1, y1 = p["bbox"]
        esq = cv2.transform(np.float32([[[x0, y0], [x1, y1]]]), M)[0]
        p["bbox"] = [float(esq[0, 0]), float(esq[0, 1]), float(esq[1, 0]), float(esq[1, 1])]
    return palabras


def _homografia(kp_doc, des_doc, kp_tpl, des_tpl, norma):
    """Homografía documento -> plantilla con emparejamiento k-NN (prueba de Lowe 0,8) y RANSAC.

    Devuelve la matriz y su calidad (inliers, proporción, error medio, escala y rotación), o solo el número de
    coincidencias si no alcanza."""
    if des_doc is None or des_tpl is None or len(kp_doc) < 10:
        return None
    matcher = cv2.BFMatcher(norma)
    pares = matcher.knnMatch(des_doc, des_tpl, k=2)
    buenos = [m for m, n in (p for p in pares if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(buenos) < 10:
        return {"inliers": 0, "matches": len(buenos)}
    src = np.float32([kp_doc[m.queryIdx].pt for m in buenos])
    dst = np.float32([kp_tpl[m.trainIdx].pt for m in buenos])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return {"inliers": 0, "matches": len(buenos)}
    mask = mask.ravel().astype(bool)
    if mask.sum() < 4:
        return {"inliers": int(mask.sum()), "matches": len(buenos)}
    proy = cv2.perspectiveTransform(src[mask][None], H)[0]
    rmse = float(np.sqrt(np.mean(np.sum((proy - dst[mask]) ** 2, axis=1))))
    return {"H": H, "inliers": int(mask.sum()), "matches": len(buenos),
            "ratio": float(mask.sum() / len(buenos)), "rmse_px": rmse,
            "escala": float(math.sqrt(abs(np.linalg.det(H[:2, :2])))),
            "rotacion": float(math.degrees(math.atan2(H[1, 0], H[0, 0])))}


def refinar_con_etiquetas(alineada, plantilla, busqueda_px=30, min_corr=0.5):
    """Segundo paso de alineación: busca cada texto fijo de la plantilla (etiquetas, títulos, pie)
    en una vecindad pequeña de la imagen ya alineada y recalcula la transformación con esas
    correspondencias, repartidas por toda la página. Corrige errores locales (p. ej. el borde
    inferior cuando la primera alineación se apoyó sobre todo en el encabezado). Las palabras cortas (< 15 pt)
    se omiten porque son poco distintivas."""
    tpl = plantilla["imagen"]
    h, w = tpl.shape
    origen, destino, desplazamientos = [], [], []
    for x0, y0, x1, y1, _ in plantilla["palabras_fijas"]:
        if (x1 - x0) < 15:
            continue
        px0, py0 = max(0, int(x0 * C.ESCALA) - 3), max(0, int(y0 * C.ESCALA) - 3)
        px1, py1 = min(w, int(x1 * C.ESCALA) + 3), min(h, int(y1 * C.ESCALA) + 3)
        parche = tpl[py0:py1, px0:px1]
        sx0, sy0 = max(0, px0 - busqueda_px), max(0, py0 - busqueda_px)
        sx1, sy1 = min(w, px1 + busqueda_px), min(h, py1 + busqueda_px)
        zona = alineada[sy0:sy1, sx0:sx1]
        if parche.size == 0 or zona.shape[0] < parche.shape[0] or zona.shape[1] < parche.shape[1] \
                or parche.std() < 5:
            continue
        res = cv2.matchTemplate(zona, parche, cv2.TM_CCOEFF_NORMED)
        _, corr, _, (mx, my) = cv2.minMaxLoc(res)
        if corr < min_corr:
            continue
        cx_t, cy_t = (px0 + px1) / 2, (py0 + py1) / 2
        cx_i, cy_i = sx0 + mx + parche.shape[1] / 2, sy0 + my + parche.shape[0] / 2
        origen.append((cx_i, cy_i))
        destino.append((cx_t, cy_t))
        desplazamientos.append(math.hypot(cx_i - cx_t, cy_i - cy_t))
    info = {"puntos": len(origen),
            "desplazamiento_mediano_px": float(np.median(desplazamientos)) if desplazamientos else None,
            "desplazamiento_max_px": float(np.max(desplazamientos)) if desplazamientos else None}
    if len(origen) < 8:
        return alineada, info | {"aplicado": False}
    H, mask = cv2.findHomography(np.float32(origen), np.float32(destino), cv2.RANSAC, 3.0)
    if H is None:
        return alineada, info | {"aplicado": False}
    refinada = cv2.warpPerspective(alineada, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return refinada, info | {"aplicado": True, "inliers": int(mask.sum())}


def evaluar_alineacion(a, foto=False):
    """Control de calidad: devuelve (ok, motivo). En fotos la hoja ocupa una parte variable de la
    imagen y puede estar más girada: la escala y la rotación admitidas son más amplias (la homografía
    las corrige; lo que importa es que muchos puntos encajen con poco error)."""
    rango = C.RANGO_ESCALA_FOTO if foto else C.RANGO_ESCALA
    max_rot = C.MAX_ROTACION_FOTO if foto else C.MAX_ROTACION_GRADOS
    if not a or a.get("inliers", 0) < C.MIN_INLIERS:
        return False, f"pocos puntos coincidentes con la plantilla ({a.get('inliers', 0) if a else 0})"
    if a["ratio"] < C.MIN_RATIO_INLIERS:
        return False, f"proporción de coincidencias baja ({a['ratio']:.2f})"
    if a["rmse_px"] > C.MAX_RMSE_PX:
        return False, f"error de alineación alto ({a['rmse_px']:.1f} px)"
    if abs(a["rotacion"]) > max_rot:
        return False, f"rotación implausible ({a['rotacion']:.1f}°)"
    if not rango[0] <= a["escala"] <= rango[1]:
        return False, f"escala implausible ({a['escala']:.2f})"
    return True, ""


def alinear_escaneado(img, foto=False):
    """Clasifica por alineación: la plantilla con más puntos coincidentes gana.

    Si la alineación ORB no pasa el control de calidad, se intenta con SIFT (más robusto y más lento)."""
    kp, des = _orb().detectAndCompute(img, None)
    resultados = {pid: _homografia(kp, des, *_FEATURES[pid], cv2.NORM_HAMMING) for pid in _PLANTILLAS}
    mejor = max(resultados, key=lambda k: (resultados[k] or {}).get("inliers", 0))
    a = resultados[mejor] or {}
    ok, motivo = evaluar_alineacion(a, foto)
    metodo = "ORB"
    if not ok:
        sift = cv2.SIFT_create(nfeatures=5000)
        kp2, des2 = sift.detectAndCompute(img, None)
        kpt, dest = sift.detectAndCompute(_PLANTILLAS[mejor]["imagen"], None)
        a2 = _homografia(kp2, des2, kpt, dest, cv2.NORM_L2)
        ok2, motivo2 = evaluar_alineacion(a2, foto)
        if ok2 or (a2 and a2.get("inliers", 0) > a.get("inliers", 0)):
            a, ok, motivo, metodo = a2, ok2, motivo2, "SIFT"
    inliers = sorted(((r or {}).get("inliers", 0) for r in resultados.values()), reverse=True)
    return mejor, a, ok, motivo, metodo, {pid: (r or {}).get("inliers", 0) for pid, r in resultados.items()}, inliers


def _palabras_no_visibles(palabras, img):
    """Palabras de la capa de texto que no se ven en la página renderizada (mismo color que el
    fondo, diminutas o tapadas por algo uniforme). Se separan del contenido."""
    visibles, ocultas = [], []
    for p in palabras:
        x0, y0, x1, y1 = (int(round(v * C.ESCALA)) for v in p["bbox"])
        region = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if region.size == 0 or (y1 - y0) < 4 or region.std() < 6:
            ocultas.append(p)
        else:
            visibles.append(p)
    return visibles, ocultas


def _imagen_pagina_completa(page):
    """True si una imagen cubre más del 80 % de la página (un escaneo guardado como PDF)."""
    area = page.rect.width * page.rect.height
    return any(pymupdf.Rect(i["bbox"]).get_area() > 0.8 * area for i in page.get_image_info())


def _objetos_pagina(page, M):
    """Imágenes, anotaciones y dibujos de la página, en coordenadas de la plantilla."""
    M = np.float32(M) if M is not None else None

    def a_plantilla(r):
        """Rectángulo en coordenadas de la plantilla (sin cambio si no hay transformación)."""
        if M is None:
            return [float(v) for v in r[:4]]
        esq = cv2.transform(np.float32([[[r[0], r[1]], [r[2], r[3]]]]), M)[0]
        return [float(esq[0, 0]), float(esq[0, 1]), float(esq[1, 0]), float(esq[1, 1])]

    return {
        "imagenes": [a_plantilla(i["bbox"]) for i in page.get_image_info()],
        "anotaciones": [[a.type[1], *a_plantilla(tuple(a.rect))] for a in page.annots()],
        "dibujos": [a_plantilla(d[:4]) + d[4:] for d in dibujos_pdf(page)],
    }


def unidades(ruta):
    """Unidades a analizar de un archivo: una por página (PDF) o una (imagen)."""
    if ruta.suffix.lower() in EXT_IMAGEN:
        return [(ruta, 0, 1)]
    with pymupdf.open(ruta) as doc:
        n = doc.page_count
    return [(ruta, i, n) for i in range(n)]


def analizar(args):
    """Analiza una unidad (ruta, origen[, página, n_páginas]) en CPU y devuelve su diccionario de resultado.

    - Una página que es una imagen se analiza como escaneo, aunque traiga capa de texto.
    - El QR se intenta leer a la resolución de trabajo y, si falla, a 300 DPI.
    - PDF nativo: clasificación por texto, geometría con palabras fijas y texto oculto en coordenadas de plantilla.
    - Escaneo o foto: alineación con la plantilla, refinamiento con etiquetas, limpieza y tinta por zona.
    - Un archivo corrupto no detiene el lote: el error queda en `error`.
    """
    ruta, origen, *resto = args
    pagina, n_paginas = (resto + [0, 1])[:2] if resto else (0, 1)
    etiqueta = ruta.name if n_paginas == 1 else f"{ruta.name}#p{pagina + 1}"
    out = {"ruta": str(ruta), "origen": origen, "carpeta": ruta.parent.name, "archivo": etiqueta,
           "pagina": pagina, "error": None}
    try:
        if ruta.suffix.lower() in EXT_IMAGEN:
            img, recortada = recortar_documento(cargar_imagen(ruta))
            out.update(meta={}, nativo=False, entrada="imagen", borde_detectado=recortada)
            palabras, doc = [], None
        else:
            doc = pymupdf.open(ruta)
            page = doc[pagina]
            out["meta"] = {k: doc.metadata.get(k, "") for k in ("producer", "creator", "creationDate", "modDate")}
            img = render_gris(page)
            palabras = palabras_pdf(page)
            ocultas = [p for p in palabras_pdf(page, incluir_ocultas=True) if p["oculta"]]
            if _imagen_pagina_completa(page):
                out["capa_texto_ignorada"] = len(palabras) + len(ocultas)
                palabras, ocultas = [], []
            else:
                palabras, no_visibles = _palabras_no_visibles(palabras, img)
                ocultas += no_visibles
            out["texto_oculto"] = [[p["texto"], *p["bbox"]] for p in ocultas][:50]
            out["entrada"] = "pdf"
            out["nativo"] = len(palabras) >= 15

        out["qr"] = leer_qr(img)
        if out["qr"] is None and doc is not None:
            out["qr"] = leer_qr(render_gris(doc[pagina], dpi=300))

        if out["nativo"]:
            pid, puntajes = clasificar_nativo(palabras)
            out["plantilla"] = pid
            out["clasificacion"] = {"metodo": "texto", "puntajes": puntajes}
            out["clasificacion_ok"] = puntajes[pid] >= C.MIN_SIMILITUD_NATIVA
            geo = geometria_nativa(palabras, _PLANTILLAS[pid])
            out["geometria"] = geo
            out["alineacion_ok"] = geo is not None and geo["residual_pt"] < 1.0
            out["alineacion_motivo"] = "" if out["alineacion_ok"] else "no se pudo ubicar el documento en la plantilla"
            out["palabras"] = transformar_palabras(palabras, geo["M"]) if geo else palabras
            out["objetos"] = _objetos_pagina(page, geo["M"] if geo else None)
            if out["texto_oculto"] and geo:
                M = np.float32(geo["M"])
                for t in out["texto_oculto"]:
                    esq = cv2.transform(np.float32([[[t[1], t[2]], [t[3], t[4]]]]), M)[0]
                    t[1:] = [float(esq[0, 0]), float(esq[0, 1]), float(esq[1, 0]), float(esq[1, 1])]
        else:
            pid, a, ok, motivo, metodo, inl, orden = alinear_escaneado(img, foto=out.get("entrada") == "imagen")
            out["plantilla"] = pid
            margen = orden[0] / max(1, orden[1]) if len(orden) > 1 else 99
            out["clasificacion"] = {"metodo": "alineacion", "inliers": inl, "margen": margen}
            out["clasificacion_ok"] = ok and margen >= 1.5
            out["alineacion"] = {k: v for k, v in (a or {}).items() if k != "H"} | {"metodo": metodo}
            out["alineacion_ok"] = ok
            out["alineacion_motivo"] = motivo
            if a and "H" in a:
                tpl = _PLANTILLAS[pid]["imagen"]
                alineada = cv2.warpPerspective(img, a["H"], (tpl.shape[1], tpl.shape[0]),
                                               flags=cv2.INTER_LINEAR, borderValue=255)
                alineada, out["refinamiento"] = refinar_con_etiquetas(alineada, _PLANTILLAS[pid])
                alineada = quitar_lineas(blanquear(alineada), _MASCARAS[pid])
                out["tinta"] = tinta_por_zona(alineada, _PLANTILLAS[pid])
                if ok:
                    out["imagen_ocr"] = alineada
        if doc is not None:
            doc.close()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out
