"""Lectura de cada archivo entregado (en procesos de CPU, sin PyTorch):

  - PDF con texto real (nativo): se extraen las palabras VISIBLES directamente (sin OCR); el texto invisible o
    tapado se separa como evidencia. Se ubica en la plantilla por sus palabras fijas.
  - PDF de imágenes e imágenes (JPG, PNG, HEIC, WEBP, TIFF de varias páginas, BMP): se alinea la imagen con la
    plantilla (homografía ORB/SIFT + refinamiento con las etiquetas), probando los 4 giros si hace falta; si en
    una foto hay dos documentos, se busca el segundo. La lectura OCR (GPU) se hace después.
  - En ambos casos se leen los códigos de máquina: timbre electrónico (PDF417) y código de barras del folio.
"""

import io
import re
from collections import Counter
from functools import lru_cache

import cv2
import numpy as np
import pillow_heif
import pymupdf
import zxingcpp
from PIL import Image, ImageOps

from .. import config as C
from .. import imagen as IM
from .. import pdf as PDF
from . import plantillas_cl as P

pillow_heif.register_heif_opener()
MAX_PIXELES_IMAGEN = 120_000_000
"""Tamaño máximo de una imagen (ancho x alto). Una foto de 108 MP cabe; una imagen "bomba" (un PNG de pocos MB que
al abrirse ocupa decenas de GB) se rechaza antes de decodificarla."""
Image.MAX_IMAGE_PIXELS = MAX_PIXELES_IMAGEN
MAX_LADO_PAGINA_PT = 2400
"""Lado máximo de una página PDF (pt, unos 85 cm): una página mayor no es un documento y renderizarla agotaría la
memoria."""

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
EXT_ADMITIDAS = EXT_IMAGEN | {".pdf"}
FORMATOS_IMAGEN = IM.FORMATOS_IMAGEN + ("HEIF",)
"""Formatos que Pillow puede abrir (ver imagen.FORMATOS_IMAGEN), más HEIF para las fotos de iPhone. Las imágenes
incrustadas en un PDF con otro formato (p. ej. JPEG 2000) se leen renderizando la página con PyMuPDF."""
LADO_TRABAJO = (1400, 2400)
"""Rango del lado mayor (px) al que se llevan las imágenes antes de alinear."""
DPI_TIMBRE_NATIVO = 400
"""Resolución con que se vuelve a buscar el timbre de un PDF nativo si a 300 DPI no se leyó: a 300 DPI se leía en un
tercio de las Hojas de Vida nativas; a 400, en todas las probadas."""
MAX_LADO_CODIGOS_PX = 4500
"""Lado máximo de la imagen en que se buscan los códigos de máquina; una foto mayor se reduce a la mitad."""

_PL = None
_ORB = None


def init_worker():
    """Inicializa el proceso lector: plantillas y sus descriptores ORB (una vez).

    La imagen para el OCR va sin blanquear ni borrar líneas: el reconocedor adoptado se entrenó con recortes tal
    cual (fotos y escaneos), y el blanqueo rompía los trazos finos (Courier)."""
    global _PL, _ORB
    cv2.setNumThreads(1)
    pillow_heif.register_heif_opener()
    _PL = P.cargar()
    orb = IM.orb()
    _ORB = {pid: orb.detectAndCompute(p["imagen"], None) for pid, p in _PL.items()}


def plantillas():
    """Plantillas cargadas por init_worker en este proceso."""
    return _PL


class _Carpeta:
    """Carpeta ficticia de un ArchivoMemoria: solo tiene nombre."""

    def __init__(self, name):
        """Carpeta con el nombre dado."""
        self.name = name


class ArchivoMemoria:
    """Archivo recibido en memoria (p. ej. por un servicio web): se usa en lugar de una ruta y nunca se escribe en disco.
    Ofrece lo que el flujo usa de una ruta: name, suffix, parent.name y read_bytes()."""

    def __init__(self, nombre, datos, carpeta):
        """Archivo `nombre` con contenido `datos` (bytes) dentro de la carpeta lógica `carpeta`."""
        self.name, self.datos, self.parent = nombre, datos, _Carpeta(carpeta)
        self.suffix = ("." + nombre.rsplit(".", 1)[-1]) if "." in nombre else ""

    def read_bytes(self):
        """Contenido del archivo."""
        return self.datos

    def __str__(self):
        """Nombre del archivo (lo que se muestra en los resultados)."""
        return self.name


def _abrible(ruta):
    """Lo que aceptan PIL y PyMuPDF: la ruta, o un flujo en memoria."""
    return io.BytesIO(ruta.datos) if isinstance(ruta, ArchivoMemoria) else ruta


def _abrir_pdf(ruta):
    """Abre un PDF desde una ruta o desde un ArchivoMemoria."""
    return pymupdf.open(stream=ruta.datos, filetype="pdf") if isinstance(ruta, ArchivoMemoria) else pymupdf.open(ruta)


def _abrir_imagen(fuente):
    """Imagen abierta con Pillow (solo FORMATOS_IMAGEN), sin decodificarla todavía. Levanta ValueError si supera
    MAX_PIXELES_IMAGEN."""
    im = Image.open(fuente, formats=FORMATOS_IMAGEN)
    if im.width * im.height > MAX_PIXELES_IMAGEN:
        im.close()
        raise ValueError(f"imagen demasiado grande ({im.width} x {im.height} px)")
    return im


def unidades(ruta):
    """(ruta, página, total): una por página de PDF o de TIFF, una por imagen."""
    suf = ruta.suffix.lower()
    if suf in EXT_IMAGEN:
        n = 1
        if suf in (".tif", ".tiff"):
            with _abrir_imagen(_abrible(ruta)) as im:
                n = getattr(im, "n_frames", 1)
        return [(ruta, i, n) for i in range(n)]
    with _abrir_pdf(ruta) as doc:
        n = doc.page_count
    return [(ruta, i, n) for i in range(n)]


def cargar_imagen(ruta, pagina=0):
    """Imagen (o página `pagina` de un TIFF) en gris, con la rotación EXIF aplicada."""
    with _abrir_imagen(_abrible(ruta)) as im:
        if pagina:
            im.seek(pagina)
            if im.width * im.height > MAX_PIXELES_IMAGEN:
                raise ValueError(f"imagen demasiado grande ({im.width} x {im.height} px)")
        return np.asarray(ImageOps.exif_transpose(im).convert("L")).copy()


def imagen_de_pagina(page):
    """Página de un PDF de imágenes: la imagen incrustada a su resolución original (o la página renderizada)."""
    infos = page.get_image_info(xrefs=True)
    grandes = [i for i in infos if pymupdf.Rect(i["bbox"]).get_area() > 0.5 * page.rect.get_area() and i["xref"]]
    if len(grandes) == 1:
        try:
            datos = page.parent.extract_image(grandes[0]["xref"])["image"]
            with _abrir_imagen(io.BytesIO(datos)) as im:
                return np.asarray(ImageOps.exif_transpose(im).convert("L")).copy()
        except Exception:
            pass
    return PDF.render_gris(page, dpi=200)


def _variantes_timbre(img_gris):
    """Versiones de una foto o escaneo en que el timbre (PDF417) se lee mejor, de la más a la menos efectiva según lo
    medido: umbral adaptativo (sombras y reflejos), umbral global de Otsu y la imagen ampliada 1,5 veces (fotos
    chicas). Se generan de a una, solo mientras el timbre no aparezca."""
    yield cv2.adaptiveThreshold(img_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    yield cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if max(img_gris.shape) < 4000:
        yield cv2.resize(img_gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


def leer_codigos(img_gris, insistir=False):
    """Timbre electrónico (PDF417) y códigos de barras (Code128) legibles en la imagen. Con `insistir`, si no aparece
    el timbre se prueba con las variantes de _variantes_timbre."""
    out = {"timbre": [], "barras": []}
    try:
        for r in zxingcpp.read_barcodes(img_gris):
            if r.format == zxingcpp.BarcodeFormat.PDF417 and r.text:
                out["timbre"].append(r.text)
            elif r.format == zxingcpp.BarcodeFormat.Code128 and r.text:
                out["barras"].append(r.text)
        for v in (_variantes_timbre(img_gris) if insistir else ()):
            if out["timbre"]:
                break
            out["timbre"] += [r.text for r in zxingcpp.read_barcodes(v, formats=zxingcpp.BarcodeFormat.PDF417) if r.text]
    except Exception:
        pass
    return out


def parsear_timbre(texto):
    """Campos FOLIO, RUN, NOM, FE y CV del XML del timbre electrónico (los que estén)."""
    campos = {}
    for clave in ("FOLIO", "RUN", "NOM", "FE", "CV"):
        m = re.search(rf"<{clave}>(.*?)</{clave}>", texto or "")
        if m:
            campos[clave] = m.group(1)
    return campos


def clasificar_texto(palabras):
    """Plantilla cuyo texto fijo más aparece en el documento."""
    textos = Counter(p["texto"] for p in palabras)
    puntajes = {}
    for pid, p in _PL.items():
        fijas = Counter(w[4] for w in p["palabras_fijas"])
        comunes = sum(min(n, textos[t]) for t, n in fijas.items())
        puntajes[pid] = comunes / max(1, sum(fijas.values()))
    mejor = max(puntajes, key=puntajes.get)
    return mejor, puntajes


Y_FIN_CABECERA_HV = 160
"""Altura (pt) donde termina la cabecera de la Hoja de Vida: debajo de "No válido para obtener ni renovar LICENCIA
CONDUCIR" empieza el contenido."""


def _es_copia(page):
    """True si la página es la versión "COPIA, no válida como certificado"."""
    t = page.get_text().upper()
    return "COPIA NO VALIDO COMO CERTIFICADO" in t or "COPIA NO VÁLIDA COMO CERTIFICADO" in t


def analizar_nativo(page, out, todas=None):
    """Lectura de un PDF nativo: palabras visibles, texto oculto, plantilla, geometría, objetos y códigos.

    `todas`: las palabras de la página con las ocultas incluidas (palabras_pdf con incluir_ocultas=True), si ya se
    extrajeron. La versión COPIA de la Hoja de Vida agrega la línea "COPIA DE" bajo la cabecera: el contenido baja
    ~12 pt pero la cabecera (logo, marco) no, así que la geometría se calcula solo con las etiquetas del contenido.
    """
    img = PDF.render_gris(page)
    todas = PDF.palabras_pdf(page, incluir_ocultas=True) if todas is None else todas
    palabras = [p for p in todas if not p["oculta"]]
    ocultas = [p for p in todas if p["oculta"]]
    palabras, no_visibles = PDF.palabras_no_visibles(palabras, img)
    ocultas += no_visibles
    pid, puntajes = clasificar_texto(palabras)
    out.update(nativo=True, plantilla=pid, clasificacion={"metodo": "texto", "puntajes": puntajes},
               clasificacion_ok=puntajes[pid] >= C.MIN_PUNTAJE_NATIVO)
    if not out["clasificacion_ok"]:
        out.update(plantilla=None, motivo="no corresponde a ninguno de los documentos solicitados")
        return out
    out["copia"] = _es_copia(page)
    tpl = _PL[pid]
    if out["copia"] and P.TIPO_DOC[pid] == "hoja_vida":
        tpl = dict(tpl, palabras_fijas=[w for w in tpl["palabras_fijas"] if w[1] > Y_FIN_CABECERA_HV])
    geo = PDF.geometria_nativa(palabras, tpl)
    out["geometria"] = geo
    out["alineacion_ok"] = geo is not None and geo["residual_pt"] < C.MAX_RESIDUAL_NATIVO_PT
    if geo:
        palabras = PDF.transformar_palabras(palabras, geo["M"])
        PDF.transformar_palabras(ocultas, geo["M"])
    out["palabras"] = [{k: p[k] for k in ("texto", "bbox", "font", "size")} | {"conf": 1.0} for p in palabras]
    out["texto_oculto"] = [[p["texto"], *p["bbox"]] for p in ocultas][:50]
    out["objetos"] = PDF.objetos_pagina(page, geo["M"] if geo else None)
    if P.TIPO_DOC[pid] == "hoja_vida":
        out["codigos"] = leer_codigos(PDF.render_gris(page, dpi=300))
        if not out["codigos"]["timbre"]:
            out["codigos"]["timbre"] = leer_codigos(PDF.render_gris(page, dpi=DPI_TIMBRE_NATIVO))["timbre"]
    return out


def _a_tamano_trabajo(img):
    """Imagen reescalada a LADO_TRABAJO y el factor aplicado. Las imágenes chicas (WhatsApp, capturas) se amplían
    antes de alinear."""
    h, w = img.shape
    lado = max(h, w)
    f = 1.0
    if lado > LADO_TRABAJO[1]:
        f = LADO_TRABAJO[1] / lado
    elif lado < LADO_TRABAJO[0]:
        f = LADO_TRABAJO[0] / lado
    if f != 1.0:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return img, f


def _alinear(img, foto, excluir=()):
    """Mejor plantilla para la imagen (ORB, respaldo SIFT). Devuelve (pid, info, ok, motivo, metodo).

    Las páginas 1 y 2 de la Hoja de Vida comparten la cabecera (los puntos ORB caen casi todos ahí): entre ellas se
    decide por cuántas etiquetas impresas del cuerpo aparecen en su lugar con cada plantilla.
    """
    kp, des = IM.orb().detectAndCompute(img, None)
    res = {pid: IM.homografia(kp, des, *_ORB[pid], cv2.NORM_HAMMING) for pid in _PL if pid not in excluir}
    if not res:
        return None, {}, False, "sin plantillas", ""
    mejor = max(res, key=lambda k: (res[k] or {}).get("inliers", 0))
    if P.TIPO_DOC[mejor] == "hoja_vida":
        cands = [k for k in ("hoja_vida_p1", "hoja_vida_p2") if k in res and (res[k] or {}).get("H") is not None]
        if len(cands) == 2:
            mejor = max(cands, key=lambda k: _puntaje_etiquetas(img, res[k]["H"], _PL[k]))
    a = res[mejor] or {}
    ok, motivo = IM.evaluar_alineacion(a, foto)
    metodo = "ORB"
    if not ok:
        sift = cv2.SIFT_create(nfeatures=5000)
        kp2, des2 = sift.detectAndCompute(img, None)
        kpt, dest = sift.detectAndCompute(_PL[mejor]["imagen"], None)
        a2 = IM.homografia(kp2, des2, kpt, dest, cv2.NORM_L2)
        ok2, motivo2 = IM.evaluar_alineacion(a2, foto)
        if ok2 or (a2 and a2.get("inliers", 0) > a.get("inliers", 0)):
            a, ok, motivo, metodo = a2, ok2, motivo2, "SIFT"
    return mejor, a, ok, motivo, metodo


def _puntaje_etiquetas(img, H, plantilla, busqueda_px=30, min_corr=0.6):
    """Fracción de las etiquetas impresas de la plantilla que se encuentran en su lugar tras alinear con H."""
    tpl = plantilla["imagen"]
    h, w = tpl.shape
    alineada = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    vistas = encontradas = 0
    for x0, y0, x1, y1, _ in plantilla["palabras_fijas"]:
        if (x1 - x0) < 15:
            continue
        px0, py0, px1, py1 = int(x0 * C.ESCALA), int(y0 * C.ESCALA), int(x1 * C.ESCALA), int(y1 * C.ESCALA)
        parche = tpl[py0:py1, px0:px1]
        zona = alineada[max(0, py0 - busqueda_px):py1 + busqueda_px, max(0, px0 - busqueda_px):px1 + busqueda_px]
        if parche.size == 0 or parche.std() < 5 or zona.shape[0] < parche.shape[0] or zona.shape[1] < parche.shape[1]:
            continue
        vistas += 1
        encontradas += cv2.minMaxLoc(cv2.matchTemplate(zona, parche, cv2.TM_CCOEFF_NORMED))[1] >= min_corr
    return encontradas / vistas if vistas else 0.0


def _refinar_H(alineada, plantilla, busqueda_px=30, min_corr=0.5):
    """Segundo paso de alineación: busca cada texto fijo de la plantilla (etiquetas, títulos, pie) en una vecindad
    pequeña de la imagen ya alineada y calcula, con esas correspondencias repartidas por toda la página, la homografía
    que corrige los errores locales (p. ej. el borde inferior cuando la primera alineación se apoyó sobre todo en el
    encabezado). Las palabras cortas (< 15 pt) se omiten porque son poco distintivas. Devuelve (H o None, info)."""
    tpl = plantilla["imagen"]
    h, w = tpl.shape
    origen, destino = [], []
    for x0, y0, x1, y1, _ in plantilla["palabras_fijas"]:
        if (x1 - x0) < 15:
            continue
        px0, py0 = max(0, int(x0 * C.ESCALA) - 3), max(0, int(y0 * C.ESCALA) - 3)
        px1, py1 = min(w, int(x1 * C.ESCALA) + 3), min(h, int(y1 * C.ESCALA) + 3)
        parche = tpl[py0:py1, px0:px1]
        sx0, sy0 = max(0, px0 - busqueda_px), max(0, py0 - busqueda_px)
        sx1, sy1 = min(w, px1 + busqueda_px), min(h, py1 + busqueda_px)
        zona = alineada[sy0:sy1, sx0:sx1]
        if parche.size == 0 or zona.shape[0] < parche.shape[0] or zona.shape[1] < parche.shape[1] or parche.std() < 5:
            continue
        _, corr, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(zona, parche, cv2.TM_CCOEFF_NORMED))
        if corr < min_corr:
            continue
        origen.append((sx0 + mx + parche.shape[1] / 2, sy0 + my + parche.shape[0] / 2))
        destino.append(((px0 + px1) / 2, (py0 + py1) / 2))
    if len(origen) < 8:
        return None, {"puntos": len(origen), "aplicado": False}
    H, mask = cv2.findHomography(np.float32(origen), np.float32(destino), cv2.RANSAC, 3.0)
    return H, {"puntos": len(origen), "aplicado": H is not None, "inliers": int(mask.sum()) if H is not None else 0}


def _girar(img, grados):
    """Imagen girada 0, 90, 180 o 270 grados (antihorario)."""
    return img if grados == 0 else np.ascontiguousarray(np.rot90(img, k={90: 1, 180: 2, 270: 3}[grados]))


def nitidez(img):
    """Varianza del laplaciano a 800 px de ancho: cuanto más baja, más borrosa la imagen."""
    chica = cv2.resize(img, (800, round(800 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(chica, cv2.CV_64F).var())


def alinear_documentos(img, foto=True):
    """Hasta 2 documentos por imagen. Para cada uno: plantilla, imagen alineada (lista para OCR) y control.

    - Se prueban los 4 giros hasta que uno alinee bien.
    - La imagen para OCR se toma de la foto ORIGINAL (sin reducir): px originales -> px de trabajo -> plantilla.
    - Se queda a la resolución de la plantilla (150 DPI). Se probó alinear a 2x (300 DPI): el lector OCR tiene
      umbrales en píxeles calibrados a 150 DPI y a 2x parte las letras finas en trozos (el reconocedor inventa
      palabras).
    - La zona que ocupa el documento encontrado se tapa para buscar un segundo; tras un AMECO, el segundo será otro
      documento.
    """
    original = img
    img, escala = _a_tamano_trabajo(img)
    encontrados = []
    trabajo = img.copy()
    excluir = set()
    for intento in range(2):
        mejor = None
        for grados in (0, 90, 270, 180):
            girada = _girar(trabajo, grados)
            pid, a, ok, motivo, metodo = _alinear(girada, foto, excluir)
            if pid is None:
                break
            cand = (ok, a.get("inliers", 0), pid, a, motivo, metodo, grados, girada)
            if mejor is None or cand[:2] > mejor[:2]:
                mejor = cand
            if ok:
                break
        if mejor is None or (intento and not mejor[0]):
            break
        ok, _, pid, a, motivo, metodo, grados, girada = mejor
        tpl = _PL[pid]
        doc = {"plantilla": pid, "rotacion_aplicada": grados, "alineacion_ok": ok, "alineacion_motivo": motivo,
               "alineacion": {k: v for k, v in a.items() if k != "H"} | {"metodo": metodo}}
        if ok:
            h, w = tpl["imagen"].shape
            alineada = cv2.warpPerspective(girada, a["H"], (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
            H_ref, ref = _refinar_H(alineada, tpl)
            H = (H_ref @ a["H"]) if H_ref is not None else a["H"]
            doc["refinamiento"] = ref
            doc["imagen_ocr"] = cv2.warpPerspective(_girar(original, grados), H @ np.diag([escala, escala, 1.0]),
                                                    (w, h), flags=cv2.INTER_AREA, borderValue=255)
            Hi = np.linalg.inv(a["H"])
            esq = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), Hi)[0]
            mascara = np.zeros(girada.shape, np.uint8)
            cv2.fillPoly(mascara, [esq.astype(np.int32)], 255)
            girada = girada.copy()
            girada[mascara > 0] = 255
            trabajo = _girar(girada, {0: 0, 90: 270, 180: 180, 270: 90}[grados])
            excluir = {pid} if P.TIPO_DOC[pid] == "ameco" else set()
        encontrados.append(doc)
        if not ok:
            break
    return encontrados


def analizar_imagen(img, out, foto):
    """Lectura de una imagen: nitidez, códigos de máquina y documentos alineados."""
    out["nitidez"] = nitidez(img)
    docs = alinear_documentos(img, foto)
    chica = img if max(img.shape) <= MAX_LADO_CODIGOS_PX else cv2.resize(img, None, fx=0.5, fy=0.5)
    es_hv = any(P.TIPO_DOC.get(d["plantilla"]) == "hoja_vida" for d in docs)
    out["codigos"] = leer_codigos(chica, insistir=es_hv)
    out["nativo"] = False
    out["documentos"] = docs
    return out


@lru_cache(maxsize=8)
def _revisiones_pdf(ruta, _firma):
    """Versiones guardadas de un PDF (marcas %%EOF; la de un PDF linealizado no cuenta). `_firma` (fecha y tamaño
    del archivo) hace que un archivo reemplazado se vuelva a contar."""
    crudo = ruta.read_bytes()
    return max(1, crudo.count(b"%%EOF") - (1 if b"/Linearized" in crudo[:1024] else 0))


def revisiones_pdf(ruta):
    """Versiones guardadas de un PDF, calculadas una vez por archivo y no una vez por página."""
    if isinstance(ruta, ArchivoMemoria):
        return _revisiones_pdf(ruta, None)
    st = ruta.stat()
    return _revisiones_pdf(ruta, (st.st_mtime_ns, st.st_size))


def analizar(args):
    """Analiza una unidad (ruta, página, total) en CPU. Un PDF cuya página es una imagen, o con menos de
    C.MIN_PALABRAS_NATIVO palabras visibles, se lee como imagen. Un archivo corrupto no detiene el lote: el error
    queda en `error`."""
    ruta, pagina, total = args
    etiqueta = ruta.name if total == 1 else f"{ruta.name}#p{pagina + 1}"
    out = {"ruta": str(ruta), "carpeta": ruta.parent.name, "archivo": ruta.name, "unidad": etiqueta,
           "pagina": pagina, "paginas_archivo": total, "error": None}
    try:
        if ruta.suffix.lower() in EXT_IMAGEN:
            out["entrada"] = "imagen"
            out["meta"] = {}
            return analizar_imagen(cargar_imagen(ruta, pagina), out, foto=True)
        doc = _abrir_pdf(ruta)
        page = doc[pagina]
        if max(page.rect.width, page.rect.height) > MAX_LADO_PAGINA_PT:
            raise ValueError(f"página demasiado grande ({page.rect.width:.0f} x {page.rect.height:.0f} pt)")
        out["entrada"] = "pdf"
        out["meta"] = {k: doc.metadata.get(k, "") for k in ("producer", "creator", "creationDate", "modDate")}
        out["revisiones_pdf"] = revisiones_pdf(ruta)
        todas = PDF.palabras_pdf(page, incluir_ocultas=True)
        visibles = sum(not p["oculta"] for p in todas)
        pagina_imagen = PDF.imagen_pagina_completa(page)
        if pagina_imagen or visibles < C.MIN_PALABRAS_NATIVO:
            out["capa_texto_ignorada"] = visibles if pagina_imagen else 0
            return analizar_imagen(imagen_de_pagina(page), out, foto=True)
        return analizar_nativo(page, out, todas)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
