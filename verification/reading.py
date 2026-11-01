"""Lectura de cada archivo entregado (en procesos de CPU, sin PyTorch):

  - PDF con texto real (nativo): se extraen las palabras VISIBLES directamente (sin OCR); el texto invisible o
    tapado se separa como evidencia. Se ubica en la plantilla por sus palabras fijas.
  - PDF de imágenes e imágenes (JPG, PNG, HEIC, WEBP, TIFF de varias páginas, BMP): se alinea la imagen con la
    plantilla (homografía ORB/SIFT + refinamiento con las etiquetas), probando los 4 giros si hace falta; si en
    una foto hay dos documentos, se busca el segundo. La lectura OCR (GPU) se hace después.
  - En ambos casos se leen los códigos de máquina: timbre electrónico (PDF417) y código de barras del folio.
"""

import io
from collections import Counter
from functools import lru_cache

import cv2
import numpy as np
import pillow_heif
import pymupdf
import zxingcpp
from PIL import Image, ImageOps

from . import config as C
from . import image as IM
from . import pdf as PDF
from .profiles import ACTIVE as PF

D, P = PF.documents, PF.templates

pillow_heif.register_heif_opener()
MAX_IMAGE_PIXELS = 120_000_000
"""Tamaño máximo de una imagen (ancho x alto). Una foto de 108 MP cabe; una imagen "bomba" (un PNG de pocos MB que
al abrirse ocupa decenas de GB) se rechaza antes de decodificarla."""
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
MAX_PAGE_SIDE_PT = 2400
"""Lado máximo de una página PDF (pt, unos 85 cm): una página mayor no es un documento y renderizarla agotaría la
memoria."""

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
ACCEPTED_EXT = IMAGE_EXT | {".pdf"}
IMAGE_FORMATS = IM.IMAGE_FORMATS + ("HEIF",)
"""Formatos que Pillow puede abrir (ver imagen.IMAGE_FORMATS), más HEIF para las fotos de iPhone. Las imágenes
incrustadas en un PDF con otro formato (p. ej. JPEG 2000) se leen renderizando la página con PyMuPDF."""
WORK_SIDE = (1400, 2400)
"""Rango del lado mayor (px) al que se llevan las imágenes antes de alinear."""
NATIVE_SEAL_DPI = 400
"""Resolución con que se vuelve a buscar el timbre de un PDF nativo si a 300 DPI no se leyó: a 300 DPI se leía en un
tercio de las Hojas de Vida nativas; a 400, en todas las probadas."""
MAX_CODES_SIDE_PX = 4500
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
    _PL = P.load()
    orb = IM.orb()
    _ORB = {pid: orb.detectAndCompute(p["imagen"], None) for pid, p in _PL.items()}


def templates():
    """Plantillas cargadas por init_worker en este proceso."""
    return _PL


class _Folder:
    """Carpeta ficticia de un ArchivoMemoria: solo tiene nombre."""

    def __init__(self, name):
        """Carpeta con el nombre dado."""
        self.name = name


class InMemoryFile:
    """Archivo recibido en memoria (p. ej. por un servicio web): se usa en lugar de una ruta y nunca se escribe en disco.
    Ofrece lo que el flujo usa de una ruta: name, suffix, parent.name y read_bytes()."""

    def __init__(self, name, data, folder):
        """Archivo `nombre` con contenido `datos` (bytes) dentro de la carpeta lógica `carpeta`."""
        self.name, self.data, self.parent = name, data, _Folder(folder)
        self.suffix = ("." + name.rsplit(".", 1)[-1]) if "." in name else ""

    def read_bytes(self):
        """Contenido del archivo."""
        return self.data

    def __str__(self):
        """Nombre del archivo (lo que se muestra en los resultados)."""
        return self.name


def _openable(path):
    """Lo que aceptan PIL y PyMuPDF: la ruta, o un flujo en memoria."""
    return io.BytesIO(path.data) if isinstance(path, InMemoryFile) else path


def _open_pdf(path):
    """Abre un PDF desde una ruta o desde un ArchivoMemoria."""
    return pymupdf.open(stream=path.data, filetype="pdf") if isinstance(path, InMemoryFile) else pymupdf.open(path)


def _open_image(font_name):
    """Imagen abierta con Pillow (solo IMAGE_FORMATS), sin decodificarla todavía. Levanta ValueError si supera
    MAX_IMAGE_PIXELS."""
    im = Image.open(font_name, formats=IMAGE_FORMATS)
    if im.width * im.height > MAX_IMAGE_PIXELS:
        im.close()
        raise ValueError(f"imagen demasiado grande ({im.width} x {im.height} px)")
    return im


def units(path):
    """(ruta, página, total): una por página de PDF o de TIFF, una por imagen."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        n = 1
        if suffix in (".tif", ".tiff"):
            with _open_image(_openable(path)) as im:
                n = getattr(im, "n_frames", 1)
        return [(path, i, n) for i in range(n)]
    with _open_pdf(path) as doc:
        n = doc.page_count
    return [(path, i, n) for i in range(n)]


def load_image(path, page_no=0):
    """Imagen (o página `pagina` de un TIFF) en gris, con la rotación EXIF aplicada."""
    with _open_image(_openable(path)) as im:
        if page_no:
            im.seek(page_no)
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"imagen demasiado grande ({im.width} x {im.height} px)")
        return np.asarray(ImageOps.exif_transpose(im).convert("L")).copy()


def page_image(page):
    """Página de un PDF de imágenes: la imagen incrustada a su resolución original (o la página renderizada)."""
    infos = page.get_image_info(xrefs=True)
    large = [i for i in infos if pymupdf.Rect(i["bbox"]).get_area() > 0.5 * page.rect.get_area() and i["xref"]]
    if len(large) == 1:
        try:
            data = page.parent.extract_image(large[0]["xref"])["image"]
            with _open_image(io.BytesIO(data)) as im:
                return np.asarray(ImageOps.exif_transpose(im).convert("L")).copy()
        except Exception:
            pass
    return PDF.render_gray(page, dpi=200)


def _seal_variants(gray):
    """Versiones de una foto o escaneo en que el timbre (PDF417) se lee mejor, de la más a la menos efectiva según lo
    medido: umbral adaptativo (sombras y reflejos), umbral global de Otsu y la imagen ampliada 1,5 veces (fotos
    chicas). Se generan de a una, solo mientras el timbre no aparezca."""
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if max(gray.shape) < 4000:
        yield cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


def read_codes(gray, persist=False):
    """Timbre electrónico (PDF417) y códigos de barras (Code128) legibles en la imagen. Con `insistir`, si no aparece
    el timbre se prueba con las variantes de _variantes_timbre."""
    out = {"timbre": [], "barras": []}
    try:
        for r in zxingcpp.read_barcodes(gray):
            if r.format == zxingcpp.BarcodeFormat.PDF417 and r.text:
                out["timbre"].append(r.text)
            elif r.format == zxingcpp.BarcodeFormat.Code128 and r.text:
                out["barras"].append(r.text)
        for v in (_seal_variants(gray) if persist else ()):
            if out["timbre"]:
                break
            out["timbre"] += [r.text for r in zxingcpp.read_barcodes(v, formats=zxingcpp.BarcodeFormat.PDF417) if r.text]
    except Exception:
        pass
    return out


def classify_text(words):
    """Plantilla cuyo texto fijo más aparece en el documento."""
    texts = Counter(p["texto"] for p in words)
    scores = {}
    for pid, p in _PL.items():
        fixed = Counter(w[4] for w in p["palabras_fijas"])
        common = sum(min(n, texts[t]) for t, n in fixed.items())
        scores[pid] = common / max(1, sum(fixed.values()))
    best = max(scores, key=scores.get)
    return best, scores


def analyze_native(page, out, all_words=None):
    """Lectura de un PDF nativo: palabras visibles, texto oculto, plantilla, geometría, objetos y códigos.

    `todas`: las palabras de la página con las ocultas incluidas (pdf_words con include_hidden=True), si ya se
    extrajeron. La versión COPIA se ubica con la plantilla que indica el perfil (documents.copy_template).
    """
    img = PDF.render_gray(page)
    all_words = PDF.pdf_words(page, include_hidden=True) if all_words is None else all_words
    words = [p for p in all_words if not p["oculta"]]
    hidden_words = [p for p in all_words if p["oculta"]]
    words, invisible = PDF.invisible_words(words, img)
    hidden_words += invisible
    pid, scores = classify_text(words)
    out.update(nativo=True, plantilla=pid, clasificacion={"metodo": "texto", "puntajes": scores},
               clasificacion_ok=scores[pid] >= C.MIN_NATIVE_SCORE)
    if not out["clasificacion_ok"]:
        out.update(plantilla=None, motivo="no corresponde a ninguno de los documentos solicitados")
        return out
    out["copia"] = D.is_copy_native(page.get_text())
    tpl = _PL[pid]
    if out["copia"]:
        tpl = D.copy_template(tpl)
    geo = PDF.native_geometry(words, tpl)
    out["geometria"] = geo
    out["alineacion_ok"] = geo is not None and geo["residual_pt"] < C.MAX_NATIVE_RESIDUAL_PT
    if geo:
        words = PDF.transform_words(words, geo["M"])
        PDF.transform_words(hidden_words, geo["M"])
    out["palabras"] = [{k: p[k] for k in ("texto", "bbox", "font", "size")} | {"conf": 1.0} for p in words]
    out["texto_oculto"] = [[p["texto"], *p["bbox"]] for p in hidden_words][:50]
    out["objetos"] = PDF.page_objects(page, geo["M"] if geo else None)
    if P.DOCUMENT_TYPE[pid] in D.SEALED_TYPES:
        out["codigos"] = read_codes(PDF.render_gray(page, dpi=300))
        if not out["codigos"]["timbre"]:
            out["codigos"]["timbre"] = read_codes(PDF.render_gray(page, dpi=NATIVE_SEAL_DPI))["timbre"]
    return out


def _to_work_size(img):
    """Imagen reescalada a WORK_SIDE y el factor aplicado. Las imágenes chicas (WhatsApp, capturas) se amplían
    antes de alinear."""
    h, w = img.shape
    side = max(h, w)
    f = 1.0
    if side > WORK_SIDE[1]:
        f = WORK_SIDE[1] / side
    elif side < WORK_SIDE[0]:
        f = WORK_SIDE[0] / side
    if f != 1.0:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return img, f


def _align(img, photo, exclude=()):
    """Mejor plantilla para la imagen (ORB, respaldo SIFT). Devuelve (pid, info, ok, motivo, metodo).

    Las páginas de un mismo documento pueden compartir la cabecera (documents.SIBLING_PAGES; los puntos ORB caen casi
    todos ahí): entre ellas se decide por cuántas etiquetas impresas del cuerpo aparecen en su lugar con cada
    plantilla.
    """
    kp, des = IM.orb().detectAndCompute(img, None)
    res = {pid: IM.homography(kp, des, *_ORB[pid], cv2.NORM_HAMMING) for pid in _PL if pid not in exclude}
    if not res:
        return None, {}, False, "sin plantillas", ""
    best = max(res, key=lambda k: (res[k] or {}).get("inliers", 0))
    if P.DOCUMENT_TYPE[best] in D.SIBLING_PAGES:
        cands = [k for k in D.SIBLING_PAGES[P.DOCUMENT_TYPE[best]] if k in res and (res[k] or {}).get("H") is not None]
        if len(cands) == 2:
            best = max(cands, key=lambda k: _label_score(img, res[k]["H"], _PL[k]))
    a = res[best] or {}
    ok, reason = IM.evaluate_alignment(a, photo)
    method = "ORB"
    if not ok:
        sift = cv2.SIFT_create(nfeatures=5000)
        kp2, des2 = sift.detectAndCompute(img, None)
        kpt, dest = sift.detectAndCompute(_PL[best]["imagen"], None)
        a2 = IM.homography(kp2, des2, kpt, dest, cv2.NORM_L2)
        ok2, reason2 = IM.evaluate_alignment(a2, photo)
        if ok2 or (a2 and a2.get("inliers", 0) > a.get("inliers", 0)):
            a, ok, reason, method = a2, ok2, reason2, "SIFT"
    return best, a, ok, reason, method


def _label_score(img, H, template, search_px=30, min_corr=0.6):
    """Fracción de las etiquetas impresas de la plantilla que se encuentran en su lugar tras alinear con H."""
    tpl = template["imagen"]
    h, w = tpl.shape
    aligned = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    seen = found = 0
    for x0, y0, x1, y1, _ in template["palabras_fijas"]:
        if (x1 - x0) < 15:
            continue
        px0, py0, px1, py1 = int(x0 * C.SCALE), int(y0 * C.SCALE), int(x1 * C.SCALE), int(y1 * C.SCALE)
        patch = tpl[py0:py1, px0:px1]
        zone = aligned[max(0, py0 - search_px):py1 + search_px, max(0, px0 - search_px):px1 + search_px]
        if patch.size == 0 or patch.std() < 5 or zone.shape[0] < patch.shape[0] or zone.shape[1] < patch.shape[1]:
            continue
        seen += 1
        found += cv2.minMaxLoc(cv2.matchTemplate(zone, patch, cv2.TM_CCOEFF_NORMED))[1] >= min_corr
    return found / seen if seen else 0.0


def _refine_H(aligned, template, search_px=30, min_corr=0.5):
    """Segundo paso de alineación: busca cada texto fijo de la plantilla (etiquetas, títulos, pie) en una vecindad
    pequeña de la imagen ya alineada y calcula, con esas correspondencias repartidas por toda la página, la homografía
    que corrige los errores locales (p. ej. el borde inferior cuando la primera alineación se apoyó sobre todo en el
    encabezado). Las palabras cortas (< 15 pt) se omiten porque son poco distintivas. Devuelve (H o None, info)."""
    tpl = template["imagen"]
    h, w = tpl.shape
    origin, destination = [], []
    for x0, y0, x1, y1, _ in template["palabras_fijas"]:
        if (x1 - x0) < 15:
            continue
        px0, py0 = max(0, int(x0 * C.SCALE) - 3), max(0, int(y0 * C.SCALE) - 3)
        px1, py1 = min(w, int(x1 * C.SCALE) + 3), min(h, int(y1 * C.SCALE) + 3)
        patch = tpl[py0:py1, px0:px1]
        sx0, sy0 = max(0, px0 - search_px), max(0, py0 - search_px)
        sx1, sy1 = min(w, px1 + search_px), min(h, py1 + search_px)
        zone = aligned[sy0:sy1, sx0:sx1]
        if patch.size == 0 or zone.shape[0] < patch.shape[0] or zone.shape[1] < patch.shape[1] or patch.std() < 5:
            continue
        _, corr, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(zone, patch, cv2.TM_CCOEFF_NORMED))
        if corr < min_corr:
            continue
        origin.append((sx0 + mx + patch.shape[1] / 2, sy0 + my + patch.shape[0] / 2))
        destination.append(((px0 + px1) / 2, (py0 + py1) / 2))
    if len(origin) < 8:
        return None, {"puntos": len(origin), "aplicado": False}
    H, mask = cv2.findHomography(np.float32(origin), np.float32(destination), cv2.RANSAC, 3.0)
    return H, {"puntos": len(origin), "aplicado": H is not None, "inliers": int(mask.sum()) if H is not None else 0}


def _rotate(img, angle):
    """Imagen girada 0, 90, 180 o 270 grados (antihorario)."""
    return img if angle == 0 else np.ascontiguousarray(np.rot90(img, k={90: 1, 180: 2, 270: 3}[angle]))


def sharpness(img):
    """Varianza del laplaciano a 800 px de ancho: cuanto más baja, más borrosa la imagen."""
    small = cv2.resize(img, (800, round(800 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def align_documents(img, photo=True):
    """Hasta 2 documentos por imagen. Para cada uno: plantilla, imagen alineada (lista para OCR) y control.

    - Se prueban los 4 giros hasta que uno alinee bien.
    - La imagen para OCR se toma de la foto ORIGINAL (sin reducir): px originales -> px de trabajo -> plantilla.
    - Se queda a la resolución de la plantilla (150 DPI). Se probó alinear a 2x (300 DPI): el lector OCR tiene
      umbrales en píxeles calibrados a 150 DPI y a 2x parte las letras finas en trozos (el reconocedor inventa
      palabras).
    - La zona que ocupa el documento encontrado se tapa para buscar un segundo; tras un tipo de
      documents.ONE_PER_IMAGE, el segundo será otro documento.
    """
    original = img
    img, scale = _to_work_size(img)
    found_docs = []
    job = img.copy()
    exclude = set()
    for attempt in range(2):
        best = None
        for angle in (0, 90, 270, 180):
            rotated = _rotate(job, angle)
            pid, a, ok, reason, method = _align(rotated, photo, exclude)
            if pid is None:
                break
            cand = (ok, a.get("inliers", 0), pid, a, reason, method, angle, rotated)
            if best is None or cand[:2] > best[:2]:
                best = cand
            if ok:
                break
        if best is None or (attempt and not best[0]):
            break
        ok, _, pid, a, reason, method, angle, rotated = best
        tpl = _PL[pid]
        doc = {"plantilla": pid, "rotacion_aplicada": angle, "alineacion_ok": ok, "alineacion_motivo": reason,
               "alineacion": {k: v for k, v in a.items() if k != "H"} | {"metodo": method}}
        if ok:
            h, w = tpl["imagen"].shape
            aligned = cv2.warpPerspective(rotated, a["H"], (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
            H_ref, ref = _refine_H(aligned, tpl)
            H = (H_ref @ a["H"]) if H_ref is not None else a["H"]
            doc["refinamiento"] = ref
            doc["imagen_ocr"] = cv2.warpPerspective(_rotate(original, angle), H @ np.diag([scale, scale, 1.0]),
                                                    (w, h), flags=cv2.INTER_AREA, borderValue=255)
            Hi = np.linalg.inv(a["H"])
            corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), Hi)[0]
            region_mask = np.zeros(rotated.shape, np.uint8)
            cv2.fillPoly(region_mask, [corners.astype(np.int32)], 255)
            rotated = rotated.copy()
            rotated[region_mask > 0] = 255
            job = _rotate(rotated, {0: 0, 90: 270, 180: 180, 270: 90}[angle])
            exclude = {pid} if P.DOCUMENT_TYPE[pid] in D.ONE_PER_IMAGE else set()
        found_docs.append(doc)
        if not ok:
            break
    return found_docs


def analyze_image(img, out, photo):
    """Lectura de una imagen: nitidez, códigos de máquina y documentos alineados."""
    out["nitidez"] = sharpness(img)
    docs = align_documents(img, photo)
    small = img if max(img.shape) <= MAX_CODES_SIDE_PX else cv2.resize(img, None, fx=0.5, fy=0.5)
    sealed = any(P.DOCUMENT_TYPE.get(d["plantilla"]) in D.SEALED_TYPES for d in docs)
    out["codigos"] = read_codes(small, persist=sealed)
    out["nativo"] = False
    out["documentos"] = docs
    return out


@lru_cache(maxsize=8)
def _pdf_revisions(path, _signature):
    """Versiones guardadas de un PDF (marcas %%EOF; la de un PDF linealizado no cuenta). `_firma` (fecha y tamaño
    del archivo) hace que un archivo reemplazado se vuelva a contar."""
    raw = path.read_bytes()
    return max(1, raw.count(b"%%EOF") - (1 if b"/Linearized" in raw[:1024] else 0))


def pdf_revisions(path):
    """Versiones guardadas de un PDF, calculadas una vez por archivo y no una vez por página."""
    if isinstance(path, InMemoryFile):
        return _pdf_revisions(path, None)
    st = path.stat()
    return _pdf_revisions(path, (st.st_mtime_ns, st.st_size))


def analyze(args):
    """Analiza una unidad (ruta, página, total) en CPU. Un PDF cuya página es una imagen, o con menos de
    C.MIN_NATIVE_WORDS palabras visibles, se lee como imagen. Un archivo corrupto no detiene el lote: el error
    queda en `error`."""
    path, page_no, total = args
    label = path.name if total == 1 else f"{path.name}#p{page_no + 1}"
    out = {"ruta": str(path), "carpeta": path.parent.name, "archivo": path.name, "unidad": label,
           "pagina": page_no, "paginas_archivo": total, "error": None}
    try:
        if path.suffix.lower() in IMAGE_EXT:
            out["entrada"] = "imagen"
            out["meta"] = {}
            return analyze_image(load_image(path, page_no), out, photo=True)
        doc = _open_pdf(path)
        page = doc[page_no]
        if max(page.rect.width, page.rect.height) > MAX_PAGE_SIDE_PT:
            raise ValueError(f"página demasiado grande ({page.rect.width:.0f} x {page.rect.height:.0f} pt)")
        out["entrada"] = "pdf"
        out["meta"] = {k: doc.metadata.get(k, "") for k in ("producer", "creator", "creationDate", "modDate")}
        out["revisiones_pdf"] = pdf_revisions(path)
        all_words = PDF.pdf_words(page, include_hidden=True)
        visible = sum(not p["oculta"] for p in all_words)
        image_page = PDF.full_page_image(page)
        if image_page or visible < C.MIN_NATIVE_WORDS:
            out["capa_texto_ignorada"] = visible if image_page else 0
            return analyze_image(page_image(page), out, photo=True)
        return analyze_native(page, out, all_words)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
