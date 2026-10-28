"""Evaluación independiente del reconocedor (no usa entrenamiento/training ni entrenamiento/testing).

1. DATOS REALES: los PDF reales se rasterizan (100/150/200 DPI) y cada palabra del texto real del PDF se
   recorta y se lee. Se informa aparte: palabras (con letras o números) y signos sueltos (":", ".", …).
2. postulant_files (2000 postulantes, otro generador y otras semillas): campos de UNA palabra (RUT, fechas,
   folios, clases, códigos…) de documentos entregados como imagen o PDF de imagen; se endereza el campo y se
   lee directo. Las etiquetas (etiquetas_ocr_v2) solo se usan para medir.

Uso: python -u -m entrenamiento.evaluar_independiente --recos actual=modelos/parseq_es_finetuned.pt nuevo=...
"""

import argparse
import io
import json
import random
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import pymupdf as fitz
import torch
from doctr.models import recognition_predictor
from PIL import Image
from rapidfuzz.distance import Levenshtein

from entrenamiento.entrenar_reconocedor import NORMALIZAR, cargar_modelo, prioridad_baja
from verificacion import config as C
from verificacion import recursos as R

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = None
IMAGENES = (".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp")


def norm(t):
    """Texto NFC con los guiones unificados y los espacios colapsados."""
    return " ".join(unicodedata.normalize("NFC", t).translate(NORMALIZAR).split())


def es_signo(t):
    """True si el texto no tiene letras ni números (un signo de puntuación suelto)."""
    return not any(c.isalnum() for c in t)


def imagen_archivo(ruta, pagina):
    """Imagen RGB de una página: la imagen incrustada de un PDF de imágenes, o la página de una imagen/TIFF."""
    if ruta.suffix.lower() == ".pdf":
        doc = fitz.open(ruta)
        if pagina >= doc.page_count or not doc[pagina].get_images():
            return None
        return Image.open(io.BytesIO(doc.extract_image(doc[pagina].get_images()[0][0])["image"])).convert("RGB")
    img = Image.open(ruta)
    if pagina:
        img.seek(pagina)
    return img.convert("RGB")


def recortes_reales(dpi, carpeta="DATOS REALES"):
    """Recortes de cada palabra del texto real de los PDF de `carpeta`, rasterizados a `dpi`, con su texto.
    Los PDF sin texto (fotos, como una licencia física) no tienen referencia y se omiten."""
    recortes, textos = [], []
    for pdf in sorted((C.ROOT / carpeta).glob("*.pdf")):
        doc = fitz.open(pdf)
        if not doc[0].get_text().strip():
            continue
        for p in doc:
            pix = p.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
            e = dpi / 72
            for x0, y0, x1, y1, w, *_ in p.get_text("words"):
                t = norm(w)
                if not t or len(t) > 30 or y1 - y0 > 25:
                    continue
                pad = (y1 - y0) * 0.15
                recortes.append(img[int((y0 - pad) * e):int((y1 + pad) * e) + 1, int((x0 - pad) * e):int((x1 + pad) * e) + 1])
                textos.append(t)
    return recortes, textos


def recortes_postulantes(n, semilla):
    """Hasta `n` campos de una palabra de documentos entregados como imagen, enderezados por su cuadrilátero.

    Las etiquetas (etiquetas_ocr_v2) y la legibilidad (ground_truth_v2) solo se usan para medir: se excluyen los
    archivos ilegibles.
    """
    gt = [json.loads(l) for l in open(C.ROOT / "ground_truth_v2" / "ground_truth.jsonl", encoding="utf-8")]
    ilegibles = {(f["carpeta"], a) for f in gt if not f["legible"] for a in f["archivos"]}
    cand = []
    for f in (C.ROOT / "etiquetas_ocr_v2").glob("*.json"):
        for e in json.loads(f.read_text(encoding="utf-8")):
            ruta = C.ROOT / "postulant_files" / f.stem / e["archivo"]
            t = norm(e["texto"] or "")
            if (ruta.suffix.lower() in IMAGENES or "imagen_en_pagina" in e) and t and " " not in t and len(t) <= 30 \
                    and (f.stem, e["archivo"]) not in ilegibles:
                cand.append((ruta, e["pagina"], e["cuadro"], t, e["campo"]))
    random.Random(semilla).shuffle(cand)
    recortes, textos, campos, cache = [], [], [], {}
    for ruta, pag, cuadro, t, campo in cand:
        if len(recortes) >= n:
            break
        if (ruta, pag) not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[(ruta, pag)] = imagen_archivo(ruta, pag)
        img = cache[(ruta, pag)]
        if img is None:
            continue
        W, H = img.size
        q = np.float32(cuadro) * [W, H]
        alto = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        ancho = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        if alto < 6 or ancho < 2 or (q[:, 0].min() < 0) or (q[:, 1].min() < 0) or q[:, 0].max() > W or q[:, 1].max() > H:
            continue
        ex, ey = (q[1] - q[0]) / ancho, (q[3] - q[0]) / alto
        mh, mw = 0.17 * alto, 0.05 * ancho
        src = np.float32([q[0] - ex * mw - ey * mh, q[1] + ex * mw - ey * mh, q[2] + ex * mw + ey * mh, q[3] - ex * mw + ey * mh])
        h_out = int(np.clip(alto * 1.35, 12, 80))
        w_out = int(np.clip(h_out * ancho / alto * 1.1, 8, 700))
        M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]]))
        recortes.append(cv2.warpPerspective(np.asarray(img), M, (w_out, h_out), borderMode=cv2.BORDER_REPLICATE))
        textos.append(t)
        campos.append(campo)
    return recortes, textos, campos


def medir(leidos, textos, filtro=None):
    """Exactitud, CER y primeros errores de las lecturas (opcionalmente solo los textos que pasan `filtro`)."""
    pares = [(a, b) for a, b in zip(leidos, textos) if filtro is None or filtro(b)]
    if not pares:
        return {"n": 0}
    return {"n": len(pares), "exactitud": sum(a == b for a, b in pares) / len(pares),
            "cer": sum(Levenshtein.distance(a, b) for a, b in pares) / max(1, sum(len(b) for _, b in pares)),
            "errores": [[b, a] for a, b in pares if a != b][:15]}


@torch.inference_mode()
def main():
    """Evalúa cada reconocedor de --recos en los PDF reales (100/150/200 DPI) y, con --postulantes, en campos de
    postulant_files; guarda los resultados en --salida."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--recos", nargs="+", required=True, help="nombre=ruta")
    ap.add_argument("--carpetas", nargs="+", default=["official_file", "DATOS REALES"],
                    help="carpetas con PDF de texto real (plantillas y documentos reales)")
    ap.add_argument("--postulantes", action="store_true", help="incluir campos de postulant_files")
    ap.add_argument("--salida", default=str(C.DIR_MODELOS / "eval_independiente.json"))
    a = ap.parse_args()
    prioridad_baja()
    with R.Candado("evaluacion_independiente"):
        R.verificar(vram_gb=4.0, ram_gb=5.0, ventana_s=20)
        R.limitar_gpu(5.0)
        recos = {n: recognition_predictor(cargar_modelo(p).cuda().eval(), batch_size=128)
                 for n, p in (x.split("=", 1) for x in a.recos)}
        res = {"postulantes": {}}
        for carpeta in a.carpetas:
            res[carpeta] = {}
            for dpi in (100, 150, 200):
                rec, txt = recortes_reales(dpi, carpeta)
                for n, r in recos.items():
                    leidos = [norm(t) for t, _ in r(rec)]
                    res[carpeta][f"{n}_{dpi}dpi"] = d = {"todo": medir(leidos, txt),
                                                         "palabras": medir(leidos, txt, lambda t: not es_signo(t)),
                                                         "signos": medir(leidos, txt, es_signo)}
                    print(f"[{carpeta} {dpi} DPI | {n}] palabras {d['palabras']['n']}: {d['palabras']['exactitud']:.4f} "
                          f"(CER {d['palabras']['cer']:.4f}) | signos sueltos {d['signos']['n']}: "
                          f"{d['signos'].get('exactitud', float('nan')):.4f} | todo {d['todo']['exactitud']:.4f}", flush=True)
        if not a.postulantes:
            Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            return
        rec, txt, campos = recortes_postulantes(a.n, 5)
        print(f"campos de una palabra de postulant_files: {len(rec)}", flush=True)
        for n, r in recos.items():
            leidos = [norm(t) for t, _ in r(rec)]
            res["postulantes"][n] = m = medir(leidos, txt)
            print(f"[postulant_files | {n}] exactitud {m['exactitud']:.4f} CER {m['cer']:.4f}", flush=True)
        Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
