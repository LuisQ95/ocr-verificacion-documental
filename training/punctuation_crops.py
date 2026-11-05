"""Recortes de signos de puntuación aislados (":", ".", "-", "/", "°", …) para el reconocedor.

El generador de documentos solo guardó palabras con letras o números, y el reconocedor nuevo nunca vio un ":"
suelto (en los documentos reales aparece en cada "Etiqueta :"). Se crean recortes con las mismas fuentes y
degradaciones, en un ZIP con el mismo formato que los demás (lote_99000.zip).

Uso: python -u -m entrenamiento.punctuation_crops
"""

import io
import json
import random
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from verification import config as C

FONTS = ["cour.ttf", "courbd.ttf", "arial.ttf", "arialbd.ttf", "verdana.ttf", "verdanab.ttf", "calibri.ttf",
           "tahoma.ttf", "times.ttf", "segoeui.ttf"]
SIGNS = ([":", ".", ",", ";", "-", "/", "°", "%", "(", ")", "'", ":-", "-", "N°"],
          [30, 12, 8, 5, 10, 6, 5, 5, 3, 3, 2, 1, 5, 5])


def crop(rnd):
    """(texto, JPEG) de un signo suelto, con fuente, tamaño, tinta, desenfoque, ruido y compresión al azar.

    El recorte tiene la altura de una línea de texto normal (como el de una palabra); a veces se reduce y se vuelve
    a ampliar para imitar una foto lejana o 72 DPI.
    """
    text = rnd.choices(*SIGNS)[0]
    f = ImageFont.truetype(f"C:\\Windows\\Fonts\\{rnd.choice(FONTS)}", rnd.randint(14, 48))
    _, y0, _, y1 = f.getbbox("Ag")
    w = int(f.getlength(text)) + 2
    height = y1 - y0
    background = tuple(rnd.randint(200, 255) for _ in range(3))
    ink = tuple(rnd.randint(0, 70) for _ in range(3))
    mh, mw = int(height * rnd.uniform(0.1, 0.25)), int(height * rnd.uniform(0.05, 0.25))
    img = Image.new("RGB", (w + 2 * mw, height + 2 * mh), background)
    ImageDraw.Draw(img).text((mw, mh - y0), text, font=f, fill=ink)
    a = np.asarray(img).astype(np.float32)
    if rnd.random() < 0.5:
        a = cv2.GaussianBlur(a, (0, 0), rnd.uniform(0.3, 1.5))
    a += np.random.default_rng(rnd.getrandbits(32)).normal(0, rnd.uniform(2, 14), a.shape)
    if rnd.random() < 0.3:
        f_ = rnd.uniform(0.35, 0.7)
        h, w_ = a.shape[:2]
        a = cv2.resize(cv2.resize(a, (max(4, int(w_ * f_)), max(4, int(h * f_)))), (w_, h))
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=rnd.randint(40, 92))
    return text, b.getvalue()


def write(split, n, seed):
    """Escribe `n` recortes en training/<split>/recortes/lote_99000.zip (mismo formato que los demás lotes)."""
    rnd = random.Random(seed)
    path = C.ROOT / "training" / split / "recortes" / "lote_99000.zip"
    meta, labels = [], {}
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            text, data = crop(rnd)
            name = f"punt_{i:06d}.jpg"
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED)
            labels[name] = text
            meta.append({"archivo": name, "texto": text, "doc": -1, "documento": "puntuacion", "valor": False,
                         "flags": [], "subcanal": "sintetico", "dpi": 0})
        zf.writestr("labels.json", json.dumps(labels, ensure_ascii=False))
        zf.writestr("meta.jsonl", "\n".join(json.dumps(m, ensure_ascii=False) for m in meta))
        zf.writestr("negativos.json", "[]")
    print(f"{split}: {n} recortes de puntuación -> {path}")


if __name__ == "__main__":
    write("training", 60000, 1)
    write("testing", 5000, 2)
