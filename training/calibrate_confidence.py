"""Calibración de la confianza del reconocedor: aprende la curva 'confianza bruta -> probabilidad de
acierto real' con recortes de validación cuyo texto se conoce (regresión isotónica), y la valida en
una mitad que no se usó para ajustarla.

Resultado: models/calibracion_confianza.json (lo usa verification/ocr.py si OCR_CALIBRATE = True).
ADVERTENCIA: los recortes son sintéticos; con datos reales conviene recalibrar con recortes reales
corregidos en la revisión humana.

Uso: python -u -m entrenamiento.calibrate_confidence
"""

import json
import random

import numpy as np
from PIL import Image

from verification import config as C
from verification import resources as R

FOLDERS = [C.ROOT / "datos_entrenamiento" / "parseq_sintetico" / "val",
            C.ROOT / "datos_entrenamiento" / "parseq_sintetico_v2" / "val"]


def isotonic(x, y):
    """Regresión isotónica (pool adjacent violators): y no decreciente en x.

    Cada bloque es [media, peso, x_min, x_max]; se fusionan bloques vecinos mientras violen el orden.
    """
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y, dtype=float)[order]
    blocks = [[y[i], 1.0, x[i], x[i]] for i in range(len(x))]
    stack = []
    for b in blocks:
        stack.append(b)
        while len(stack) > 1 and stack[-2][0] > stack[-1][0]:
            b2, b1 = stack.pop(), stack.pop()
            w = b1[1] + b2[1]
            stack.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    xs = np.array([(b[2] + b[3]) / 2 for b in stack])
    ys = np.array([b[0] for b in stack])
    return xs, ys


def ece(conf, hit, bins=10):
    """Error de calibración esperado: diferencia media entre confianza y acierto real por tramos."""
    conf, hit = np.asarray(conf), np.asarray(hit, dtype=float)
    total = 0.0
    for a in range(bins):
        m = (conf >= a / bins) & (conf < (a + 1) / bins if a < bins - 1 else conf <= 1)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return total


def main():
    """Calibración del flujo anterior con los recortes de `CARPETAS`; la curva final se ajusta con todos los datos.

    Para el reconocedor actual se usa training/calibrate_recognizer.py (recortes de training/testing).
    """
    with R.RunLock("calibracion"):
        R.verify(vram_gb=C.OCR_VRAM_BUDGET_GB, ram_gb=C.RAM_BUDGET_GB, window_s=10)
        R.limit_gpu(C.OCR_VRAM_BUDGET_GB)
        from verification.ocr import OCRReader
        reader = OCRReader(restrict=False, calibrate=False)

        items = []
        for d in FOLDERS:
            labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
            items += [(d / "images" / k, v) for k, v in labels.items()]
        random.Random(1).shuffle(items)
        conf, hit = [], []
        for i in range(0, len(items), 512):
            batch = items[i:i + 512]
            crops = [np.repeat(np.asarray(Image.open(p).convert("L"))[:, :, None], 3, axis=2) for p, _ in batch]
            for (text, c), (_, real) in zip(reader.recognize(crops), batch):
                conf.append(c)
                hit.append(text == real)
        conf, hit = np.array(conf), np.array(hit)
        half = len(conf) // 2
        xs, ys = isotonic(conf[:half], hit[:half])
        cal = np.interp(conf[half:], xs, ys)

        print(f"recortes: {len(conf)} (ajuste {half}, validación {len(conf) - half}) | acierto global {hit.mean():.3f}")
        print(f"error de calibración (validación): sin calibrar {ece(conf[half:], hit[half:]):.3f} -> "
              f"calibrada {ece(cal, hit[half:]):.3f}")
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            m = conf[half:] >= threshold
            mc = cal >= threshold
            print(f"  confianza >= {threshold:.2f}: bruta -> acierto real {hit[half:][m].mean():.3f} "
                  f"({m.mean():.0%} de lecturas) | calibrada -> {hit[half:][mc].mean():.3f} ({mc.mean():.0%})")
        xs_f, ys_f = isotonic(conf, hit)
        C.CONFIDENCE_CALIBRATION.write_text(json.dumps({"x": xs_f.tolist(), "y": ys_f.tolist(), "n": int(len(conf)),
                                                  "fuente": "recortes sintéticos de validación v1+v2"}), encoding="utf-8")
        print(f"Curva guardada en {C.CONFIDENCE_CALIBRATION} ({len(xs_f)} puntos)")


if __name__ == "__main__":
    main()
