"""Calibra la confianza de un reconocedor con los recortes de training/testing (regresión isotónica).

Mide la confianza tal como la calcula el lector en producción (verification/ocr.py), ajusta la curva con una
mitad de la muestra y comprueba el error de calibración con la otra mitad.

Uso: python -u -m training.calibrate_recognizer --weights models/parseq_candidato.pt
     [--splits testing testing_v2]   (testing_v2: nombres nunca vistos y texto aleatorio; la confianza debe valer
                                      también para lo que el modelo no memorizó)
"""

import argparse
import io
import json
import mmap
import random
from pathlib import Path

import numpy as np
from PIL import Image

from training.calibrate_confidence import ece, isotonic
from training.train_recognizer import DIR, index, zip_index, low_priority
from verification import config as C
from verification import resources as R


def main():
    """Mide la confianza en una muestra repartida entre los conjuntos, ajusta la curva con una mitad, informa el error
    de calibración en la otra (las dos mitades mezclan todos los conjuntos) y guarda la curva ajustada con todo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--output", default=str(C.MODELS_DIR / "confidence_calibration.json"))
    ap.add_argument("--splits", nargs="+", default=["testing"], help="conjuntos de recortes (se reparte --n entre ellos)")
    a = ap.parse_args()
    low_priority()
    with R.RunLock("calibracion"):
        R.verify(vram_gb=3.0, ram_gb=4.0, window_s=20)
        R.limit_gpu(4.0)
        from verification.ocr import OCRReader
        reader = OCRReader(weights_path=Path(a.weights), restrict=False, calibrate=False)
        items = []
        for s in a.splits:
            it = index(s)["items"]
            random.Random(11).shuffle(it)
            items += [(s, x) for x in it[:a.n // len(a.splits)]]
        random.Random(12).shuffle(items)
        paths = {s: [DIR / s / "recortes" / z for z in zip_index(s)] for s in a.splits}
        mm = {}
        conf, hit = [], []
        for i in range(0, len(items), 1024):
            batch = [x for _, x in items[i:i + 1024]]
            crops = []
            for s, (zi, off, n, *_) in items[i:i + 1024]:
                if (s, zi) not in mm:
                    with open(paths[s][zi], "rb") as f:
                        mm[s, zi] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                g = np.asarray(Image.open(io.BytesIO(mm[s, zi][off:off + n])).convert("L"))
                crops.append(np.repeat(g[:, :, None], 3, axis=2))
            for (text, c), it in zip(reader.recognize(crops), batch):
                conf.append(c)
                hit.append(text == it[3])
        conf, hit = np.array(conf), np.array(hit)
        half = len(conf) // 2
        xs, ys = isotonic(conf[:half], hit[:half])
        cal = np.interp(conf[half:], xs, ys)
        print(f"recortes {len(conf)} | acierto {hit.mean():.4f} | error de calibración (mitad no usada): "
              f"sin calibrar {ece(conf[half:], hit[half:]):.4f} -> calibrada {ece(cal, hit[half:]):.4f}")
        for u in (0.5, 0.7, 0.8, 0.9, 0.95):
            m, mc = conf[half:] >= u, cal >= u
            print(f"  confianza >= {u:.2f}: bruta -> acierto {hit[half:][m].mean():.4f} ({m.mean():.0%}) | "
                  f"calibrada -> {hit[half:][mc].mean():.4f} ({mc.mean():.0%})")
        xs_f, ys_f = isotonic(conf, hit)
        Path(a.output).write_text(json.dumps({"x": xs_f.tolist(), "y": ys_f.tolist(), "n": int(len(conf)),
                                              "pesos": a.weights,
                                              "fuente": f"training/{'+'.join(a.splits)} (recortes)"}),
                                  encoding="utf-8")
        print(f"Curva guardada en {a.output} ({len(xs_f)} puntos)")


if __name__ == "__main__":
    main()
