"""Calibración de la confianza del reconocedor: aprende la curva 'confianza bruta -> probabilidad de
acierto real' con recortes de validación cuyo texto se conoce (regresión isotónica), y la valida en
una mitad que no se usó para ajustarla.

Resultado: modelos/calibracion_confianza.json (lo usa verificacion/ocr.py si OCR_CALIBRAR = True).
ADVERTENCIA: los recortes son sintéticos; con datos reales conviene recalibrar con recortes reales
corregidos en la revisión humana.

Uso: python -u -m entrenamiento.calibrar_confianza
"""

import json
import random

import numpy as np
from PIL import Image

from verificacion import config as C
from verificacion import recursos as R

CARPETAS = [C.ROOT / "datos_entrenamiento" / "parseq_sintetico" / "val",
            C.ROOT / "datos_entrenamiento" / "parseq_sintetico_v2" / "val"]


def isotonica(x, y):
    """Regresión isotónica (pool adjacent violators): y no decreciente en x.

    Cada bloque es [media, peso, x_min, x_max]; se fusionan bloques vecinos mientras violen el orden.
    """
    orden = np.argsort(x)
    x, y = np.asarray(x)[orden], np.asarray(y, dtype=float)[orden]
    bloques = [[y[i], 1.0, x[i], x[i]] for i in range(len(x))]
    pila = []
    for b in bloques:
        pila.append(b)
        while len(pila) > 1 and pila[-2][0] > pila[-1][0]:
            b2, b1 = pila.pop(), pila.pop()
            w = b1[1] + b2[1]
            pila.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    xs = np.array([(b[2] + b[3]) / 2 for b in pila])
    ys = np.array([b[0] for b in pila])
    return xs, ys


def ece(conf, acierto, bins=10):
    """Error de calibración esperado: diferencia media entre confianza y acierto real por tramos."""
    conf, acierto = np.asarray(conf), np.asarray(acierto, dtype=float)
    total = 0.0
    for a in range(bins):
        m = (conf >= a / bins) & (conf < (a + 1) / bins if a < bins - 1 else conf <= 1)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - acierto[m].mean())
    return total


def main():
    """Calibración del flujo anterior con los recortes de `CARPETAS`; la curva final se ajusta con todos los datos.

    Para el reconocedor actual se usa entrenamiento/calibrar_reconocedor.py (recortes de entrenamiento/testing).
    """
    with R.Candado("calibracion"):
        R.verificar(vram_gb=C.PRESUPUESTO_VRAM_OCR_GB, ram_gb=C.PRESUPUESTO_RAM_GB, ventana_s=10)
        R.limitar_gpu(C.PRESUPUESTO_VRAM_OCR_GB)
        from verificacion.ocr import LectorOCR
        lector = LectorOCR(restringir=False, calibrar=False)

        items = []
        for d in CARPETAS:
            etiquetas = json.loads((d / "labels.json").read_text(encoding="utf-8"))
            items += [(d / "images" / k, v) for k, v in etiquetas.items()]
        random.Random(1).shuffle(items)
        conf, acierto = [], []
        for i in range(0, len(items), 512):
            lote = items[i:i + 512]
            recortes = [np.repeat(np.asarray(Image.open(p).convert("L"))[:, :, None], 3, axis=2) for p, _ in lote]
            for (texto, c), (_, real) in zip(lector.reconocer(recortes), lote):
                conf.append(c)
                acierto.append(texto == real)
        conf, acierto = np.array(conf), np.array(acierto)
        mitad = len(conf) // 2
        xs, ys = isotonica(conf[:mitad], acierto[:mitad])
        cal = np.interp(conf[mitad:], xs, ys)

        print(f"recortes: {len(conf)} (ajuste {mitad}, validación {len(conf) - mitad}) | acierto global {acierto.mean():.3f}")
        print(f"error de calibración (validación): sin calibrar {ece(conf[mitad:], acierto[mitad:]):.3f} -> "
              f"calibrada {ece(cal, acierto[mitad:]):.3f}")
        for umbral in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            m = conf[mitad:] >= umbral
            mc = cal >= umbral
            print(f"  confianza >= {umbral:.2f}: bruta -> acierto real {acierto[mitad:][m].mean():.3f} "
                  f"({m.mean():.0%} de lecturas) | calibrada -> {acierto[mitad:][mc].mean():.3f} ({mc.mean():.0%})")
        xs_f, ys_f = isotonica(conf, acierto)
        C.CALIBRACION_CONF.write_text(json.dumps({"x": xs_f.tolist(), "y": ys_f.tolist(), "n": int(len(conf)),
                                                  "fuente": "recortes sintéticos de validación v1+v2"}), encoding="utf-8")
        print(f"Curva guardada en {C.CALIBRACION_CONF} ({len(xs_f)} puntos)")


if __name__ == "__main__":
    main()
