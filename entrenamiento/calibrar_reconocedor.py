"""Calibra la confianza de un reconocedor con los recortes de entrenamiento/testing (regresión isotónica).

Mide la confianza tal como la calcula el lector en producción (verificacion/ocr.py), ajusta la curva con una
mitad de la muestra y comprueba el error de calibración con la otra mitad.

Uso: python -u -m entrenamiento.calibrar_reconocedor --pesos modelos/parseq_cl_candidato.pt
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

from entrenamiento.calibrar_confianza import ece, isotonica
from entrenamiento.entrenar_reconocedor import DIR, indice, indice_zips, prioridad_baja
from verificacion import config as C
from verificacion import recursos as R


def main():
    """Mide la confianza en una muestra repartida entre los conjuntos, ajusta la curva con una mitad, informa el error
    de calibración en la otra (las dos mitades mezclan todos los conjuntos) y guarda la curva ajustada con todo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--salida", default=str(C.DIR_MODELOS / "calibracion_confianza_cl.json"))
    ap.add_argument("--splits", nargs="+", default=["testing"], help="conjuntos de recortes (se reparte --n entre ellos)")
    a = ap.parse_args()
    prioridad_baja()
    with R.Candado("calibracion"):
        R.verificar(vram_gb=3.0, ram_gb=4.0, ventana_s=20)
        R.limitar_gpu(4.0)
        from verificacion.ocr import LectorOCR
        lector = LectorOCR(ruta_pesos=Path(a.pesos), restringir=False, calibrar=False)
        items = []
        for s in a.splits:
            it = indice(s)["items"]
            random.Random(11).shuffle(it)
            items += [(s, x) for x in it[:a.n // len(a.splits)]]
        random.Random(12).shuffle(items)
        rutas = {s: [DIR / s / "recortes" / z for z in indice_zips(s)] for s in a.splits}
        mm = {}
        conf, acierto = [], []
        for i in range(0, len(items), 1024):
            lote = [x for _, x in items[i:i + 1024]]
            recortes = []
            for s, (zi, off, n, *_) in items[i:i + 1024]:
                if (s, zi) not in mm:
                    with open(rutas[s][zi], "rb") as f:
                        mm[s, zi] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                g = np.asarray(Image.open(io.BytesIO(mm[s, zi][off:off + n])).convert("L"))
                recortes.append(np.repeat(g[:, :, None], 3, axis=2))
            for (texto, c), it in zip(lector.reconocer(recortes), lote):
                conf.append(c)
                acierto.append(texto == it[3])
        conf, acierto = np.array(conf), np.array(acierto)
        mitad = len(conf) // 2
        xs, ys = isotonica(conf[:mitad], acierto[:mitad])
        cal = np.interp(conf[mitad:], xs, ys)
        print(f"recortes {len(conf)} | acierto {acierto.mean():.4f} | error de calibración (mitad no usada): "
              f"sin calibrar {ece(conf[mitad:], acierto[mitad:]):.4f} -> calibrada {ece(cal, acierto[mitad:]):.4f}")
        for u in (0.5, 0.7, 0.8, 0.9, 0.95):
            m, mc = conf[mitad:] >= u, cal >= u
            print(f"  confianza >= {u:.2f}: bruta -> acierto {acierto[mitad:][m].mean():.4f} ({m.mean():.0%}) | "
                  f"calibrada -> {acierto[mitad:][mc].mean():.4f} ({mc.mean():.0%})")
        xs_f, ys_f = isotonica(conf, acierto)
        Path(a.salida).write_text(json.dumps({"x": xs_f.tolist(), "y": ys_f.tolist(), "n": int(len(conf)),
                                              "pesos": a.pesos,
                                              "fuente": f"entrenamiento/{'+'.join(a.splits)} (recortes)"}),
                                  encoding="utf-8")
        print(f"Curva guardada en {a.salida} ({len(xs_f)} puntos)")


if __name__ == "__main__":
    main()
