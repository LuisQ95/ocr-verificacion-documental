"""Entrena el reconocedor PARSeq con los recortes de entrenamiento/training y lo valida con entrenamiento/testing.

Modos:
  barrido   prueba varias tasas de aprendizaje con pocos pasos y elige la mejor (validación en testing)
  entrenar  entrenamiento completo con todos los recortes de training; guarda el mejor según testing
  evaluar   evalúa unos pesos con TODOS los recortes de testing, desglosado por canal, DPI, documento, etc.

Los recortes están en ZIP sin compresión: cada imagen se lee directo por su posición dentro del ZIP
(mmap), sin descomprimir, lo que permite barajar al azar 2 millones de recortes.

Recursos: candado de ejecución única, chequeo con ventana de observación, límite duro de VRAM calculado
según lo libre, prioridad baja de CPU y vigía de memoria (si el sistema se queda sin margen, guarda, libera
los lectores de datos y espera a que la memoria se recupere; si no se recupera en 3 h se detiene y
--reanudar continúa).

Uso:
  python -u -m entrenamiento.entrenar_reconocedor --modo evaluar --pesos modelos/parseq_es_finetuned.pt
  python -u -m entrenamiento.entrenar_reconocedor --modo barrido --lrs 2e-4 4e-4 8e-4
  python -u -m entrenamiento.entrenar_reconocedor --modo entrenar --lr 4e-4 --epochs 4
"""

import argparse
import ctypes
import gc
import itertools
import json
import math
import mmap
import pickle
import random
import struct
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from doctr.datasets import VOCABS
from doctr.models import parseq
from doctr.models.preprocessor import PreProcessor
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader, Dataset

from verificacion import config as C
from verificacion import recursos as R

DIR = C.ROOT / "entrenamiento"
VOCAB = set(VOCABS["spanish"])
NORMALIZAR = str.maketrans({"\xad": "-", "‐": "-", "‑": "-", "‒": "-", ";": ";"})
CANDIDATO = C.DIR_MODELOS / "parseq_cl_candidato.pt"


def prioridad_baja():
    """Prioridad BELOW_NORMAL en Windows: entrena, pero cede la CPU a lo que el usuario esté usando."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def limpiar(t):
    """Etiqueta normalizada, o None si está vacía, pasa de 30 caracteres o tiene caracteres fuera del vocabulario."""
    t = unicodedata.normalize("NFC", t).translate(NORMALIZAR)
    return t if 0 < len(t) <= 30 and all(c in VOCAB for c in t) else None


def indice(split):
    """[(zip, offset, tamaño, texto, documento, subcanal, dpi, valor, flags)] con caché en disco.

    Lee todos los ZIP de la carpeta: lote_* (conjunto anterior) o docs_/nombres_/aleatorio_* (conjunto v2). La caché
    se rehace si cambia la lista de ZIP.
    """
    cache = DIR / split / "indice_recortes.pkl"
    zips = sorted((DIR / split / "recortes").glob("*.zip"))
    if cache.exists():
        datos = pickle.loads(cache.read_bytes())
        if datos["zips"] == [z.name for z in zips]:
            return datos
    items, descartados = [], 0
    for zi, z in enumerate(zips):
        with zipfile.ZipFile(z) as zf, open(z, "rb") as fh:
            meta = {m["archivo"]: m for m in (json.loads(l) for l in zf.read("meta.jsonl").decode("utf-8").splitlines() if l)}
            for info in zf.infolist():
                m = meta.get(info.filename)
                if m is None:
                    continue
                t = limpiar(m["texto"])
                if t is None:
                    descartados += 1
                    continue
                fh.seek(info.header_offset)
                cab = fh.read(30)
                n_nom, n_extra = struct.unpack("<HH", cab[26:30])
                items.append((zi, info.header_offset + 30 + n_nom + n_extra, info.file_size, t, m["documento"],
                              m["subcanal"], m["dpi"], m["valor"], tuple(m["flags"])))
    datos = {"zips": [z.name for z in zips], "items": items, "descartados": descartados}
    cache.write_bytes(pickle.dumps(datos))
    return datos


class Recortes(Dataset):
    """Guarda el índice en arreglos compactos: en Windows cada proceso lector recibe una copia, y una lista de
    2 millones de tuplas Python multiplicada por los lectores agota la memoria del sistema."""

    def __init__(self, split, items, cfg):
        """Recortes `items` del conjunto `split`, preprocesados según la configuración `cfg` del modelo."""
        self.rutas =[DIR / split / "recortes" / z for z in indice_zips(split)]
        self.zi = np.array([it[0] for it in items], np.int16)
        self.off = np.array([it[1] for it in items], np.int64)
        self.tam = np.array([it[2] for it in items], np.int32)
        textos = [it[3].encode("utf-8") for it in items]
        self.t_fin = np.cumsum([len(t) for t in textos]).astype(np.int64)
        self.t_blob = b"".join(textos)
        self.cfg = cfg
        self.mm = {}
        self.pre = None

    def __len__(self):
        """Cantidad de recortes."""
        return len(self.zi)

    def _texto(self, i):
        """Etiqueta del recorte `i`."""
        ini = self.t_fin[i - 1] if i else 0
        return self.t_blob[ini:self.t_fin[i]].decode("utf-8")

    def _bytes(self, zi, off, n):
        """Bytes de una imagen dentro del ZIP `zi`, leídos por mmap (el ZIP se abre una vez por lector)."""
        m = self.mm.get(zi)
        if m is None:
            with open(self.rutas[zi], "rb") as f:
                m = self.mm[zi] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return m[off:off + n]

    def __getitem__(self, i):
        """(tensor normalizado, etiqueta, índice) del recorte `i`."""
        if self.pre is None:
            self.pre = PreProcessor(self.cfg["input_shape"][1:], batch_size=1, mean=self.cfg["mean"], std=self.cfg["std"],
                                    preserve_aspect_ratio=True, symmetric_pad=False)
        import io
        texto = self._texto(i)
        img = np.asarray(Image.open(io.BytesIO(self._bytes(int(self.zi[i]), int(self.off[i]), int(self.tam[i])))).convert("L"))
        x = self.pre.normalize(self.pre.sample_transforms(np.repeat(img[:, :, None], 3, axis=2)))
        return x, texto, i


_ZIPS = {}


def indice_zips(split):
    """Nombres de los ZIP de recortes del conjunto, en el mismo orden que usa `indice`."""
    if split not in _ZIPS:
        _ZIPS[split] = [z.name for z in sorted((DIR / split / "recortes").glob("*.zip"))]
    return _ZIPS[split]


def _collate(batch):
    """Lote: tensores apilados, etiquetas e índices."""
    xs, ts, ids = zip(*batch)
    return torch.stack(xs), list(ts), list(ids)


def _init_worker(_):
    """Cada lector de datos corre con prioridad baja."""
    prioridad_baja()


def cargador(partes, cfg, bs, workers, barajar=False):
    """partes: [(split, items)]: varios conjuntos se mezclan (p. ej. training_v2 + repaso de training).
    Los lectores de entrenamiento se mantienen vivos; los de validación se crean y se liberan en cada uso."""
    ds = [Recortes(s, it, cfg) for s, it in partes]
    return DataLoader(ds[0] if len(ds) == 1 else torch.utils.data.ConcatDataset(ds), batch_size=bs, shuffle=barajar,
                      drop_last=barajar, num_workers=workers, collate_fn=_collate, pin_memory=True,
                      worker_init_fn=_init_worker, persistent_workers=barajar and workers > 0)


def cargar_modelo(ruta=None):
    """ruta None o 'generico': PARSeq preentrenado de docTR (nunca vio los datos del proyecto); la capa de salida y
    el embedding de caracteres se reinician porque el vocabulario cambia a español."""
    if ruta == "generico":
        ruta = None
    m = parseq(pretrained=ruta is None, vocab=VOCABS["spanish"])
    if ruta:
        m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return m


def _sin_tildes(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


@torch.inference_mode()
def evaluar(modelo, loader, items, desglose=False):
    """Exactitud por palabra, CER y primeros 40 errores. Con `desglose`, también por documento, subcanal, rango de
    DPI, tipo de valor, marcas y palabras con tilde o Ñ. Cada acumulador es [exactas, n, distancia, largo]."""
    modelo.eval()
    tot = defaultdict(lambda: [0, 0, 0, 0])
    errores = []
    for x, textos, ids in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = modelo(x.cuda(non_blocking=True))["preds"]
        for (p, _), t, i in zip(preds, textos, ids):
            ok, d = p == t, Levenshtein.distance(p, t)
            claves = ["total"]
            if desglose:
                _, _, _, _, doc, sub, dpi, valor, flags = items[i]
                rango = "<100" if dpi < 100 else "100-199" if dpi < 200 else "200-299" if dpi < 300 else "300+"
                claves += [f"documento={doc}", f"subcanal={sub}", f"dpi={rango}", f"valor={valor}"] + \
                          [f"marca={f}" for f in flags] + (["sin_marcas"] if not flags else []) + \
                          (["tiene_tilde_o_ñ"] if _sin_tildes(t) != t else [])
            for k in claves:
                a = tot[k]
                a[0] += ok
                a[1] += 1
                a[2] += d
                a[3] += len(t)
            if not ok and len(errores) < 40:
                errores.append([t, p])
    res = {k: {"exactitud": v[0] / v[1], "cer": v[2] / max(1, v[3]), "n": v[1]} for k, v in sorted(tot.items())}
    return {"exactitud": res["total"]["exactitud"], "cer": res["total"]["cer"], "desglose": res, "errores": errores}


def muestra(items, n, semilla):
    """Hasta `n` elementos al azar (reproducible con `semilla`)."""
    rnd =random.Random(semilla)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    return [items[i] for i in idx[:n]]


def validar(modelo, validaciones, log):
    """validaciones: [(split, loader, items)]. Criterio = promedio de la exactitud de cada conjunto (así el mejor
    modelo debe leer bien los nombres nunca vistos Y no empeorar en el conjunto anterior)."""
    res = {s: evaluar(modelo, ld, it) for s, ld, it, _ in validaciones}
    criterio = [res[s] for s, _, _, c in validaciones if c]
    for s, m in res.items():
        log(f"    validación {s}: exactitud {m['exactitud']:.4f} CER {m['cer']:.4f}")
    return {"exactitud": sum(m["exactitud"] for m in criterio) / len(criterio),
            "cer": sum(m["cer"] for m in criterio) / len(criterio),
            "por_conjunto": {s: {"exactitud": m["exactitud"], "cer": m["cer"]} for s, m in res.items()}}


def vocabulario_v1():
    """Palabras de nombres/apellidos del conjunto anterior (aparecen miles de veces: el modelo las memorizó)."""
    sys.path.insert(0, str(C.ROOT))
    from scripts.sinteticos_reales import entrenamiento_ocr as E
    from scripts.sinteticos_reales import valores as V
    palabras = set()
    for w in V.NOMBRES_M + V.NOMBRES_F + V.APELLIDOS + E.NOMBRES_RAROS + E.APELLIDOS_RAROS:
        palabras.update(p for p in w.upper().replace("'", " ").replace("-", " ").split() if len(p) > 1)
    return sorted(palabras)


def reserva_v1(log):
    """20 % de los nombres/apellidos del conjunto anterior queda fuera del entrenamiento (solo para medir)."""
    todas = vocabulario_v1()
    reservadas = set(random.Random(20261027).sample(todas, round(len(todas) * 0.2)))
    (C.DIR_MODELOS / "reserva_vocabulario_v1.json").write_text(json.dumps(sorted(reservadas), ensure_ascii=False, indent=0),
                                                                encoding="utf-8")
    log(f"Vocabulario anterior: {len(todas)} palabras de nombres; {len(reservadas)} reservadas para medir")
    return set(todas), reservadas


def repaso_v1(items, nombres, reservadas, tope, semilla=11):
    """Datos anteriores redistribuidos: fuera las palabras reservadas; las palabras de nombres (sobrerrepresentadas)
    con un máximo de `tope` recortes cada una; todo lo demás (fechas, RUT, notas, textos fijos) completo."""
    rnd = random.Random(semilla)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    cuenta, out = defaultdict(int), []
    for i in idx:
        t = items[i][3].upper()
        if t in reservadas:
            continue
        if t in nombres:
            cuenta[t] += 1
            if cuenta[t] > tope:
                continue
        out.append(items[i])
    return out


def esperar_memoria(necesaria_gb, log, max_h=3.0, estable_s=120):
    """Pausa: espera a que la memoria comprometida libre vuelva a `necesaria_gb` durante `estable_s` seguidos
    (otro programa la soltó). True si se recuperó; False si pasaron `max_h` horas."""
    fin, desde = time.time() + max_h * 3600, None
    while time.time() < fin:
        libre = R.memoria()["comprometida_libre_gb"]
        desde = (desde or time.time()) if libre >= necesaria_gb else None
        if desde and time.time() - desde >= estable_s:
            log(f"  memoria recuperada ({libre:.1f} GB libres): se reanuda")
            return True
        time.sleep(10)
    return False


def entrenar(modelo, train_partes, validaciones, args, pasos_max=None, salida=None, checkpoint=None, log=print):
    """Entrena con AdamW y OneCycle; valida al final de cada época y guarda en `salida` el mejor según el criterio.

    - `pasos_max`: entrenamiento corto (barrido o afinado) de una sola época de esa cantidad de pasos.
    - Punto de control cada 4 000 pasos y al final de cada época; con --reanudar se continúa desde él. Al reanudar a
      mitad de época se hace un recorrido nuevo (barajado) para los pasos que faltan.
    - Pausa por memoria: si el vigía alerta, se guarda, se liberan los lectores de datos (~1,9 GB cada uno) y se
      espera a que el otro programa suelte memoria; la pausa no cuenta para la velocidad informada, que se mide
      desde el inicio de esta sesión (no cuenta pasos de una sesión anterior).
    """
    vigia = R.Vigia().iniciar()
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    train = cargador(train_partes, modelo.cfg, args.bs, args.workers, barajar=True)
    pasos_ep = len(train)
    total = pasos_max or args.epochs * pasos_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.05 if not pasos_max else 0.15)
    paso, ep0, historial, mejor = 0, 1, [], -1.0
    if checkpoint and args.reanudar and checkpoint.exists():
        e = torch.load(checkpoint, map_location="cpu", weights_only=False)
        modelo.load_state_dict(e["modelo"])
        opt.load_state_dict(e["opt"])
        sched.load_state_dict(e["sched"])
        paso, ep0, historial, mejor = e["paso"], e["epoca"], e["historial"], e["mejor"]
        log(f"Reanudando en el paso {paso} (época {ep0})")

    def guardar(ep):
        """Punto de control (modelo, optimizador, planificador, paso, época, historial y mejor)."""
        if checkpoint:
            torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "paso": paso, "epoca": ep, "historial": historial, "mejor": mejor}, checkpoint)

    t0, paso0 = time.perf_counter(), paso
    for ep in range(ep0, (args.epochs if not pasos_max else 1) + 1):
        modelo.train()
        suma, n = 0.0, 0
        hechos = paso - (ep - 1) * pasos_ep if not pasos_max else 0
        while hechos < pasos_ep and not (pasos_max and paso >= pasos_max):
            pausa = False
            for x, textos, _ in itertools.islice(train, pasos_ep - hechos):
                if vigia.alerta:
                    pausa = True
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = modelo(x.cuda(non_blocking=True), textos)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
                opt.step()
                sched.step()
                paso += 1
                hechos += 1
                if math.isfinite(loss.item()):
                    suma += loss.item()
                    n += 1
                if paso % 500 == 0:
                    el = time.perf_counter() - t0
                    log(f"  paso {paso}/{total} | pérdida {suma / max(1, n):.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                        f"{el / 60:.1f} min | {(paso - paso0) * args.bs / el:.0f} recortes/s | VRAM pico "
                        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
                if checkpoint and paso % 4000 == 0:
                    guardar(ep)
                if pasos_max and paso >= pasos_max:
                    break
            if not pausa:
                break
            guardar(ep)
            log(f"PAUSA por memoria del sistema en el paso {paso}: {vigia.alerta}. Se liberan los lectores y se espera")
            vigia.detener()
            del train
            gc.collect()
            t_pausa = time.perf_counter()
            if not esperar_memoria(args.workers * 1.9 + C.MARGEN_COMPROMETIDA_GB, log):
                log("DETENIDO: la memoria no se recuperó en 3 h. Reanudar con --reanudar")
                return None
            t0 += time.perf_counter() - t_pausa
            train = cargador(train_partes, modelo.cfg, args.bs, args.workers, barajar=True)
            vigia = R.Vigia().iniciar()
        m = validar(modelo, validaciones, log)
        historial.append({"epoca": ep, "paso": paso, "perdida": suma / max(1, n), "exactitud": m["exactitud"], "cer": m["cer"],
                          "por_conjunto": m["por_conjunto"], "minutos": round((time.perf_counter() - t0) / 60, 1)})
        log(f"[época {ep}] pérdida {suma / max(1, n):.4f} | validación (promedio): exactitud {m['exactitud']:.4f} "
            f"CER {m['cer']:.4f}")
        if salida and m["exactitud"] > mejor:
            mejor = m["exactitud"]
            torch.save(modelo.state_dict(), salida)
            log(f"  mejor hasta ahora -> {salida.name}")
        guardar(ep + 1)
        if pasos_max:
            break
    vigia.detener()
    return {"historial": historial, "mejor": mejor}


def presupuesto_vram(log):
    """Usa lo que esté libre en la GPU (peor momento observado) dejando el margen configurado."""
    peor = float("inf")
    fin = time.time() + 30
    while time.time() < fin:
        g = R.gpu()
        if g is None:
            raise R.RecursosInsuficientes("no se pudo consultar la GPU")
        peor = min(peor, g["vram_libre_gb"])
        time.sleep(2)
    gb = round(min(12.0, peor - C.MARGEN_VRAM_GB - 0.5), 1)
    log(f"VRAM libre en el peor momento {peor:.1f} GB -> presupuesto {gb} GB (margen {C.MARGEN_VRAM_GB} GB)")
    return gb


def main():
    """Línea de comandos de los tres modos (ver el módulo).

    - Recursos: el proceso principal con CUDA y lotes de 256 compromete ~11 GB; cada lector de datos importa
      PyTorch/docTR y compromete ~1,9 GB, así que su número se ajusta a la memoria libre (más 2 de validación).
    - Validación: cada `--val-splits` es un criterio para elegir el mejor punto; "split:clase" filtra por clase
      (nombres, aleatorio o docs). Con --repaso-v1 se agregan los datos anteriores redistribuidos y se informa la
      exactitud en los recortes de las palabras anteriores reservadas.
    - --lr-auto usa la mejor tasa del último barrido.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["barrido", "entrenar", "evaluar"], required=True)
    ap.add_argument("--pesos", default=str(C.MODELO_PARSEQ), help="pesos de partida / a evaluar")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-4, 4e-4, 8e-4])
    ap.add_argument("--pasos", type=int, default=2500, help="pasos por configuración en el barrido")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=60000, help="recortes de testing para validar en cada época")
    ap.add_argument("--salida", default=str(CANDIDATO))
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--pasos-afinado", type=int, default=0, help="modo entrenar: afinado corto de N pasos")
    ap.add_argument("--lr-auto", action="store_true", help="modo entrenar: usar la mejor tasa del último barrido")
    ap.add_argument("--train-splits", nargs="+", default=["training"], help="conjuntos de entrenamiento")
    ap.add_argument("--val-splits", nargs="+", default=["testing"],
                    help="conjuntos de validación (criterio); 'split:clase' con clase nombres|aleatorio|docs")
    ap.add_argument("--repaso-v1", action="store_true",
                    help="agregar los datos anteriores redistribuidos (sin nombres reservados, nombres con tope)")
    ap.add_argument("--tope-nombre", type=int, default=300, help="máximo de recortes por palabra de nombre anterior")
    ap.add_argument("--split-eval", default="testing", help="modo evaluar: conjunto a evaluar")
    ap.add_argument("--solo-reservadas-v1", action="store_true", help="modo evaluar: solo nombres anteriores reservados")
    args = ap.parse_args()
    prioridad_baja()
    def log(s):
        """Imprime al instante (el registro se sigue mientras corre)."""
        print(s, flush=True)

    try:
        with R.Candado("reconocedor"):
            gb = presupuesto_vram(log)
            if gb < 3:
                raise R.RecursosInsuficientes(f"presupuesto de VRAM {gb} GB insuficiente")
            R.verificar(vram_gb=gb, ram_gb=6.0, ventana_s=30, log=log)
            libre = R.memoria()["comprometida_libre_gb"]
            posibles = int((libre - C.MARGEN_COMPROMETIDA_GB - 12.0) // 1.9) - 2
            args.workers = max(2, min(args.workers, posibles))
            log(f"Memoria comprometida libre {libre:.1f} GB -> {args.workers} lectores de entrenamiento + 2 de validación")
            R.limitar_gpu(gb)
            torch.backends.cudnn.benchmark = True
            t0 = time.time()
            base = cargar_modelo(args.pesos)
            if args.modo == "evaluar":
                modelo = base.cuda()
                ev = indice(args.split_eval)
                items_ev = ev["items"]
                if args.solo_reservadas_v1:
                    _, reservadas = reserva_v1(log)
                    items_ev = [it for it in items_ev if it[3].upper() in reservadas]
                m = evaluar(modelo, cargador([(args.split_eval, items_ev)], modelo.cfg, 512, args.workers + 2), items_ev,
                            desglose=True)
                destino = C.DIR_MODELOS / f"eval_reconocedor_{Path(args.pesos).stem}{args.etiqueta}.json"
                destino.write_text(json.dumps({"pesos": args.pesos, "split": args.split_eval, **m}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
                log(f"[{Path(args.pesos).name}] {args.split_eval} ({len(items_ev)} recortes): exactitud "
                    f"{m['exactitud']:.4f} CER {m['cer']:.4f} -> {destino.name}")
                return
            train_partes = [(s, indice(s)["items"]) for s in args.train_splits]
            validaciones = []
            for spec in args.val_splits:
                s, _, clase = spec.partition(":")
                it = [x for x in indice(s)["items"] if not clase or (
                      x[4] not in ("nombres", "aleatorio") if clase == "docs" else x[4] == clase)]
                it = muestra(it, args.val_n if args.modo == "entrenar" else 20000, 7)
                validaciones.append((spec, cargador([(s, it)], base.cfg, 512, 2), it, True))
            if args.repaso_v1:
                nombres_v1, reservadas = reserva_v1(log)
                train_partes.append(("training", repaso_v1(indice("training")["items"], nombres_v1, reservadas,
                                                           args.tope_nombre)))
                res_items = [it for it in indice("testing")["items"] if it[3].upper() in reservadas]
                validaciones.append(("v1_reservadas", cargador([("testing", res_items)], base.cfg, 512, 2), res_items, False))
            for s, it in train_partes:
                log(f"  entrenamiento: {s} {len(it)} recortes")
            for s, _, it, c in validaciones:
                log(f"  validación: {s} {len(it)} recortes{' (criterio)' if c else ' (solo se informa)'}")
            log(f"Índices listos en {time.time() - t0:.0f}s")
            if args.modo == "barrido":
                resultados = []
                for lr in args.lrs:
                    args.lr = lr
                    modelo = cargar_modelo(args.pesos).cuda()
                    log(f"--- barrido lr={lr:g} bs={args.bs} pasos={args.pasos}")
                    r = entrenar(modelo, train_partes, validaciones, args, pasos_max=args.pasos, log=log)
                    if r is None:
                        return
                    resultados.append({"lr": lr, **r["historial"][-1]})
                    del modelo
                    torch.cuda.empty_cache()
                mejor = max(resultados, key=lambda r: r["exactitud"])
                (C.DIR_MODELOS / "barrido_reconocedor.json").write_text(json.dumps({"resultados": resultados, "mejor": mejor},
                                                                                     indent=1), encoding="utf-8")
                log(f"Mejor tasa de aprendizaje: {mejor['lr']:g} (exactitud {mejor['exactitud']:.4f})")
                return
            if args.lr_auto:
                args.lr = json.loads((C.DIR_MODELOS / "barrido_reconocedor.json").read_text(encoding="utf-8"))["mejor"]["lr"]
                log(f"Tasa de aprendizaje elegida por el barrido: {args.lr:g}")
            modelo = base.cuda()
            salida = Path(args.salida)
            r = entrenar(modelo, train_partes, validaciones, args, salida=salida,
                         checkpoint=salida.with_name(salida.stem + "_checkpoint.pt"), log=log,
                         pasos_max=args.pasos_afinado or None)
            if r:
                salida.with_name(salida.stem + "_metricas.json").write_text(json.dumps({**r, "args": vars(args)}, indent=1),
                                                                            encoding="utf-8")
                log(f"Terminado. Mejor exactitud en validación: {r['mejor']:.4f} -> {salida}")
    except R.RecursosInsuficientes as e:
        log(f"NO SE LANZÓ: {e}")
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
