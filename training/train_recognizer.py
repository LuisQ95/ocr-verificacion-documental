"""Entrena el reconocedor PARSeq con los recortes de training/training y lo valida con training/testing.

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
  python -u -m training.train_recognizer --mode evaluar --weights models/parseq_es_finetuned.pt
  python -u -m training.train_recognizer --mode barrido --lrs 2e-4 4e-4 8e-4
  python -u -m training.train_recognizer --mode entrenar --lr 4e-4 --epochs 4
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

from verification import config as C
from verification import resources as R

DIR = C.ROOT / "entrenamiento"
VOCAB = set(VOCABS["spanish"])
NORMALIZE = str.maketrans({"\xad": "-", "‐": "-", "‑": "-", "‒": "-", ";": ";"})
CANDIDATE = C.MODELS_DIR / "parseq_candidato.pt"


def low_priority():
    """Prioridad BELOW_NORMAL en Windows: entrena, pero cede la CPU a lo que el usuario esté usando."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def clean(t):
    """Etiqueta normalizada, o None si está vacía, pasa de 30 caracteres o tiene caracteres fuera del vocabulario."""
    t = unicodedata.normalize("NFC", t).translate(NORMALIZE)
    return t if 0 < len(t) <= 30 and all(c in VOCAB for c in t) else None


def index(split):
    """[(zip, offset, tamaño, texto, documento, subcanal, dpi, valor, flags)] con caché en disco.

    Lee todos los ZIP de la carpeta: lote_* (conjunto anterior) o docs_/nombres_/aleatorio_* (conjunto v2). La caché
    se rehace si cambia la lista de ZIP.
    """
    cache = DIR / split / "indice_recortes.pkl"
    zips = sorted((DIR / split / "recortes").glob("*.zip"))
    if cache.exists():
        data = pickle.loads(cache.read_bytes())
        if data["zips"] == [z.name for z in zips]:
            return data
    items, discarded = [], 0
    for zi, z in enumerate(zips):
        with zipfile.ZipFile(z) as zf, open(z, "rb") as fh:
            meta = {m["archivo"]: m for m in (json.loads(l) for l in zf.read("meta.jsonl").decode("utf-8").splitlines() if l)}
            for info in zf.infolist():
                m = meta.get(info.filename)
                if m is None:
                    continue
                t = clean(m["texto"])
                if t is None:
                    discarded += 1
                    continue
                fh.seek(info.header_offset)
                header = fh.read(30)
                n_names, n_extra = struct.unpack("<HH", header[26:30])
                items.append((zi, info.header_offset + 30 + n_names + n_extra, info.file_size, t, m["documento"],
                              m["subcanal"], m["dpi"], m["valor"], tuple(m["flags"])))
    data = {"zips": [z.name for z in zips], "items": items, "descartados": discarded}
    cache.write_bytes(pickle.dumps(data))
    return data


class Crops(Dataset):
    """Guarda el índice en arreglos compactos: en Windows cada proceso lector recibe una copia, y una lista de
    2 millones de tuplas Python multiplicada por los lectores agota la memoria del sistema."""

    def __init__(self, split, items, cfg):
        """Recortes `items` del conjunto `split`, preprocesados según la configuración `cfg` del modelo."""
        self.paths =[DIR / split / "recortes" / z for z in zip_index(split)]
        self.zi = np.array([it[0] for it in items], np.int16)
        self.off = np.array([it[1] for it in items], np.int64)
        self.size_ = np.array([it[2] for it in items], np.int32)
        texts = [it[3].encode("utf-8") for it in items]
        self.t_end = np.cumsum([len(t) for t in texts]).astype(np.int64)
        self.t_blob = b"".join(texts)
        self.cfg = cfg
        self.mm = {}
        self.pre = None

    def __len__(self):
        """Cantidad de recortes."""
        return len(self.zi)

    def _text(self, i):
        """Etiqueta del recorte `i`."""
        start_i = self.t_end[i - 1] if i else 0
        return self.t_blob[start_i:self.t_end[i]].decode("utf-8")

    def _bytes(self, zi, off, n):
        """Bytes de una imagen dentro del ZIP `zi`, leídos por mmap (el ZIP se abre una vez por lector)."""
        m = self.mm.get(zi)
        if m is None:
            with open(self.paths[zi], "rb") as f:
                m = self.mm[zi] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return m[off:off + n]

    def __getitem__(self, i):
        """(tensor normalizado, etiqueta, índice) del recorte `i`."""
        if self.pre is None:
            self.pre = PreProcessor(self.cfg["input_shape"][1:], batch_size=1, mean=self.cfg["mean"], std=self.cfg["std"],
                                    preserve_aspect_ratio=True, symmetric_pad=False)
        import io
        text = self._text(i)
        crop_bytes = self._bytes(int(self.zi[i]), int(self.off[i]), int(self.size_[i]))
        img = np.asarray(Image.open(io.BytesIO(crop_bytes)).convert("L"))
        x = self.pre.normalize(self.pre.sample_transforms(np.repeat(img[:, :, None], 3, axis=2)))
        return x, text, i


_ZIPS = {}


def zip_index(split):
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
    low_priority()


def make_loader(parts, cfg, bs, workers, do_shuffle=False):
    """partes: [(split, items)]: varios conjuntos se mezclan (p. ej. training_v2 + repaso de training).
    Los lectores de entrenamiento se mantienen vivos; los de validación se crean y se liberan en cada uso."""
    ds = [Crops(s, it, cfg) for s, it in parts]
    return DataLoader(ds[0] if len(ds) == 1 else torch.utils.data.ConcatDataset(ds), batch_size=bs, shuffle=do_shuffle,
                      drop_last=do_shuffle, num_workers=workers, collate_fn=_collate, pin_memory=True,
                      worker_init_fn=_init_worker, persistent_workers=do_shuffle and workers > 0)


def load_model(path=None):
    """ruta None o 'generico': PARSeq preentrenado de docTR (nunca vio los datos del proyecto); la capa de salida y
    el embedding de caracteres se reinician porque el vocabulario cambia a español."""
    if path == "generico":
        path = None
    m = parseq(pretrained=path is None, vocab=VOCABS["spanish"])
    if path:
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m


def _strip_accents(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


@torch.inference_mode()
def evaluate(model, loader, items, breakdown=False):
    """Exactitud por palabra, CER y primeros 40 errores. Con `desglose`, también por documento, subcanal, rango de
    DPI, tipo de valor, marcas y palabras con tilde o Ñ. Cada acumulador es [exactas, n, distancia, largo]."""
    model.eval()
    tot = defaultdict(lambda: [0, 0, 0, 0])
    errors = []
    for x, texts, ids in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(x.cuda(non_blocking=True))["preds"]
        for (p, _), t, i in zip(preds, texts, ids):
            ok, d = p == t, Levenshtein.distance(p, t)
            keys = ["total"]
            if breakdown:
                _, _, _, _, doc, sub, dpi, value, flags = items[i]
                span = "<100" if dpi < 100 else "100-199" if dpi < 200 else "200-299" if dpi < 300 else "300+"
                keys += [f"documento={doc}", f"subcanal={sub}", f"dpi={span}", f"valor={value}"] + \
                          [f"marca={f}" for f in flags] + (["sin_marcas"] if not flags else []) + \
                          (["tiene_tilde_o_ñ"] if _strip_accents(t) != t else [])
            for k in keys:
                a = tot[k]
                a[0] += ok
                a[1] += 1
                a[2] += d
                a[3] += len(t)
            if not ok and len(errors) < 40:
                errors.append([t, p])
    res = {k: {"exactitud": v[0] / v[1], "cer": v[2] / max(1, v[3]), "n": v[1]} for k, v in sorted(tot.items())}
    return {"exactitud": res["total"]["exactitud"], "cer": res["total"]["cer"], "desglose": res, "errores": errors}


def sample(items, n, seed):
    """Hasta `n` elementos al azar (reproducible con `semilla`)."""
    rnd =random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    return [items[i] for i in idx[:n]]


def validate(model, validations, log):
    """validaciones: [(split, loader, items)]. Criterio = promedio de la exactitud de cada conjunto (así el mejor
    modelo debe leer bien los nombres nunca vistos Y no empeorar en el conjunto anterior)."""
    res = {s: evaluate(model, ld, it) for s, ld, it, _ in validations}
    criterion = [res[s] for s, _, _, c in validations if c]
    for s, m in res.items():
        log(f"    validación {s}: exactitud {m['exactitud']:.4f} CER {m['cer']:.4f}")
    return {"exactitud": sum(m["exactitud"] for m in criterion) / len(criterion),
            "cer": sum(m["cer"] for m in criterion) / len(criterion),
            "por_conjunto": {s: {"exactitud": m["exactitud"], "cer": m["cer"]} for s, m in res.items()}}


def vocabulary_v1():
    """Palabras de nombres/apellidos del conjunto anterior (aparecen miles de veces: el modelo las memorizó).

    Usa el generador de datos sintéticos, que no se publica en el repositorio (ver RECOGNIZER_CARD.md)."""
    sys.path.insert(0, str(C.ROOT))
    from scripts.synthetic import ocr_training as E
    from scripts.synthetic import values as V
    words = set()
    for w in V.MALE_NAMES + V.FEMALE_NAMES + V.SURNAMES + E.RARE_NAMES + E.RARE_SURNAMES:
        words.update(p for p in w.upper().replace("'", " ").replace("-", " ").split() if len(p) > 1)
    return sorted(words)


def reserve_v1(log):
    """20 % de los nombres/apellidos del conjunto anterior queda fuera del entrenamiento (solo para medir)."""
    all_words = vocabulary_v1()
    reserved = set(random.Random(20261027).sample(all_words, round(len(all_words) * 0.2)))
    (C.MODELS_DIR / "reserva_vocabulario_v1.json").write_text(json.dumps(sorted(reserved), ensure_ascii=False, indent=0),
                                                                encoding="utf-8")
    log(f"Vocabulario anterior: {len(all_words)} palabras de nombres; {len(reserved)} reservadas para medir")
    return set(all_words), reserved


def review_v1(items, names, reserved, cap, seed=11):
    """Datos anteriores redistribuidos: fuera las palabras reservadas; las palabras de nombres (sobrerrepresentadas)
    con un máximo de `tope` recortes cada una; todo lo demás (fechas, RUT, notas, textos fijos) completo."""
    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    count, out = defaultdict(int), []
    for i in idx:
        t = items[i][3].upper()
        if t in reserved:
            continue
        if t in names:
            count[t] += 1
            if count[t] > cap:
                continue
        out.append(items[i])
    return out


def wait_for_memory(required_gb, log, max_h=3.0, stable_s=120):
    """Pausa: espera a que la memoria comprometida libre vuelva a `required_gb` durante `stable_s` seguidos
    (otro programa la soltó). True si se recuperó; False si pasaron `max_h` horas."""
    end, from_ = time.time() + max_h * 3600, None
    while time.time() < end:
        free = R.memory()["comprometida_libre_gb"]
        from_ = (from_ or time.time()) if free >= required_gb else None
        if from_ and time.time() - from_ >= stable_s:
            log(f"  memoria recuperada ({free:.1f} GB libres): se reanuda")
            return True
        time.sleep(10)
    return False


def run_training(model, train_parts, validations, args, max_steps=None, output=None, checkpoint=None, log=print):
    """Entrena con AdamW y OneCycle; valida al final de cada época y guarda en `salida` el mejor según el criterio.

    - `max_steps`: entrenamiento corto (barrido o afinado) de una sola época de esa cantidad de pasos.
    - Punto de control cada 4 000 pasos y al final de cada época; con --reanudar se continúa desde él. Al reanudar a
      mitad de época se hace un recorrido nuevo (barajado) para los pasos que faltan.
    - Pausa por memoria: si el vigía alerta, se guarda, se liberan los lectores de datos (~1,9 GB cada uno) y se
      espera a que el otro programa suelte memoria; la pausa no cuenta para la velocidad informada, que se mide
      desde el inicio de esta sesión (no cuenta pasos de una sesión anterior).
    """
    watchdog = R.Watchdog().start()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train = make_loader(train_parts, model.cfg, args.bs, args.workers, do_shuffle=True)
    steps_per_epoch = len(train)
    total = max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.05 if not max_steps else 0.15)
    step, ep0, history, best = 0, 1, [], -1.0
    if checkpoint and args.resume and checkpoint.exists():
        e = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(e["modelo"])
        opt.load_state_dict(e["opt"])
        sched.load_state_dict(e["sched"])
        step, ep0, history, best = e["paso"], e["epoca"], e["historial"], e["mejor"]
        log(f"Reanudando en el paso {step} (época {ep0})")

    def save(ep):
        """Punto de control (modelo, optimizador, planificador, paso, época, historial y mejor)."""
        if checkpoint:
            torch.save({"modelo": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "paso": step, "epoca": ep, "historial": history, "mejor": best}, checkpoint)

    t0, step0 = time.perf_counter(), step
    for ep in range(ep0, (args.epochs if not max_steps else 1) + 1):
        model.train()
        total_sum, n = 0.0, 0
        done = step - (ep - 1) * steps_per_epoch if not max_steps else 0
        while done < steps_per_epoch and not (max_steps and step >= max_steps):
            pause = False
            for x, texts, _ in itertools.islice(train, steps_per_epoch - done):
                if watchdog.alert:
                    pause = True
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x.cuda(non_blocking=True), texts)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                step += 1
                done += 1
                if math.isfinite(loss.item()):
                    total_sum += loss.item()
                    n += 1
                if step % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    log(f"  paso {step}/{total} | pérdida {total_sum / max(1, n):.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                        f"{elapsed / 60:.1f} min | {(step - step0) * args.bs / elapsed:.0f} recortes/s | VRAM pico "
                        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
                if checkpoint and step % 4000 == 0:
                    save(ep)
                if max_steps and step >= max_steps:
                    break
            if not pause:
                break
            save(ep)
            log(f"PAUSA por memoria del sistema en el paso {step}: {watchdog.alert}. Se liberan los lectores y se espera")
            watchdog.stop()
            del train
            gc.collect()
            t_pause = time.perf_counter()
            if not wait_for_memory(args.workers * 1.9 + C.COMMIT_MARGIN_GB, log):
                log("DETENIDO: la memoria no se recuperó en 3 h. Reanudar con --reanudar")
                return None
            t0 += time.perf_counter() - t_pause
            train = make_loader(train_parts, model.cfg, args.bs, args.workers, do_shuffle=True)
            watchdog = R.Watchdog().start()
        m = validate(model, validations, log)
        history.append({"epoca": ep, "paso": step, "perdida": total_sum / max(1, n), "exactitud": m["exactitud"],
                        "cer": m["cer"], "por_conjunto": m["por_conjunto"],
                        "minutos": round((time.perf_counter() - t0) / 60, 1)})
        log(f"[época {ep}] pérdida {total_sum / max(1, n):.4f} | validación (promedio): exactitud {m['exactitud']:.4f} "
            f"CER {m['cer']:.4f}")
        if output and m["exactitud"] > best:
            best = m["exactitud"]
            torch.save(model.state_dict(), output)
            log(f"  mejor hasta ahora -> {output.name}")
        save(ep + 1)
        if max_steps:
            break
    watchdog.stop()
    return {"historial": history, "mejor": best}


def vram_budget(log):
    """Usa lo que esté libre en la GPU (peor momento observado) dejando el margen configurado."""
    worst = float("inf")
    end = time.time() + 30
    while time.time() < end:
        g = R.gpu()
        if g is None:
            raise R.InsufficientResources("no se pudo consultar la GPU")
        worst = min(worst, g["vram_libre_gb"])
        time.sleep(2)
    gb = round(min(12.0, worst - C.VRAM_MARGIN_GB - 0.5), 1)
    log(f"VRAM libre en el peor momento {worst:.1f} GB -> presupuesto {gb} GB (margen {C.VRAM_MARGIN_GB} GB)")
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
    ap.add_argument("--mode", choices=["barrido", "entrenar", "evaluar"], required=True)
    ap.add_argument("--weights", default=str(C.PARSEQ_MODEL), help="pesos de partida / a evaluar")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-4, 4e-4, 8e-4])
    ap.add_argument("--steps", type=int, default=2500, help="pasos por configuración en el barrido")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=60000, help="recortes de testing para validar en cada época")
    ap.add_argument("--output", default=str(CANDIDATE))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--finetune-steps", type=int, default=0, help="modo entrenar: afinado corto de N pasos")
    ap.add_argument("--lr-auto", action="store_true", help="modo entrenar: usar la mejor tasa del último barrido")
    ap.add_argument("--train-splits", nargs="+", default=["training"], help="conjuntos de entrenamiento")
    ap.add_argument("--val-splits", nargs="+", default=["testing"],
                    help="conjuntos de validación (criterio); 'split:clase' con clase nombres|aleatorio|docs")
    ap.add_argument("--review-v1", action="store_true",
                    help="agregar los datos anteriores redistribuidos (sin nombres reservados, nombres con tope)")
    ap.add_argument("--name-cap", type=int, default=300, help="máximo de recortes por palabra de nombre anterior")
    ap.add_argument("--split-eval", default="testing", help="modo evaluar: conjunto a evaluar")
    ap.add_argument("--only-reserved-v1", action="store_true", help="modo evaluar: solo nombres anteriores reservados")
    args = ap.parse_args()
    low_priority()
    def log(s):
        """Imprime al instante (el registro se sigue mientras corre)."""
        print(s, flush=True)

    try:
        with R.RunLock("reconocedor"):
            gb = vram_budget(log)
            if gb < 3:
                raise R.InsufficientResources(f"presupuesto de VRAM {gb} GB insuficiente")
            R.verify(vram_gb=gb, ram_gb=6.0, window_s=30, log=log)
            free = R.memory()["comprometida_libre_gb"]
            possible = int((free - C.COMMIT_MARGIN_GB - 12.0) // 1.9) - 2
            args.workers = max(2, min(args.workers, possible))
            log(f"Memoria comprometida libre {free:.1f} GB -> {args.workers} lectores de entrenamiento + 2 de validación")
            R.limit_gpu(gb)
            torch.backends.cudnn.benchmark = True
            t0 = time.time()
            base = load_model(args.weights)
            if args.mode == "evaluar":
                model = base.cuda()
                ev = index(args.split_eval)
                eval_items = ev["items"]
                if args.only_reserved_v1:
                    _, reserved = reserve_v1(log)
                    eval_items = [it for it in eval_items if it[3].upper() in reserved]
                m = evaluate(model, make_loader([(args.split_eval, eval_items)], model.cfg, 512, args.workers + 2), eval_items,
                            breakdown=True)
                destination = C.MODELS_DIR / f"eval_reconocedor_{Path(args.weights).stem}{args.label}.json"
                destination.write_text(json.dumps({"pesos": args.weights, "split": args.split_eval, **m}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
                log(f"[{Path(args.weights).name}] {args.split_eval} ({len(eval_items)} recortes): exactitud "
                    f"{m['exactitud']:.4f} CER {m['cer']:.4f} -> {destination.name}")
                return
            train_parts = [(s, index(s)["items"]) for s in args.train_splits]
            validations = []
            for spec in args.val_splits:
                s, _, lclass = spec.partition(":")
                it = [x for x in index(s)["items"] if not lclass or (
                      x[4] not in ("nombres", "aleatorio") if lclass == "docs" else x[4] == lclass)]
                it = sample(it, args.val_n if args.mode == "entrenar" else 20000, 7)
                validations.append((spec, make_loader([(s, it)], base.cfg, 512, 2), it, True))
            if args.review_v1:
                names_v1, reserved = reserve_v1(log)
                train_parts.append(("training", review_v1(index("training")["items"], names_v1, reserved,
                                                           args.name_cap)))
                res_items = [it for it in index("testing")["items"] if it[3].upper() in reserved]
                validations.append(("v1_reservadas", make_loader([("testing", res_items)], base.cfg, 512, 2),
                                    res_items, False))
            for s, it in train_parts:
                log(f"  entrenamiento: {s} {len(it)} recortes")
            for s, _, it, c in validations:
                log(f"  validación: {s} {len(it)} recortes{' (criterio)' if c else ' (solo se informa)'}")
            log(f"Índices listos en {time.time() - t0:.0f}s")
            if args.mode == "barrido":
                results = []
                for lr in args.lrs:
                    args.lr = lr
                    model = load_model(args.weights).cuda()
                    log(f"--- barrido lr={lr:g} bs={args.bs} pasos={args.steps}")
                    r = run_training(model, train_parts, validations, args, max_steps=args.steps, log=log)
                    if r is None:
                        return
                    results.append({"lr": lr, **r["historial"][-1]})
                    del model
                    torch.cuda.empty_cache()
                best = max(results, key=lambda r: r["exactitud"])
                (C.MODELS_DIR / "barrido_reconocedor.json").write_text(json.dumps({"resultados": results, "mejor": best},
                                                                                     indent=1), encoding="utf-8")
                log(f"Mejor tasa de aprendizaje: {best['lr']:g} (exactitud {best['exactitud']:.4f})")
                return
            if args.lr_auto:
                args.lr = json.loads((C.MODELS_DIR / "barrido_reconocedor.json").read_text(encoding="utf-8"))["mejor"]["lr"]
                log(f"Tasa de aprendizaje elegida por el barrido: {args.lr:g}")
            model = base.cuda()
            output = Path(args.output)
            r = run_training(model, train_parts, validations, args, output=output,
                         checkpoint=output.with_name(output.stem + "_checkpoint.pt"), log=log,
                         max_steps=args.finetune_steps or None)
            if r:
                output.with_name(output.stem + "_metricas.json").write_text(json.dumps({**r, "args": vars(args)}, indent=1),
                                                                            encoding="utf-8")
                log(f"Terminado. Mejor exactitud en validación: {r['mejor']:.4f} -> {output}")
    except R.InsufficientResources as e:
        log(f"NO SE LANZÓ: {e}")
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
