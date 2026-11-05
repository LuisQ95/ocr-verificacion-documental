"""Afina el detector de texto (DBNet de docTR) con los documentos completos de training/training y lo
compara con el detector actual en training/testing.

Cada ejemplo es una ventana de 1024x1024 de un documento entregado (foto, escaneo, captura, PDF de imagen),
reescalada para que las palabras midan 18–34 px de alto: el detector aprende a encontrar texto con fondos,
perspectiva, sellos, resaltador, oclusiones, etc. Las palabras ocluidas no se marcan como texto.

Modos:
  entrenar  afina (columna vertebral con tasa baja, cabezas con tasa normal) y guarda el mejor según testing
  evaluar   detector actual vs candidato en las mismas ventanas de testing: precisión, exhaustividad, F1 y
            lectura de punta a punta (palabras leídas exactas) con el reconocedor indicado

Uso:
  python -u -m training.train_detector --mode entrenar --epochs 3
  python -u -m training.train_detector --mode evaluar --reco models/parseq_candidato.pt
"""

import argparse
import ctypes
import gzip
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pillow_heif
import pymupdf as fitz
import torch
from doctr.datasets import VOCABS
from doctr.models import db_resnet50, parseq
from doctr.models.preprocessor import PreProcessor
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from verification import config as C
from verification import resources as R

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = None
DIR = C.ROOT / "training"
SIDE = 1024
CANDIDATE = C.MODELS_DIR / "dbnet_cl_candidato.pt"
EXCLUDE = {"ocluido"}


def low_priority():
    """Prioridad BELOW_NORMAL en Windows: cede la CPU a lo que el usuario esté usando."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def pages(split):
    """[(ruta, número de página dentro del archivo, cuadros normalizados [(4x2), texto])] de documentos guardados."""
    out = []
    for z in sorted((DIR / split / "etiquetas").glob("*.jsonl.gz")):
        for line in gzip.open(z, "rt", encoding="utf-8"):
            r = json.loads(line)
            if not r["guardado"] or "ilegible" in r["efectos"]:
                continue
            seen_one = {}
            for pg in r["paginas"]:
                arch_file = pg.get("archivo")
                if not arch_file:
                    continue
                n = seen_one.get(arch_file, 0)
                seen_one[arch_file] = n + 1
                words = [(w["cuadro"], w["texto"]) for w in pg["palabras"] if not set(w["flags"]) & EXCLUDE]
                if words:
                    out.append((str(DIR / split / "documentos" / arch_file), n, words))
    return out


def load_image(path, n):
    """Página `n` como imagen RGB: la imagen incrustada de un PDF de imagen, o la página de una imagen/TIFF."""
    if path.endswith(".pdf"):
        doc = fitz.open(path)
        xref = doc[n].get_images()[0][0]
        return Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")
    img = Image.open(path)
    if n:
        img.seek(n)
    return img.convert("RGB")


def window(img, words, rnd):
    """Reescala para que las palabras midan 18–34 px y recorta 1024x1024 alrededor de una palabra al azar.

    Devuelve la ventana, las cajas normalizadas de las palabras que quedan dentro (al menos 60 % de su área) y sus
    textos. Los cuadriláteros de las palabras se manejan como arreglo (N, 4, 2) en px.
    """
    W, H = img.size
    q = np.array([c for c, _ in words], np.float32) * [W, H]
    tall = (np.linalg.norm(q[:, 3] - q[:, 0], axis=1) + np.linalg.norm(q[:, 2] - q[:, 1], axis=1)) / 2
    s = rnd.uniform(18, 34) / max(2.0, float(np.median(tall)))
    s = min(s, 4.0)
    img = img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.Resampling.BILINEAR)
    q = q * s
    cx, cy = q[rnd.randrange(len(q))].mean(0)
    x0 = int(np.clip(cx - rnd.uniform(0.2, 0.8) * SIDE, 0, max(0, img.width - SIDE)))
    y0 = int(np.clip(cy - rnd.uniform(0.2, 0.8) * SIDE, 0, max(0, img.height - SIDE)))
    canvas = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    canvas.paste(img.crop((x0, y0, x0 + SIDE, y0 + SIDE)), (0, 0))
    q = q - [x0, y0]
    boxes, texts = [], []
    for quad, (_, t) in zip(q, words):
        bx0, by0 = quad.min(0)
        bx1, by1 = quad.max(0)
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, SIDE), min(by1, SIDE)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * area:
            continue
        boxes.append([cx0 / SIDE, cy0 / SIDE, cx1 / SIDE, cy1 / SIDE])
        texts.append(t)
    return np.asarray(canvas), np.array(boxes, np.float32).reshape(-1, 4), texts


class Windows(Dataset):
    """`n` ventanas al azar (reproducibles con `semilla`) de las páginas `items`."""

    def __init__(self, items, n, seed, cfg):
        """Ventanas de `items` normalizadas según la configuración `cfg` del detector."""
        self.items, self.n, self.seed, self.cfg = items, n, seed, cfg

    def __len__(self):
        """Cantidad de ventanas."""
        return self.n

    def __getitem__(self, i):
        """(tensor normalizado, objetivo {"words": cajas}, imagen, textos) de la ventana `i`."""
        rnd = random.Random(self.seed * 1_000_003 + i)
        path, k, words = self.items[rnd.randrange(len(self.items))]
        arr, boxes, texts = window(load_image(path, k), words, rnd)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(self.cfg["mean"]).view(3, 1, 1)) / torch.tensor(self.cfg["std"]).view(3, 1, 1)
        return x, {"words": boxes}, arr, texts


def _collate(b):
    """Lote: tensores apilados, objetivos, imágenes y textos."""
    xs, ts, arrs, txt = zip(*b)
    return torch.stack(xs), list(ts), list(arrs), list(txt)


def _init(_):
    """Cada lector corre con prioridad baja y con el lector HEIC registrado."""
    low_priority()
    pillow_heif.register_heif_opener()


def detector(path=None):
    """DBNet preentrenado de docTR, con los pesos de `ruta` si se dan."""
    m = db_resnet50(pretrained=True)
    if path:
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m


def iou(a, b):
    """Matriz de intersección sobre unión entre las cajas `a` (N, 4) y `b` (M, 4)."""
    x0, y0 =np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


@torch.inference_mode()
def evaluate(det, reco, loader):
    """Precisión, exhaustividad y F1 de las cajas (emparejadas con IoU ≥ 0,5) y lectura de punta a punta: cada caja
    detectada se recorta y se lee con el reconocedor."""
    det.eval()
    reco.eval()
    pre_r = PreProcessor(reco.cfg["input_shape"][1:], batch_size=256, mean=reco.cfg["mean"], std=reco.cfg["std"],
                         preserve_aspect_ratio=True, symmetric_pad=False)
    tp = fp = fn = read_count = total_gt = 0
    for x, targets, arrs, texts in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = det(x.cuda(non_blocking=True), return_preds=True)["preds"]
        for p, obj, arr, txt in zip(preds, targets, arrs, texts):
            gt = obj["words"]
            pr = p["words"][:, :4] if len(p["words"]) else np.zeros((0, 4), np.float32)
            total_gt += len(gt)
            if len(pr) == 0:
                fn += len(gt)
                continue
            if len(gt) == 0:
                fp += len(pr)
                continue
            m = iou(pr, gt)
            used_p, used_g, pairs = set(), set(), []
            for idx in np.argsort(-m, axis=None):
                i, j = divmod(int(idx), m.shape[1])
                if m[i, j] < 0.5:
                    break
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                pairs.append((i, j))
            tp += len(pairs)
            fp += len(pr) - len(pairs)
            fn += len(gt) - len(pairs)
            crops = []
            for i, _ in pairs:
                bx0, by0, bx1, by1 = (pr[i] * SIDE).astype(int)
                crops.append(arr[max(0, by0):max(by0 + 1, by1), max(0, bx0):max(bx0 + 1, bx1)])
            if crops:
                read_values = []
                for batch in pre_r(crops):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        read_values += reco(batch.cuda())["preds"]
                read_count += sum(t == txt[j] for (t, _), (_, j) in zip(read_values, pairs))
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": prec, "exhaustividad": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "palabras_leidas_exactas": read_count / max(1, total_gt), "palabras": total_gt}


def recognizer(path):
    """PARSeq (vocabulario español) con los pesos de `ruta`, en la GPU."""
    m = parseq(pretrained=False, vocab=VOCABS["spanish"])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.cuda()


def main():
    """Línea de comandos de los dos modos (ver el módulo).

    - Cada lector carga fotos de hasta 4032 px y páginas de 600 DPI: se reserva ~3,5 GB por lector.
    - Se congelan las primeras etapas de la red base (ya saben detectar bordes y formas): menos VRAM. Solo layer4 se
      entrena, con una tasa 10 veces menor que las cabezas.
    - La BatchNorm de la red base queda fija: lotes tan chicos la descalibran.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["entrenar", "evaluar"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--windows", type=int, default=16000, help="ventanas por época")
    ap.add_argument("--val-windows", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reco", default=str(C.PARSEQ_MODEL))
    ap.add_argument("--det", default=str(CANDIDATE))
    ap.add_argument("--vram-max", type=float, default=6.0,
                    help="tope de VRAM: en Windows lo reservado por CUDA también cuenta como memoria del sistema")
    args = ap.parse_args()
    low_priority()
    def log(s):
        """Imprime al instante (el registro se sigue mientras corre)."""
        print(s, flush=True)

    try:
        with R.RunLock("detector"):
            g = R.gpu()
            gb = round(min(args.vram_max, g["vram_libre_gb"] - C.VRAM_MARGIN_GB - 0.5), 1)
            R.verify(vram_gb=gb, ram_gb=6.0, window_s=30, log=log)
            free = R.memory()["comprometida_libre_gb"]
            args.workers = max(1, min(args.workers, int((free - C.COMMIT_MARGIN_GB - 12.0) // 3.5) - 1))
            log(f"VRAM presupuesto {gb} GB | memoria comprometida libre {free:.1f} GB -> {args.workers} lectores")
            R.limit_gpu(gb)
            torch.backends.cudnn.benchmark = True
            t0 = time.time()
            tr, te = pages("training"), pages("testing")
            log(f"Páginas: training {len(tr)} | testing {len(te)} ({time.time() - t0:.0f}s)")
            cfg = db_resnet50(pretrained=False).cfg
            val = DataLoader(Windows(te, args.val_windows, 99, cfg), batch_size=args.bs, num_workers=2,
                             collate_fn=_collate, worker_init_fn=_init)
            reco = recognizer(args.reco)
            if args.mode == "evaluar":
                res = {}
                for name, path in (("actual", None), ("candidato", args.det)):
                    if path and not Path(path).exists():
                        continue
                    res[name] = evaluate(detector(path).cuda(), reco, val)
                    log(f"[{name}] {json.dumps({k: round(v, 4) for k, v in res[name].items()})}")
                (C.MODELS_DIR / "eval_detector.json").write_text(json.dumps({"reco": args.reco, **res}, indent=1),
                                                                   encoding="utf-8")
                return
            model = detector().cuda()
            base = evaluate(model, reco, val)
            log(f"[detector actual] {json.dumps({k: round(v, 4) for k, v in base.items()})}")
            for name, p in model.feat_extractor.named_parameters():
                if not name.startswith("layer4"):
                    p.requires_grad = False
            column = [p for n_, p in model.feat_extractor.named_parameters() if p.requires_grad]
            ids = {id(p) for p in model.feat_extractor.parameters()}
            rest = [p for p in model.parameters() if id(p) not in ids]
            opt = torch.optim.AdamW([{"params": column, "lr": args.lr * 0.1}, {"params": rest, "lr": args.lr}],
                                    weight_decay=1e-4)
            steps = args.epochs * (args.windows // args.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.1, args.lr], total_steps=steps, pct_start=0.1)
            watchdog = R.Watchdog().start()
            best, history = base["f1"], []
            for ep in range(1, args.epochs + 1):
                data = DataLoader(Windows(tr, args.windows, ep, cfg), batch_size=args.bs, shuffle=False,
                                   num_workers=args.workers, collate_fn=_collate, worker_init_fn=_init, drop_last=True)
                model.train()
                model.feat_extractor.eval()
                total_sum, n, te0 = 0.0, 0, time.time()
                for k, (x, obj, _, _) in enumerate(data, 1):
                    if watchdog.alert:
                        log(f"DETENIDO por memoria del sistema: {watchdog.alert}")
                        return
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(x.cuda(non_blocking=True), obj)["loss"]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                    opt.step()
                    sched.step()
                    total_sum += loss.item()
                    n += 1
                    if k % 200 == 0:
                        log(f"  época {ep} paso {k}/{len(data)} | pérdida {total_sum / n:.4f}"
                            f" | {(time.time() - te0) / 60:.1f} min"
                            f" | VRAM pico {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
                m = evaluate(model, reco, val)
                history.append({"epoca": ep, "perdida": total_sum / max(1, n), **m})
                log(f"[época {ep}] {json.dumps({k: round(v, 4) for k, v in m.items()})}")
                if m["f1"] > best:
                    best = m["f1"]
                    torch.save(model.state_dict(), CANDIDATE)
                    log(f"  mejor F1 hasta ahora -> {CANDIDATE.name}")
            watchdog.stop()
            (C.MODELS_DIR / "dbnet_cl_candidato_metricas.json").write_text(
                json.dumps({"base": base, "historial": history, "args": vars(args)}, indent=1), encoding="utf-8")
            log(f"Terminado. F1 base {base['f1']:.4f} -> mejor {best:.4f}")
    except R.InsufficientResources as e:
        log(f"NO SE LANZÓ: {e}")
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
