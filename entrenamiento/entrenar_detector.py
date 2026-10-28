"""Afina el detector de texto (DBNet de docTR) con los documentos completos de entrenamiento/training y lo
compara con el detector actual en entrenamiento/testing.

Cada ejemplo es una ventana de 1024x1024 de un documento entregado (foto, escaneo, captura, PDF de imagen),
reescalada para que las palabras midan 18–34 px de alto: el detector aprende a encontrar texto con fondos,
perspectiva, sellos, resaltador, oclusiones, etc. Las palabras ocluidas no se marcan como texto.

Modos:
  entrenar  afina (columna vertebral con tasa baja, cabezas con tasa normal) y guarda el mejor según testing
  evaluar   detector actual vs candidato en las mismas ventanas de testing: precisión, exhaustividad, F1 y
            lectura de punta a punta (palabras leídas exactas) con el reconocedor indicado

Uso:
  python -u -m entrenamiento.entrenar_detector --modo entrenar --epochs 3
  python -u -m entrenamiento.entrenar_detector --modo evaluar --reco modelos/parseq_cl_candidato.pt
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

from verificacion import config as C
from verificacion import recursos as R

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = None
DIR = C.ROOT / "entrenamiento"
LADO = 1024
CANDIDATO = C.DIR_MODELOS / "dbnet_cl_candidato.pt"
EXCLUIR = {"ocluido"}


def prioridad_baja():
    """Prioridad BELOW_NORMAL en Windows: cede la CPU a lo que el usuario esté usando."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def paginas(split):
    """[(ruta, número de página dentro del archivo, cuadros normalizados [(4x2), texto])] de documentos guardados."""
    out = []
    for z in sorted((DIR / split / "etiquetas").glob("*.jsonl.gz")):
        for linea in gzip.open(z, "rt", encoding="utf-8"):
            r = json.loads(linea)
            if not r["guardado"] or "ilegible" in r["efectos"]:
                continue
            visto = {}
            for pg in r["paginas"]:
                arch = pg.get("archivo")
                if not arch:
                    continue
                n = visto.get(arch, 0)
                visto[arch] = n + 1
                palabras = [(w["cuadro"], w["texto"]) for w in pg["palabras"] if not set(w["flags"]) & EXCLUIR]
                if palabras:
                    out.append((str(DIR / split / "documentos" / arch), n, palabras))
    return out


def cargar_imagen(ruta, n):
    """Página `n` como imagen RGB: la imagen incrustada de un PDF de imagen, o la página de una imagen/TIFF."""
    if ruta.endswith(".pdf"):
        doc = fitz.open(ruta)
        xref = doc[n].get_images()[0][0]
        return Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")
    img = Image.open(ruta)
    if n:
        img.seek(n)
    return img.convert("RGB")


def ventana(img, palabras, rnd):
    """Reescala para que las palabras midan 18–34 px y recorta 1024x1024 alrededor de una palabra al azar.

    Devuelve la ventana, las cajas normalizadas de las palabras que quedan dentro (al menos 60 % de su área) y sus
    textos. Los cuadriláteros de las palabras se manejan como arreglo (N, 4, 2) en px.
    """
    W, H = img.size
    q = np.array([c for c, _ in palabras], np.float32) * [W, H]
    altos = (np.linalg.norm(q[:, 3] - q[:, 0], axis=1) + np.linalg.norm(q[:, 2] - q[:, 1], axis=1)) / 2
    s = rnd.uniform(18, 34) / max(2.0, float(np.median(altos)))
    s = min(s, 4.0)
    img = img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.Resampling.BILINEAR)
    q = q * s
    cx, cy = q[rnd.randrange(len(q))].mean(0)
    x0 = int(np.clip(cx - rnd.uniform(0.2, 0.8) * LADO, 0, max(0, img.width - LADO)))
    y0 = int(np.clip(cy - rnd.uniform(0.2, 0.8) * LADO, 0, max(0, img.height - LADO)))
    lienzo = Image.new("RGB", (LADO, LADO), (0, 0, 0))
    lienzo.paste(img.crop((x0, y0, x0 + LADO, y0 + LADO)), (0, 0))
    q = q - [x0, y0]
    cajas, textos = [], []
    for quad, (_, t) in zip(q, palabras):
        bx0, by0 = quad.min(0)
        bx1, by1 = quad.max(0)
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, LADO), min(by1, LADO)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * area:
            continue
        cajas.append([cx0 / LADO, cy0 / LADO, cx1 / LADO, cy1 / LADO])
        textos.append(t)
    return np.asarray(lienzo), np.array(cajas, np.float32).reshape(-1, 4), textos


class Ventanas(Dataset):
    """`n` ventanas al azar (reproducibles con `semilla`) de las páginas `items`."""

    def __init__(self, items, n, semilla, cfg):
        """Ventanas de `items` normalizadas según la configuración `cfg` del detector."""
        self.items, self.n, self.semilla, self.cfg = items, n, semilla, cfg

    def __len__(self):
        """Cantidad de ventanas."""
        return self.n

    def __getitem__(self, i):
        """(tensor normalizado, objetivo {"words": cajas}, imagen, textos) de la ventana `i`."""
        rnd = random.Random(self.semilla * 1_000_003 + i)
        ruta, k, palabras = self.items[rnd.randrange(len(self.items))]
        arr, cajas, textos = ventana(cargar_imagen(ruta, k), palabras, rnd)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(self.cfg["mean"]).view(3, 1, 1)) / torch.tensor(self.cfg["std"]).view(3, 1, 1)
        return x, {"words": cajas}, arr, textos


def _collate(b):
    """Lote: tensores apilados, objetivos, imágenes y textos."""
    xs, ts, arrs, txt = zip(*b)
    return torch.stack(xs), list(ts), list(arrs), list(txt)


def _init(_):
    """Cada lector corre con prioridad baja y con el lector HEIC registrado."""
    prioridad_baja()
    pillow_heif.register_heif_opener()


def detector(ruta=None):
    """DBNet preentrenado de docTR, con los pesos de `ruta` si se dan."""
    m = db_resnet50(pretrained=True)
    if ruta:
        m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
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
def evaluar(det, reco, loader):
    """Precisión, exhaustividad y F1 de las cajas (emparejadas con IoU ≥ 0,5) y lectura de punta a punta: cada caja
    detectada se recorta y se lee con el reconocedor."""
    det.eval()
    reco.eval()
    pre_r = PreProcessor(reco.cfg["input_shape"][1:], batch_size=256, mean=reco.cfg["mean"], std=reco.cfg["std"],
                         preserve_aspect_ratio=True, symmetric_pad=False)
    tp = fp = fn = leidas = total_gt = 0
    for x, objetivos, arrs, textos in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = det(x.cuda(non_blocking=True), return_preds=True)["preds"]
        for p, obj, arr, txt in zip(preds, objetivos, arrs, textos):
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
            usados_p, usados_g, pares = set(), set(), []
            for idx in np.argsort(-m, axis=None):
                i, j = divmod(int(idx), m.shape[1])
                if m[i, j] < 0.5:
                    break
                if i in usados_p or j in usados_g:
                    continue
                usados_p.add(i)
                usados_g.add(j)
                pares.append((i, j))
            tp += len(pares)
            fp += len(pr) - len(pares)
            fn += len(gt) - len(pares)
            recortes = []
            for i, _ in pares:
                bx0, by0, bx1, by1 = (pr[i] * LADO).astype(int)
                recortes.append(arr[max(0, by0):max(by0 + 1, by1), max(0, bx0):max(bx0 + 1, bx1)])
            if recortes:
                leidos = []
                for lote in pre_r(recortes):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        leidos += reco(lote.cuda())["preds"]
                leidas += sum(t == txt[j] for (t, _), (_, j) in zip(leidos, pares))
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": prec, "exhaustividad": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "palabras_leidas_exactas": leidas / max(1, total_gt), "palabras": total_gt}


def reconocedor(ruta):
    """PARSeq (vocabulario español) con los pesos de `ruta`, en la GPU."""
    m = parseq(pretrained=False, vocab=VOCABS["spanish"])
    m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return m.cuda()


def main():
    """Línea de comandos de los dos modos (ver el módulo).

    - Cada lector carga fotos de hasta 4032 px y páginas de 600 DPI: se reserva ~3,5 GB por lector.
    - Se congelan las primeras etapas de la red base (ya saben detectar bordes y formas): menos VRAM. Solo layer4 se
      entrena, con una tasa 10 veces menor que las cabezas.
    - La BatchNorm de la red base queda fija: lotes tan chicos la descalibran.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["entrenar", "evaluar"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ventanas", type=int, default=16000, help="ventanas por época")
    ap.add_argument("--val-ventanas", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reco", default=str(C.MODELO_PARSEQ))
    ap.add_argument("--det", default=str(CANDIDATO))
    ap.add_argument("--vram-max", type=float, default=6.0,
                    help="tope de VRAM: en Windows lo reservado por CUDA también cuenta como memoria del sistema")
    args = ap.parse_args()
    prioridad_baja()
    def log(s):
        """Imprime al instante (el registro se sigue mientras corre)."""
        print(s, flush=True)

    try:
        with R.Candado("detector"):
            g = R.gpu()
            gb = round(min(args.vram_max, g["vram_libre_gb"] - C.MARGEN_VRAM_GB - 0.5), 1)
            R.verificar(vram_gb=gb, ram_gb=6.0, ventana_s=30, log=log)
            libre = R.memoria()["comprometida_libre_gb"]
            args.workers = max(1, min(args.workers, int((libre - C.MARGEN_COMPROMETIDA_GB - 12.0) // 3.5) - 1))
            log(f"VRAM presupuesto {gb} GB | memoria comprometida libre {libre:.1f} GB -> {args.workers} lectores")
            R.limitar_gpu(gb)
            torch.backends.cudnn.benchmark = True
            t0 = time.time()
            tr, te = paginas("training"), paginas("testing")
            log(f"Páginas: training {len(tr)} | testing {len(te)} ({time.time() - t0:.0f}s)")
            cfg = db_resnet50(pretrained=False).cfg
            val = DataLoader(Ventanas(te, args.val_ventanas, 99, cfg), batch_size=args.bs, num_workers=2,
                             collate_fn=_collate, worker_init_fn=_init)
            reco = reconocedor(args.reco)
            if args.modo == "evaluar":
                res = {}
                for nombre, ruta in (("actual", None), ("candidato", args.det)):
                    if ruta and not Path(ruta).exists():
                        continue
                    res[nombre] = evaluar(detector(ruta).cuda(), reco, val)
                    log(f"[{nombre}] {json.dumps({k: round(v, 4) for k, v in res[nombre].items()})}")
                (C.DIR_MODELOS / "eval_detector.json").write_text(json.dumps({"reco": args.reco, **res}, indent=1),
                                                                   encoding="utf-8")
                return
            modelo = detector().cuda()
            base = evaluar(modelo, reco, val)
            log(f"[detector actual] {json.dumps({k: round(v, 4) for k, v in base.items()})}")
            for nombre, p in modelo.feat_extractor.named_parameters():
                if not nombre.startswith("layer4"):
                    p.requires_grad = False
            columna = [p for n_, p in modelo.feat_extractor.named_parameters() if p.requires_grad]
            ids = {id(p) for p in modelo.feat_extractor.parameters()}
            resto = [p for p in modelo.parameters() if id(p) not in ids]
            opt = torch.optim.AdamW([{"params": columna, "lr": args.lr * 0.1}, {"params": resto, "lr": args.lr}],
                                    weight_decay=1e-4)
            pasos = args.epochs * (args.ventanas // args.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.1, args.lr], total_steps=pasos, pct_start=0.1)
            vigia = R.Vigia().iniciar()
            mejor, historial = base["f1"], []
            for ep in range(1, args.epochs + 1):
                datos = DataLoader(Ventanas(tr, args.ventanas, ep, cfg), batch_size=args.bs, shuffle=False,
                                   num_workers=args.workers, collate_fn=_collate, worker_init_fn=_init, drop_last=True)
                modelo.train()
                modelo.feat_extractor.eval()
                suma, n, te0 = 0.0, 0, time.time()
                for k, (x, obj, _, _) in enumerate(datos, 1):
                    if vigia.alerta:
                        log(f"DETENIDO por memoria del sistema: {vigia.alerta}")
                        return
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = modelo(x.cuda(non_blocking=True), obj)["loss"]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 5.0)
                    opt.step()
                    sched.step()
                    suma += loss.item()
                    n += 1
                    if k % 200 == 0:
                        log(f"  época {ep} paso {k}/{len(datos)} | pérdida {suma / n:.4f}"
                            f" | {(time.time() - te0) / 60:.1f} min"
                            f" | VRAM pico {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
                m = evaluar(modelo, reco, val)
                historial.append({"epoca": ep, "perdida": suma / max(1, n), **m})
                log(f"[época {ep}] {json.dumps({k: round(v, 4) for k, v in m.items()})}")
                if m["f1"] > mejor:
                    mejor = m["f1"]
                    torch.save(modelo.state_dict(), CANDIDATO)
                    log(f"  mejor F1 hasta ahora -> {CANDIDATO.name}")
            vigia.detener()
            (C.DIR_MODELOS / "dbnet_cl_candidato_metricas.json").write_text(
                json.dumps({"base": base, "historial": historial, "args": vars(args)}, indent=1), encoding="utf-8")
            log(f"Terminado. F1 base {base['f1']:.4f} -> mejor {mejor:.4f}")
    except R.RecursosInsuficientes as e:
        log(f"NO SE LANZÓ: {e}")
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
