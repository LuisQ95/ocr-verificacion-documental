"""OCR en GPU con docTR: detección (DBNet) + reconocimiento (PARSeq afinado en español).

Entre la detección y el reconocimiento hay pasos propios para evitar errores conocidos:
  1. Franjas horizontales solapadas: el texto pequeño llega al detector a buen tamaño.
  2. Duplicados: una palabra detectada en dos franjas se conserva una sola vez.
  3. Cajas fusionadas: si el detector une dos líneas cercanas en una caja, se separan por el
     perfil horizontal de tinta y cada línea se reconoce por separado.
  4. Lectura de respaldo por zona: si una zona de la plantilla tiene tinta pero el detector no
     encontró texto ahí (p. ej. un "0" aislado), se recorta la tinta y se reconoce directamente.
  5. Caracteres sueltos: en las zonas de texto libre (secciones de filas variables de la plantilla), la tinta del tamaño
     de 1 a 3 caracteres que el detector dejó fuera de toda caja (la clase "D" de una licencia, el día "9" de una
     fecha) se recorta y se reconoce; solo se conserva una lectura con confianza suficiente.
"""

import math

import cv2
import numpy as np
import torch
from doctr.datasets import VOCABS
from doctr.models import db_resnet50, detection_predictor, parseq, recognition_predictor

from . import config as C
from .boxes import iou

STRIP_HEIGHT = 704
OVERLAP = 100
INK_THRESHOLD = 170
"""Gris por debajo de esto se considera tinta (incluye texto fino y gris)."""
MIN_CONF_LOOSE = 0.5
"""Confianza mínima (calibrada) para conservar la lectura de un carácter suelto: más abajo suele ser una mancha."""


DIGITS = set("0123456789")
UPPERCASE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTERS = UPPERCASE | set("ÁÉÍÓÚÜÑabcdefghijklmnopqrstuvwxyzáéíóúüñ")


def zone_set(zone):
    """Caracteres que puede contener un campo según su tipo (None = sin restricción).
    Un valor falsificado sigue siendo válido para su campo (un DNI falso son 8 dígitos), así que
    restringir no oculta fraudes: solo evita confusiones de forma ('O' por '0', 'l' por 'I')."""
    kind, name = zone.get("tipo"), zone.get("nombre", "")
    if kind == "fecha":
        return DIGITS | {"/"}
    if kind == "numero":
        return DIGITS | {".", ","}
    if kind == "identidad":
        return DIGITS if "dni" in name else LETTERS | set(",.'-")
    if kind == "codigo":
        return DIGITS | UPPERCASE | set("-/.")
    if kind == "categorico" and zone.get("validos"):
        return set("".join(zone["validos"]).replace(" ", ""))
    return None


def load_calibration(path=C.CONFIDENCE_CALIBRATION):
    """Curva confianza bruta -> probabilidad de acierto (ajustada con recortes de texto conocido)."""
    if not path.exists():
        return None
    import json
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array(d["x"], dtype=float), np.array(d["y"], dtype=float)


class _RestrictedHead(torch.nn.Module):
    """Envuelve la capa de salida de PARSeq: suma una máscara (-inf en caracteres no permitidos) en
    CADA paso de la decodificación y del refinamiento, y guarda las salidas sin máscara para calcular
    la confianza con lo que el modelo realmente creía."""

    def __init__(self, head):
        """Envuelve `cabeza` (la capa lineal de salida de PARSeq), sin máscara al inicio."""
        super().__init__()
        self.head = head
        self.region_mask = None
        self.raw = None

    def forward(self, x):
        """Salida de la cabeza con la máscara sumada (si hay); guarda la salida cruda en `crudo`."""
        out = self.head(x)
        self.raw = out
        return out if self.region_mask is None else out + self.region_mask.to(out.dtype)


def load_recognizer(weights_path=C.PARSEQ_MODEL):
    """PARSeq afinado si existe; si no, el preentrenado de docTR (vocabulario francés, sin Ñ/Á/Í/Ó/Ú)."""
    if weights_path and weights_path.exists():
        model = parseq(pretrained=False, vocab=VOCABS["spanish"])
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        return model, weights_path.stem
    return parseq(pretrained=True), "parseq_preentrenado"


def _join_diacritics(boxes):
    """El detector a veces separa la fila de tildes de una palabra ("Ó" de "TECNOLÓGICA") en una caja
    muy delgada, que se lee como basura y deja la palabra sin tilde. Esas franjas se reincorporan a
    la palabra que tienen justo debajo."""
    boxes = list(boxes)
    for c in sorted(boxes, key=lambda c: c[3] - c[1]):
        if c not in boxes:
            continue
        height = c[3] - c[1]
        below = [u for u in boxes if u is not c and (u[3] - u[1]) > 2 * height and -2 <= u[1] - c[3] <= 4
                  and min(c[2], u[2]) - max(c[0], u[0]) >= 0.5 * (c[2] - c[0])]
        if below:
            u = max(below, key=lambda u: min(c[2], u[2]) - max(c[0], u[0]))
            i = boxes.index(u)
            boxes[i] = (min(u[0], c[0]), min(u[1], c[1]), max(u[2], c[2]), u[3], u[4])
            boxes.remove(c)
    return boxes


def _segments(crop_img, max_aspect=5.0):
    """Divide un recorte muy alargado en tramos por columnas sin tinta entre caracteres.
    PARSeq lee a 32x128 px: un código largo comprimido a ese ancho deja ~8 px por carácter y el
    modelo puede 'entrar en bucle' y repetir dígitos ("REG-339390-199390-1993")."""
    h, w = crop_img.shape[:2]
    if h == 0 or w / h <= max_aspect:
        return [crop_img]
    empty_cols = (crop_img < INK_THRESHOLD).sum(axis=0) == 0
    parts = math.ceil(w / h / (max_aspect - 1))
    cuts = []
    for k in range(1, parts):
        ideal = k * w // parts
        cands = [x for x in range(max(1, ideal - h), min(w - 1, ideal + h)) if empty_cols[x]]
        if cands:
            cuts.append(min(cands, key=lambda x: abs(x - ideal)))
    if not cuts:
        return [crop_img]
    edges = [0] + sorted(set(cuts)) + [w]
    return [crop_img[:, a:b] for a, b in zip(edges, edges[1:]) if b - a > 2]


def _split_lines(img, box):
    """Divide una caja en varias si contiene más de una línea de texto (filas sin tinta entre medio).

    Se descartan motas de menos de 4 px. Una franja delgada pegada a la de abajo es la fila de tildes de una
    mayúscula ("OYARZÚN"), no otra línea: se une a ella, porque leída sola el reconocedor inventa texto.
    """
    x0, y0, x1, y1 = box
    rows = (img[y0:y1, x0:x1] < INK_THRESHOLD).sum(axis=1) > 0
    spans, home = [], None
    for i, present in enumerate(rows):
        if present and home is None:
            home = i
        elif not present and home is not None:
            spans.append((home, i))
            home = None
    if home is not None:
        spans.append((home, len(rows)))
    spans = [t for t in spans if t[1] - t[0] >= 4]
    if len(spans) < 2:
        return [box]
    max_height = max(b - a for a, b in spans)
    merged = [spans[0]]
    for a, b in spans[1:]:
        pa, pb = merged[-1]
        if a - pb <= 4 and min(pb - pa, b - a) < 0.45 * max_height:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    spans = merged
    if len(spans) < 2:
        return [box]
    return [(x0, max(0, y0 + a - 1), x1, min(img.shape[0], y0 + b + 1)) for a, b in spans]


class OCRReader:
    """Detector DBNet + reconocedor PARSeq, con los pasos propios descritos en el módulo."""

    def __init__(self, weights_path=C.PARSEQ_MODEL, device=None, detector_path=C.DBNET_MODEL,
                 restrict=None, calibrate=None):
        """Carga los modelos en `dispositivo` (GPU si hay).

        `restringir` y `calibrar` en None toman los valores de config. La detección usa lotes de 2 porque cada
        franja reserva bloques de ~512 MB: así el pico de VRAM queda dentro de C.OCR_VRAM_BUDGET_GB. Los
        recortes más alargados que `max_aspect` se parten en tramos (el modelo antiguo se trababa con palabras
        largas; el entrenado con los documentos del perfil lee palabras completas y conviene no partirlas).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        reco, self.model_name = load_recognizer(weights_path)
        self.restrict = C.OCR_RESTRICT if restrict is None else restrict
        self.calibration = load_calibration() if (C.OCR_CALIBRATE if calibrate is None else calibrate) else None
        if self.restrict:
            self.model_name += " + restricción por campo"
        if self.calibration is not None:
            self.model_name += " + confianza calibrada"
        reco.head = _RestrictedHead(reco.head)
        self.reco_model = reco
        self.vocab = reco.vocab
        finetuned = bool(detector_path and detector_path.exists())
        detector = db_resnet50(pretrained=not finetuned)
        if finetuned:
            detector.load_state_dict(torch.load(detector_path, map_location="cpu", weights_only=True))
            self.model_name += f" + {detector_path.stem}"
        self.det = detection_predictor(arch=detector, assume_straight_pages=True,
                                       preserve_aspect_ratio=True, symmetric_pad=True,
                                       batch_size=2).to(self.device).eval()
        self.reco = recognition_predictor(arch=reco, batch_size=256).to(self.device).eval()
        self.max_aspect = 5.0

    @staticmethod
    def _strips(height):
        """Franjas solapadas; cada caja pertenece al 'núcleo' de una sola franja."""
        n = max(1, math.ceil((height - OVERLAP) / (STRIP_HEIGHT - OVERLAP)))
        step = (height - STRIP_HEIGHT) / (n - 1) if n > 1 else 0
        starts = [round(i * step) for i in range(n)]
        final = [min(height, y0 + STRIP_HEIGHT) for y0 in starts]
        cuts = [0] + [(starts[i + 1] + final[i]) / 2 for i in range(len(starts) - 1)] + [height]
        return [(y0, y1, cuts[i], cuts[i + 1]) for i, (y0, y1) in enumerate(zip(starts, final))]

    def _detect(self, images):
        """Cajas de texto (en px) de cada página: detección por franjas, sin duplicados entre franjas, con las
        tildes reincorporadas y las cajas de varias líneas separadas."""
        chunks, index = [], []
        for n, img in enumerate(images):
            for y0, y1, c0, c1 in self._strips(img.shape[0]):
                chunks.append(np.repeat(img[y0:y1, :, None], 3, axis=2))
                index.append((n, y0, c0, c1))
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            preds = self.det(chunks)

        boxes = [[] for _ in images]
        for (n, y0, c0, c1), pred, chunk in zip(index, preds, chunks):
            h, w = chunk.shape[:2]
            arr = next(iter(pred.values())) if isinstance(pred, dict) else pred
            for x_0, y_0, x_1, y_1, score in np.asarray(arr):
                py0, py1 = y0 + y_0 * h, y0 + y_1 * h
                if c0 <= (py0 + py1) / 2 < c1:
                    boxes[n].append((x_0 * w, py0, x_1 * w, py1, float(score)))

        output = []
        for img, cs in zip(images, boxes):
            unique = []
            for c in sorted(cs, key=lambda c: -c[4]):
                if all(iou(c, u) < 0.3 for u in unique):
                    unique.append(c)
            unique = _join_diacritics(unique)
            final = []
            for x0, y0, x1, y1, _ in unique:
                box = (max(0, int(x0) - 1), max(0, int(y0) - 1),
                        min(img.shape[1], int(math.ceil(x1)) + 1), min(img.shape[0], int(math.ceil(y1)) + 1))
                if box[2] - box[0] > 2 and box[3] - box[1] > 2:
                    final += _split_lines(img, box)
            output.append(final)
        return output

    @torch.inference_mode()
    def recognize(self, crops, sets=None):
        """Lee recortes (RGB uint8). Con restricción, cada recorte solo puede producir caracteres de su
        conjunto; la confianza es la media de la probabilidad SIN máscara de los caracteres elegidos."""
        if not crops:
            return []
        sets = sets or [None] * len(crops)
        if not self.restrict:
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                readings = self.reco(crops)
            return [(t, self._calibrate(c)) for t, c in readings]

        head, eos = self.reco_model.head, len(self.vocab)
        idx = {ch: i for i, ch in enumerate(self.vocab)}
        output, i = [], 0
        for batch in self.reco.pre_processor(crops):
            n = batch.shape[0]
            region_mask = torch.zeros((n, 1, eos + 1))
            for k in range(n):
                cj = sets[i + k]
                if cj:
                    region_mask[k, 0, :] = -1e4
                    region_mask[k, 0, [idx[c] for c in cj if c in idx] + [eos]] = 0
            head.region_mask = region_mask.to(self.device)
            try:
                with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    out = self.reco_model(batch.to(self.device), return_model_output=True)
            finally:
                head.region_mask = None
            chosen = out["out_map"].float().argmax(-1)
            raw_prob = torch.softmax(head.raw.float(), -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            for k in range(n):
                row = chosen[k].tolist()
                length = row.index(eos) if eos in row else len(row)
                text = "".join(self.vocab[j] for j in row[:length] if j < eos)
                conf = raw_prob[k, :length].clip(0, 1).mean().item() if length else 0.0
                output.append((text, self._calibrate(conf)))
            i += n
        return output

    def _calibrate(self, conf):
        """Confianza bruta -> probabilidad de acierto según la curva de calibración (si está activa)."""
        if self.calibration is None:
            return float(conf)
        return float(np.interp(conf, *self.calibration))

    @staticmethod
    def _zone_of(box, zones):
        """Zona de la plantilla que contiene el centro de la caja (None si ninguna)."""
        cx, cy = (box[0] + box[2]) / 2 / C.SCALE, (box[1] + box[3]) / 2 / C.SCALE
        return next((z for z in zones if isinstance(z, dict)
                     and z["rect"][0] <= cx <= z["rect"][2] and z["rect"][1] <= cy <= z["rect"][3]), None)

    @staticmethod
    def _fallback_zones(img, boxes, zones):
        """Zonas con tinta donde no se detectó ninguna caja: se recorta la tinta para leerla directo.
        La tinta que ya pertenece a una caja detectada (p. ej. el final de una palabra larga que
        invade la columna vecina) se ignora."""
        extra = []
        centers = [((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) for c in boxes]
        img = img.copy()
        for x0, y0, x1, y1 in boxes:
            img[y0:y1, x0:x1] = 255
        for zone in zones:
            rect = zone["rect"] if isinstance(zone, dict) else zone
            x0, y0, x1, y1 = (int(v * C.SCALE) for v in rect)
            x0, y0, x1, y1 = max(0, x0 + 1), max(0, y0 + 1), min(img.shape[1], x1 - 1), min(img.shape[0], y1 - 1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in centers):
                continue
            ys, xs = np.where(img[y0:y1, x0:x1] < INK_THRESHOLD)
            if len(xs) < 12:
                continue
            extra.append((max(0, x0 + xs.min() - 3), max(0, y0 + ys.min() - 3),
                          min(img.shape[1], x0 + xs.max() + 4), min(img.shape[0], y0 + ys.max() + 4)))
        return extra

    @staticmethod
    def _loose(img, boxes, rect):
        """Cajas de la tinta del tamaño de 1 a 3 caracteres que quedó fuera de toda caja detectada dentro de `rect`
        (puntos). La altura de referencia es la mediana de las cajas detectadas en el rectángulo: una mancha mucho
        más baja o más alta, o más ancha que 3 caracteres (una línea, una firma), no es un carácter."""
        x0, y0, x1, y1 = (int(v * C.SCALE) for v in rect)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
        inside = [c for c in boxes if x0 <= (c[0] + c[2]) / 2 <= x1 and y0 <= (c[1] + c[3]) / 2 <= y1]
        if len(inside) < 3 or x1 - x0 < 10 or y1 - y0 < 10:
            return []
        height = float(np.median([c[3] - c[1] for c in inside]))
        ink = (img[y0:y1, x0:x1] < INK_THRESHOLD).astype(np.uint8)
        for cx0, cy0, cx1, cy1 in boxes:
            ink[max(0, cy0 - y0 - 2):max(0, cy1 - y0 + 2), max(0, cx0 - x0 - 2):max(0, cx1 - x0 + 2)] = 0
        joined_line = cv2.dilate(ink, np.ones((1, max(3, round(height * 0.35))), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(joined_line, connectivity=8)
        output = []
        for i in range(1, n):
            sx, sy, sw, sh, _ = stats[i]
            if not 0.55 * height <= sh <= 1.4 * height or not 0.2 * height <= sw <= 3 * height:
                continue
            if ink[sy:sy + sh, sx:sx + sw].sum() < 12:
                continue
            output.append((max(0, x0 + sx - 3), max(0, y0 + sy - 3),
                           min(img.shape[1], x0 + sx + sw + 3), min(img.shape[0], y0 + sy + sh + 3)))
        return output

    @torch.inference_mode()
    def read(self, gray_images, zones_by_page=None, free_by_page=None):
        """Recibe páginas alineadas (gris, coordenadas de plantilla) y, opcionalmente, los rectángulos
        de las zonas de cada página (en puntos) y los de sus zonas de texto libre (donde se buscan caracteres
        sueltos). Devuelve por página una lista de palabras {texto, conf, bbox, respaldo} con bbox en puntos PDF
        de la plantilla.

        Un recorte dividido en tramos se reconstruye concatenando sus lecturas (con la confianza mínima). Las
        lecturas con mayoría de guiones o puntuación se descartan: son restos de marca de agua, líneas o manchas.
        """
        boxes = self._detect(gray_images)
        crops, index, sets = [], [], []
        for n, (img, cs) in enumerate(zip(gray_images, boxes)):
            zones = zones_by_page[n] if zones_by_page else []
            extra = self._fallback_zones(img, cs, zones) if zones else []
            loose = [s for r in (free_by_page[n] if free_by_page else []) for s in self._loose(img, cs, r)]
            for box, fallback in [(c, False) for c in cs] + [(c, True) for c in extra] + [(c, "suelto") for c in loose]:
                x0, y0, x1, y1 = box
                zone = self._zone_of(box, zones) if self.restrict and zones else None
                segs = _segments(img[y0:y1, x0:x1], self.max_aspect)
                crops += [np.repeat(s[:, :, None], 3, axis=2) for s in segs]
                sets += [zone_set(zone) if zone else None] * len(segs)
                index.append((n, box, fallback, len(segs)))

        output = [[] for _ in gray_images]
        if not crops:
            return output
        readings_s = self.recognize(crops, sets)
        readings, i = [], 0
        for *_, k in index:
            parts = readings_s[i:i + k]
            readings.append(("".join(t for t, _ in parts), min(c for _, c in parts)))
            i += k
        for (n, (x0, y0, x1, y1), fallback, _), (text, conf) in zip(index, readings):
            if fallback == "suelto" and (conf < MIN_CONF_LOOSE or not text.isalnum()):
                continue
            if text and sum(ch.isalnum() for ch in text) >= len(text) / 2:
                output[n].append({"texto": text, "conf": float(conf), "respaldo": fallback,
                                  "bbox": [x0 / C.SCALE, y0 / C.SCALE, x1 / C.SCALE, y1 / C.SCALE]})
        return output
