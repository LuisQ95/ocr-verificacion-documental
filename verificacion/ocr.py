"""OCR en GPU con docTR: detección (DBNet) + reconocimiento (PARSeq afinado en español).

Entre la detección y el reconocimiento hay pasos propios para evitar errores conocidos:
  1. Franjas horizontales solapadas: el texto pequeño llega al detector a buen tamaño.
  2. Duplicados: una palabra detectada en dos franjas se conserva una sola vez.
  3. Cajas fusionadas: si el detector une dos líneas cercanas en una caja, se separan por el
     perfil horizontal de tinta y cada línea se reconoce por separado.
  4. Lectura de respaldo por zona: si una zona de la plantilla tiene tinta pero el detector no
     encontró texto ahí (p. ej. un "0" aislado), se recorta la tinta y se reconoce directamente.
"""

import math

import numpy as np
import torch
from doctr.datasets import VOCABS
from doctr.models import db_resnet50, detection_predictor, parseq, recognition_predictor

from . import config as C

ALTO_FRANJA = 704
SOLAPE = 100
UMBRAL_TINTA = 170
"""Gris por debajo de esto se considera tinta (incluye texto fino y gris)."""


DIGITOS = set("0123456789")
MAYUSCULAS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETRAS = MAYUSCULAS | set("ÁÉÍÓÚÜÑabcdefghijklmnopqrstuvwxyzáéíóúüñ")


def conjunto_zona(zona):
    """Caracteres que puede contener un campo según su tipo (None = sin restricción).
    Un valor falsificado sigue siendo válido para su campo (un DNI falso son 8 dígitos), así que
    restringir no oculta fraudes: solo evita confusiones de forma ('O' por '0', 'l' por 'I')."""
    tipo, nombre = zona.get("tipo"), zona.get("nombre", "")
    if tipo == "fecha":
        return DIGITOS | {"/"}
    if tipo == "numero":
        return DIGITOS | {".", ","}
    if tipo == "identidad":
        return DIGITOS if "dni" in nombre else LETRAS | set(",.'-")
    if tipo == "codigo":
        return DIGITOS | MAYUSCULAS | set("-/.")
    if tipo == "categorico" and zona.get("validos"):
        return set("".join(zona["validos"]).replace(" ", ""))
    return None


def zonas_para_ocr(plantilla, vocab=None):
    """Zonas de una plantilla con lo que el lector necesita: rectángulo, tipo y valores válidos."""
    vocab = vocab or {}
    return [{"rect": z["rect"], "tipo": z.get("tipo"), "nombre": nombre,
             "validos": vocab.get((plantilla["id"], z.get("columna") or nombre))}
            for nombre, z in plantilla["zonas"].items()]


def cargar_calibracion(ruta=C.CALIBRACION_CONF):
    """Curva confianza bruta -> probabilidad de acierto (ajustada con recortes de texto conocido)."""
    if not ruta.exists():
        return None
    import json
    d = json.loads(ruta.read_text(encoding="utf-8"))
    return np.array(d["x"], dtype=float), np.array(d["y"], dtype=float)


class _CabezaRestringida(torch.nn.Module):
    """Envuelve la capa de salida de PARSeq: suma una máscara (-inf en caracteres no permitidos) en
    CADA paso de la decodificación y del refinamiento, y guarda las salidas sin máscara para calcular
    la confianza con lo que el modelo realmente creía."""

    def __init__(self, cabeza):
        """Envuelve `cabeza` (la capa lineal de salida de PARSeq), sin máscara al inicio."""
        super().__init__()
        self.cabeza = cabeza
        self.mascara = None
        self.crudo = None

    def forward(self, x):
        """Salida de la cabeza con la máscara sumada (si hay); guarda la salida cruda en `crudo`."""
        out = self.cabeza(x)
        self.crudo = out
        return out if self.mascara is None else out + self.mascara.to(out.dtype)


def cargar_reconocedor(ruta_pesos=C.MODELO_PARSEQ):
    """PARSeq afinado si existe; si no, el preentrenado de docTR (vocabulario francés, sin Ñ/Á/Í/Ó/Ú)."""
    if ruta_pesos and ruta_pesos.exists():
        modelo = parseq(pretrained=False, vocab=VOCABS["spanish"])
        modelo.load_state_dict(torch.load(ruta_pesos, map_location="cpu", weights_only=True))
        return modelo, ruta_pesos.stem
    return parseq(pretrained=True), "parseq_preentrenado"


def _iou(a, b):
    """Intersección sobre unión de dos cajas (x0, y0, x1, y1, ...)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _unir_diacriticos(cajas):
    """El detector a veces separa la fila de tildes de una palabra ("Ó" de "TECNOLÓGICA") en una caja
    muy delgada, que se lee como basura y deja la palabra sin tilde. Esas franjas se reincorporan a
    la palabra que tienen justo debajo."""
    cajas = list(cajas)
    resto = []
    for c in sorted(cajas, key=lambda c: c[3] - c[1]):
        if c not in cajas:
            continue
        alto = c[3] - c[1]
        debajo = [u for u in cajas if u is not c and (u[3] - u[1]) > 2 * alto and -2 <= u[1] - c[3] <= 4
                  and min(c[2], u[2]) - max(c[0], u[0]) >= 0.5 * (c[2] - c[0])]
        if debajo:
            u = max(debajo, key=lambda u: min(c[2], u[2]) - max(c[0], u[0]))
            i = cajas.index(u)
            cajas[i] = (min(u[0], c[0]), min(u[1], c[1]), max(u[2], c[2]), u[3], u[4])
            cajas.remove(c)
    return cajas


def _segmentos(img_recorte, max_aspecto=5.0):
    """Divide un recorte muy alargado en tramos por columnas sin tinta entre caracteres.
    PARSeq lee a 32x128 px: un código largo comprimido a ese ancho deja ~8 px por carácter y el
    modelo puede 'entrar en bucle' y repetir dígitos ("REG-339390-199390-1993")."""
    h, w = img_recorte.shape[:2]
    if h == 0 or w / h <= max_aspecto:
        return [img_recorte]
    vacias = (img_recorte < UMBRAL_TINTA).sum(axis=0) == 0
    partes = math.ceil(w / h / (max_aspecto - 1))
    cortes = []
    for k in range(1, partes):
        ideal = k * w // partes
        cands = [x for x in range(max(1, ideal - h), min(w - 1, ideal + h)) if vacias[x]]
        if cands:
            cortes.append(min(cands, key=lambda x: abs(x - ideal)))
    if not cortes:
        return [img_recorte]
    bordes = [0] + sorted(set(cortes)) + [w]
    return [img_recorte[:, a:b] for a, b in zip(bordes, bordes[1:]) if b - a > 2]


def _separar_lineas(img, caja):
    """Divide una caja en varias si contiene más de una línea de texto (filas sin tinta entre medio).

    Se descartan motas de menos de 4 px. Una franja delgada pegada a la de abajo es la fila de tildes de una
    mayúscula ("OYARZÚN"), no otra línea: se une a ella, porque leída sola el reconocedor inventa texto.
    """
    x0, y0, x1, y1 = caja
    filas = (img[y0:y1, x0:x1] < UMBRAL_TINTA).sum(axis=1) > 0
    tramos, inicio = [], None
    for i, hay in enumerate(filas):
        if hay and inicio is None:
            inicio = i
        elif not hay and inicio is not None:
            tramos.append((inicio, i))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, len(filas)))
    tramos = [t for t in tramos if t[1] - t[0] >= 4]
    if len(tramos) < 2:
        return [caja]
    alto_max = max(b - a for a, b in tramos)
    unidos = [tramos[0]]
    for a, b in tramos[1:]:
        pa, pb = unidos[-1]
        if a - pb <= 4 and min(pb - pa, b - a) < 0.45 * alto_max:
            unidos[-1] = (pa, b)
        else:
            unidos.append((a, b))
    tramos = unidos
    if len(tramos) < 2:
        return [caja]
    return [(x0, max(0, y0 + a - 1), x1, min(img.shape[0], y0 + b + 1)) for a, b in tramos]


class LectorOCR:
    """Detector DBNet + reconocedor PARSeq, con los pasos propios descritos en el módulo."""

    def __init__(self, ruta_pesos=C.MODELO_PARSEQ, dispositivo=None, ruta_detector=C.MODELO_DBNET,
                 restringir=None, calibrar=None):
        """Carga los modelos en `dispositivo` (GPU si hay).

        `restringir` y `calibrar` en None toman los valores de config. La detección usa lotes de 2 porque cada
        franja reserva bloques de ~512 MB: así el pico de VRAM queda dentro de C.PRESUPUESTO_VRAM_OCR_GB. Los
        recortes más alargados que `max_aspecto` se parten en tramos (el modelo antiguo se trababa con palabras
        largas; el entrenado con documentos chilenos lee palabras completas y conviene no partirlas).
        """
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        reco, self.nombre_modelo = cargar_reconocedor(ruta_pesos)
        self.restringir = C.OCR_RESTRINGIR if restringir is None else restringir
        self.calibracion = cargar_calibracion() if (C.OCR_CALIBRAR if calibrar is None else calibrar) else None
        if self.restringir:
            self.nombre_modelo += " + restricción por campo"
        if self.calibracion is not None:
            self.nombre_modelo += " + confianza calibrada"
        reco.head = _CabezaRestringida(reco.head)
        self.modelo_reco = reco
        self.vocab = reco.vocab
        detector = db_resnet50(pretrained=True)
        if ruta_detector and ruta_detector.exists():
            detector.load_state_dict(torch.load(ruta_detector, map_location="cpu", weights_only=True))
            self.nombre_modelo += f" + {ruta_detector.stem}"
        self.det = detection_predictor(arch=detector, assume_straight_pages=True,
                                       preserve_aspect_ratio=True, symmetric_pad=True,
                                       batch_size=2).to(self.dispositivo).eval()
        self.reco = recognition_predictor(arch=reco, batch_size=256).to(self.dispositivo).eval()
        self.max_aspecto = 5.0

    @staticmethod
    def _franjas(alto):
        """Franjas solapadas; cada caja pertenece al 'núcleo' de una sola franja."""
        n = max(1, math.ceil((alto - SOLAPE) / (ALTO_FRANJA - SOLAPE)))
        paso = (alto - ALTO_FRANJA) / (n - 1) if n > 1 else 0
        inicios = [round(i * paso) for i in range(n)]
        finales = [min(alto, y0 + ALTO_FRANJA) for y0 in inicios]
        cortes = [0] + [(inicios[i + 1] + finales[i]) / 2 for i in range(len(inicios) - 1)] + [alto]
        return [(y0, y1, cortes[i], cortes[i + 1]) for i, (y0, y1) in enumerate(zip(inicios, finales))]

    def _detectar(self, imagenes):
        """Cajas de texto (en px) de cada página: detección por franjas, sin duplicados entre franjas, con las
        tildes reincorporadas y las cajas de varias líneas separadas."""
        trozos, indice = [], []
        for n, img in enumerate(imagenes):
            for y0, y1, c0, c1 in self._franjas(img.shape[0]):
                trozos.append(np.repeat(img[y0:y1, :, None], 3, axis=2))
                indice.append((n, y0, c0, c1))
        with torch.autocast(self.dispositivo, dtype=torch.float16, enabled=self.dispositivo == "cuda"):
            preds = self.det(trozos)

        cajas = [[] for _ in imagenes]
        for (n, y0, c0, c1), pred, trozo in zip(indice, preds, trozos):
            h, w = trozo.shape[:2]
            arr = next(iter(pred.values())) if isinstance(pred, dict) else pred
            for x_0, y_0, x_1, y_1, score in np.asarray(arr):
                py0, py1 = y0 + y_0 * h, y0 + y_1 * h
                if c0 <= (py0 + py1) / 2 < c1:
                    cajas[n].append((x_0 * w, py0, x_1 * w, py1, float(score)))

        salida = []
        for img, cs in zip(imagenes, cajas):
            unicas = []
            for c in sorted(cs, key=lambda c: -c[4]):
                if all(_iou(c, u) < 0.3 for u in unicas):
                    unicas.append(c)
            unicas = _unir_diacriticos(unicas)
            finales = []
            for x0, y0, x1, y1, _ in unicas:
                caja = (max(0, int(x0) - 1), max(0, int(y0) - 1),
                        min(img.shape[1], int(math.ceil(x1)) + 1), min(img.shape[0], int(math.ceil(y1)) + 1))
                if caja[2] - caja[0] > 2 and caja[3] - caja[1] > 2:
                    finales += _separar_lineas(img, caja)
            salida.append(finales)
        return salida

    @torch.inference_mode()
    def reconocer(self, recortes, conjuntos=None):
        """Lee recortes (RGB uint8). Con restricción, cada recorte solo puede producir caracteres de su
        conjunto; la confianza es la media de la probabilidad SIN máscara de los caracteres elegidos."""
        if not recortes:
            return []
        conjuntos = conjuntos or [None] * len(recortes)
        if not self.restringir:
            with torch.autocast(self.dispositivo, dtype=torch.float16, enabled=self.dispositivo == "cuda"):
                lecturas = self.reco(recortes)
            return [(t, self._calibrar(c)) for t, c in lecturas]

        cabeza, eos = self.modelo_reco.head, len(self.vocab)
        idx = {ch: i for i, ch in enumerate(self.vocab)}
        salida, i = [], 0
        for lote in self.reco.pre_processor(recortes):
            n = lote.shape[0]
            mascara = torch.zeros((n, 1, eos + 1))
            for k in range(n):
                cj = conjuntos[i + k]
                if cj:
                    mascara[k, 0, :] = -1e4
                    mascara[k, 0, [idx[c] for c in cj if c in idx] + [eos]] = 0
            cabeza.mascara = mascara.to(self.dispositivo)
            try:
                with torch.autocast(self.dispositivo, dtype=torch.float16, enabled=self.dispositivo == "cuda"):
                    out = self.modelo_reco(lote.to(self.dispositivo), return_model_output=True)
            finally:
                cabeza.mascara = None
            elegidos = out["out_map"].float().argmax(-1)
            prob_cruda = torch.softmax(cabeza.crudo.float(), -1).gather(-1, elegidos.unsqueeze(-1)).squeeze(-1)
            for k in range(n):
                fila = elegidos[k].tolist()
                largo = fila.index(eos) if eos in fila else len(fila)
                texto = "".join(self.vocab[j] for j in fila[:largo] if j < eos)
                conf = prob_cruda[k, :largo].clip(0, 1).mean().item() if largo else 0.0
                salida.append((texto, self._calibrar(conf)))
            i += n
        return salida

    def _calibrar(self, conf):
        """Confianza bruta -> probabilidad de acierto según la curva de calibración (si está activa)."""
        if self.calibracion is None:
            return float(conf)
        return float(np.interp(conf, *self.calibracion))

    @staticmethod
    def _zona_de(caja, zonas):
        """Zona de la plantilla que contiene el centro de la caja (None si ninguna)."""
        cx, cy = (caja[0] + caja[2]) / 2 / C.ESCALA, (caja[1] + caja[3]) / 2 / C.ESCALA
        return next((z for z in zonas if isinstance(z, dict)
                     and z["rect"][0] <= cx <= z["rect"][2] and z["rect"][1] <= cy <= z["rect"][3]), None)

    @staticmethod
    def _respaldo_zonas(img, cajas, zonas):
        """Zonas con tinta donde no se detectó ninguna caja: se recorta la tinta para leerla directo.
        La tinta que ya pertenece a una caja detectada (p. ej. el final de una palabra larga que
        invade la columna vecina) se ignora."""
        extra = []
        centros = [((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) for c in cajas]
        img = img.copy()
        for x0, y0, x1, y1 in cajas:
            img[y0:y1, x0:x1] = 255
        for zona in zonas:
            rect = zona["rect"] if isinstance(zona, dict) else zona
            x0, y0, x1, y1 = (int(v * C.ESCALA) for v in rect)
            x0, y0, x1, y1 = max(0, x0 + 1), max(0, y0 + 1), min(img.shape[1], x1 - 1), min(img.shape[0], y1 - 1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in centros):
                continue
            ys, xs = np.where(img[y0:y1, x0:x1] < UMBRAL_TINTA)
            if len(xs) < 12:
                continue
            extra.append((max(0, x0 + xs.min() - 3), max(0, y0 + ys.min() - 3),
                          min(img.shape[1], x0 + xs.max() + 4), min(img.shape[0], y0 + ys.max() + 4)))
        return extra

    @torch.inference_mode()
    def leer(self, imagenes_gris, zonas_por_pagina=None):
        """Recibe páginas alineadas (gris, coordenadas de plantilla) y, opcionalmente, los rectángulos
        de las zonas de cada página (en puntos). Devuelve por página una lista de palabras
        {texto, conf, bbox, respaldo} con bbox en puntos PDF de la plantilla.

        Un recorte dividido en tramos se reconstruye concatenando sus lecturas (con la confianza mínima). Las
        lecturas con mayoría de guiones o puntuación se descartan: son restos de marca de agua, líneas o manchas.
        """
        cajas = self._detectar(imagenes_gris)
        recortes, indice, conjuntos = [], [], []
        for n, (img, cs) in enumerate(zip(imagenes_gris, cajas)):
            zonas = zonas_por_pagina[n] if zonas_por_pagina else []
            extra = self._respaldo_zonas(img, cs, zonas) if zonas else []
            for caja, respaldo in [(c, False) for c in cs] + [(c, True) for c in extra]:
                x0, y0, x1, y1 = caja
                zona = self._zona_de(caja, zonas) if self.restringir and zonas else None
                segs = _segmentos(img[y0:y1, x0:x1], self.max_aspecto)
                recortes += [np.repeat(s[:, :, None], 3, axis=2) for s in segs]
                conjuntos += [conjunto_zona(zona) if zona else None] * len(segs)
                indice.append((n, caja, respaldo, len(segs)))

        salida = [[] for _ in imagenes_gris]
        if not recortes:
            return salida
        lecturas_seg = self.reconocer(recortes, conjuntos)
        lecturas, i = [], 0
        for *_, k in indice:
            partes = lecturas_seg[i:i + k]
            lecturas.append(("".join(t for t, _ in partes), min(c for _, c in partes)))
            i += k
        for (n, (x0, y0, x1, y1), respaldo, _), (texto, conf) in zip(indice, lecturas):
            if texto and sum(ch.isalnum() for ch in texto) >= len(texto) / 2:
                salida[n].append({"texto": texto, "conf": float(conf), "respaldo": respaldo,
                                  "bbox": [x0 / C.ESCALA, y0 / C.ESCALA, x1 / C.ESCALA, y1 / C.ESCALA]})
        return salida
