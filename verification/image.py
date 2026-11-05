"""Imágenes de documentos (fotos y escaneos): formatos admitidos y alineación con la plantilla por puntos
característicos (ORB o SIFT) y homografía, con su control de calidad."""

import math

import cv2
import numpy as np

from . import config as C

IMAGE_FORMATS = ("JPEG", "PNG", "TIFF", "BMP", "WEBP")
"""Formatos que Pillow puede abrir: solo los de fotos y escaneos. Pillow reconoce el formato por el contenido, no
por la extensión, y la mayoría de sus fallas de seguridad están en decodificadores de formatos poco usados (FPX, EPS,
GIF, JPEG 2000, SPIDER...): con esta lista un archivo malicioso disfrazado de foto no llega a ellos."""


def orb():
    """Detector ORB con los parámetros usados para documentos (muchos puntos, umbral FAST bajo)."""
    return cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10)


def homography(kp_doc, des_doc, kp_tpl, des_tpl, norma):
    """Homografía documento -> plantilla con emparejamiento k-NN (prueba de Lowe 0,8) y RANSAC.

    Devuelve la matriz y su calidad (inliers, proporción, error medio, escala y rotación), solo el número de
    coincidencias si no alcanza, o None si no hay puntos suficientes."""
    if des_doc is None or des_tpl is None or len(kp_doc) < 10:
        return None
    pairs = cv2.BFMatcher(norma).knnMatch(des_doc, des_tpl, k=2)
    good_ones = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good_ones) < 10:
        return {"inliers": 0, "matches": len(good_ones)}
    src = np.float32([kp_doc[m.queryIdx].pt for m in good_ones])
    dst = np.float32([kp_tpl[m.trainIdx].pt for m in good_ones])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return {"inliers": 0, "matches": len(good_ones)}
    mask = mask.ravel().astype(bool)
    if mask.sum() < 4:
        return {"inliers": int(mask.sum()), "matches": len(good_ones)}
    proj = cv2.perspectiveTransform(src[mask][None], H)[0]
    rmse = float(np.sqrt(np.mean(np.sum((proj - dst[mask]) ** 2, axis=1))))
    return {"H": H, "inliers": int(mask.sum()), "matches": len(good_ones),
            "ratio": float(mask.sum() / len(good_ones)), "rmse_px": rmse,
            "escala": float(math.sqrt(abs(np.linalg.det(H[:2, :2])))),
            "rotacion": float(math.degrees(math.atan2(H[1, 0], H[0, 0])))}


def evaluate_alignment(a, photo=False):
    """Control de calidad: devuelve (ok, motivo). En fotos la hoja ocupa una parte variable de la
    imagen y puede estar más girada: la escala y la rotación admitidas son más amplias (la homografía
    las corrige; lo que importa es que muchos puntos encajen con poco error)."""
    span = C.PHOTO_SCALE_RANGE if photo else C.SCALE_RANGE
    max_rot = C.MAX_PHOTO_ROTATION if photo else C.MAX_ROTATION_DEGREES
    if not a or a.get("inliers", 0) < C.MIN_INLIERS:
        return False, f"pocos puntos coincidentes con la plantilla ({a.get('inliers', 0) if a else 0})"
    if a["ratio"] < C.MIN_RATIO_INLIERS:
        return False, f"proporción de coincidencias baja ({a['ratio']:.2f})"
    if a["rmse_px"] > C.MAX_RMSE_PX:
        return False, f"error de alineación alto ({a['rmse_px']:.1f} px)"
    if abs(a["rotacion"]) > max_rot:
        return False, f"rotación implausible ({a['rotacion']:.1f}°)"
    if not span[0] <= a["escala"] <= span[1]:
        return False, f"escala implausible ({a['escala']:.2f})"
    return True, ""
