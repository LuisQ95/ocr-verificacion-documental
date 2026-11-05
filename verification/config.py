"""Rutas y umbrales del pipeline. Los umbrales están aquí para poder ajustarlos sin tocar la lógica.

Cada constante lleva su explicación como docstring de atributo (la cadena que la sigue), que muestran los editores y
las herramientas de documentación.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "official_file"
"""Plantillas en blanco de cada documento."""
APPLICANTS_DIR = ROOT / "postulant_files"
OFFICIAL_DATA_DIR = ROOT / "webscrap_files"
"""Datos de la fuente oficial (JSON), una carpeta por postulante."""
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

PROFILE = os.environ.get("VERIFICATION_PROFILE", "operator_certification")
"""Perfil de documentos (verification/profiles/<perfil>): plantillas, normalización, reglas y requisitos propios de
un conjunto de documentos. Se elige con la variable de entorno VERIFICATION_PROFILE."""

PARSEQ_MODEL = MODELS_DIR / "parseq_v4.pt"
"""Reconocedor adoptado el 2026-10-28 (ver training/RECOGNIZER_CARD.md): parte del PARSeq genérico de docTR y
se entrenó con un vocabulario amplio de nombres (20 % reservado para la prueba), texto aleatorio y los datos anteriores
redistribuidos, para que lea en vez de memorizar. Verificador completo en 600 postulantes: genuinos acusados 8 -> 2
con la misma detección (455 vs 456 de 520). El anterior se borró el 2026-10-31."""
DBNET_MODEL = MODELS_DIR / "dbnet_es_finetuned.pt"
"""Detector afinado; solo se usa si existe (el afinado no mejoró y no se adoptó)."""
CONFIDENCE_CALIBRATION = MODELS_DIR / "confidence_calibration_v4.json"
"""Curva que convierte la confianza del reconocedor en probabilidad de acierto (ajustada con training/testing y
testing_v2, que incluye nombres nunca vistos)."""
CANDIDATE_VARIABLES = ("OCR_PARSEQ_MODEL", "OCR_CALIBRATION")
"""OCR_PARSEQ_MODEL / OCR_CALIBRATION (variables de entorno, rutas relativas a la raíz) permiten medir un candidato
con el pipeline completo antes de adoptarlo, sin cambiar lo adoptado."""
if os.environ.get("OCR_PARSEQ_MODEL"):
    PARSEQ_MODEL = ROOT / os.environ["OCR_PARSEQ_MODEL"]
if os.environ.get("OCR_CALIBRATION"):
    CONFIDENCE_CALIBRATION = ROOT / os.environ["OCR_CALIBRATION"]
APPLICANT_REGISTRY = APPLICANTS_DIR / "candidatos.xlsx"
"""Identidad declarada (número de identidad) y cargo de cada carpeta de postulante."""

DPI = 150
"""Resolución de trabajo para alineación y OCR."""
SCALE = DPI / 72
"""Píxeles por punto PDF."""

MIN_NATIVE_WORDS = 15
"""Palabras visibles que debe tener una página PDF para leerla como texto (nativo); con menos, se lee como imagen."""
MIN_NATIVE_SCORE = 0.45
"""Fracción de palabras fijas de una plantilla que debe tener un PDF nativo para reconocerse como ese documento."""
MAX_NATIVE_RESIDUAL_PT = 2.0
"""Error mediano (pt) de las etiquetas de un PDF nativo ubicado sobre su plantilla para aceptar la ubicación."""
MIN_OCR_LABEL_COVERAGE = 0.25
"""Fracción de etiquetas del formato que deben leerse en una página OCR; con menos, la página es ilegible."""
MIN_INLIERS = 60
"""Puntos que deben encajar con la plantilla para aceptar la alineación."""
MIN_RATIO_INLIERS = 0.20
MAX_RMSE_PX = 3.0
"""Error medio de reproyección máximo tras alinear."""
MAX_ROTATION_DEGREES = 10
SCALE_RANGE = (0.85, 1.15)
PHOTO_SCALE_RANGE = (0.4, 2.5)
"""En fotos la hoja ocupa una parte variable de la imagen: se admite un rango de escala más amplio."""
MAX_PHOTO_ROTATION = 20

OCR_HIGH_CONF = 0.80
"""Por encima, una diferencia leída por OCR se considera real; por debajo, dudosa."""
OCR_BATCH_PAGES = 24
OCR_RESTRICT = False
"""Restringir caracteres por tipo de campo al decodificar. Medido el 2026-10-17 sobre 3 conjuntos de oficiales
escaneados (180 documentos c/u): sin mejora en falsas alarmas y más casos en revisión, por eso no se adoptó."""
OCR_CALIBRATE = True
"""Convertir la confianza en probabilidad de acierto (CONFIDENCE_CALIBRATION). Medido el 2026-10-17: falsas alarmas en
escaneos difíciles 2,8 % -> 1,7 % y 3,9 % -> 2,2 %, sin cambio en escaneos normales, por eso se adoptó."""

OCR_VRAM_BUDGET_GB = 2.5
"""VRAM del pipeline (detector + reconocedor en inferencia), para convivir con otros procesos del equipo."""
TRAIN_VRAM_BUDGET_GB = 4.0
"""VRAM del ajuste fino con lotes de 64."""
RAM_BUDGET_GB = 5.0
"""Memoria comprometida total que pueden sumar los procesos propios."""
VRAM_MARGIN_GB = 2.0
"""VRAM que se deja libre además del presupuesto."""
COMMIT_MARGIN_GB = 4.0
OBSERVATION_WINDOW_S = 120
"""El chequeo previo observa GPU y memoria este tiempo para captar los picos de otros procesos."""
WATCH_INTERVAL_S = 5
CRITICAL_COMMIT_GB = 1.5
"""Si la memoria comprometida libre del sistema baja de esto, el proceso propio pausa o se detiene."""

MIN_SHARPNESS = 12.0
"""Varianza del laplaciano bajo la cual una foto se considera ilegible."""
READ_WORKERS = 6
"""Procesos de CPU para la lectura de documentos (sin PyTorch, ~250 MB cada uno)."""
