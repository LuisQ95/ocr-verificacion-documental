"""Rutas y umbrales del pipeline. Los umbrales están aquí para poder ajustarlos sin tocar la lógica.

Cada constante lleva su explicación como docstring de atributo (la cadena que la sigue), que muestran los editores y
las herramientas de documentación.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_OFICIAL = ROOT / "official_file"
"""Plantillas en blanco de cada documento."""
DIR_POSTULANTES = ROOT / "postulant_files"
DIR_WEB = ROOT / "webscrap_files"
"""Datos de la fuente oficial (JSON), una carpeta por postulante."""
DIR_MODELOS = ROOT / "modelos"
DIR_ZONAS = DIR_MODELOS / "plantillas"
"""Zonas deducidas de cada plantilla."""
DIR_RESULTADOS = ROOT / "resultados"

MODELO_PARSEQ = DIR_MODELOS / "parseq_cl_v4.pt"
"""Reconocedor adoptado el 2026-10-28 (ver entrenamiento/FICHA_RECONOCEDOR.md): parte del PARSeq genérico de docTR y
se entrenó con un vocabulario amplio de nombres (20 % reservado para la prueba), texto aleatorio y los datos anteriores
redistribuidos, para que lea en vez de memorizar. Verificador completo en 600 postulantes: genuinos acusados 8 -> 2
con la misma detección (455 vs 456 de 520). El anterior, parseq_cl_puro.pt, se conserva como respaldo."""
MODELO_DBNET = DIR_MODELOS / "dbnet_es_finetuned.pt"
"""Detector afinado; solo se usa si existe (el afinado no mejoró y no se adoptó)."""
CALIBRACION_CONF = DIR_MODELOS / "calibracion_confianza_cl_v4.json"
"""Curva que convierte la confianza del reconocedor en probabilidad de acierto (ajustada con entrenamiento/testing y
testing_v2, que incluye nombres nunca vistos)."""
VARIABLES_CANDIDATO = ("OCR_MODELO_PARSEQ", "OCR_CALIBRACION")
"""OCR_MODELO_PARSEQ / OCR_CALIBRACION (variables de entorno, rutas relativas a la raíz) permiten medir un candidato
con el pipeline completo antes de adoptarlo, sin cambiar lo adoptado."""
if os.environ.get("OCR_MODELO_PARSEQ"):
    MODELO_PARSEQ = ROOT / os.environ["OCR_MODELO_PARSEQ"]
if os.environ.get("OCR_CALIBRACION"):
    CALIBRACION_CONF = ROOT / os.environ["OCR_CALIBRACION"]
VOCABULARIO = DIR_MODELOS / "vocabulario.json"
"""Valores válidos de los campos categóricos, tomados de los documentos oficiales."""
REGISTRO_POSTULANTES = DIR_POSTULANTES / "candidatos.xlsx"
"""Identidad declarada (RUT) y cargo de cada carpeta de postulante."""

DPI = 150
"""Resolución de trabajo para alineación y OCR."""
ESCALA = DPI / 72
"""Píxeles por punto PDF."""

MIN_SIMILITUD_NATIVA = 0.6
"""Fracción de palabras fijas de la plantilla que debe tener un PDF nativo para reconocerse como ese documento."""
MIN_INLIERS = 60
"""Puntos que deben encajar con la plantilla para aceptar la alineación."""
MIN_RATIO_INLIERS = 0.20
MAX_RMSE_PX = 3.0
"""Error medio de reproyección máximo tras alinear."""
MAX_ROTACION_GRADOS = 10
RANGO_ESCALA = (0.85, 1.15)
RANGO_ESCALA_FOTO = (0.4, 2.5)
"""En fotos la hoja ocupa una parte variable de la imagen: se admite un rango de escala más amplio."""
MAX_ROTACION_FOTO = 20
MAX_RESIDUAL_ETIQUETAS_PT = 3.0
"""Verificación posterior a la alineación: distancia máxima entre las etiquetas leídas y su posición en la plantilla."""
MIN_ETIQUETAS_VERIFICADAS = 5

UMBRAL_BLANQUEO = 215
"""Los píxeles más claros que esto se vuelven blancos (quita marca de agua y ruido)."""
CONF_OCR_ALTA = 0.80
"""Por encima, una diferencia leída por OCR se considera real; por debajo, dudosa."""
MIN_TINTA_ZONA = 0.004
"""Fracción de píxeles oscuros para considerar que una zona tiene texto."""
OCR_BATCH_PAGINAS = 24
OCR_RESTRINGIR = False
"""Restringir caracteres por tipo de campo al decodificar. Medido el 2026-10-17 sobre 3 conjuntos de oficiales
escaneados (180 documentos c/u): sin mejora en falsas alarmas y más casos en revisión, por eso no se adoptó."""
OCR_CALIBRAR = True
"""Convertir la confianza en probabilidad de acierto (CALIBRACION_CONF). Medido el 2026-10-17: falsas alarmas en
escaneos difíciles 2,8 % -> 1,7 % y 3,9 % -> 2,2 %, sin cambio en escaneos normales, por eso se adoptó."""

PRESUPUESTO_VRAM_OCR_GB = 2.5
"""VRAM del pipeline (detector + reconocedor en inferencia), para convivir con otros procesos del equipo."""
PRESUPUESTO_VRAM_TRAIN_GB = 4.0
"""VRAM del ajuste fino con lotes de 64."""
PRESUPUESTO_RAM_GB = 5.0
"""Memoria comprometida total que pueden sumar los procesos propios."""
MARGEN_VRAM_GB = 2.0
"""VRAM que se deja libre además del presupuesto."""
MARGEN_COMPROMETIDA_GB = 4.0
VENTANA_OBSERVACION_S = 120
"""El chequeo previo observa GPU y memoria este tiempo para captar los picos de otros procesos."""
INTERVALO_VIGIA_S = 5
COMPROMETIDA_CRITICA_GB = 1.5
"""Si la memoria comprometida libre del sistema baja de esto, el proceso propio pausa o se detiene."""
PIPELINE_WORKERS = 6
"""Procesos de CPU del pipeline genérico (sin PyTorch, ~250 MB cada uno)."""

FUZZY_TEXTO_LARGO = 90
"""Similitud mínima para tolerar ruido de OCR en textos largos."""
LARGO_TEXTO_TOLERANTE = 14
TOLERANCIA_GEOMETRIA_PT = 0.5
"""Desplazamiento global aceptable en un PDF nativo."""
CAMPOS_TECNICOS = ("codigo_de_verificacion", "integridad")
"""Códigos sin significado de negocio que también viajan en el QR: en escaneos se validan por el QR."""

HVID_DIAS_VIGENCIA = 30
"""Una Hoja de Vida más antigua que esto queda DESACTUALIZADA (ver config/requisitos_cl.yaml)."""
NITIDEZ_MINIMA = 12.0
"""Varianza del laplaciano bajo la cual una foto se considera ilegible."""
CHILE_WORKERS = 6
"""Procesos de CPU para la lectura del flujo chileno (sin PyTorch)."""
