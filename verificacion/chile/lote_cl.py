"""Verificación de un conjunto de postulantes para el visor (misma interfaz que verificacion.lote)."""

from datetime import date

from .. import config as C
from . import pipeline_cl as PL


def verificar_lote(carpetas, registro, dir_postulantes=C.DIR_POSTULANTES, dir_oficiales=C.DIR_WEB,
                   segundo_lector=False, usar_gpu=True, progreso=None, log=print, cancelado=None, hoy=None):
    """Devuelve (personas, documentos, no_admitidos, meta). `progreso(fase, hechos, total)`.
    El segundo lector del flujo anterior no se usa: las lecturas dudosas quedan como OBSERVADO."""
    progreso = progreso or (lambda *a: None)

    def avance(fase, hechos, total):
        """Traduce las fases del pipeline a los textos de progreso del visor."""
        progreso("leyendo documentos" if fase == "lectura" else "leyendo imágenes con OCR", hechos, total)

    personas, documentos, no_adm, meta = PL.verificar(
        list(carpetas), registro, dir_postulantes, dir_oficiales, min(C.CHILE_WORKERS, max(1, len(carpetas) * 3)),
        usar_gpu, log=log, progreso=avance, cancelado=cancelado or (lambda: False), hoy=hoy or date.today())
    meta["segundo_lector"] = "no se usa en el flujo chileno"
    return personas, documentos, no_adm, meta
