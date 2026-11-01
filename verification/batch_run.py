"""Verificación de un conjunto de postulantes para el visor (misma interfaz que verification.batch_run)."""

from datetime import date

from . import config as C
from . import pipeline as PL


def verify_batch(folders, registry, applicants_dir=C.APPLICANTS_DIR, official_dir=C.OFFICIAL_DATA_DIR,
                   second_reader=False, use_gpu=True, progress=None, log=print, cancelled=None, today=None):
    """Devuelve (personas, documentos, no_admitidos, meta). `progreso(fase, hechos, total)`.
    El segundo lector del flujo anterior no se usa: las lecturas dudosas quedan como OBSERVADO."""
    progress = progress or (lambda *a: None)

    def on_progress(phase, done, total):
        """Traduce las fases del pipeline a los textos de progreso del visor."""
        progress("leyendo documentos" if phase == "lectura" else "leyendo imágenes con OCR", done, total)

    people, documents, rejected, meta = PL.verify(
        list(folders), registry, applicants_dir, official_dir, min(C.READ_WORKERS, max(1, len(folders) * 3)),
        use_gpu, log=log, progress=on_progress, cancelled=cancelled or (lambda: False), today=today or date.today())
    meta["segundo_lector"] = "no se usa"
    return people, documents, rejected, meta
