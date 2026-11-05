"""Restricción de caracteres por campo y calibración de la confianza (sin GPU)."""

import numpy as np

from training.calibrate_confidence import isotonic
from verification.ocr import zone_set


def test_sets_by_type():
    """Cada tipo de campo admite solo sus caracteres; sin valores conocidos un categórico no se restringe."""
    assert zone_set({"tipo": "identidad", "nombre": "dni"}) == set("0123456789")
    assert "0" not in zone_set({"tipo": "identidad", "nombre": "apellidos"})
    assert "Ñ" in zone_set({"tipo": "identidad", "nombre": "apellidos"})
    assert zone_set({"tipo": "fecha"}) == set("0123456789/")
    assert zone_set({"tipo": "categorico", "validos": {"A-IIb", "A-I"}}) == set("A-Ib")
    assert zone_set({"tipo": "descriptivo"}) is None
    assert zone_set({"tipo": "categorico"}) is None


def test_isotonic_is_monotonic_and_corrects_overconfidence():
    """Un modelo que dice 0,9 pero acierta el 60 % queda calibrado cerca de 0,6, con una curva monótona."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 5000)
    hit = rng.random(5000) < (conf - 0.3)
    xs, ys = isotonic(conf, hit)
    assert np.all(np.diff(ys) >= -1e-9)
    assert abs(np.interp(0.9, xs, ys) - 0.6) < 0.08


def test_ocr_calls_use_doctr_parameter_names():
    """Los argumentos con nombre que el lector pasa a docTR existen en sus funciones (un nombre traducido por error
    solo fallaría al cargar el modelo)."""
    import ast
    import inspect
    from pathlib import Path

    from doctr.models import detection_predictor, recognition_predictor
    known = {"detection_predictor": detection_predictor, "recognition_predictor": recognition_predictor}
    tree = ast.parse((Path(__file__).parents[1] / "verification" / "ocr.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") in known]
    assert len(calls) == 2
    for c in calls:
        params = inspect.signature(known[c.func.id]).parameters
        assert all(k.arg in params or "kwargs" in params for k in c.keywords if k.arg != "arch") \
            and any(k.arg == "arch" for k in c.keywords)
