"""Restricción de caracteres por campo y calibración de la confianza (sin GPU)."""

import numpy as np

from entrenamiento.calibrar_confianza import isotonica
from verificacion.ocr import conjunto_zona


def test_conjuntos_por_tipo():
    """Cada tipo de campo admite solo sus caracteres; sin valores conocidos un categórico no se restringe."""
    assert conjunto_zona({"tipo": "identidad", "nombre": "dni"}) == set("0123456789")
    assert "0" not in conjunto_zona({"tipo": "identidad", "nombre": "apellidos"})
    assert "Ñ" in conjunto_zona({"tipo": "identidad", "nombre": "apellidos"})
    assert conjunto_zona({"tipo": "fecha"}) == set("0123456789/")
    assert conjunto_zona({"tipo": "categorico", "validos": {"A-IIb", "A-I"}}) == set("A-Ib")
    assert conjunto_zona({"tipo": "descriptivo"}) is None
    assert conjunto_zona({"tipo": "categorico"}) is None


def test_isotonica_es_monotona_y_corrige_exceso_de_confianza():
    """Un modelo que dice 0,9 pero acierta el 60 % queda calibrado cerca de 0,6, con una curva monótona."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 5000)
    acierto = rng.random(5000) < (conf - 0.3)
    xs, ys = isotonica(conf, acierto)
    assert np.all(np.diff(ys) >= -1e-9)
    assert abs(np.interp(0.9, xs, ys) - 0.6) < 0.08
