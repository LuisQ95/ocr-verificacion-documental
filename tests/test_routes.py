"""Rutas de los dos visores: cada parámetro de la ruta ("/api/x/{nombre}") es un parámetro de su función."""

import inspect
import re

import pytest


@pytest.mark.parametrize("module", ["viewer.server", "evaluation_viewer.server"])
def test_path_parameters_match_functions(module):
    """Un parámetro de ruta sin su argumento en la función haría que FastAPI lo pida como consulta y la ruta
    respondería 422."""
    app = __import__(module, fromlist=["app"]).app
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        names = set(inspect.signature(endpoint).parameters)
        assert set(re.findall(r"{(\w+)}", route.path)) <= names, route.path
