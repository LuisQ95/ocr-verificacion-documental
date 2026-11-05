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


def test_safe_path_only_opens_files_inside_the_data_folders(tmp_path, monkeypatch):
    """Solo se abren archivos de las carpetas de datos: ni rutas con "..", ni una carpeta hermana con el mismo inicio."""
    import os

    from fastapi import HTTPException

    from viewer import server as V
    data = tmp_path / "data"
    sibling = tmp_path / "data_other"
    data.mkdir()
    sibling.mkdir()
    (data / "ok.pdf").write_bytes(b"x")
    (sibling / "secret.pdf").write_bytes(b"x")
    monkeypatch.setattr(V, "ALLOWED_ROOTS", (os.path.realpath(data) + os.sep,))
    assert V._safe_path(data / "ok.pdf").name == "ok.pdf"
    for bad in (sibling / "secret.pdf", f"{data}/../data_other/secret.pdf", data / "no_existe.pdf", data):
        with pytest.raises(HTTPException):
            V._safe_path(bad)
