"""Visor de pruebas: identidad, JSON oficiales, acceso (IP + contraseña) y verificación en memoria de punta a punta.

Las pruebas de punta a punta usan la demo local `DATOS REALES/demo` (PDF nativos: no usan OCR ni GPU). Esa carpeta
contiene documentos de personas reales y no se versiona: si no está, esas pruebas se omiten. Los valores esperados
(nombre, RUT, números de certificado) se leen de sus JSON oficiales, nunca se escriben en el código.
"""

import base64
import io
import json
import re
import time
import zipfile
from datetime import date

import pytest

from verification import config as C
from verification import in_memory as M
from verification.profiles.operator_certification import documents as D

DEMO = C.ROOT / "DATOS REALES" / "demo"


def _demo_case():
    """Primer postulante de la demo con Hoja de Vida y certificados AMECO Rev.02 y Rev.03.

    Devuelve un diccionario con la carpeta de documentos, los JSON oficiales (ruta y contenido) y los PDF del
    postulante identificados por contenido del nombre de archivo; None si la demo no está disponible.
    """
    if not (DEMO / "webscrap_files").exists():
        return None
    for web in sorted(p for p in (DEMO / "webscrap_files").iterdir() if p.is_dir()):
        post = DEMO / "postulant_files" / web.name
        rec = web / "hoja_vida_conductor.json"
        certificates = {}
        for j in web.glob("ameco_*.json"):
            data = json.loads(j.read_text(encoding="utf-8"))
            certificates["rev03" if "03" in data.get("Formato", "") else "rev02"] = (j, data)
        if not (rec.exists() and post.exists() and {"rev02", "rev03"} <= set(certificates)):
            continue

        def pdf(key):
            """PDF del postulante cuyo nombre contiene `clave`."""
            return next(f for f in post.iterdir() if key in f.name and f.suffix.lower() == ".pdf")

        words = web.name.split()
        return {"post": post, "nombres": " ".join(words[:2]), "apellidos": " ".join(words[2:]),
                "etiqueta": web.name, "hv": (rec, json.loads(rec.read_text(encoding="utf-8"))), "ameco": certificates,
                "pdf_hv": pdf("HVID"),
                "pdf_rev03": pdf(re.sub(r"\D", "", certificates["rev03"][1]["N°"])),
                "pdf_rev02": pdf(re.sub(r"\D", "", certificates["rev02"][1]["N°"]))}
    return None


CASE = _demo_case()
with_demo = pytest.mark.skipif(CASE is None, reason="sin la demo local de DATOS REALES")


def test_names_and_surnames():
    """Nombres y apellidos: 1 a 3 nombres, 1 a 2 apellidos (con partículas), solo letras."""
    identity, errors, warns = M.validate_identity("Alejandra  Bárbara", "Alarcón Sepúlveda", "12.345.678-5")
    assert not errors and not warns and identity["nombre_completo"] == "ALEJANDRA BÁRBARA ALARCÓN SEPÚLVEDA"
    assert identity["rut"] == "12.345.678-5"
    assert not M.validate_identity("MARIA DE LOS ANGELES", "DE LA CRUZ PEREZ", "12345678-5")[1]
    assert M.validate_identity("A B C D", "PEREZ", "12345678-5")[1]
    assert M.validate_identity("JUAN", "PEREZ SOTO ROJAS", "12345678-5")[1]
    assert M.validate_identity("JUAN", "", "12345678-5")[1]
    assert M.validate_identity("JUAN2", "PEREZ", "12345678-5")[1]


def test_single_surname_warns_about_namesakes():
    """Con un solo apellido se acepta, pero se advierte del riesgo de homónimos."""
    _, errors, warns = M.validate_identity("LUIS", "QUISPE", "11.111.111-1")
    assert not errors and any("un solo apellido" in a for a in warns)


def test_id_and_date():
    """El RUT exige dígito verificador válido; la fecha de nacimiento es opcional pero no puede ser futura."""
    assert M.validate_identity("LUIS", "QUISPE", "12.345.678-4")[1]
    assert M.validate_identity("LUIS", "QUISPE", "")[1]
    identity, errors, _ = M.validate_identity("LUIS", "QUISPE", "11111111-1", "1995-05-02")
    assert not errors and identity["fecha_nacimiento"] == date(1995, 5, 2)
    assert M.validate_identity("LUIS", "QUISPE", "11111111-1", "2999-01-01")[1]


def test_official_json_by_content():
    """Los JSON oficiales se clasifican por su contenido, no por el nombre del archivo; los inválidos se advierten."""
    rec = json.dumps({"Nombre": "X", "R.U.N.": "1-9", "Licencias registradas": []}).encode()
    cert = json.dumps({"N°": "AMCH-1", "Formato": "Rev.03"}).encode()
    of, warns = D.read_official_bytes([("cualquier_nombre.json", rec), ("otro.json", cert), ("roto.json", b"{no"),
                                ("ajeno.json", b'{"a": 1}')])
    assert of["hoja_vida"]["Nombre"] == "X" and "AMCH-1" in of["ameco"] and len(warns) == 2


@pytest.fixture(scope="module")
def viewer():
    """Servidor del visor con una IP de prueba permitida y una contraseña conocida."""
    from fastapi.testclient import TestClient

    from evaluation_viewer import access as A
    from evaluation_viewer import server as S
    S._cfg = {"host": "127.0.0.1", "puerto": 8010, "ips_permitidas": ["testclient"],
              "clave": A.key_hash("clave-de-prueba-123")}
    with TestClient(S.app) as c:
        yield S, c


def test_ip_not_allowed(viewer):
    """Una IP fuera de la lista no ve ni la página ni el inicio de sesión."""
    S, c = viewer
    S._cfg["ips_permitidas"] = []
    try:
        assert c.get("/").status_code == 403 and c.post("/api/login", json={"clave": "x"}).status_code == 403
    finally:
        S._cfg["ips_permitidas"] = ["testclient"]


def test_access_change_without_restart(viewer, tmp_path, monkeypatch):
    """Al guardar access.json, las IP permitidas nuevas rigen desde la petición siguiente."""
    from evaluation_viewer import access as A
    S, c = viewer
    stored = (S._cfg, S._cfg_version)
    path = tmp_path / "access.json"
    monkeypatch.setattr(A, "CONFIG_PATH", path)
    try:
        A.save(dict(S._cfg, ips_permitidas=[]), path)
        assert c.get("/").status_code == 403
        A.save(dict(S._cfg, ips_permitidas=["testclient"]), path)
        S._cfg_version = None
        assert c.get("/").status_code == 200
    finally:
        S._cfg, S._cfg_version = stored


def test_requests_without_files_have_max_size(viewer):
    """El inicio de sesión no acepta cuerpos grandes (no se procesa una contraseña de 100 KB)."""
    _, c = viewer
    assert c.post("/api/login", json={"clave": "x" * 100_000}).status_code == 413


def test_password(viewer):
    """Sin sesión no hay acceso; la contraseña incorrecta se rechaza y la correcta abre sesión."""
    _, c = viewer
    assert c.get("/api/sesion").status_code == 401
    assert c.post("/api/login", json={"clave": "incorrecta"}).status_code == 401
    assert c.post("/api/login", json={"clave": "clave-de-prueba-123"}).status_code == 200
    assert c.get("/api/sesion").status_code == 200


def _b64(path):
    """Archivo como lo envía el navegador: nombre y contenido en base64."""
    return {"nombre": path.name, "base64": base64.b64encode(path.read_bytes()).decode()}


def _verify(c, body):
    """Lanza una verificación y espera a que el trabajo termine (máx. 150 s)."""
    r = c.post("/api/verificar", json=body)
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    for _ in range(300):
        t = c.get(f"/api/trabajo/{tid}").json()
        if t["estado"] in ("terminado", "error"):
            return t
        time.sleep(0.5)
    raise TimeoutError


@with_demo
def test_in_memory_verification_missing_and_namesakes(viewer):
    """De punta a punta: documento faltante, dictámenes, ZIP con los JSON extraídos y su confianza, y homónimos
    distinguidos por fecha de nacimiento."""
    _, c = viewer
    c.post("/api/login", json={"clave": "clave-de-prueba-123"})
    rec_json, rec_of = CASE["hv"]
    cert_json, cert_of = CASE["ameco"]["rev03"]
    body = {"nombres": CASE["nombres"], "apellidos": CASE["apellidos"], "rut": cert_of["RUT"], "fecha_nacimiento": "",
              "cargo": "Operador de Grúa Móvil",
              "postulante": [_b64(CASE["pdf_hv"]), _b64(CASE["pdf_rev03"])],
              "oficiales": [_b64(rec_json), _b64(cert_json)]}
    assert _verify(c, body)["estado"] == "terminado"
    r = c.get("/api/resultados").json()[0]
    states = {e["tipo"]: e["estado"] for e in r["esperados"]}
    assert states == {"hoja_vida": "ENTREGADO", "ameco_rev02": M.MISSING, "ameco_rev03": "ENTREGADO"}
    assert {d["tipo"]: d["veredicto"] for d in r["documentos"]} == \
        {"hoja_vida_p1": "DESACTUALIZADO", "ameco_rev03": "AUTÉNTICO"}
    assert re.fullmatch(r"\d{2}-\d{2}-\d{4}", r["fecha_nacimiento"]) and r["fecha_nacimiento_fuente"] == "JSON de Hoja de Vida"
    assert r["etiqueta"] == CASE["etiqueta"]
    x = c.get(f"/api/zip/{r['id']}")
    assert x.status_code == 200
    with zipfile.ZipFile(io.BytesIO(x.content)) as z:
        names = z.namelist()
        assert {"LEAME.txt", "resumen.json", "diferencias.json", "reporte_verificacion.xlsx"} <= set(names)
        extracted = [n for n in names if n.startswith("datos_extraidos/")]
        assert len(extracted) == 2
        certificate = json.loads(z.read(next(n for n in extracted if "ameco" in n)))
        assert certificate["N°"] == cert_of["N°"] and certificate["RUT"] == cert_of["RUT"]
        assert certificate["Licencia de Conductor"]["Clase"] == cert_of["Licencia de Conductor"]["Clase"]
        assert certificate["Exámenes"]["Conocim. General"] == cert_of["Exámenes"]["Conocim. General"]
        rec = json.loads(z.read(next(n for n in extracted if "hoja_vida" in n)))
        assert len(rec["Licencias registradas"]) == len(rec_of["Licencias registradas"])
        assert rec["Antecedentes"]["RNC"] == rec_of["Antecedentes"]["RNC"]
        assert rec["_control"]["metodo"].startswith("lectura directa")
        assert certificate["_confianza"]["N°"] == {"confianza": 1.0, "nivel": "ALTA"}
        assert certificate["_confianza"]["Exámenes"]["Conocim. General"]["nivel"] == "ALTA"
        assert [x["nivel"] for x in rec["_confianza"]["Licencias registradas"]] == ["ALTA"] * len(rec["Licencias registradas"])
        assert rec["_control"]["resumen_confianza"]["BAJA"] == 0
    d = next(d for d in r["documentos"] if d["tipo"] == "ameco_rev03")
    assert {"campo": "N°", "valor": cert_of["N°"], "confianza": 1.0, "nivel": "ALTA"} in d["campos_extraidos"]

    other = dict(body, rut="11.111.111-1", fecha_nacimiento="2001-10-10", oficiales=[],
                postulante=[_b64(CASE["pdf_rev03"])])
    assert _verify(c, other)["estado"] == "terminado"
    labels = sorted(x["etiqueta"] for x in c.get("/api/resultados").json())
    assert labels == sorted([f"{CASE['etiqueta']} (10-10-2001)", f"{CASE['etiqueta']} ({r['fecha_nacimiento']})"])
    fresh = c.get("/api/resultados").json()[0]
    assert fresh["documentos"][0]["veredicto"] == "SIN DATO OFICIAL"

    assert c.post("/api/borrar").status_code == 200 and c.get("/api/resultados").json() == []


@with_demo
def test_without_official_data_only_extracts(viewer):
    """Sin JSON oficiales no se analiza: se entregan solo los datos extraídos, con el cargo en mayúsculas."""
    _, c = viewer
    c.post("/api/login", json={"clave": "clave-de-prueba-123"})
    cert_of = CASE["ameco"]["rev02"][1]
    body = {"nombres": CASE["nombres"], "apellidos": CASE["apellidos"], "rut": cert_of["RUT"],
              "cargo": "supervisor de faena", "oficiales": [], "postulante": [_b64(CASE["pdf_rev02"])]}
    assert _verify(c, body)["estado"] == "terminado"
    r = c.get("/api/resultados").json()[0]
    assert r["sin_datos_oficiales"] and r["veredicto"] == "SIN DATO OFICIAL" and r["aptitud"] == "SIN DATO OFICIAL"
    assert r["cargo"] == "SUPERVISOR DE FAENA"
    d = r["documentos"][0]
    assert d["veredicto"] == "SIN DATO OFICIAL" and d["campos"] == [] and d["json_extraido"]["N°"] == cert_of["N°"]
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api/zip/{r['id']}").content)) as z:
        assert "diferencias.json" not in z.namelist()
        assert json.loads(z.read("resumen.json"))["autenticidad"] == "SIN DATO OFICIAL"
    c.post("/api/borrar")
