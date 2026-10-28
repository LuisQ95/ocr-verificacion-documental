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

from verificacion import config as C
from verificacion.chile import memoria_cl as M

DEMO = C.ROOT / "DATOS REALES" / "demo"


def _caso_demo():
    """Primer postulante de la demo con Hoja de Vida y certificados AMECO Rev.02 y Rev.03.

    Devuelve un diccionario con la carpeta de documentos, los JSON oficiales (ruta y contenido) y los PDF del
    postulante identificados por contenido del nombre de archivo; None si la demo no está disponible.
    """
    if not (DEMO / "webscrap_files").exists():
        return None
    for web in sorted(p for p in (DEMO / "webscrap_files").iterdir() if p.is_dir()):
        post = DEMO / "postulant_files" / web.name
        hv = web / "hoja_vida_conductor.json"
        amecos = {}
        for j in web.glob("ameco_*.json"):
            datos = json.loads(j.read_text(encoding="utf-8"))
            amecos["rev03" if "03" in datos.get("Formato", "") else "rev02"] = (j, datos)
        if not (hv.exists() and post.exists() and {"rev02", "rev03"} <= set(amecos)):
            continue

        def pdf(clave):
            """PDF del postulante cuyo nombre contiene `clave`."""
            return next(f for f in post.iterdir() if clave in f.name and f.suffix.lower() == ".pdf")

        palabras = web.name.split()
        return {"post": post, "nombres": " ".join(palabras[:2]), "apellidos": " ".join(palabras[2:]),
                "etiqueta": web.name, "hv": (hv, json.loads(hv.read_text(encoding="utf-8"))), "ameco": amecos,
                "pdf_hv": pdf("HVID"),
                "pdf_rev03": pdf(re.sub(r"\D", "", amecos["rev03"][1]["N°"])),
                "pdf_rev02": pdf(re.sub(r"\D", "", amecos["rev02"][1]["N°"]))}
    return None


CASO = _caso_demo()
con_demo = pytest.mark.skipif(CASO is None, reason="sin la demo local de DATOS REALES")


def test_nombres_y_apellidos():
    """Nombres y apellidos: 1 a 3 nombres, 1 a 2 apellidos (con partículas), solo letras."""
    ident, errores, adv = M.validar_identidad("Alejandra  Bárbara", "Alarcón Sepúlveda", "12.345.678-5")
    assert not errores and not adv and ident["nombre_completo"] == "ALEJANDRA BÁRBARA ALARCÓN SEPÚLVEDA"
    assert ident["rut"] == "12.345.678-5"
    assert not M.validar_identidad("MARIA DE LOS ANGELES", "DE LA CRUZ PEREZ", "12345678-5")[1]
    assert M.validar_identidad("A B C D", "PEREZ", "12345678-5")[1]
    assert M.validar_identidad("JUAN", "PEREZ SOTO ROJAS", "12345678-5")[1]
    assert M.validar_identidad("JUAN", "", "12345678-5")[1]
    assert M.validar_identidad("JUAN2", "PEREZ", "12345678-5")[1]


def test_un_apellido_advierte_homonimos():
    """Con un solo apellido se acepta, pero se advierte del riesgo de homónimos."""
    _, errores, adv = M.validar_identidad("LUIS", "QUISPE", "11.111.111-1")
    assert not errores and any("un solo apellido" in a for a in adv)


def test_rut_y_fecha():
    """El RUT exige dígito verificador válido; la fecha de nacimiento es opcional pero no puede ser futura."""
    assert M.validar_identidad("LUIS", "QUISPE", "12.345.678-4")[1]
    assert M.validar_identidad("LUIS", "QUISPE", "")[1]
    ident, errores, _ = M.validar_identidad("LUIS", "QUISPE", "11111111-1", "1995-05-02")
    assert not errores and ident["fecha_nacimiento"] == date(1995, 5, 2)
    assert M.validar_identidad("LUIS", "QUISPE", "11111111-1", "2999-01-01")[1]


def test_json_oficiales_por_contenido():
    """Los JSON oficiales se clasifican por su contenido, no por el nombre del archivo; los inválidos se advierten."""
    hv = json.dumps({"Nombre": "X", "R.U.N.": "1-9", "Licencias registradas": []}).encode()
    am = json.dumps({"N°": "AMCH-1", "Formato": "Rev.03"}).encode()
    of, adv = M.leer_oficiales([("cualquier_nombre.json", hv), ("otro.json", am), ("roto.json", b"{no"),
                                ("ajeno.json", b'{"a": 1}')])
    assert of["hoja_vida"]["Nombre"] == "X" and "AMCH-1" in of["ameco"] and len(adv) == 2


@pytest.fixture(scope="module")
def visor():
    """Servidor del visor con una IP de prueba permitida y una contraseña conocida."""
    from fastapi.testclient import TestClient

    from visor_pruebas import acceso as A
    from visor_pruebas import servidor as S
    S._cfg = {"host": "127.0.0.1", "puerto": 8010, "ips_permitidas": ["testclient"],
              "clave": A.hash_clave("clave-de-prueba-123")}
    with TestClient(S.app) as c:
        yield S, c


def test_ip_no_permitida(visor):
    """Una IP fuera de la lista no ve ni la página ni el inicio de sesión."""
    S, c = visor
    S._cfg["ips_permitidas"] = []
    try:
        assert c.get("/").status_code == 403 and c.post("/api/login", json={"clave": "x"}).status_code == 403
    finally:
        S._cfg["ips_permitidas"] = ["testclient"]


def test_contrasena(visor):
    """Sin sesión no hay acceso; la contraseña incorrecta se rechaza y la correcta abre sesión."""
    _, c = visor
    assert c.get("/api/sesion").status_code == 401
    assert c.post("/api/login", json={"clave": "incorrecta"}).status_code == 401
    assert c.post("/api/login", json={"clave": "clave-de-prueba-123"}).status_code == 200
    assert c.get("/api/sesion").status_code == 200


def _b64(ruta):
    """Archivo como lo envía el navegador: nombre y contenido en base64."""
    return {"nombre": ruta.name, "base64": base64.b64encode(ruta.read_bytes()).decode()}


def _verificar(c, cuerpo):
    """Lanza una verificación y espera a que el trabajo termine (máx. 150 s)."""
    r = c.post("/api/verificar", json=cuerpo)
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    for _ in range(300):
        t = c.get(f"/api/trabajo/{tid}").json()
        if t["estado"] in ("terminado", "error"):
            return t
        time.sleep(0.5)
    raise TimeoutError


@con_demo
def test_verificacion_en_memoria_faltante_y_homonimos(visor):
    """De punta a punta: documento faltante, dictámenes, ZIP con los JSON extraídos y su confianza, y homónimos
    distinguidos por fecha de nacimiento."""
    _, c = visor
    c.post("/api/login", json={"clave": "clave-de-prueba-123"})
    hv_json, hv_of = CASO["hv"]
    am_json, am_of = CASO["ameco"]["rev03"]
    cuerpo = {"nombres": CASO["nombres"], "apellidos": CASO["apellidos"], "rut": am_of["RUT"], "fecha_nacimiento": "",
              "cargo": "Operador de Grúa Móvil",
              "postulante": [_b64(CASO["pdf_hv"]), _b64(CASO["pdf_rev03"])],
              "oficiales": [_b64(hv_json), _b64(am_json)]}
    assert _verificar(c, cuerpo)["estado"] == "terminado"
    r = c.get("/api/resultados").json()[0]
    estados = {e["tipo"]: e["estado"] for e in r["esperados"]}
    assert estados == {"hoja_vida": "ENTREGADO", "ameco_rev02": M.FALTANTE, "ameco_rev03": "ENTREGADO"}
    assert {d["tipo"]: d["veredicto"] for d in r["documentos"]} == \
        {"hoja_vida_p1": "DESACTUALIZADO", "ameco_rev03": "AUTÉNTICO"}
    assert re.fullmatch(r"\d{2}-\d{2}-\d{4}", r["fecha_nacimiento"]) and r["fecha_nacimiento_fuente"] == "JSON de Hoja de Vida"
    assert r["etiqueta"] == CASO["etiqueta"]
    x = c.get(f"/api/zip/{r['id']}")
    assert x.status_code == 200
    with zipfile.ZipFile(io.BytesIO(x.content)) as z:
        nombres = z.namelist()
        assert {"LEAME.txt", "resumen.json", "diferencias.json", "reporte_verificacion.xlsx"} <= set(nombres)
        extraidos = [n for n in nombres if n.startswith("datos_extraidos/")]
        assert len(extraidos) == 2
        ameco = json.loads(z.read(next(n for n in extraidos if "ameco" in n)))
        assert ameco["N°"] == am_of["N°"] and ameco["RUT"] == am_of["RUT"]
        assert ameco["Licencia de Conductor"]["Clase"] == am_of["Licencia de Conductor"]["Clase"]
        assert ameco["Exámenes"]["Conocim. General"] == am_of["Exámenes"]["Conocim. General"]
        hv = json.loads(z.read(next(n for n in extraidos if "hoja_vida" in n)))
        assert len(hv["Licencias registradas"]) == len(hv_of["Licencias registradas"])
        assert hv["Antecedentes"]["RNC"] == hv_of["Antecedentes"]["RNC"]
        assert hv["_control"]["metodo"].startswith("lectura directa")
        assert ameco["_confianza"]["N°"] == {"confianza": 1.0, "nivel": "ALTA"}
        assert ameco["_confianza"]["Exámenes"]["Conocim. General"]["nivel"] == "ALTA"
        assert [x["nivel"] for x in hv["_confianza"]["Licencias registradas"]] == ["ALTA"] * len(hv["Licencias registradas"])
        assert hv["_control"]["resumen_confianza"]["BAJA"] == 0
    d = next(d for d in r["documentos"] if d["tipo"] == "ameco_rev03")
    assert {"campo": "N°", "valor": am_of["N°"], "confianza": 1.0, "nivel": "ALTA"} in d["campos_extraidos"]

    otro = dict(cuerpo, rut="11.111.111-1", fecha_nacimiento="2001-10-10", oficiales=[],
                postulante=[_b64(CASO["pdf_rev03"])])
    assert _verificar(c, otro)["estado"] == "terminado"
    etiquetas = sorted(x["etiqueta"] for x in c.get("/api/resultados").json())
    assert etiquetas == sorted([f"{CASO['etiqueta']} (10-10-2001)", f"{CASO['etiqueta']} ({r['fecha_nacimiento']})"])
    nuevo = c.get("/api/resultados").json()[0]
    assert nuevo["documentos"][0]["veredicto"] == "SIN DATO OFICIAL"

    assert c.post("/api/borrar").status_code == 200 and c.get("/api/resultados").json() == []


@con_demo
def test_sin_datos_oficiales_solo_extrae(visor):
    """Sin JSON oficiales no se analiza: se entregan solo los datos extraídos, con el cargo en mayúsculas."""
    _, c = visor
    c.post("/api/login", json={"clave": "clave-de-prueba-123"})
    am_of = CASO["ameco"]["rev02"][1]
    cuerpo = {"nombres": CASO["nombres"], "apellidos": CASO["apellidos"], "rut": am_of["RUT"],
              "cargo": "supervisor de faena", "oficiales": [], "postulante": [_b64(CASO["pdf_rev02"])]}
    assert _verificar(c, cuerpo)["estado"] == "terminado"
    r = c.get("/api/resultados").json()[0]
    assert r["sin_datos_oficiales"] and r["veredicto"] == "SIN DATO OFICIAL" and r["aptitud"] == "SIN DATO OFICIAL"
    assert r["cargo"] == "SUPERVISOR DE FAENA"
    d = r["documentos"][0]
    assert d["veredicto"] == "SIN DATO OFICIAL" and d["campos"] == [] and d["json_extraido"]["N°"] == am_of["N°"]
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api/zip/{r['id']}").content)) as z:
        assert "diferencias.json" not in z.namelist()
        assert json.loads(z.read("resumen.json"))["autenticidad"] == "SIN DATO OFICIAL"
    c.post("/api/borrar")
