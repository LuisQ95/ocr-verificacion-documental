"""Lógica del pipeline sin imágenes ni OCR: unidades leídas -> páginas, agrupación de páginas en documentos, elección
del JSON oficial de cada certificado, nombres únicos, y las piezas comunes de comparación (campos y cajas)."""

from verificacion import cajas
from verificacion import campos as KC
from verificacion import config as C
from verificacion.chile import pipeline_cl as PL


def _pagina(plantilla, archivo, ilegible=False):
    """Página mínima tal como la produce paginas_de_unidad."""
    return {"plantilla": plantilla, "archivo": archivo, "unidad": archivo, "carpeta": "c", "ilegible": ilegible}


def test_unidad_con_error_es_ilegible():
    """Un archivo que no se pudo leer queda como página ilegible con el motivo."""
    p, = PL.paginas_de_unidad({"archivo": "a.pdf", "error": "ValueError: roto"})
    assert p["ilegible"] and "roto" in p["motivo_ilegible"] and p["plantilla"] is None


def test_imagen_sin_alinear_borrosa_o_nitida():
    """Sin alineación, una imagen borrosa es ilegible y una nítida es un documento no reconocido."""
    doc = {"plantilla": "ameco_rev03", "alineacion_ok": False, "alineacion_motivo": "pocos puntos"}
    borrosa, = PL.paginas_de_unidad({"archivo": "f.jpg", "documentos": [doc], "nitidez": C.NITIDEZ_MINIMA - 1})
    nitida, = PL.paginas_de_unidad({"archivo": "f.jpg", "documentos": [doc], "nitidez": C.NITIDEZ_MINIMA + 50})
    assert borrosa["ilegible"] and not borrosa.get("no_reconocido")
    assert nitida["no_reconocido"] and not nitida.get("ilegible")
    assert borrosa["plantilla"] is None and borrosa["plantilla_probable"] == "ameco_rev03"


def test_imagen_sin_documentos_es_ilegible():
    """Una imagen donde no se encontró ningún documento queda como una página ilegible."""
    p, = PL.paginas_de_unidad({"archivo": "f.jpg", "documentos": []})
    assert p["ilegible"]


def test_hoja_de_vida_se_junta_por_archivo_y_luego_por_orden():
    """Las páginas 1 y 2 del mismo archivo forman un documento; las sueltas se emparejan entre archivos."""
    ps = [_pagina("hoja_vida_p1", "a.pdf"), _pagina("hoja_vida_p2", "a.pdf"),
          _pagina("hoja_vida_p1", "p1.jpg"), _pagina("hoja_vida_p2", "p2.jpg"), _pagina("ameco_rev03", "c.pdf")]
    docs = PL.agrupar(ps)
    juntas = sorted(tuple(p["archivo"] for p in d) for d in docs if len(d) == 2)
    assert juntas == [("a.pdf", "a.pdf"), ("p1.jpg", "p2.jpg")]
    assert [[p["archivo"] for p in d] for d in docs if len(d) == 1] == [["c.pdf"]]


def test_hojas_de_vida_en_fotos_sueltas_se_juntan_por_folio():
    """Dos Hojas de Vida en fotos sueltas: cada página 1 se junta con la página 2 de su misma emisión (folio leído
    con confusiones de OCR), no con la siguiente en orden."""
    def pagina(plantilla, archivo, folio):
        """Página con el folio leído."""
        return _pagina(plantilla, archivo) | {"valores": {"folio": {"texto": folio}}}

    ps = [pagina("hoja_vida_p1", "1.jpg", "500012345678"), pagina("hoja_vida_p1", "2.jpg", "500098765432"),
          pagina("hoja_vida_p2", "3.jpg", "5OOO98765432"), pagina("hoja_vida_p2", "4.jpg", "500012345678")]
    assert sorted(tuple(p["archivo"] for p in d) for d in PL.agrupar(ps)) == [("1.jpg", "4.jpg"), ("2.jpg", "3.jpg")]


def test_hoja_de_vida_ilegible_no_se_empareja():
    """Una página ilegible queda sola: no se junta con la otra página."""
    docs = PL.agrupar([_pagina("hoja_vida_p1", "a.jpg"), _pagina("hoja_vida_p2", "b.jpg", ilegible=True)])
    assert sorted(len(d) for d in docs) == [1, 1]


def test_json_del_certificado_por_numero_y_por_version():
    """El JSON del certificado se elige por el N° leído (tolerando O/0 en OCR) o, si no, por la única versión."""
    oficiales = {"ameco": {"AMCH-1001": {"Formato": "Rev.03"}, "AMCH-2002": {"Formato": "Rev.02"}}}

    def doc(numero, plantilla, es_ocr):
        """Documento con el N° leído."""
        return {"valores": {"numero_certificado": {"texto": numero}}, "plantilla": plantilla, "es_ocr": es_ocr}

    assert PL._json_ameco(doc("AMCH-1001", "ameco_rev03", False), oficiales)[0] == "AMCH-1001"
    assert PL._json_ameco(doc("AMCH-1OO1", "ameco_rev03", True), oficiales)[0] == "AMCH-1001"
    assert PL._json_ameco(doc("ilegible", "ameco_rev02", True), oficiales)[0] == "AMCH-2002"
    dos_rev03 = {"ameco": {"A-1": {"Formato": "Rev.03"}, "A-2": {"Formato": "Rev.03"}}}
    assert PL._json_ameco(doc("", "ameco_rev03", True), dos_rev03) == (None, None)


def test_nombres_unicos():
    """Los documentos con el mismo nombre en una carpeta se numeran desde el segundo."""
    filas = PL.nombres_unicos([{"archivo": "x.jpg"}, {"archivo": "x.jpg"}, {"archivo": "y.pdf"}, {"archivo": "x.jpg"}])
    assert [f["archivo"] for f in filas] == ["x.jpg", "x.jpg (2)", "y.pdf", "x.jpg (3)"]


def test_negaciones():
    """Aparecer o desaparecer un NO, SIN o prefijo negativo cambia el sentido; el ruido pegado a NO no."""
    assert KC.cambia_negacion("REGISTRA", "NO REGISTRA")
    assert KC.cambia_negacion("COMPLETA", "INCOMPLETA")
    assert not KC.cambia_negacion("NO REGISTRA", "NOJ REGISTRA")
    assert not KC.cambia_negacion("SIN ANTECEDENTES", "SUN ANTECEDENTES")


def test_asignar_palabras_separa_etiquetas_y_ordena_lineas():
    """Las etiquetas del formato no son valores; el valor de una zona se arma en orden de lectura."""
    plantilla = {"palabras_fijas": [(10, 10, 40, 20, "Nombre")],
                 "zonas": {"nombre": {"rect": [50, 5, 300, 40]}}}
    palabras = [{"texto": "SOTO", "bbox": [120, 10, 160, 20], "conf": 0.9},
                {"texto": "JUAN", "bbox": [60, 10.5, 100, 20.5], "conf": 0.95},
                {"texto": "Nombre", "bbox": [10, 10, 40, 20]},
                {"texto": "PEREZ", "bbox": [60, 25, 100, 35], "conf": 0.7}]
    r = KC.asignar_palabras(palabras, plantilla)
    assert r["valores"]["nombre"]["texto"] == "JUAN SOTO PEREZ"
    assert r["valores"]["nombre"]["conf"] == 0.7 and r["cobertura_etiquetas"] == 1.0


def test_fuente_sin_prefijo_de_subconjunto():
    """El prefijo de 6 mayúsculas que agregan los PDF a las fuentes incrustadas no cuenta."""
    assert KC.fuente_base("ABCDEF+ArialMT") == "ArialMT" and KC.fuente_base("Courier") == "Courier"


def test_cajas():
    """Centro, contención, intersección e IoU (los elementos extra de la caja se ignoran)."""
    assert cajas.centro((0, 0, 10, 20)) == (5, 10)
    assert cajas.dentro((5, 5), (0, 0, 10, 10)) and not cajas.dentro((11, 5), (0, 0, 10, 10))
    assert cajas.intersecta((0, 0, 10, 10), (5, 5, 15, 15)) and not cajas.intersecta((0, 0, 10, 10), (10, 0, 20, 10))
    assert cajas.iou((0, 0, 10, 10, 0.9), (0, 0, 10, 10, 0.5)) == 1.0
    assert abs(cajas.iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert cajas.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
