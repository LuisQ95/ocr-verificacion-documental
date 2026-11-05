"""Lógica del pipeline sin imágenes ni OCR: unidades leídas -> páginas, agrupación de páginas en documentos, elección
del JSON oficial de cada certificado, nombres únicos, y las piezas comunes de comparación (campos y cajas)."""

from verification import boxes
from verification import fields as KC
from verification import config as C
from verification import pipeline as PL
from verification.profiles.operator_certification import documents as D


def _page(template, file, illegible=False):
    """Página mínima tal como la produce pages_of_unit."""
    return {"plantilla": template, "archivo": file, "unidad": file, "carpeta": "c", "ilegible": illegible}


def test_unit_with_error_is_illegible():
    """Un archivo que no se pudo leer queda como página ilegible con el motivo."""
    p, = PL.pages_of_unit({"archivo": "a.pdf", "error": "ValueError: roto"})
    assert p["ilegible"] and "roto" in p["motivo_ilegible"] and p["plantilla"] is None


def test_unaligned_image_blurry_or_sharp():
    """Sin alineación, una imagen borrosa es ilegible y una nítida es un documento no reconocido."""
    doc = {"plantilla": "ameco_rev03", "alineacion_ok": False, "alineacion_motivo": "pocos puntos"}
    blurry, = PL.pages_of_unit({"archivo": "f.jpg", "documentos": [doc], "nitidez": C.MIN_SHARPNESS - 1})
    sharp_ok, = PL.pages_of_unit({"archivo": "f.jpg", "documentos": [doc], "nitidez": C.MIN_SHARPNESS + 50})
    assert blurry["ilegible"] and not blurry.get("no_reconocido")
    assert sharp_ok["no_reconocido"] and not sharp_ok.get("ilegible")
    assert blurry["plantilla"] is None and blurry["plantilla_probable"] == "ameco_rev03"


def test_image_without_documents_is_illegible():
    """Una imagen donde no se encontró ningún documento queda como una página ilegible."""
    p, = PL.pages_of_unit({"archivo": "f.jpg", "documentos": []})
    assert p["ilegible"]


def test_record_joined_by_file_then_by_order():
    """Las páginas 1 y 2 del mismo archivo forman un documento; las sueltas se emparejan entre archivos."""
    ps = [_page("hoja_vida_p1", "a.pdf"), _page("hoja_vida_p2", "a.pdf"),
          _page("hoja_vida_p1", "p1.jpg"), _page("hoja_vida_p2", "p2.jpg"), _page("ameco_rev03", "c.pdf")]
    docs = D.group_pages(ps)
    joined = sorted(tuple(p["archivo"] for p in d) for d in docs if len(d) == 2)
    assert joined == [("a.pdf", "a.pdf"), ("p1.jpg", "p2.jpg")]
    assert [[p["archivo"] for p in d] for d in docs if len(d) == 1] == [["c.pdf"]]


def test_records_in_loose_photos_joined_by_folio():
    """Dos Hojas de Vida en fotos sueltas: cada página 1 se junta con la página 2 de su misma emisión (folio leído
    con confusiones de OCR), no con la siguiente en orden."""
    def page_no(template, file, folio):
        """Página con el folio leído."""
        return _page(template, file) | {"valores": {"folio": {"texto": folio}}}

    ps = [page_no("hoja_vida_p1", "1.jpg", "500012345678"), page_no("hoja_vida_p1", "2.jpg", "500098765432"),
          page_no("hoja_vida_p2", "3.jpg", "5OOO98765432"), page_no("hoja_vida_p2", "4.jpg", "500012345678")]
    assert sorted(tuple(p["archivo"] for p in d) for d in D.group_pages(ps)) == [("1.jpg", "4.jpg"), ("2.jpg", "3.jpg")]


def test_illegible_record_is_not_paired():
    """Una página ilegible queda sola: no se junta con la otra página."""
    docs = D.group_pages([_page("hoja_vida_p1", "a.jpg"), _page("hoja_vida_p2", "b.jpg", illegible=True)])
    assert sorted(len(d) for d in docs) == [1, 1]


def test_certificate_json_by_number_and_version():
    """El JSON del certificado se elige por el N° leído (tolerando O/0 en OCR) o, si no, por la única versión."""
    official_data = {"ameco": {"AMCH-1001": {"Formato": "Rev.03"}, "AMCH-2002": {"Formato": "Rev.02"}}}

    def doc(number, template, is_ocr):
        """Documento con el N° leído."""
        return {"valores": {"numero_certificado": {"texto": number}}, "plantilla": template, "es_ocr": is_ocr}

    assert D._certificate_json(doc("AMCH-1001", "ameco_rev03", False), official_data)[0] == "AMCH-1001"
    assert D._certificate_json(doc("AMCH-1OO1", "ameco_rev03", True), official_data)[0] == "AMCH-1001"
    assert D._certificate_json(doc("ilegible", "ameco_rev02", True), official_data)[0] == "AMCH-2002"
    two_rev03 = {"ameco": {"A-1": {"Formato": "Rev.03"}, "A-2": {"Formato": "Rev.03"}}}
    assert D._certificate_json(doc("", "ameco_rev03", True), two_rev03) == (None, None)


def test_unique_names():
    """Los documentos con el mismo nombre en una carpeta se numeran desde el segundo."""
    rows = PL.unique_names([{"archivo": "x.jpg"}, {"archivo": "x.jpg"}, {"archivo": "y.pdf"}, {"archivo": "x.jpg"}])
    assert [f["archivo"] for f in rows] == ["x.jpg", "x.jpg (2)", "y.pdf", "x.jpg (3)"]


def test_negations():
    """Aparecer o desaparecer un NO, SIN o prefijo negativo cambia el sentido; el ruido pegado a NO no."""
    assert KC.negation_changes("REGISTRA", "NO REGISTRA")
    assert KC.negation_changes("COMPLETA", "INCOMPLETA")
    assert not KC.negation_changes("NO REGISTRA", "NOJ REGISTRA")
    assert not KC.negation_changes("SIN ANTECEDENTES", "SUN ANTECEDENTES")


def test_assign_words_separates_labels_and_orders_lines():
    """Las etiquetas del formato no son valores; el valor de una zona se arma en orden de lectura."""
    template = {"palabras_fijas": [(10, 10, 40, 20, "Nombre")],
                 "zonas": {"nombre": {"rect": [50, 5, 300, 40]}}}
    words = [{"texto": "SOTO", "bbox": [120, 10, 160, 20], "conf": 0.9},
                {"texto": "JUAN", "bbox": [60, 10.5, 100, 20.5], "conf": 0.95},
                {"texto": "Nombre", "bbox": [10, 10, 40, 20]},
                {"texto": "PEREZ", "bbox": [60, 25, 100, 35], "conf": 0.7}]
    r = KC.assign_words(words, template)
    assert r["valores"]["nombre"]["texto"] == "JUAN SOTO PEREZ"
    assert r["valores"]["nombre"]["conf"] == 0.7 and r["cobertura_etiquetas"] == 1.0


def test_font_without_subset_prefix():
    """El prefijo de 6 mayúsculas que agregan los PDF a las fuentes incrustadas no cuenta."""
    assert KC.base_font("ABCDEF+ArialMT") == "ArialMT" and KC.base_font("Courier") == "Courier"


def test_boxes():
    """Centro, contención, intersección e IoU (los elementos extra de la caja se ignoran)."""
    assert boxes.center((0, 0, 10, 20)) == (5, 10)
    assert boxes.inside((5, 5), (0, 0, 10, 10)) and not boxes.inside((11, 5), (0, 0, 10, 10))
    assert boxes.intersects((0, 0, 10, 10), (5, 5, 15, 15)) and not boxes.intersects((0, 0, 10, 10), (10, 0, 20, 10))
    assert boxes.iou((0, 0, 10, 10, 0.9), (0, 0, 10, 10, 0.5)) == 1.0
    assert abs(boxes.iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert boxes.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
