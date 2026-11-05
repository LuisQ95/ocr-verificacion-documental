"""Entradas de imagen (sin OCR): formatos, páginas múltiples, rotación y tipo de documento.

Se usan las plantillas en blanco de `official_file/` como "foto" del documento. Esas plantillas no se versionan:
si faltan, las pruebas que las necesitan se omiten.
"""

import numpy as np
import pillow_heif
import pymupdf
import pytest
from PIL import Image

from verification import config as C
from verification import pdf as PDF
from verification import reading as L
from verification.profiles.operator_certification import normalize as N

TEMPLATES = ("hoja_vida_conductor.pdf", "ameco_certificado_competencia_rev03.pdf")
requires_templates = pytest.mark.skipif(not all((C.TEMPLATES_DIR / p).exists() for p in TEMPLATES),
                                         reason="sin las plantillas oficiales en official_file/")


@pytest.fixture(scope="module")
def record_pages():
    """Las dos páginas de la Hoja de Vida en blanco, rasterizadas a 200 DPI en gris."""
    L.init_worker()
    with pymupdf.open(C.TEMPLATES_DIR / "hoja_vida_conductor.pdf") as doc:
        return [PDF.render_gray(doc[i], dpi=200) for i in range(2)]


@pytest.fixture(scope="module")
def certificate_rev03():
    """Certificado AMECO Rev.03 en blanco, rasterizado a 200 DPI en gris."""
    L.init_worker()
    with pymupdf.open(C.TEMPLATES_DIR / "ameco_certificado_competencia_rev03.pdf") as doc:
        return PDF.render_gray(doc[0], dpi=200)


@requires_templates
def test_record_pages_1_and_2_are_distinguished(record_pages):
    """Ambas páginas comparten la cabecera: el tipo se decide por las etiquetas del cuerpo."""
    assert [L.align_documents(p)[0]["plantilla"] for p in record_pages] == ["hoja_vida_p1", "hoja_vida_p2"]


@requires_templates
def test_certificate(certificate_rev03):
    """El certificado AMECO Rev.03 se identifica y se alinea con su plantilla."""
    d = L.align_documents(certificate_rev03)[0]
    assert d["plantilla"] == "ameco_rev03" and d["alineacion_ok"]


@requires_templates
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotations(record_pages, k):
    """Una página girada 90°, 180° o 270° se endereza antes de alinear."""
    d = L.align_documents(np.ascontiguousarray(np.rot90(record_pages[0], k=k)))[0]
    assert d["plantilla"] == "hoja_vida_p1" and d["alineacion_ok"] and d["rotacion_aplicada"] != 0


@requires_templates
def test_heic_image(tmp_path, record_pages):
    """Las fotos HEIC (iPhone) se abren igual que cualquier imagen."""
    pillow_heif.register_heif_opener()
    path = tmp_path / "foto.heic"
    Image.fromarray(record_pages[0]).convert("RGB").save(path, quality=90)
    img = L.load_image(path)
    assert img.shape == record_pages[0].shape
    assert L.align_documents(img)[0]["plantilla"] == "hoja_vida_p1"


@requires_templates
def test_multipage_tiff(tmp_path, record_pages):
    """Un TIFF de varias páginas se separa en una unidad por página."""
    path = tmp_path / "escaneo.tiff"
    p1, p2 = (Image.fromarray(p) for p in record_pages)
    p1.save(path, save_all=True, append_images=[p2])
    units = L.units(path)
    assert [(u[1], u[2]) for u in units] == [(0, 2), (1, 2)]
    assert np.array_equal(L.load_image(path, 1), record_pages[1])


@requires_templates
def test_image_without_document_is_not_identified():
    """Una imagen que no es ningún documento conocido no se alinea con ninguna plantilla."""
    noise = np.random.default_rng(0).integers(0, 255, (1800, 1300), dtype=np.uint8)
    docs = L.align_documents(noise)
    assert not docs or not docs[0]["alineacion_ok"]


def test_too_large_image_is_rejected(tmp_path):
    """Una imagen "bomba" (pocos KB en disco, más de MAX_IMAGE_PIXELS al abrirse) se rechaza sin decodificarla, y
    el lote sigue: el error queda en la unidad."""
    path = tmp_path / "bomba.png"
    Image.new("1", (12_000, 11_000), 1).save(path)
    with pytest.raises(ValueError, match="demasiado grande"):
        L.load_image(path)
    assert "demasiado grande" in str(L.analyze((path, 0, 1))["error"])


def test_giant_pdf_page_is_rejected(tmp_path):
    """Una página PDF de varios metros no se renderiza: queda como error de lectura."""
    path = tmp_path / "gigante.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=10_000, height=10_000)
        doc.save(path)
    assert "demasiado grande" in str(L.analyze((path, 0, 1))["error"])


@requires_templates
def test_page_limit_in_evaluation_viewer(tmp_path):
    """Un archivo con más páginas que MAX_PAGES_PER_FILE no se lee: queda entre los no admitidos."""
    from verification import in_memory as M
    buf = tmp_path / "muchas.tiff"
    pages = [Image.new("L", (100, 100), 255) for _ in range(M.MAX_PAGES_PER_FILE + 1)]
    pages[0].save(buf, save_all=True, append_images=pages[1:])
    identity, _, _ = M.validate_identity("JUAN", "PEREZ SOTO", "12.345.678-5")
    r = M.verify_in_memory(identity, [("muchas.tiff", buf.read_bytes())], [], use_gpu=False)
    assert [x["archivo"] for x in r["no_admitidos"]] == ["muchas.tiff"]
    assert "páginas" in r["no_admitidos"][0]["motivo"] and not r["documentos"]


def test_electronic_seal():
    """El timbre electrónico (XML del código de barras) se separa en sus campos."""
    t = N.parse_seal("<TED><DD><FOLIO>500012345678</FOLIO><RUN>12345678-5</RUN><NOM>JUAN PEREZ</NOM>"
                         "<FE>2026-10-01T10:00:00</FE><CV>ABC123</CV></DD></TED>")
    assert t == {"FOLIO": "500012345678", "RUN": "12345678-5", "NOM": "JUAN PEREZ", "FE": "2026-10-01T10:00:00",
                 "CV": "ABC123"}
