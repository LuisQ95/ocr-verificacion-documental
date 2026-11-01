"""Entradas de imagen del flujo chileno (sin OCR): formatos, páginas múltiples, rotación y tipo de documento.

Se usan las plantillas en blanco de `official_file/` como "foto" del documento. Esas plantillas no se versionan:
si faltan, las pruebas que las necesitan se omiten.
"""

import numpy as np
import pillow_heif
import pymupdf
import pytest
from PIL import Image

from verificacion import config as C
from verificacion import pdf as PDF
from verificacion.chile import lectura_cl as L

PLANTILLAS = ("hoja_vida_conductor.pdf", "ameco_certificado_competencia_rev03.pdf")
requiere_plantillas = pytest.mark.skipif(not all((C.DIR_OFICIAL / p).exists() for p in PLANTILLAS),
                                         reason="sin las plantillas oficiales en official_file/")


@pytest.fixture(scope="module")
def paginas_hv():
    """Las dos páginas de la Hoja de Vida en blanco, rasterizadas a 200 DPI en gris."""
    L.init_worker()
    with pymupdf.open(C.DIR_OFICIAL / "hoja_vida_conductor.pdf") as doc:
        return [PDF.render_gris(doc[i], dpi=200) for i in range(2)]


@pytest.fixture(scope="module")
def ameco_rev03():
    """Certificado AMECO Rev.03 en blanco, rasterizado a 200 DPI en gris."""
    L.init_worker()
    with pymupdf.open(C.DIR_OFICIAL / "ameco_certificado_competencia_rev03.pdf") as doc:
        return PDF.render_gris(doc[0], dpi=200)


@requiere_plantillas
def test_pagina_1_y_2_de_la_hoja_de_vida_se_distinguen(paginas_hv):
    """Ambas páginas comparten la cabecera: el tipo se decide por las etiquetas del cuerpo."""
    assert [L.alinear_documentos(p)[0]["plantilla"] for p in paginas_hv] == ["hoja_vida_p1", "hoja_vida_p2"]


@requiere_plantillas
def test_ameco(ameco_rev03):
    """El certificado AMECO Rev.03 se identifica y se alinea con su plantilla."""
    d = L.alinear_documentos(ameco_rev03)[0]
    assert d["plantilla"] == "ameco_rev03" and d["alineacion_ok"]


@requiere_plantillas
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotaciones(paginas_hv, k):
    """Una página girada 90°, 180° o 270° se endereza antes de alinear."""
    d = L.alinear_documentos(np.ascontiguousarray(np.rot90(paginas_hv[0], k=k)))[0]
    assert d["plantilla"] == "hoja_vida_p1" and d["alineacion_ok"] and d["rotacion_aplicada"] != 0


@requiere_plantillas
def test_imagen_heic(tmp_path, paginas_hv):
    """Las fotos HEIC (iPhone) se abren igual que cualquier imagen."""
    pillow_heif.register_heif_opener()
    ruta = tmp_path / "foto.heic"
    Image.fromarray(paginas_hv[0]).convert("RGB").save(ruta, quality=90)
    img = L.cargar_imagen(ruta)
    assert img.shape == paginas_hv[0].shape
    assert L.alinear_documentos(img)[0]["plantilla"] == "hoja_vida_p1"


@requiere_plantillas
def test_tiff_de_varias_paginas(tmp_path, paginas_hv):
    """Un TIFF de varias páginas se separa en una unidad por página."""
    ruta = tmp_path / "escaneo.tiff"
    p1, p2 = (Image.fromarray(p) for p in paginas_hv)
    p1.save(ruta, save_all=True, append_images=[p2])
    unidades = L.unidades(ruta)
    assert [(u[1], u[2]) for u in unidades] == [(0, 2), (1, 2)]
    assert np.array_equal(L.cargar_imagen(ruta, 1), paginas_hv[1])


@requiere_plantillas
def test_imagen_sin_documento_no_se_identifica():
    """Una imagen que no es ningún documento conocido no se alinea con ninguna plantilla."""
    ruido = np.random.default_rng(0).integers(0, 255, (1800, 1300), dtype=np.uint8)
    docs = L.alinear_documentos(ruido)
    assert not docs or not docs[0]["alineacion_ok"]


def test_imagen_demasiado_grande_se_rechaza(tmp_path):
    """Una imagen "bomba" (pocos KB en disco, más de MAX_PIXELES_IMAGEN al abrirse) se rechaza sin decodificarla, y
    el lote sigue: el error queda en la unidad."""
    ruta = tmp_path / "bomba.png"
    Image.new("1", (12_000, 11_000), 1).save(ruta)
    with pytest.raises(ValueError, match="demasiado grande"):
        L.cargar_imagen(ruta)
    assert "demasiado grande" in str(L.analizar((ruta, 0, 1))["error"])


def test_pagina_pdf_gigante_se_rechaza(tmp_path):
    """Una página PDF de varios metros no se renderiza: queda como error de lectura."""
    ruta = tmp_path / "gigante.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=10_000, height=10_000)
        doc.save(ruta)
    assert "demasiado grande" in str(L.analizar((ruta, 0, 1))["error"])


@requiere_plantillas
def test_limite_de_paginas_en_el_visor_de_pruebas(tmp_path):
    """Un archivo con más páginas que MAX_PAGINAS_ARCHIVO no se lee: queda entre los no admitidos."""
    from verificacion.chile import memoria_cl as M
    buf = tmp_path / "muchas.tiff"
    paginas = [Image.new("L", (100, 100), 255) for _ in range(M.MAX_PAGINAS_ARCHIVO + 1)]
    paginas[0].save(buf, save_all=True, append_images=paginas[1:])
    ident, _, _ = M.validar_identidad("JUAN", "PEREZ SOTO", "12.345.678-5")
    r = M.verificar_en_memoria(ident, [("muchas.tiff", buf.read_bytes())], [], usar_gpu=False)
    assert [x["archivo"] for x in r["no_admitidos"]] == ["muchas.tiff"]
    assert "páginas" in r["no_admitidos"][0]["motivo"] and not r["documentos"]


def test_timbre_electronico():
    """El timbre electrónico (XML del código de barras) se separa en sus campos."""
    t = L.parsear_timbre("<TED><DD><FOLIO>500012345678</FOLIO><RUN>12345678-5</RUN><NOM>JUAN PEREZ</NOM>"
                         "<FE>2026-10-01T10:00:00</FE><CV>ABC123</CV></DD></TED>")
    assert t == {"FOLIO": "500012345678", "RUN": "12345678-5", "NOM": "JUAN PEREZ", "FE": "2026-10-01T10:00:00",
                 "CV": "ABC123"}
