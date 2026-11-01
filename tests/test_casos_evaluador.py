"""Medición de los casos del evaluador con resultados y respuestas inventados (sin datos reales)."""

import json
import zipfile

from auditoria import casos_evaluador as CE


def _zip(carpeta, nombre, autenticidad, aptitud, documentos):
    """Un .zip como el que descarga el visor de pruebas, con solo su resumen.json."""
    with zipfile.ZipFile(carpeta / f"{nombre}.zip", "w") as z:
        z.writestr("resumen.json", json.dumps({
            "postulante": {"nombres": nombre.split()[0], "apellidos": " ".join(nombre.split()[1:])},
            "autenticidad": autenticidad, "aptitud": aptitud,
            "documentos": [{"tipo": t, "dictamen": d} for t, d in documentos]}))


def test_kit_y_casos_reales(tmp_path):
    """El kit se compara caso a caso; los casos reales dan detección, revisión y acusaciones por documento."""
    _zip(tmp_path, "JUAN PEREZ SOTO", "AUTÉNTICO", "APTO", [("hoja_vida_p1", "AUTÉNTICO"), ("ameco_rev03", "AUTÉNTICO")])
    _zip(tmp_path, "ANA ROJAS DIAZ", "FALSIFICADO", "NO APTO", [("hoja_vida_p1", "FALSIFICADO"), ("ameco_rev02", "OBSERVADO")])
    kit = tmp_path / "kit.md"
    kit.write_text("### Caso 01 — Todo auténtico\n- Postulante sintético: Juan Pérez Soto\n"
                   "- Dictamen por documento (hoy): ameco_rev03: **AUTÉNTICO**; hoja_vida_p1: **AUTÉNTICO**\n"
                   "- Autenticidad: **AUTÉNTICO** · Aptitud: **APTO**\n"
                   "### Caso 02 — Falsificado\n- Postulante sintético: ANA ROJAS DIAZ\n"
                   "- Dictamen por documento (hoy): ameco_rev02: **AUTÉNTICO**; hoja_vida_p1: **FALSIFICADO**\n"
                   "- Autenticidad: **FALSIFICADO** · Aptitud: **NO APTO**\n", encoding="utf-8")
    resultados = CE.leer_zips(tmp_path)
    informe = CE.comparar_kit(resultados, CE.leer_kit(kit))
    assert informe[0] == "Kit: 1 de 2 casos iguales a lo esperado"
    assert "igual" in informe[1] and "DISTINTO" in informe[2]

    verdad = tmp_path / "verdad.csv"
    verdad.write_text("nombre;tipo;verdad\nANA ROJAS DIAZ;hoja_vida;FALSIFICADO\nANA ROJAS DIAZ;ameco_rev02;GENUINO\n"
                      "JUAN PEREZ SOTO;ameco_rev03;FALSIFICADO\nOTRO NOMBRE;hoja_vida;GENUINO\n", encoding="utf-8")
    lineas = CE.medir_reales(resultados, CE.leer_verdad(verdad))
    assert "Falsificaciones: 2 | detectadas 1 | a revisión 0 | escapadas 1" in lineas[1]
    assert "Genuinos: 1 | acusados 0 | a revisión 1" in lineas[2]
    assert "1 postulante(s)" in lineas[3]
