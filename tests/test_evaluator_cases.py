"""Medición de los casos del evaluador con resultados y respuestas inventados (sin datos reales)."""

import json
import zipfile

from audit import evaluator_cases as CE


def _zip(folder, name, authenticity, fitness, documents):
    """Un .zip como el que descarga el visor de pruebas, con solo su resumen.json."""
    with zipfile.ZipFile(folder / f"{name}.zip", "w") as z:
        z.writestr("resumen.json", json.dumps({
            "postulante": {"nombres": name.split()[0], "apellidos": " ".join(name.split()[1:])},
            "autenticidad": authenticity, "aptitud": fitness,
            "documentos": [{"tipo": t, "dictamen": d} for t, d in documents]}))


def test_kit_and_real_cases(tmp_path):
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
    results = CE.read_zips(tmp_path)
    report_lines = CE.compare_kit(results, CE.read_kit(kit))
    assert report_lines[0] == "Kit: 1 de 2 casos iguales a lo esperado"
    assert "igual" in report_lines[1] and "DISTINTO" in report_lines[2]

    truth = tmp_path / "verdad.csv"
    truth.write_text("nombre;tipo;verdad\nANA ROJAS DIAZ;hoja_vida;FALSIFICADO\nANA ROJAS DIAZ;ameco_rev02;GENUINO\n"
                      "JUAN PEREZ SOTO;ameco_rev03;FALSIFICADO\nOTRO NOMBRE;hoja_vida;GENUINO\n", encoding="utf-8")
    lines = CE.measure_real(results, CE.read_truth(truth))
    assert "Falsificaciones: 2 | detectadas 1 | a revisión 0 | escapadas 1" in lines[1]
    assert "Genuinos: 1 | acusados 0 | a revisión 1" in lines[2]
    assert "1 postulante(s)" in lines[3]
