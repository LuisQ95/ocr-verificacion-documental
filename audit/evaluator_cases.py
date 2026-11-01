"""Medición con los casos del evaluador externo: compara los resultados que entrega el visor de pruebas (los .zip
descargados) con lo esperado. Solo mide: nada de lo que se lee aquí alimenta reglas ni modelos.

Fuentes de lo esperado (archivos locales, nunca versionados porque pueden tener datos de personas):
  --kit       evaluation_viewer/kit_answers.md: dictamen esperado de cada documento de los casos del kit.
  --verdad    CSV con la verdad de casos reales, separado por ';' y con encabezado:
                  nombre;tipo;verdad
                  JUAN PEREZ SOTO;hoja_vida;GENUINO
                  JUAN PEREZ SOTO;ameco_rev03;FALSIFICADO
              tipo: hoja_vida, ameco_rev02 o ameco_rev03; verdad: GENUINO o FALSIFICADO.

Uso:
  python -m auditoria.evaluator_cases --zips <carpeta con los .zip> [--kit evaluation_viewer/kit_answers.md]
                                      [--truth "DATOS REALES/casos_evaluador.csv"]
"""

import argparse
import csv
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

GENUINE, FAKE = "GENUINO", "FALSIFICADO"
REVIEW = {"OBSERVADO", "NO LEGIBLE"}
"""Dictámenes que mandan el documento a revisión humana (no acusan ni aprueban)."""


def key(name):
    """Nombre en mayúsculas, sin tildes y con un espacio entre palabras (para cruzar fuentes)."""
    t = "".join(c for c in unicodedata.normalize("NFD", name or "") if unicodedata.category(c) != "Mn")
    return " ".join(t.upper().split())


def base_type(kind):
    """hoja_vida_p1 / hoja_vida_p2 -> hoja_vida; los demás tipos quedan igual."""
    return "hoja_vida" if (kind or "").startswith("hoja_vida") else kind


def read_zips(folder):
    """{nombre del postulante: resumen.json} de cada .zip del visor de pruebas (el más reciente gana)."""
    out = {}
    for z in sorted(Path(folder).glob("*.zip"), key=lambda p: p.stat().st_mtime):
        with zipfile.ZipFile(z) as f:
            if "resumen.json" not in f.namelist():
                continue
            r = json.loads(f.read("resumen.json"))
        p = r.get("postulante", {})
        out[key(f"{p.get('nombres', '')} {p.get('apellidos', '')}")] = r | {"_zip": z.name}
    return out


def read_kit(path):
    """{nombre: {"caso", "documentos": Counter((tipo, dictamen)), "autenticidad", "aptitud"}} de kit_answers.md."""
    cases, current = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = {"caso": line[4:].strip(), "documentos": Counter()}
        elif current and "Postulante sintético:" in line:
            cases[key(line.split(":", 1)[1])] = current
        elif current and line.startswith("- Dictamen por documento"):
            for kind, ruling in re.findall(r"(\w+): \*\*([^*]+)\*\*", line.split(":", 1)[1]):
                current["documentos"][(base_type(kind), ruling)] += 1
        elif current and line.startswith("- Autenticidad:"):
            m = re.search(r"Autenticidad: \*\*([^*]+)\*\* · Aptitud: \*\*([^*]+)\*\*", line)
            if m:
                current["autenticidad"], current["aptitud"] = m.groups()
    return cases


def read_truth(path):
    """{nombre: [(tipo, verdad)]} del CSV de casos reales."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            truth = row["verdad"].strip().upper()
            if truth not in (GENUINE, FAKE):
                raise ValueError(f"verdad desconocida: {row['verdad']!r} (use GENUINO o FALSIFICADO)")
            out.setdefault(key(row["nombre"]), []).append((base_type(row["tipo"].strip()), truth))
    return out


def compare_kit(results, kit):
    """Líneas de informe: cada caso del kit, si coincide con lo esperado y en qué difiere."""
    lines, equal_n = [], 0
    for name, exp_ in sorted(kit.items(), key=lambda x: x[1]["caso"]):
        r = results.get(name)
        if r is None:
            lines.append(f"  {exp_['caso']}: sin resultado del evaluador")
            continue
        obtained = Counter((base_type(d["tipo"]), d["dictamen"]) for d in r["documentos"])
        delta = []
        if obtained != exp_["documentos"]:
            delta.append(f"documentos {dict(obtained)} (esperado {dict(exp_['documentos'])})")
        for field in ("autenticidad", "aptitud"):
            if exp_.get(field) and r.get(field) != exp_[field]:
                delta.append(f"{field} {r.get(field)} (esperado {exp_[field]})")
        equal_n += not delta
        lines.append(f"  {exp_['caso']}: " + ("igual" if not delta else "DISTINTO: " + "; ".join(delta)))
    return [f"Kit: {equal_n} de {len(kit)} casos iguales a lo esperado"] + lines


def measure_real(results, truth):
    """Métricas por documento de los casos reales: falsificaciones detectadas, a revisión y escapadas; genuinos
    acusados y a revisión. Cada documento esperado se cruza con un documento del mismo tipo del resultado."""
    m = Counter()
    no_result = []
    for name, docs in truth.items():
        r = results.get(name)
        if r is None:
            no_result.append(name)
            continue
        available = [(base_type(d["tipo"]), d["dictamen"]) for d in r["documentos"]]
        for kind, v in docs:
            ruling = next((d for t, d in available if t == kind), None)
            if ruling is None:
                m[(v, "no encontrado")] += 1
                continue
            available.remove((kind, ruling))
            if ruling == "FALSIFICADO":
                m[(v, "acusado")] += 1
            elif ruling in REVIEW:
                m[(v, "a revisión")] += 1
            else:
                m[(v, "aprobado")] += 1
    forged_n, gen = sum(n for (v, _), n in m.items() if v == FAKE), sum(n for (v, _), n in m.items() if v == GENUINE)
    lines = ["Casos reales (por documento):",
              f"  Falsificaciones: {forged_n} | detectadas {m[(FAKE, 'acusado')]} | a revisión {m[(FAKE, 'a revisión')]} | "
              f"escapadas {m[(FAKE, 'aprobado')]} | no encontradas {m[(FAKE, 'no encontrado')]}",
              f"  Genuinos: {gen} | acusados {m[(GENUINE, 'acusado')]} | a revisión {m[(GENUINE, 'a revisión')]} | "
              f"aprobados {m[(GENUINE, 'aprobado')]} | no encontrados {m[(GENUINE, 'no encontrado')]}"]
    if no_result:
        lines.append(f"  Sin resultado del evaluador: {len(no_result)} postulante(s)")
    return lines


def main():
    """Línea de comandos."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="carpeta con los .zip descargados del visor de pruebas")
    ap.add_argument("--kit", default=None, help="kit_answers.md")
    ap.add_argument("--truth", default=None, help="CSV nombre;tipo;verdad de los casos reales")
    a = ap.parse_args()
    results = read_zips(a.zips)
    print(f"{len(results)} resultado(s) leídos de {a.zips}")
    if a.kit:
        print("\n".join(compare_kit(results, read_kit(a.kit))))
    if a.truth:
        print("\n".join(measure_real(results, read_truth(a.truth))))


if __name__ == "__main__":
    main()
