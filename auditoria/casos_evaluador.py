"""Medición con los casos del evaluador externo: compara los resultados que entrega el visor de pruebas (los .zip
descargados) con lo esperado. Solo mide: nada de lo que se lee aquí alimenta reglas ni modelos.

Fuentes de lo esperado (archivos locales, nunca versionados porque pueden tener datos de personas):
  --kit       visor_pruebas/kit_respuestas.md: dictamen esperado de cada documento de los casos del kit.
  --verdad    CSV con la verdad de casos reales, separado por ';' y con encabezado:
                  nombre;tipo;verdad
                  JUAN PEREZ SOTO;hoja_vida;GENUINO
                  JUAN PEREZ SOTO;ameco_rev03;FALSIFICADO
              tipo: hoja_vida, ameco_rev02 o ameco_rev03; verdad: GENUINO o FALSIFICADO.

Uso:
  python -m auditoria.casos_evaluador --zips <carpeta con los .zip> [--kit visor_pruebas/kit_respuestas.md]
                                      [--verdad "DATOS REALES/casos_evaluador.csv"]
"""

import argparse
import csv
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

GENUINO, FALSO = "GENUINO", "FALSIFICADO"
REVISION = {"OBSERVADO", "NO LEGIBLE"}
"""Dictámenes que mandan el documento a revisión humana (no acusan ni aprueban)."""


def clave(nombre):
    """Nombre en mayúsculas, sin tildes y con un espacio entre palabras (para cruzar fuentes)."""
    t = "".join(c for c in unicodedata.normalize("NFD", nombre or "") if unicodedata.category(c) != "Mn")
    return " ".join(t.upper().split())


def tipo_base(tipo):
    """hoja_vida_p1 / hoja_vida_p2 -> hoja_vida; los demás tipos quedan igual."""
    return "hoja_vida" if (tipo or "").startswith("hoja_vida") else tipo


def leer_zips(carpeta):
    """{nombre del postulante: resumen.json} de cada .zip del visor de pruebas (el más reciente gana)."""
    out = {}
    for z in sorted(Path(carpeta).glob("*.zip"), key=lambda p: p.stat().st_mtime):
        with zipfile.ZipFile(z) as f:
            if "resumen.json" not in f.namelist():
                continue
            r = json.loads(f.read("resumen.json"))
        p = r.get("postulante", {})
        out[clave(f"{p.get('nombres', '')} {p.get('apellidos', '')}")] = r | {"_zip": z.name}
    return out


def leer_kit(ruta):
    """{nombre: {"caso", "documentos": Counter((tipo, dictamen)), "autenticidad", "aptitud"}} de kit_respuestas.md."""
    casos, actual = {}, {}
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if linea.startswith("### "):
            actual = {"caso": linea[4:].strip(), "documentos": Counter()}
        elif actual and "Postulante sintético:" in linea:
            casos[clave(linea.split(":", 1)[1])] = actual
        elif actual and linea.startswith("- Dictamen por documento"):
            for tipo, dictamen in re.findall(r"(\w+): \*\*([^*]+)\*\*", linea.split(":", 1)[1]):
                actual["documentos"][(tipo_base(tipo), dictamen)] += 1
        elif actual and linea.startswith("- Autenticidad:"):
            m = re.search(r"Autenticidad: \*\*([^*]+)\*\* · Aptitud: \*\*([^*]+)\*\*", linea)
            if m:
                actual["autenticidad"], actual["aptitud"] = m.groups()
    return casos


def leer_verdad(ruta):
    """{nombre: [(tipo, verdad)]} del CSV de casos reales."""
    out = {}
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f, delimiter=";"):
            verdad = fila["verdad"].strip().upper()
            if verdad not in (GENUINO, FALSO):
                raise ValueError(f"verdad desconocida: {fila['verdad']!r} (use GENUINO o FALSIFICADO)")
            out.setdefault(clave(fila["nombre"]), []).append((tipo_base(fila["tipo"].strip()), verdad))
    return out


def comparar_kit(resultados, kit):
    """Líneas de informe: cada caso del kit, si coincide con lo esperado y en qué difiere."""
    lineas, iguales = [], 0
    for nombre, esp in sorted(kit.items(), key=lambda x: x[1]["caso"]):
        r = resultados.get(nombre)
        if r is None:
            lineas.append(f"  {esp['caso']}: sin resultado del evaluador")
            continue
        obtenidos = Counter((tipo_base(d["tipo"]), d["dictamen"]) for d in r["documentos"])
        dif = []
        if obtenidos != esp["documentos"]:
            dif.append(f"documentos {dict(obtenidos)} (esperado {dict(esp['documentos'])})")
        for campo in ("autenticidad", "aptitud"):
            if esp.get(campo) and r.get(campo) != esp[campo]:
                dif.append(f"{campo} {r.get(campo)} (esperado {esp[campo]})")
        iguales += not dif
        lineas.append(f"  {esp['caso']}: " + ("igual" if not dif else "DISTINTO: " + "; ".join(dif)))
    return [f"Kit: {iguales} de {len(kit)} casos iguales a lo esperado"] + lineas


def medir_reales(resultados, verdad):
    """Métricas por documento de los casos reales: falsificaciones detectadas, a revisión y escapadas; genuinos
    acusados y a revisión. Cada documento esperado se cruza con un documento del mismo tipo del resultado."""
    m = Counter()
    sin_resultado = []
    for nombre, docs in verdad.items():
        r = resultados.get(nombre)
        if r is None:
            sin_resultado.append(nombre)
            continue
        disponibles = [(tipo_base(d["tipo"]), d["dictamen"]) for d in r["documentos"]]
        for tipo, v in docs:
            dictamen = next((d for t, d in disponibles if t == tipo), None)
            if dictamen is None:
                m[(v, "no encontrado")] += 1
                continue
            disponibles.remove((tipo, dictamen))
            if dictamen == "FALSIFICADO":
                m[(v, "acusado")] += 1
            elif dictamen in REVISION:
                m[(v, "a revisión")] += 1
            else:
                m[(v, "aprobado")] += 1
    fal, gen = sum(n for (v, _), n in m.items() if v == FALSO), sum(n for (v, _), n in m.items() if v == GENUINO)
    lineas = ["Casos reales (por documento):",
              f"  Falsificaciones: {fal} | detectadas {m[(FALSO, 'acusado')]} | a revisión {m[(FALSO, 'a revisión')]} | "
              f"escapadas {m[(FALSO, 'aprobado')]} | no encontradas {m[(FALSO, 'no encontrado')]}",
              f"  Genuinos: {gen} | acusados {m[(GENUINO, 'acusado')]} | a revisión {m[(GENUINO, 'a revisión')]} | "
              f"aprobados {m[(GENUINO, 'aprobado')]} | no encontrados {m[(GENUINO, 'no encontrado')]}"]
    if sin_resultado:
        lineas.append(f"  Sin resultado del evaluador: {len(sin_resultado)} postulante(s)")
    return lineas


def main():
    """Línea de comandos."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="carpeta con los .zip descargados del visor de pruebas")
    ap.add_argument("--kit", default=None, help="kit_respuestas.md")
    ap.add_argument("--verdad", default=None, help="CSV nombre;tipo;verdad de los casos reales")
    a = ap.parse_args()
    resultados = leer_zips(a.zips)
    print(f"{len(resultados)} resultado(s) leídos de {a.zips}")
    if a.kit:
        print("\n".join(comparar_kit(resultados, leer_kit(a.kit))))
    if a.verdad:
        print("\n".join(medir_reales(resultados, leer_verdad(a.verdad))))


if __name__ == "__main__":
    main()
