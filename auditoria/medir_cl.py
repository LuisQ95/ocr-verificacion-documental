"""Medición SOLO DE LECTURA del flujo chileno contra ground_truth_v2 (nunca alimenta reglas ni modelos).

Empareja cada documento del ground truth (persona + tipo) con el dictamen del pipeline (mismo archivo y tipo) y
reporta: falsificaciones detectadas, genuinos acusados, y el tratamiento de cada escenario (ilegible, copia,
desactualizada, documento de otra persona...).

Uso: python -m auditoria.medir_cl --resultados resultados/chile [--subconjunto ajuste|reservado] [--detalle]
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIPO = {"hoja_vida_p1": "hoja_vida", "hoja_vida_p2": "hoja_vida", "ameco_rev02": "rev02", "ameco_rev03": "rev03"}


def wilson(k, n, z=1.96):
    """Intervalo de confianza de Wilson (95 % por defecto) para la proporción k/n."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - m), min(1, c + m)


def pct(k, n):
    """"k/n = p% (IC95 a%–b%)"."""
    a, b = wilson(k, n)
    return f"{k}/{n} = {k / max(1, n):.1%} (IC95 {a:.1%}–{b:.1%})"


def subconjunto(n, cual):
    """True si el postulante N pertenece al subconjunto: impares = ajuste, pares = reservado (None = todos)."""
    return cual is None or (cual == "ajuste") == (n % 2 == 1)


def main():
    """Empareja ground truth y dictámenes y reporta detección, falsas alarmas, técnicas, canales y escenarios.

    Si un documento no se identificó (ilegible sin tipo o mal clasificado), se busca su dictamen como documento
    "desconocido". Una "falsificación" del generador sin ningún cambio real (p. ej. agregar una clase que ya tenía)
    es indistinguible del original: se excluye de la medición y se informa aparte.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default=str(ROOT / "resultados" / "chile"))
    ap.add_argument("--subconjunto", choices=["ajuste", "reservado"], default=None)
    ap.add_argument("--detalle", action="store_true")
    a = ap.parse_args()
    gt = [json.loads(l) for l in open(ROOT / "ground_truth_v2" / "ground_truth.jsonl", encoding="utf-8")]
    res = [json.loads(l) for l in open(Path(a.resultados) / "resultados.jsonl", encoding="utf-8")]
    por_carpeta = defaultdict(list)
    for r in res:
        por_carpeta[r["carpeta"]].append(r)
    carpetas = set(por_carpeta)
    filas = []
    for g in gt:
        if g["carpeta"] not in carpetas or not subconjunto(g["n"], a.subconjunto) or not g["entregado"]:
            continue
        cand = [r for r in por_carpeta[g["carpeta"]] if TIPO.get(r["tipo_documento"]) == g["documento"]
                and set(r["archivos"]) & set(g["archivos"])]
        if not cand:
            cand = [r for r in por_carpeta[g["carpeta"]] if set(r["archivos"]) & set(g["archivos"])
                    and r["tipo_documento"] in ("desconocido", None)]
        v = cand[0]["veredicto"] if cand else "SIN DICTAMEN"
        filas.append((g, v, cand[0] if cand else None))
    sin_cambio = [(g, v, r) for g, v, r in filas if g["falsificado"] and not g["campos_modificados"]
                  and g["tipo_falsificacion"] == "contenido"]
    falsos = [(g, v, r) for g, v, r in filas if g["falsificado"] and (g, v, r) not in sin_cambio]
    genuinos = [(g, v, r) for g, v, r in filas if not g["falsificado"]]
    if sin_cambio:
        print(f"(excluidas {len(sin_cambio)} falsificaciones del generador sin ningún cambio real)")
    print(f"Documentos medidos: {len(filas)} ({len(falsos)} falsificados, {len(genuinos)} genuinos) | "
          f"subconjunto: {a.subconjunto or 'todos'}")
    print("FALSIFICADOS")
    print("  detectados (FALSIFICADO)          ", pct(sum(v == "FALSIFICADO" for _, v, _ in falsos), len(falsos)))
    print("  detectados o a revisión (no AUTÉNTICO)", pct(sum(v != "AUTÉNTICO" for _, v, _ in falsos), len(falsos)))
    print("  escapados (AUTÉNTICO)             ", pct(sum(v == "AUTÉNTICO" for _, v, _ in falsos), len(falsos)))
    print("GENUINOS")
    print("  acusados (FALSIFICADO)            ", pct(sum(v == "FALSIFICADO" for _, v, _ in genuinos), len(genuinos)))
    print("  a revisión (OBSERVADO)            ", pct(sum(v == "OBSERVADO" for _, v, _ in genuinos), len(genuinos)))
    print("  dictámenes de genuinos:", dict(Counter(v for _, v, _ in genuinos).most_common()))
    print("\nPor técnica de falsificación (detectados / total):")
    por = defaultdict(lambda: [0, 0])
    for g, v, _ in falsos:
        k = f"{g['tipo_falsificacion']}/{g['tecnica'] or g['motivo']}"
        por[k][0] += v == "FALSIFICADO"
        por[k][1] += 1
    for k, (d, n) in sorted(por.items(), key=lambda x: -x[1][1]):
        print(f"  {k:45} {d}/{n}")
    print("\nPor canal (falsos detectados | genuinos acusados):")
    for canal in sorted({g["subcanal"] or "?" for g, _, _ in filas}):
        f = [v for g, v, _ in falsos if g["subcanal"] == canal]
        gg = [v for g, v, _ in genuinos if g["subcanal"] == canal]
        print(f"  {canal:22} falsos {sum(x == 'FALSIFICADO' for x in f)}/{len(f)} | genuinos acusados "
              f"{sum(x == 'FALSIFICADO' for x in gg)}/{len(gg)} | a revisión {sum(x == 'OBSERVADO' for x in gg)}/{len(gg)}")
    print("\nEscenarios en genuinos (dictamen obtenido):")
    esc = defaultdict(Counter)
    for g, v, _ in genuinos:
        for e in g["escenarios"]:
            esc[e][v] += 1
    for e, c in sorted(esc.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {e:34} {dict(c)}")
    if a.detalle:
        print("\nDETALLE de errores (falsos escapados y genuinos no auténticos):")
        for g, v, r in filas:
            mal = (g["falsificado"] and v == "AUTÉNTICO") or (not g["falsificado"] and v not in ("AUTÉNTICO", "DESACTUALIZADO", "NO VÁLIDO"))
            if mal:
                print(f"- {g['carpeta']} | {g['documento']} | {g['subcanal']} | falso={g['falsificado']} "
                      f"({g['motivo']}/{g['tecnica']}) | escenarios={g['escenarios']} -> {v}")
                for m in (r or {}).get("motivos", [])[:6]:
                    print(f"     · {m[:220]}")


if __name__ == "__main__":
    main()
