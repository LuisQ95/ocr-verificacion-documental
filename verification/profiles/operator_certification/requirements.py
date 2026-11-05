"""Aptitud del postulante según requirements.yaml, con los datos oficiales (JSON) de los documentos que
presentó y que resultaron auténticos."""

from datetime import date
from pathlib import Path

import yaml

from ...fields import FIT, INCOMPLETE, NOT_EVALUABLE, NOT_FIT
from . import rules as K
from .normalize import text_key, date_value, strip_accents

CONFIG_PATH = Path(__file__).with_name("requirements.yaml")
"""Requisitos de aptitud por cargo."""


def load(path=CONFIG_PATH):
    """Requisitos por cargo desde el YAML."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _plus_years(d, n):
    """La fecha `d` desplazada `n` años (un 29 de febrero pasa al 28 si el año no es bisiesto)."""
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def _license(js, req, today):
    """Motivos de NO APTO por licencia y antecedentes (datos oficiales actuales de la Hoja de Vida)."""
    reasons = []
    latest = {}
    for l in js.get("Licencias registradas", []):
        if l["Registro"] == "ULTIMA":
            f = date_value(l["Fecha otorgamiento"])
            if f and (l["Clase"] not in latest or f > latest[l["Clase"]]):
                latest[l["Clase"]] = f
    rl = req["licencia"]
    for c in rl.get("todas", []):
        if c not in latest:
            reasons.append(f"no tiene licencia clase {c}")
    if rl.get("alguna") and not any(c in latest for c in rl["alguna"]):
        reasons.append(f"no tiene ninguna clase profesional requerida ({', '.join(rl['alguna'])})")
    if rl.get("vigente"):
        required_classes = [c for c in rl.get("todas", []) + rl.get("alguna", []) if c in latest]
        for c in required_classes:
            years = rl["anios_control_profesional"] if c.startswith("A") else rl["anios_control_otras"]
            if _plus_years(latest[c], years) < today:
                reasons.append(f"licencia clase {c} con control vencido (última renovación {latest[c]})")
    ra = req["antecedentes"]
    limit = _plus_years(today, -ra["sin_suspensiones_anios"])
    for a in js.get("Anotaciones", []):
        t = strip_accents(a["Tipo"]).upper()
        f = date_value(a["Fecha anotación"])
        if ("SUSPENSION" in t or "CANCELACION" in t or "GRAVISIMA" in t) and f and f >= limit:
            reasons.append(f"{a['Tipo'].title()} el {f} (últimos {ra['sin_suspensiones_anios']} años)")
    if ra.get("rnc_sin_antecedentes") and js.get("Antecedentes", {}).get("RNC") != "SIN ANTECEDENTES":
        reasons.append(f"Registro Nacional de Conductores: {js.get('Antecedentes', {}).get('RNC', 'sin dato')}")
    return reasons


def evaluate_person(position, documents, official_data, req, today):
    """documentos: filas dictaminadas del postulante; oficiales: {'hoja_vida': json, 'ameco': {N°: json}}.

    Devuelve (aptitud, motivos). La Hoja de Vida puede venir como hoja_vida_p1 o hoja_vida_p2. El cargo se compara
    sin mayúsculas ni tildes; un cargo no configurado acepta un certificado AMECO de cualquier equipo.
    """
    missing_n, not_fit, review = [], [], []
    rec = [d for d in documents if d["tipo_documento"].startswith("hoja_vida")]
    rec_ok = [d for d in rec if d["veredicto"] in (K.AUTHENTIC,)]
    if not rec:
        missing_n.append("no presentó la Hoja de Vida del Conductor")
    elif not rec_ok:
        v = {d["veredicto"] for d in rec}
        if K.FORGED in v:
            missing_n.append("la Hoja de Vida presentada está falsificada")
        elif K.NOT_VALID in v:
            missing_n.append("presentó una COPIA no válida de la Hoja de Vida: se requiere el original")
        elif K.OUTDATED in v:
            missing_n.append(f"la Hoja de Vida tiene más de {req['dias_vigencia_hoja_vida']} días: presentar una actualizada")
        elif K.ILLEGIBLE in v:
            missing_n.append("la Hoja de Vida no se puede leer: presentar una legible")
        elif v == {K.NO_OFFICIAL_DATA}:
            review.append("la Hoja de Vida no tiene dato oficial (JSON): no se verificó")
        else:
            review.append("la Hoja de Vida requiere revisión humana")
    if official_data.get("hoja_vida"):
        not_fit += _license(official_data["hoja_vida"], req, today)

    positions = {text_key(strip_accents(k)): v for k, v in req["puestos"].items()}
    grade = []
    if text_key(strip_accents(position or "")) in positions:
        keys = positions[text_key(strip_accents(position))]
    else:
        keys = []
        grade = [f"el cargo '{position}' no tiene requisitos configurados: se acepta un certificado AMECO de cualquier equipo"]
    certificates = [d for d in documents if d["tipo_documento"].startswith("ameco")]
    valid_set = []
    for d in certificates:
        js = official_data.get("ameco", {}).get(d.get("archivo_oficial_clave"))
        if js is None or d["veredicto"] not in (K.AUTHENTIC,):
            continue
        operator_ = text_key(js.get("Operador Equipo", ""))
        if keys and not any(text_key(c) in operator_ for c in keys):
            continue
        valid_set.append(js)
    if not certificates:
        missing_n.append("no presentó certificado AMECO")
    elif not valid_set:
        if any(d["veredicto"] == K.FORGED for d in certificates):
            missing_n.append("el certificado AMECO presentado está falsificado")
        elif any(d["veredicto"] in (K.FLAGGED, K.ILLEGIBLE) for d in certificates):
            review.append("el certificado AMECO requiere revisión humana o no se puede leer")
        elif any(d["veredicto"] == K.NO_OFFICIAL_DATA for d in certificates):
            review.append("el certificado AMECO no tiene dato oficial (JSON): no se verificó")
        else:
            missing_n.append(f"no presentó un certificado AMECO para el cargo ({', '.join(keys) or 'cualquiera'})")
    else:
        fit_ones = [js for js in valid_set if js.get("Estado evaluación") == req["ameco"]["estado"]
                 and (not req["ameco"]["vigente"] or (date_value(js.get("Fecha Expiración", "")) or date.min) >= today)]
        if not fit_ones:
            for js in valid_set:
                if js.get("Estado evaluación") != req["ameco"]["estado"]:
                    not_fit.append(f"certificado AMECO {js['N°']}: {js.get('Estado evaluación')}")
                else:
                    not_fit.append(f"certificado AMECO {js['N°']} vencido ({js.get('Fecha Expiración')})")
    if not_fit:
        return NOT_FIT, not_fit + missing_n + review + grade
    if missing_n:
        return INCOMPLETE, missing_n + review + grade
    if review:
        return NOT_EVALUABLE, review + grade
    return FIT, ["cumple todos los requisitos del cargo"] + grade
