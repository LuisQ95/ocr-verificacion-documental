"""Aptitud del postulante según config/requisitos_cl.yaml, con los datos oficiales (JSON) de los documentos que
presentó y que resultaron auténticos."""

from datetime import date

import yaml

from .. import config as C
from . import comparar_cl as K
from .normalizar_cl import clave_texto, fecha, sin_tildes

RUTA = C.ROOT / "config" / "requisitos_cl.yaml"
APTO, NO_APTO, INCOMPLETO, NO_EVALUABLE = "APTO", "NO APTO", "INCOMPLETO", "NO EVALUABLE"


def cargar(ruta=RUTA):
    """Requisitos por cargo desde el YAML."""
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def _mas_anios(d, n):
    """La fecha `d` desplazada `n` años (un 29 de febrero pasa al 28 si el año no es bisiesto)."""
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def _licencia(js, req, hoy):
    """Motivos de NO APTO por licencia y antecedentes (datos oficiales actuales de la Hoja de Vida)."""
    motivos = []
    ultimas = {}
    for l in js.get("Licencias registradas", []):
        if l["Registro"] == "ULTIMA":
            f = fecha(l["Fecha otorgamiento"])
            if f and (l["Clase"] not in ultimas or f > ultimas[l["Clase"]]):
                ultimas[l["Clase"]] = f
    rl = req["licencia"]
    for c in rl.get("todas", []):
        if c not in ultimas:
            motivos.append(f"no tiene licencia clase {c}")
    if rl.get("alguna") and not any(c in ultimas for c in rl["alguna"]):
        motivos.append(f"no tiene ninguna clase profesional requerida ({', '.join(rl['alguna'])})")
    if rl.get("vigente"):
        exigidas = [c for c in rl.get("todas", []) + rl.get("alguna", []) if c in ultimas]
        for c in exigidas:
            anios = rl["anios_control_profesional"] if c.startswith("A") else rl["anios_control_otras"]
            if _mas_anios(ultimas[c], anios) < hoy:
                motivos.append(f"licencia clase {c} con control vencido (última renovación {ultimas[c]})")
    ra = req["antecedentes"]
    limite = _mas_anios(hoy, -ra["sin_suspensiones_anios"])
    for a in js.get("Anotaciones", []):
        t = sin_tildes(a["Tipo"]).upper()
        f = fecha(a["Fecha anotación"])
        if ("SUSPENSION" in t or "CANCELACION" in t or "GRAVISIMA" in t) and f and f >= limite:
            motivos.append(f"{a['Tipo'].title()} el {f} (últimos {ra['sin_suspensiones_anios']} años)")
    if ra.get("rnc_sin_antecedentes") and js.get("Antecedentes", {}).get("RNC") != "SIN ANTECEDENTES":
        motivos.append(f"Registro Nacional de Conductores: {js.get('Antecedentes', {}).get('RNC', 'sin dato')}")
    return motivos


def evaluar_persona(puesto, documentos, oficiales, req, hoy):
    """documentos: filas dictaminadas del postulante; oficiales: {'hoja_vida': json, 'ameco': {N°: json}}.

    Devuelve (aptitud, motivos). La Hoja de Vida puede venir como hoja_vida_p1 o hoja_vida_p2. El cargo se compara
    sin mayúsculas ni tildes; un cargo no configurado acepta un certificado AMECO de cualquier equipo.
    """
    faltan, no_apto, revisar = [], [], []
    hv = [d for d in documentos if d["tipo_documento"].startswith("hoja_vida")]
    hv_ok = [d for d in hv if d["veredicto"] in (K.AUTENTICO,)]
    if not hv:
        faltan.append("no presentó la Hoja de Vida del Conductor")
    elif not hv_ok:
        v = {d["veredicto"] for d in hv}
        if K.FALSIFICADO in v:
            faltan.append("la Hoja de Vida presentada está falsificada")
        elif K.NO_VALIDO in v:
            faltan.append("presentó una COPIA no válida de la Hoja de Vida: se requiere el original")
        elif K.DESACTUALIZADO in v:
            faltan.append(f"la Hoja de Vida tiene más de {req['dias_vigencia_hoja_vida']} días: presentar una actualizada")
        elif K.NO_LEGIBLE in v:
            faltan.append("la Hoja de Vida no se puede leer: presentar una legible")
        elif v == {K.SIN_DATO_OFICIAL}:
            revisar.append("la Hoja de Vida no tiene dato oficial (JSON): no se verificó")
        else:
            revisar.append("la Hoja de Vida requiere revisión humana")
    if oficiales.get("hoja_vida"):
        no_apto += _licencia(oficiales["hoja_vida"], req, hoy)

    puestos = {clave_texto(sin_tildes(k)): v for k, v in req["puestos"].items()}
    nota = []
    if clave_texto(sin_tildes(puesto or "")) in puestos:
        claves = puestos[clave_texto(sin_tildes(puesto))]
    else:
        claves = []
        nota = [f"el cargo '{puesto}' no tiene requisitos configurados: se acepta un certificado AMECO de cualquier equipo"]
    amecos = [d for d in documentos if d["tipo_documento"].startswith("ameco")]
    validos = []
    for d in amecos:
        js = oficiales.get("ameco", {}).get(d.get("archivo_oficial_clave"))
        if js is None or d["veredicto"] not in (K.AUTENTICO,):
            continue
        operador = clave_texto(js.get("Operador Equipo", ""))
        if claves and not any(clave_texto(c) in operador for c in claves):
            continue
        validos.append(js)
    if not amecos:
        faltan.append("no presentó certificado AMECO")
    elif not validos:
        if any(d["veredicto"] == K.FALSIFICADO for d in amecos):
            faltan.append("el certificado AMECO presentado está falsificado")
        elif any(d["veredicto"] in (K.OBSERVADO, K.NO_LEGIBLE) for d in amecos):
            revisar.append("el certificado AMECO requiere revisión humana o no se puede leer")
        elif any(d["veredicto"] == K.SIN_DATO_OFICIAL for d in amecos):
            revisar.append("el certificado AMECO no tiene dato oficial (JSON): no se verificó")
        else:
            faltan.append(f"no presentó un certificado AMECO para el cargo ({', '.join(claves) or 'cualquiera'})")
    else:
        aptos = [js for js in validos if js.get("Estado evaluación") == req["ameco"]["estado"]
                 and (not req["ameco"]["vigente"] or (fecha(js.get("Fecha Expiración", "")) or date.min) >= hoy)]
        if not aptos:
            for js in validos:
                if js.get("Estado evaluación") != req["ameco"]["estado"]:
                    no_apto.append(f"certificado AMECO {js['N°']}: {js.get('Estado evaluación')}")
                else:
                    no_apto.append(f"certificado AMECO {js['N°']} vencido ({js.get('Fecha Expiración')})")
    if no_apto:
        return NO_APTO, no_apto + faltan + revisar + nota
    if faltan:
        return INCOMPLETO, faltan + revisar + nota
    if revisar:
        return NO_EVALUABLE, revisar + nota
    return APTO, ["cumple todos los requisitos del cargo"] + nota
