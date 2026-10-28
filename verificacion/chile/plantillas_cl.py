"""Plantillas chilenas: páginas de official_file/, zonas de cada campo (en puntos PDF de la plantilla), su tipo
de comparación y la clave del dato oficial (JSON) con que se compara.

Las zonas salen de la disposición de los documentos reales (DATOS REALES): posición, alineación y tamaño de
cada valor, con margen para valores más largos. Ids de página:
  hoja_vida_p1, hoja_vida_p2  -> Hoja de Vida del Conductor (Registro Civil), 2 páginas
  ameco_rev02, ameco_rev03     -> Certificado de Competencia AMECO (dos versiones del mismo documento)
"""

import pymupdf
import numpy as np

from .. import config as C
from ..plantillas import dibujos_pdf, palabras_pdf

ARCHIVOS = {"hoja_vida_p1": ("hoja_vida_conductor.pdf", 0), "hoja_vida_p2": ("hoja_vida_conductor.pdf", 1),
            "ameco_rev02": ("ameco_certificado_competencia_rev02.pdf", 0),
            "ameco_rev03": ("ameco_certificado_competencia_rev03.pdf", 0)}
TIPO_DOC = {"hoja_vida_p1": "hoja_vida", "hoja_vida_p2": "hoja_vida", "ameco_rev02": "ameco", "ameco_rev03": "ameco"}
NOMBRE_DOC = {"hoja_vida": "Hoja de Vida del Conductor", "ameco": "Certificado de Competencia AMECO"}


def _hvid(p):
    """Zonas de la página `p` (0 o 1) de la Hoja de Vida; en la página 2 los datos de cabecera bajan 8 pt."""
    d = 8 * p
    z = {
        "folio": ([490, 38, 557, 50], "codigo", None),
        "codigo_verificacion": ([478, 65, 557, 77], "codigo", None),
        "folio_bajo_barra": ([468, 119, 557, 129], "codigo", None),
        "nombre": ([205, 167 + d, 557, 182 + d], "identidad", "Nombre"),
        "run": ([205, 197 + d, 294, 210 + d], "rut", "R.U.N."),
        "fecha_nacimiento": ([378, 197 + d, 557, 210 + d], "fecha", "Fecha nacimiento"),
        "fecha_emision": ([210, 526, 557, 541], "fecha", None),
    }
    if p == 0:
        z["domicilio"] = ([205, 210, 557, 223], "descriptivo", "Domicilio")
        z["comuna"] = ([205, 222, 557, 235], "categorico", "Comuna")
        z["cuerpo"] = ([84, 244, 557, 486], "cuerpo", None)
    else:
        z["cuerpo"] = ([84, 218, 557, 516], "cuerpo", None)
    return z


ZONAS = {
    "hoja_vida_p1": _hvid(0),
    "hoja_vida_p2": _hvid(1),
    "ameco_rev03": {
        "numero_certificado": ([483, 80, 580, 93], "codigo", "N°"),
        "nombre": ([111, 130, 560, 142], "identidad", "Nombre"),
        "rut": ([111, 148, 300, 160], "rut", "RUT"),
        "empresa": ([111, 166, 560, 178], "descriptivo", "Empresa"),
        "estado_evaluacion": ([364, 184, 480, 196], "categorico", "Estado evaluación"),
        "tipo_operador": ([34, 202, 560, 214], "descriptivo", "Operador Equipo"),
        "descripcion_equipo": ([58, 224, 556, 256], "descriptivo", "Equipo"),
        "nota_conocimiento_general": ([38, 289, 129, 300], "numero", "Exámenes/Conocim. General"),
        "nota_senaleticas": ([129, 289, 215, 300], "numero", "Exámenes/Señaléticas"),
        "nota_parcial": ([215, 289, 290, 300], "numero", "Exámenes/Nota Parcial"),
        "nota_refuerzo_brechas": ([290, 289, 365, 300], "numero", "Exámenes/Refuerzo Brechas"),
        "evaluacion_practica": ([365, 289, 470, 300], "categorico", "Exámenes/Evaluación Práctica"),
        "nota_aprobacion_final": ([470, 289, 578, 300], "categorico", "Exámenes/Nota Aprobación Final"),
        "licencia_clase": ([147, 322, 420, 333], "clases", "Licencia de Conductor/Clase"),
        "licencia_municipalidad": ([147, 334, 420, 345], "categorico", "Licencia de Conductor/Municipalidad"),
        "licencia_ley": ([147, 345, 420, 356], "codigo", "Licencia de Conductor/Ley"),
        "licencia_restricciones": ([147, 357, 420, 368], "descriptivo", "Licencia de Conductor/Restricciones"),
        "licencia_fecha_control": ([147, 369, 420, 380], "fecha", "Licencia de Conductor/Fecha control"),
        "fecha_certificacion": ([147, 394, 420, 405], "fecha", "Fecha Certificación"),
        "fecha_expiracion": ([147, 405, 420, 416], "fecha", "Fecha Expiración"),
        "evaluador1_nombre": ([100, 562, 272, 573], "identidad", "Evaluadores[0]"),
        "evaluador2_nombre": ([345, 562, 515, 573], "identidad", "Evaluadores[1]"),
    },
    "ameco_rev02": {
        "numero_certificado": ([486, 118, 585, 130], "codigo", "N°"),
        "nombre": ([121, 118, 465, 130], "identidad", "Nombre"),
        "rut": ([121, 131, 330, 143], "rut", "RUT"),
        "empresa": ([121, 143, 465, 155], "descriptivo", "Empresa"),
        "estado_evaluacion": ([445, 168, 565, 181], "categorico", "Estado evaluación"),
        "tipo_operador": ([34, 189, 565, 202], "descriptivo", "Operador Equipo"),
        "descripcion_equipo": ([34, 210, 565, 252], "descriptivo", "Equipo"),
        "nota_conocimiento_general": ([35, 312, 121, 325], "numero", "Exámenes/Conocim. General"),
        "nota_senaleticas": ([121, 312, 203, 325], "numero", "Exámenes/Señaléticas"),
        "nota_parcial": ([203, 312, 280, 325], "numero", "Exámenes/Nota Parcial"),
        "nota_refuerzo_brechas": ([280, 312, 364, 325], "numero", "Exámenes/Refuerzo Brechas"),
        "evaluacion_practica": ([364, 312, 467, 325], "categorico", "Exámenes/Evaluación Práctica"),
        "nota_aprobacion_final": ([467, 312, 558, 326], "categorico", "Exámenes/Nota Aprobación Final"),
        "licencia_clase": ([164, 350, 420, 362], "clases", "Licencia de Conductor/Clase"),
        "licencia_municipalidad": ([164, 362, 420, 374], "categorico", "Licencia de Conductor/Municipalidad"),
        "licencia_ley": ([164, 374, 420, 386], "codigo", "Licencia de Conductor/Ley"),
        "licencia_restricciones": ([164, 386, 420, 398], "descriptivo", "Licencia de Conductor/Restricciones"),
        "licencia_fecha_control": ([164, 398, 261, 410], "fecha", "Licencia de Conductor/Fecha control"),
        "fecha_certificacion": ([164, 417, 420, 429], "fecha", "Fecha Certificación"),
        "fecha_expiracion": ([164, 434, 420, 446], "fecha", "Fecha Expiración"),
        "evaluador1_nombre": ([40, 639, 225, 652], "identidad", "Evaluadores[0]"),
        "evaluador2_nombre": ([370, 639, 550, 652], "identidad", "Evaluadores[1]"),
        "evaluador_de_competencia": ([467, 692, 592, 702], "identidad", "Evaluadores[1]"),
        "pie_nombre": ([200, 748, 425, 758], "identidad", "Nombre"),
        "pie_rut": ([250, 759, 375, 768], "rut", "RUT"),
    },
}
"""zona -> (rect [x0, y0, x1, y1] en pt, tipo de comparación, clave JSON | None).

- tipos: identidad, rut, codigo, fecha, numero, clases, categorico, descriptivo, cuerpo (filas variables)
- claves JSON: "Nombre", "Licencia de Conductor/Clase" (anidada), "Exámenes/Señaléticas", "Evaluadores[0]"...
- clave None: dato propio de la emisión (no se compara con el JSON)
"""

IMAGENES_ESPERADAS = {
    "hoja_vida_p1": [[92.7, 617, 270.3, 695]],
    "hoja_vida_p2": [[106.3, 617, 256.7, 676], [397.8, 617, 547.8, 699.5]],
    "ameco_rev03": [[142.5, 492.9, 227.6, 535.1], [377.8, 482.5, 480.0, 545.3]],
    "ameco_rev02": [[57.6, 557.6, 220.0, 662.4], [388.2, 564.2, 526.3, 641.2], [50.6, 692.0, 113.9, 753.8],
                    [145.6, 693.4, 204.2, 761.4]],
}
"""Imágenes que llevan los documentos genuinos además de las de la plantilla (cambian en cada documento): timbre
electrónico, firmas de los evaluadores y del jefe de archivo, QR de credenciales del evaluador."""

CLAVES_TEXTO = {
    "hoja_vida_p1": ["HOJA", "VIDA", "CONDUCTOR", "LICENCIAS", "REGISTRADAS", "Domicilio", "Comuna"],
    "hoja_vida_p2": ["HOJA", "VIDA", "CONDUCTOR", "Página", "Impreso", "REGION"],
    "ameco_rev02": ["CERTIFICADO", "COMPETENCIA", "Rev.02", "Superintendente", "Perfil", "Ocupacional", "EXAMENES"],
    "ameco_rev03": ["CERTIFICADO", "COMPETENCIA", "Rev.", "03", "EXÁMENES", "Conductor", "AMECO"],
}
"""Palabras que identifican cada página aunque se lea solo una parte (clasificación por texto)."""

_CACHE = None


def cargar():
    """Plantillas con: imagen (gris, a C.DPI), tamaño en pt, palabras fijas, dibujos y zonas."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out = {}
    for pid, (archivo, n) in ARCHIVOS.items():
        doc = pymupdf.open(C.DIR_OFICIAL / archivo)
        page = doc[n]
        pix = page.get_pixmap(dpi=C.DPI, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        fijas = [(*p["bbox"], p["texto"]) for p in palabras_pdf(page)]
        imagenes = [list(i["bbox"]) for i in page.get_image_info()] + IMAGENES_ESPERADAS[pid]
        out[pid] = {
            "id": pid, "tipo_documento": TIPO_DOC[pid], "pagina": n, "archivo": archivo,
            "ancho_pt": page.rect.width, "alto_pt": page.rect.height, "imagen": img,
            "palabras_fijas": fijas, "dibujos": dibujos_pdf(page), "imagenes_esperadas": imagenes,
            "zonas": {k: {"rect": r, "tipo": t, "clave": c} for k, (r, t, c) in ZONAS[pid].items()},
        }
    _CACHE = out
    return out
