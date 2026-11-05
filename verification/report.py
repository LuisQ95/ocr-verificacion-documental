"""Salida del pipeline: base SQLite (para consultas) y reporte Excel (para revisión humana)."""

import json
import sqlite3
from collections import Counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from . import config as C
from .fields import (AUTHENTIC, DIFFERENT, DOUBTFUL, EQUAL, FIT, FLAGGED, FORGED, INCOMPLETE, NOT_APPLICABLE,
                     NOT_EVALUABLE, NOT_FIT, OUTDATED, OUTDATED_FIELD)
from .profiles import ACTIVE as PF

COLORS = {FORGED: "F4C7C3", FLAGGED: "FCE8B2", AUTHENTIC: "B7E1CD", OUTDATED: "C9DAF8",
           DIFFERENT: "F4C7C3", DOUBTFUL: "FCE8B2", OUTDATED_FIELD: "C9DAF8",
           FIT: "B7E1CD", NOT_FIT: "F4C7C3", INCOMPLETE: "FCE8B2", NOT_EVALUABLE: "FCE8B2",
           "NO LEGIBLE": "FCE8B2", "NO VÁLIDO": "C9DAF8", "NO RECONOCIDO": "EEEEEE", "SIN DATO OFICIAL": "EEEEEE"}


def _sqlite(path, people, documents, meta, rejected_files):
    """Base SQLite (se rehace desde cero) con la ejecución, personas, documentos, campos, señales y no admitidos."""
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE ejecucion (clave TEXT PRIMARY KEY, valor TEXT);
        CREATE TABLE personas (carpeta TEXT PRIMARY KEY, dni TEXT, declarado INTEGER, veredicto TEXT,
            documentos INTEGER, falsificados INTEGER, observados INTEGER, desactualizados INTEGER,
            tipos_falsificados TEXT, puesto TEXT, aptitud TEXT, motivos_aptitud TEXT);
        CREATE TABLE documentos (carpeta TEXT, archivo TEXT, tipo_documento TEXT, formato TEXT, veredicto TEXT,
            motivos TEXT, misma_instancia INTEGER, archivo_oficial TEXT, alineacion_ok INTEGER, alineacion TEXT,
            residual_etiquetas_pt REAL, producer TEXT, prioridad_revision TEXT, PRIMARY KEY (carpeta, archivo));
        CREATE TABLE campos (carpeta TEXT, archivo TEXT, zona TEXT, oficial TEXT, postulante TEXT, estado TEXT,
            motivo TEXT, conf REAL);
        CREATE TABLE senales (carpeta TEXT, archivo TEXT, tipo TEXT, severidad TEXT, detalle TEXT);
        CREATE TABLE no_admitidos (carpeta TEXT, archivo TEXT, origen TEXT, motivo TEXT);
    """)
    conn.executemany("INSERT INTO ejecucion VALUES (?, ?)", [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()])
    conn.executemany("INSERT INTO personas VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(p["carpeta"], p["dni"], int(p["declarado"]), p["veredicto"], p["documentos"], p["falsificados"],
                      p["observados"], p["desactualizados"], ", ".join(p["tipos_falsificados"]), p["puesto"],
                      p["aptitud"], " | ".join(p["motivos_aptitud"])) for p in people])
    for d in documents:
        conn.execute("INSERT INTO documentos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d["carpeta"], d["archivo"], d["tipo_documento"], d["formato"], d["veredicto"],
                     " | ".join(d["motivos"]), None if d["misma_instancia"] is None else int(d["misma_instancia"]),
                     d["archivo_oficial"], int(bool(d["alineacion_ok"])),
                     json.dumps(d["alineacion"], ensure_ascii=False), d["residual_etiquetas_pt"], d["producer"],
                     d.get("prioridad_revision", "")))
        conn.executemany("INSERT INTO campos VALUES (?,?,?,?,?,?,?,?)",
                        [(d["carpeta"], d["archivo"], c["zona"], c["oficial"], c["postulante"], c["estado"],
                          c["motivo"], c["conf"]) for c in d["campos"]])
        conn.executemany("INSERT INTO senales VALUES (?,?,?,?,?)",
                        [(d["carpeta"], d["archivo"], s["tipo"], s["severidad"], s["detalle"]) for s in d["senales"]])
    conn.executemany("INSERT INTO no_admitidos VALUES (?,?,?,?)",
                    [(x["carpeta"], x["archivo"], x["origen"], x["motivo"]) for x in rejected_files])
    conn.commit()
    conn.close()


def _sheet(ws, headers, rows, cols_color=()):
    """Llena una hoja: encabezado con estilo, filas, color por estado en `cols_color`, filtro y anchos."""
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1A264D")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in rows:
        ws.append(row)
        for i in cols_color:
            color = COLORS.get(row[i])
            if color:
                ws.cell(ws.max_row, i + 1).fill = PatternFill("solid", fgColor=color)
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        width = max(len(str(c.value or "")) for c in col[:300])
        ws.column_dimensions[col[0].column_letter].width = min(70, max(10, width + 2))


def _count_table(ws, title, counter):
    """Agrega a la hoja una tabla de conteo (valor, cantidad) coloreada por valor."""
    ws.append([])
    ws.append([title, "Cantidad"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    for v, n in counter.most_common():
        ws.append([v, n])
        ws.cell(ws.max_row, 1).fill = PatternFill("solid", fgColor=COLORS.get(v, "FFFFFF"))


def excel(path, people, documents, meta, rejected_files):
    """Reporte Excel: resumen con guía de lectura y conteos, y hojas de personas, documentos, diferencias, señales
    y no admitidos."""
    wb = Workbook()
    ws = wb.active
    ws.title = "resumen"
    ws.append(["Reporte de verificación de documentos"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    for key, text in (
            ("Cómo leer este reporte", ""),
            ("Autenticidad", "¿Los documentos son verdaderos? AUTÉNTICO = coinciden con el dato oficial (JSON); "
                             "FALSIFICADO = algún dato o elemento no coincide; OBSERVADO = requiere revisión humana "
                             "(lectura dudosa; la columna 'Prioridad de revisión' pone primero, con ALTA, los que tienen "
                             "un indicio técnico o un valor distinto leído con claridad); NO LEGIBLE = no se pudo leer "
                             "(no se acusa); NO VÁLIDO = copia no válida como certificado; DESACTUALIZADO = genuino "
                             "pero antiguo; NO RECONOCIDO = archivo que no es ninguno de los documentos solicitados; "
                             "SIN DATO OFICIAL = no se entregó su JSON oficial: no se verificó, solo se extrajeron "
                             "sus datos."),
            ("Aptitud", "¿Cumple los requisitos del puesto (requirements.yaml del perfil de documentos)? Se evalúa "
                        "con los datos oficiales. APTO / NO APTO / INCOMPLETO (falta un documento o el presentado es falso). "
                        "Un postulante puede ser AUTÉNTICO y NO APTO: sus documentos son verdaderos pero no "
                        "cumple un requisito (p. ej. licencia vencida)."),
            ("Detalle", "Hoja 'personas': motivos de aptitud. Hoja 'diferencias': campo por campo, oficial vs "
                        "postulante. Hoja 'senales': evidencia adicional (QR, objetos superpuestos, identidad...).")):
        ws.append([key, text])
        ws.cell(ws.max_row, 1).font = Font(bold=True)
        ws.cell(ws.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.append([])
    for key, value in meta.items():
        ws.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    _count_table(ws, "Autenticidad por postulante", Counter(p["veredicto"] for p in people))
    _count_table(ws, "Aptitud por postulante", Counter(p["aptitud"] for p in people))
    _count_table(ws, "Veredicto por documento", Counter(d["veredicto"] for d in documents))
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 90

    _sheet(wb.create_sheet("personas"),
          ["Carpeta", f"{PF.documents.ID_LABEL} / documento de identidad", "Identidad declarada", "Autenticidad", "Aptitud", "Motivos de aptitud",
           "Documentos",
           "Falsificados", "Observados", "Desactualizados", "Documentos falsificados", "Puesto"],
          [[p["carpeta"], p["dni"], "SÍ" if p["declarado"] else "NO", p["veredicto"], p["aptitud"],
            " | ".join(p["motivos_aptitud"]), p["documentos"], p["falsificados"], p["observados"],
            p["desactualizados"], ", ".join(p["tipos_falsificados"]), p["puesto"]] for p in people],
          cols_color=(3, 4))

    _sheet(wb.create_sheet("documentos"),
          ["Carpeta", "Archivo", "Tipo de documento", "Formato", "Veredicto", "Prioridad de revisión", "Motivos",
           "Mismo documento que el oficial", "Archivo oficial"],
          [[d["carpeta"], d["archivo"], d["tipo_documento"], d["formato"], d["veredicto"],
            d.get("prioridad_revision", ""),
            " | ".join(d["motivos"])[:2000],
            "" if d["misma_instancia"] is None else ("SÍ" if d["misma_instancia"] else "NO (otra emisión)"),
            d["archivo_oficial"]] for d in documents], cols_color=(4,))

    _sheet(wb.create_sheet("diferencias"),
          ["Carpeta", "Archivo", "Campo", "Valor oficial", "Valor postulante", "Estado", "Motivo", "Confianza OCR"],
          [[d["carpeta"], d["archivo"], c["zona"], c["oficial"], c["postulante"], c["estado"], c["motivo"], c["conf"]]
           for d in documents for c in d["campos"] if c["estado"] not in (EQUAL, NOT_APPLICABLE)], cols_color=(5,))

    _sheet(wb.create_sheet("senales"),
          ["Carpeta", "Archivo", "Tipo", "Severidad", "Detalle"],
          [[d["carpeta"], d["archivo"], s["tipo"], s["severidad"], s["detalle"]]
           for d in documents for s in d["senales"]])

    _sheet(wb.create_sheet("no_admitidos"), ["Carpeta", "Archivo", "Origen", "Motivo"],
          [[x["carpeta"], x["archivo"], x["origen"], x["motivo"]] for x in rejected_files])
    wb.save(path)


def write_report(people, documents, meta, output=C.RESULTS_DIR, rejected_files=()):
    """Escribe verification.db y verification_report.xlsx en `salida`; devuelve sus rutas."""
    output.mkdir(parents=True, exist_ok=True)
    db = output / "verification.db"
    xlsx = output / "verification_report.xlsx"
    _sqlite(db, people, documents, meta, list(rejected_files))
    excel(xlsx, people, documents, meta, list(rejected_files))
    return db, xlsx
