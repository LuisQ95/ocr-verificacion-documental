"""Asignación de palabras a zonas, comparación campo a campo y dictamen por documento.

Reglas de comparación (cada zona tiene un "tipo" y un "cambio", ver plantillas.py):
  - PDF nativo: comparación exacta (solo se ignoran espacios y mayúsculas).
  - Escaneo (OCR): se toleran diferencias que produce el escaneo y que nadie falsifica (tildes,
    I/l/1, O/0, puntuación pegada a los bordes), pero según el tipo de campo:
      identidad, codigo  -> sin similitud aproximada
      numero             -> el separador decimal se conserva ("220.00" != "2200.0")
      fecha              -> si el OCR no da una fecha, es lectura fallida (DUDOSO)
      categorico         -> si se lee OTRO valor válido es DIFERENTE; si se lee un valor
                            inexistente es lectura fallida (DUDOSO)
      descriptivo        -> similitud aproximada solo si no cambia ninguna negación
  - Negaciones: si aparece/desaparece "NO", "SIN", "NI" o un prefijo IN-/DES-/IM- entre los dos
    valores ("REGISTRA" vs "NO REGISTRA", "COMPLETA" vs "INCOMPLETA"), siempre es DIFERENTE.
  - "Campo vacío" vs "texto no leído": se mide la tinta de la zona antes de acusar.
  - Una diferencia leída con confianza baja es DUDOSA (revisión humana), no FALSIFICADA.

Modo de comparación según el documento oficial disponible:
  - misma instancia (mismo código de verificación / QR): todo debe coincidir.
  - estado actual (el oficial es una consulta más reciente): la identidad debe coincidir; los
    campos de instancia (emisión, código, hash) no se comparan; los temporales que difieren son
    DESACTUALIZADO si el documento del postulante es más antiguo. En tablas con fechas, un registro
    oficial anterior a la emisión del documento del postulante que no aparece en él es FALSIFICADO.
"""

import json
import re
import statistics
import unicodedata
from collections import Counter
from datetime import date

from rapidfuzz import fuzz

from . import config as C

CONFUNDIBLES = str.maketrans({"l": "I", "1": "I", "|": "I", "i": "I", "0": "O", "o": "O"})
CONFUNDIBLES_NUM = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "i": "1"})
PUNTUACION_BORDE = ".,:;-_°'\"`´()[]{}|"
NEGACIONES = {"NO", "SIN", "NI", "NUNCA", "NINGUN", "NINGUNA", "NINGUNO"}
PREFIJOS_NEGACION = ("IN", "IM", "DES", "NO")

IGUAL, TOLERADO, DIFERENTE, DUDOSO = "IGUAL", "IGUAL_TOLERADO_OCR", "DIFERENTE", "DUDOSO"
DESACTUALIZADO_C, NO_APLICA = "DESACTUALIZADO", "NO_APLICA"
TIPOS_NO_COMPARABLE = ("archivo", "alineacion", "clasificacion")
AUTENTICO, OBSERVADO, FALSIFICADO, DESACTUALIZADO = "AUTÉNTICO", "OBSERVADO", "FALSIFICADO", "DESACTUALIZADO"


def _sin_tildes(t):
    """Texto sin tildes ni diéresis."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def clave(texto, confundibles=False):
    """Forma canónica genérica (sin tipo). Con confundibles=True (solo OCR) iguala I/l/1/|, O/0,
    tildes, puntuación de borde y separadores. Para comparar campos se usa normalizar()."""
    t = texto or ""
    if confundibles:
        t = _sin_tildes(t.translate(CONFUNDIBLES)).upper()
        t = "".join(tok.strip(PUNTUACION_BORDE) for tok in t.split())
        return re.sub(r"[.,]", "", t)
    return re.sub(r"\s+", "", t.upper())


def tokens(texto):
    """Palabras normalizadas para OCR (sin tildes, confundibles igualados, sin puntuación de borde)."""
    t = _sin_tildes((texto or "").translate(CONFUNDIBLES)).upper()
    return [x for x in (tok.strip(PUNTUACION_BORDE).replace(".", "").replace(",", "") for tok in t.split()) if x]


def normalizar(texto, tipo, ocr):
    """Forma canónica de un valor para compararlo según su tipo de campo.

    Sin OCR solo se ignoran espacios y mayúsculas. Con OCR, en números se igualan los confundibles con dígitos y se
    reparan separadores decimales mal leídos ("440:00", "440/00" -> "440.00") o perdidos ("2200 00" ->
    "2200.00"); en lo demás se usan los `tokens` normalizados.
    """
    if not ocr:
        return re.sub(r"\s+", "", (texto or "").upper())
    if tipo == "numero":
        t = _sin_tildes((texto or "").translate(CONFUNDIBLES_NUM)).upper().replace(",", ".")
        t = " ".join(tok.strip(PUNTUACION_BORDE.replace(".", "")) for tok in t.split()).strip(".")
        t = re.sub(r"(?<=\d)[:;/](?=\d)", ".", t)
        t = re.sub(r"(?<=\d) (?=\d{2}\b)", ".", t)
        return re.sub(r"\s+", "", t)
    return "".join(tokens(texto))


def _neg(t):
    """'NOJ' (ruido pegado a 'NO') cuenta como 'NO' y 'SUN' (una letra mal leída) como 'SIN';
    'NOREGISTRA' no (es otra palabra)."""
    for n in NEGACIONES:
        if t.startswith(n) and len(t) - len(n) <= 1:
            return n
        if len(n) >= 3 and len(t) == len(n) and sum(a != b for a, b in zip(t, n)) == 1:
            return n
    return t


def cambia_negacion(a, b):
    """True si entre `a` y `b` aparece o desaparece una negación ("NO", "SIN"...) o un prefijo negativo
    ("COMPLETA" vs "INCOMPLETA")."""
    ta, tb = [_neg(t) for t in tokens(a)], [_neg(t) for t in tokens(b)]
    if Counter(t for t in ta if t in NEGACIONES) != Counter(t for t in tb if t in NEGACIONES):
        return True
    solo_a, solo_b = set(ta) - set(tb), set(tb) - set(ta)
    return any(x == p + y or y == p + x for x in solo_a for y in solo_b for p in PREFIJOS_NEGACION)


def fuente_base(nombre):
    """Quita el prefijo de subconjunto que los PDF reales agregan a las fuentes ("ABCDEF+Arial")."""
    return re.sub(r"^[A-Z]{6}\+", "", nombre or "")


def fecha(texto):
    """Primera fecha dd/mm/aaaa del texto (None si no hay o no es válida)."""
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", texto or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _centro(b):
    """Centro de una caja (x0, y0, x1, y1)."""
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _dentro(c, r):
    """True si el punto `c` está dentro del rectángulo `r`."""
    return r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]


def _intersecta(a, b):
    """True si los rectángulos `a` y `b` se superponen."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _orden_lectura(palabras):
    """Agrupa por línea real (centros verticales cercanos) y ordena de izquierda a derecha.
    Evita que una leve inclinación del escaneo separe palabras de la misma línea."""
    if not palabras:
        return []
    altura = statistics.median(p["bbox"][3] - p["bbox"][1] for p in palabras)
    lineas = []
    for p in sorted(palabras, key=lambda p: _centro(p["bbox"])[1]):
        cy = _centro(p["bbox"])[1]
        if lineas and abs(cy - lineas[-1]["cy"]) <= altura * 0.5:
            lineas[-1]["palabras"].append(p)
            lineas[-1]["cy"] = statistics.mean(_centro(q["bbox"])[1] for q in lineas[-1]["palabras"])
        else:
            lineas.append({"cy": cy, "palabras": [p]})
    return [p for linea in lineas for p in sorted(linea["palabras"], key=lambda p: p["bbox"][0])]


def asignar_palabras(palabras, plantilla):
    """Separa las palabras fijas de la plantilla (etiquetas) del contenido y asigna el contenido a las zonas.

    Devuelve el texto de cada zona (en orden de lectura, con su confianza mínima y sus fuentes), las palabras fuera
    de zonas y cuán bien encajan las etiquetas (cobertura y residual), que sirven de control de calidad.
    La marca de agua se elimina en la imagen (corrección de iluminación + blanqueo), no aquí: un filtro por
    similitud de texto con sus palabras descartaba datos reales ("POLICIALES" ~ "OFICIAL").
    """
    fijas = [(_centro(w[:4]), w[4].upper()) for w in plantilla["palabras_fijas"]]
    distancias, encontradas = [], set()
    contenido = []
    for p in palabras:
        c = _centro(p["bbox"])
        txt = p["texto"].upper()
        es_fija = False
        for i, (cf, tf) in enumerate(fijas):
            d = ((c[0] - cf[0]) ** 2 + (c[1] - cf[1]) ** 2) ** 0.5
            if d < 6 and (txt == tf or fuzz.ratio(txt, tf) >= 70):
                es_fija = True
                if fuzz.ratio(txt, tf) >= 85:
                    distancias.append(d)
                    encontradas.add(i)
                break
        if not es_fija:
            contenido.append(p)

    valores, fuera = {}, []
    zonas = plantilla["zonas"]
    for p in contenido:
        c = _centro(p["bbox"])
        zona = next((n for n, z in zonas.items() if _dentro(c, z["rect"])), None)
        if zona is None:
            fuera.append(p["texto"])
            continue
        valores.setdefault(zona, []).append(p)

    salida = {}
    for zona, ps in valores.items():
        ps = _orden_lectura(ps)
        salida[zona] = {
            "texto": " ".join(p["texto"] for p in ps),
            "conf": min(p.get("conf", 1.0) for p in ps),
            "fuentes": sorted({fuente_base(p["font"]) for p in ps if p.get("font")}),
        }
    return {
        "valores": salida,
        "fuera_de_zonas": fuera,
        "cobertura_etiquetas": len(encontradas) / max(1, len(fijas)),
        "residual_etiquetas_pt": statistics.median(distancias) if distancias else None,
        "n_etiquetas": len(distancias),
    }


def dni_documento(doc):
    """DNI (8 dígitos) leídos en el documento: del campo "dni" y del QR."""
    candidatos = []
    v = (doc.get("valores") or {}).get("dni")
    if v and re.fullmatch(r"\d{8}", clave(v["texto"])):
        candidatos.append(clave(v["texto"]))
    qr = doc.get("qr") or {}
    if re.fullmatch(r"\d{8}", str(qr.get("dni", ""))):
        candidatos.append(str(qr["dni"]))
    return candidatos


def dni_persona(docs):
    """DNI más votado entre todos los documentos de una persona (None si no hay)."""
    votos = Counter(d for doc in docs for d in dni_documento(doc))
    return votos.most_common(1)[0][0] if votos else None


def cargar_vocabulario():
    """Vocabulario guardado {(plantilla, campo): valores válidos}; vacío si no existe."""
    if not C.VOCABULARIO.exists():
        return {}
    datos = json.loads(C.VOCABULARIO.read_text(encoding="utf-8"))
    return {tuple(k.split("|", 1)): set(v) for k, v in datos.items()}


def guardar_vocabulario(vocab):
    """Guarda el vocabulario en C.VOCABULARIO con claves "plantilla|campo"."""
    C.VOCABULARIO.write_text(json.dumps({f"{a}|{b}": sorted(v) for (a, b), v in vocab.items()},
                                        ensure_ascii=False, indent=1), encoding="utf-8")


def unir_vocabularios(*vocabs):
    """Unión de varios vocabularios, campo por campo."""
    salida = {}
    for v in vocabs:
        for k, valores in v.items():
            salida.setdefault(k, set()).update(valores)
    return salida


def vocabulario(oficiales, plantillas):
    """Valores válidos de cada campo categórico, observados en los documentos oficiales."""
    vocab = {}
    for r in oficiales:
        p = plantillas.get(r.get("plantilla"))
        if not p:
            continue
        for zona, v in (r.get("valores") or {}).items():
            z = p["zonas"].get(zona)
            if z and z.get("tipo") == "categorico" and v["texto"]:
                vocab.setdefault((p["id"], z.get("columna") or zona), set()).add(v["texto"])
    return vocab


def _comparar_valor(of, po, es_ocr, conf, tinta, tipo="descriptivo", validos=None):
    """(estado, motivo) de un campo: oficial `of` contra lo leído `po`, con las reglas del módulo.

    - Zona sin texto: si tiene tinta, el OCR falló (DUDOSO). Un valor de 1-2 caracteres puede perderse en un mal
      escaneo (DUDOSO); borrar un registro real se ve en las demás celdas de la fila.
    - Un número falsificado sigue siendo un número: si el OCR no leyó dígitos, la lectura falló (DUDOSO).
    """
    if clave(of) == clave(po):
        return IGUAL, ""
    if not es_ocr:
        return DIFERENTE, ""
    if not clave(po):
        if tinta >= C.MIN_TINTA_ZONA:
            return DUDOSO, "hay tinta en la zona pero el OCR no leyó texto"
        if len(clave(of)) <= 2:
            return DUDOSO, "valor muy corto no leído"
        return DIFERENTE, "el campo está vacío en el documento del postulante"
    if cambia_negacion(of, po):
        return (DIFERENTE, "cambia una negación (NO / SIN / IN- / DES-)") if conf >= 0.5 else \
            (DUDOSO, f"posible cambio de negación leído con confianza baja ({conf:.2f})")
    no, np_ = normalizar(of, tipo, True), normalizar(po, tipo, True)
    if no == np_:
        return TOLERADO, "solo difieren tildes, caracteres confundibles o puntuación del escaneo"
    if re.fullmatch(r"[\d.,/]+", clave(of)) and not any(ch.isdigit() for ch in po):
        return DUDOSO, "valor numérico ilegible (el OCR no leyó dígitos)"
    if tipo == "fecha" and fecha(of) and not fecha(po.translate(CONFUNDIBLES_NUM)):
        return DUDOSO, "fecha ilegible (el OCR no leyó una fecha válida)"
    if tipo == "categorico" and validos:
        validos_n = {normalizar(v, tipo, True): v for v in validos}
        if np_ in validos_n:
            return (DIFERENTE, f"se lee otro valor válido: '{validos_n[np_]}'") if conf >= 0.5 else \
                (DUDOSO, f"otro valor válido leído con confianza baja ({conf:.2f})")
        return DUDOSO, "valor no reconocido entre los valores válidos del campo (lectura fallida)"
    if tipo == "descriptivo" and fuzz.ratio(no, np_) >= C.FUZZY_TEXTO_LARGO and len(no) >= C.LARGO_TEXTO_TOLERANTE:
        return TOLERADO, "texto descriptivo casi idéntico (ruido de OCR)"
    if conf < C.CONF_OCR_ALTA:
        return DUDOSO, f"lectura con confianza baja ({conf:.2f})"
    return DIFERENTE, ""


def _misma_instancia(post, ofi, plantilla):
    """¿El oficial es el mismo documento que presentó el postulante (mismo código / QR)?

    Ante la duda se elige la comparación estricta: un documento con fecha de emisión posterior a la
    consulta oficial no puede ser una emisión legítima más nueva, y en escaneos el código se compara
    con tolerancia de OCR (si no, un código mal leído o alterado evitaría comparar la emisión). Si ningún código
    legible coincide, solo es "otra emisión" cuando el documento del postulante es más antiguo."""
    fe_post, fe_ofi = _fecha_emision(post, plantilla), _fecha_emision(ofi, plantilla)
    if fe_post and fe_ofi and fe_post >= fe_ofi:
        return True
    qp, qo = post.get("qr") or {}, ofi.get("qr") or {}
    if qp.get("cod") and qo.get("cod"):
        return qp["cod"] == qo["cod"]
    zonas_inst = [n for n, z in plantilla["zonas"].items() if z.get("cambio") == "instancia" and z.get("tipo") == "codigo"]
    if not zonas_inst:
        return True
    es_ocr = not post.get("nativo")
    vp, vo = post.get("valores", {}), ofi.get("valores", {})
    for z in zonas_inst:
        a, b = normalizar(vo.get(z, {}).get("texto", ""), "codigo", es_ocr), \
            normalizar(vp.get(z, {}).get("texto", ""), "codigo", es_ocr)
        if a and (a == b or (es_ocr and fuzz.ratio(a, b) >= 85)):
            return True
    return not (fe_post and fe_ofi and fe_post < fe_ofi)


def _fecha_emision(doc, plantilla):
    """Fecha de emisión leída en el documento (zona de instancia de tipo fecha con "emision" en el nombre)."""
    for n, z in plantilla["zonas"].items():
        if z.get("cambio") == "instancia" and z.get("tipo") == "fecha" and "emision" in n:
            return fecha((doc.get("valores") or {}).get(n, {}).get("texto", ""))
    return None


def _celda(nombre_tabla, fila, col):
    """Nombre de la zona de una celda de tabla."""
    return f"{nombre_tabla}_f{fila}_{col}"


def _comparar_tabla_estado_actual(post, ofi, plantilla, nombre_tabla, t, fe_post, es_ocr):
    """Tablas en modo 'estado actual': las filas pueden haberse desplazado por registros nuevos.
    Se comparan como conjuntos: un registro oficial anterior a la emisión del documento del
    postulante debe estar en él."""
    vp, vo = post.get("valores", {}), ofi.get("valores", {})
    zonas = plantilla["zonas"]

    def filas(valores):
        """(fila, celdas, firma normalizada) de cada fila no vacía de la tabla."""
        salida = []
        for f in range(1, t["filas"] + 1):
            celdas = {c: valores.get(_celda(nombre_tabla, f, c), {}).get("texto", "") for c in t["columnas"]}
            if any(celdas.values()):
                firma = "|".join(normalizar(celdas[c], zonas[_celda(nombre_tabla, f, c)].get("tipo"), es_ocr)
                                 for c in t["columnas"] if c != "n")
                salida.append((f, celdas, firma))
        return salida

    del_post = {firma for _, _, firma in filas(vp)}
    col_fecha = next((c for c in t["columnas"] if zonas[_celda(nombre_tabla, 1, c)].get("tipo") == "fecha"), None)
    campos = []
    for f, celdas, firma in filas(vo):
        f_reg = fecha(celdas.get(col_fecha, "")) if col_fecha else None
        if firma in del_post:
            estado, motivo = IGUAL, ""
        elif f_reg and fe_post and f_reg > fe_post:
            estado, motivo = DESACTUALIZADO_C, "registro posterior a la emisión del documento presentado"
        else:
            estado, motivo = DIFERENTE, "registro oficial anterior a la emisión que no figura en el documento presentado"
        for c in t["columnas"]:
            campos.append({"zona": _celda(nombre_tabla, f, c), "oficial": celdas[c], "postulante": "",
                           "estado": estado, "motivo": motivo, "conf": 1.0})
    oficiales = {firma for _, _, firma in filas(vo)}
    for f, celdas, firma in filas(vp):
        if firma not in oficiales:
            for c in t["columnas"]:
                campos.append({"zona": _celda(nombre_tabla, f, c), "oficial": "", "postulante": celdas[c],
                               "estado": DIFERENTE, "motivo": "registro que no existe en la fuente oficial", "conf": 1.0})
    return campos


def comparar_campos(post, ofi, plantilla):
    """Compara campo a campo el documento del postulante con el oficial. Devuelve (campos, misma_instancia).

    - Si el oficial es otra emisión, los campos de instancia no se comparan y un dato temporal distinto queda
      DESACTUALIZADO cuando el documento del postulante es más antiguo.
    - Códigos técnicos en escaneos: si el QR del postulante es idéntico al oficial, quedan verificados por una
      lectura de máquina (más fiable que el OCR de una letra Courier fina).
    - Tablas de la misma instancia: si la fila completa coincide, las diferencias por celda son cortes de palabra
      del OCR (cada celda se normaliza según su tipo: los decimales se conservan).
    """
    es_ocr = not post.get("nativo")
    vp, vo = post.get("valores", {}), ofi.get("valores", {})
    tinta = post.get("tinta", {})
    vocab = plantilla.get("vocabulario", {})
    misma = _misma_instancia(post, ofi, plantilla)
    fe_post, fe_ofi = _fecha_emision(post, plantilla), _fecha_emision(ofi, plantilla)
    post_mas_antiguo = bool(fe_post and fe_ofi and fe_post < fe_ofi)

    qr_valido = es_ocr and post.get("qr") and post.get("qr") == ofi.get("qr")
    valores_qr = {clave(str(v)) for v in (post.get("qr") or {}).values()}

    campos = {}
    tablas_estado_actual = set() if misma else set(plantilla.get("tablas", {}))
    for zona, z in plantilla["zonas"].items():
        if z.get("tabla") in tablas_estado_actual:
            continue
        of = vo.get(zona, {}).get("texto", "")
        po_d = vp.get(zona, {})
        po = po_d.get("texto", "")
        conf = po_d.get("conf", 1.0)
        base = {"zona": zona, "oficial": of, "postulante": po, "conf": round(conf, 3)}
        if not misma and z.get("cambio") == "instancia":
            campos[zona] = base | {"estado": NO_APLICA, "motivo": "el oficial es otra emisión del documento"}
            continue
        if qr_valido and zona in C.CAMPOS_TECNICOS and clave(of) in valores_qr:
            igual = clave(of) == clave(po)
            campos[zona] = base | {"estado": IGUAL if igual else TOLERADO,
                                   "motivo": "" if igual else "verificado por el QR (idéntico al oficial)"}
            continue
        estado, motivo = _comparar_valor(of, po, es_ocr, conf, tinta.get(zona, 0.0), z.get("tipo", "descriptivo"),
                                         vocab.get((plantilla["id"], z.get("columna") or zona)))
        if estado == DIFERENTE and not misma and z.get("cambio") == "temporal":
            if post_mas_antiguo:
                estado, motivo = DESACTUALIZADO_C, "el dato cambió después de la emisión del documento presentado"
            elif not fe_post:
                estado, motivo = DUDOSO, "otra emisión del documento y sin fecha de emisión legible para comparar"
        campos[zona] = base | {"estado": estado, "motivo": motivo}

    for nombre_tabla, t in plantilla.get("tablas", {}).items():
        if nombre_tabla in tablas_estado_actual:
            continue
        for f in range(1, t["filas"] + 1):
            celdas = [_celda(nombre_tabla, f, c) for c in t["columnas"]]
            tipos = [plantilla["zonas"][c].get("tipo") for c in celdas]
            fila_o = "".join(normalizar(campos[c]["oficial"], tp, es_ocr) for c, tp in zip(celdas, tipos))
            fila_p = "".join(normalizar(campos[c]["postulante"], tp, es_ocr) for c, tp in zip(celdas, tipos))
            if fila_o == fila_p and not any(cambia_negacion(campos[c]["oficial"], campos[c]["postulante"]) for c in celdas):
                for c in celdas:
                    if campos[c]["estado"] not in (IGUAL, NO_APLICA):
                        campos[c].update(estado=TOLERADO, motivo="la fila completa coincide")

    salida = list(campos.values())
    for nombre_tabla in tablas_estado_actual:
        salida += _comparar_tabla_estado_actual(post, ofi, plantilla, nombre_tabla, plantilla["tablas"][nombre_tabla],
                                                fe_post, es_ocr)
    return salida, misma


def _senal(tipo, severidad, detalle):
    """Señal forense: severidad "fraude", "observacion" o "info"."""
    return {"tipo": tipo, "severidad": severidad, "detalle": detalle}


def _zonas_tocadas(rect, plantilla):
    """Zonas de la plantilla que el rectángulo toca."""
    return [n for n, z in plantilla["zonas"].items() if _intersecta(rect, z["rect"])]


def _dibujo_de_plantilla(d, dibujos_tpl, tol=1.5):
    """True si el dibujo `d` es uno de la plantilla (misma posición ±`tol` pt y mismo relleno)."""
    for t in dibujos_tpl:
        if all(abs(d[i] - t[i]) <= tol for i in range(4)):
            if (d[4] is None) == (t[4] is None) and (d[4] is None or max(abs(a - b) for a, b in zip(d[4], t[4])) < 0.03):
                return True
    return False


def senales_objetos(post, plantilla):
    """PDF nativo: objetos que tapan o alteran lo que se ve sin cambiar la capa de texto."""
    s = []
    obj = post.get("objetos") or {}
    for r in obj.get("imagenes", []):
        tocadas = _zonas_tocadas(r, plantilla)
        if tocadas:
            s.append(_senal("objeto_superpuesto", "fraude",
                            f"imagen superpuesta sobre campo(s) de datos: {', '.join(tocadas[:5])}"))
    for tipo, *r in obj.get("anotaciones", []):
        tocadas = _zonas_tocadas(r, plantilla)
        s.append(_senal("anotacion", "fraude" if tocadas else "observacion",
                        f"anotación '{tipo}' agregada al PDF" + (f" sobre: {', '.join(tocadas[:5])}" if tocadas else "")))
    ajenos = [d for d in obj.get("dibujos", []) if not _dibujo_de_plantilla(d, plantilla.get("dibujos", []))]
    tocadas = sorted({n for d in ajenos for n in _zonas_tocadas(d[:4], plantilla)})
    if tocadas:
        s.append(_senal("objeto_superpuesto", "fraude",
                        f"{len(ajenos)} objeto(s) gráfico(s) que no son de la plantilla sobre: {', '.join(tocadas[:6])}"))
    ocultas = post.get("texto_oculto") or []
    en_zonas = sorted({n for t in ocultas for n in _zonas_tocadas(t[1:5], plantilla)})
    if en_zonas:
        s.append(_senal("texto_oculto", "fraude",
                        f"texto invisible o tapado dentro de campo(s) de datos: {', '.join(en_zonas[:6])}"))
    elif ocultas:
        s.append(_senal("texto_oculto", "observacion", f"{len(ocultas)} palabra(s) de texto invisible en el PDF"))
    return s


def senales_forenses(post, ofi, plantilla):
    """Señales más allá de los campos: QR, coherencia de lo impreso con su propio QR, capa de texto, objetos
    superpuestos y, en PDF nativos, metadatos, geometría y tipografías.

    - En escaneos, los códigos técnicos del QR (cod, h) van en Courier fino y su OCR es poco fiable: no se exigen.
    - Si el oficial tiene QR y el del postulante no se puede leer (o fue tapado), se observa.
    - Los metadatos solos no prueban un cambio de contenido (unir, comprimir o reguardar un PDF genuino también los
      cambia): se informan como contexto; la evidencia está en lo demás.
    """
    s = []
    es_ocr = not post.get("nativo")
    qp, qo = post.get("qr"), ofi.get("qr")

    if qp and qo and qp != qo and _misma_instancia(post, ofi, plantilla):
        s.append(_senal("qr", "fraude", "el QR del documento no coincide con el QR del documento oficial"))
    if qp:
        textos = {clave(v["texto"], es_ocr) for v in post.get("valores", {}).values()}
        omitir = {"t", "cod", "h"} if es_ocr else {"t"}
        faltan = [f"{k}={v}" for k, v in qp.items()
                  if k not in omitir and clave(str(v), es_ocr) and clave(str(v), es_ocr) not in textos]
        if faltan:
            s.append(_senal("qr_vs_impreso", "observacion" if es_ocr else "fraude",
                            "lo impreso no coincide con su propio QR: " + ", ".join(faltan[:5])))
    elif qo:
        s.append(_senal("qr", "observacion", "el QR del documento no se puede leer (el oficial sí tiene QR)"))

    if post.get("capa_texto_ignorada"):
        s.append(_senal("capa_texto", "info", f"la página es una imagen; se ignoró su capa de texto "
                                              f"({post['capa_texto_ignorada']} palabras)"))

    if not es_ocr:
        s += senales_objetos(post, plantilla)
    if not es_ocr and ofi.get("nativo"):
        mp, mo = post.get("meta", {}), ofi.get("meta", {})
        if mp.get("producer") != mo.get("producer"):
            s.append(_senal("metadatos", "info",
                            f"PDF producido con '{mp.get('producer')}' (oficial: '{mo.get('producer')}')"))
        if mp.get("modDate") and mp.get("modDate") != mp.get("creationDate") \
                and mo.get("modDate") == mo.get("creationDate"):
            s.append(_senal("metadatos", "info", "PDF modificado después de su creación"))
        geo = post.get("geometria") or {}
        desplazado = abs(geo.get("dx", 0)) > C.TOLERANCIA_GEOMETRIA_PT or abs(geo.get("dy", 0)) > C.TOLERANCIA_GEOMETRIA_PT \
            or abs(geo.get("escala", 1) - 1) > 0.002
        if desplazado:
            s.append(_senal("geometria", "observacion",
                            f"el contenido está desplazado respecto a la plantilla "
                            f"(dx={geo['dx']:.1f} pt, dy={geo['dy']:.1f} pt, escala={geo['escala']:.3f}): "
                            "posible documento recreado"))
        fp, fo = post.get("valores", {}), ofi.get("valores", {})
        distintas = [z for z in fp if z in fo and fp[z].get("fuentes") and fo[z].get("fuentes")
                     and fp[z]["fuentes"] != fo[z]["fuentes"]]
        if distintas:
            s.append(_senal("fuentes", "observacion",
                            f"tipografía distinta a la oficial en {len(distintas)} campo(s): "
                            + ", ".join(distintas[:6])))
    return s


def control_calidad(post):
    """Problemas que impiden confiar en la lectura (se reportan como OBSERVADO).

    En escaneados las plantillas comparten encabezado y pie: el margen de la clasificación puede ser bajo aunque el
    tipo sea correcto. Si las etiquetas leídas lo confirman (cobertura ≥ 60 %), no es una observación."""
    s = []
    if post.get("error"):
        s.append(_senal("archivo", "observacion", f"no se pudo leer el archivo: {post['error']}"))
        return s
    if not post.get("alineacion_ok"):
        s.append(_senal("alineacion", "observacion",
                        f"alineación deficiente con la plantilla: {post.get('alineacion_motivo')}"))
    if not post.get("nativo") and post.get("alineacion_ok"):
        r = post.get("residual_etiquetas_pt")
        if post.get("n_etiquetas", 0) < C.MIN_ETIQUETAS_VERIFICADAS:
            s.append(_senal("alineacion", "observacion", "no se pudieron verificar suficientes etiquetas tras alinear"))
        elif r is not None and r > C.MAX_RESIDUAL_ETIQUETAS_PT:
            s.append(_senal("alineacion", "observacion", f"las etiquetas quedaron desplazadas {r:.1f} pt tras alinear"))
        if post.get("cobertura_etiquetas", 0) < 0.5:
            s.append(_senal("clasificacion", "observacion",
                            f"el formato no coincide bien con la plantilla (cobertura {post['cobertura_etiquetas']:.0%})"))
    confirmado_por_etiquetas = not post.get("nativo") and post.get("cobertura_etiquetas", 0) >= 0.6
    if not post.get("clasificacion_ok") and not confirmado_por_etiquetas:
        s.append(_senal("clasificacion", "observacion", "tipo de documento identificado con poco margen"))
    return s


def veredicto(campos, senales):
    """FALSIFICADO > OBSERVADO > DESACTUALIZADO > AUTÉNTICO, con sus motivos."""
    diferentes = [c for c in campos if c["estado"] == DIFERENTE]
    dudosos = [c for c in campos if c["estado"] == DUDOSO]
    desactualizados = [c for c in campos if c["estado"] == DESACTUALIZADO_C]
    if diferentes or any(x["severidad"] == "fraude" for x in senales):
        v = FALSIFICADO
    elif dudosos or any(x["severidad"] == "observacion" for x in senales):
        v = OBSERVADO
    elif desactualizados:
        v = DESACTUALIZADO
    else:
        v = AUTENTICO
    motivos = [f"{c['zona']}: oficial '{c['oficial']}' vs postulante '{c['postulante']}'"
               + (f" ({c['motivo']})" if c["motivo"] else "") for c in diferentes]
    motivos += [x["detalle"] for x in senales if x["severidad"] in ("fraude", "observacion")]
    motivos += [f"{c['zona']}: {c['motivo']}" for c in dudosos]
    motivos += [f"{c['zona']}: desactualizado ({c['motivo']})" for c in desactualizados[:5]]
    return v, motivos


def dictaminar(post, ofi, plantilla, senales_extra=()):
    """Dictamen de un documento: control de calidad, comparación de campos y señales forenses.

    Sin documento oficial, en un PDF nativo igual se buscan objetos que tapan o esconden datos (son evidencia por sí
    mismos). Si no se confirmó el tipo de documento o la alineación, no se comparan campos (se compararía contra
    la zona equivocada): el documento queda OBSERVADO.
    """
    senales = control_calidad(post) + list(senales_extra)
    campos, misma = [], None
    if ofi is None:
        senales.append(_senal("fuente_oficial", "observacion", "no hay documento oficial para comparar"))
        if post.get("nativo"):
            senales += senales_objetos(post, plantilla)
    elif not any(x["tipo"] in TIPOS_NO_COMPARABLE for x in senales):
        campos, misma = comparar_campos(post, ofi, plantilla)
        senales += senales_forenses(post, ofi, plantilla)
        if not misma:
            senales.append(_senal("instancia", "info", "el documento oficial es otra emisión: se compararon "
                                                       "identidad y datos vigentes, no los datos de emisión"))

    v, motivos = veredicto(campos, senales)
    return {"veredicto": v, "campos": campos, "senales": senales, "motivos": motivos, "misma_instancia": misma}
