"""Reglas del perfil operator_certification que no necesitan plantillas ni OCR: normalización, comparación,
dictamen y aptitud."""

from datetime import date

from verification.profiles.operator_certification import rules as K
from verification.profiles.operator_certification import record_body as HV
from verification.profiles.operator_certification import normalize as N
from verification.profiles.operator_certification import requirements as Q

VALID_ID = "12.345.678-5"
OTHER_ID = "11.111.111-1"


def test_dates_in_all_formats():
    """Fechas numéricas, con mes escrito o abreviado, con el separador perdido por el OCR ("29-dic2025") o con el
    mes mal leído ("Novienbre"). Si el día es ilegible o falta, no se inventa una fecha."""
    assert N.date_value("05-01-2028") == date(2028, 1, 5)
    assert N.date_value("31 de agosto de 2024") == date(2024, 8, 31)
    assert N.date_value("22-jul-2026") == date(2026, 7, 22)
    assert N.date_value("29-dic2025") == date(2025, 12, 29)
    assert N.date_value("5 Novienbre 2019") == date(2019, 11, 5)
    assert N.date_value("8. de diciembre de 2025") == date(2025, 12, 8)
    assert N.date_value("2: de diciembre del 2027") == date(2027, 12, 2)
    assert N.date_value("100% 18290 NOVIEMBRE 2012") is None
    assert N.date_value("MARZO 2023") is None


def test_id_and_check_digit():
    """El RUT se normaliza sin puntos; una lectura parcial o con dígito verificador incorrecto no es válida."""
    assert N.national_id("12.345.678-5") == N.national_id("12345678-5") == "12345678-5"
    assert N.national_id_valid("12345678-5") and N.national_id_valid(N.national_id(OTHER_ID))
    assert not N.national_id_valid("12345678-4")
    assert not N.national_id_valid("345678-5")
    assert N.national_id_ocr("l2.345.678-5") == "12345678-5"


def test_classes_ignore_reading_leftovers():
    """Las clases de licencia ignoran palabras vecinas y clases inexistentes como "A44"."""
    assert N.classes("B-D-A2") == {"B", "D", "A2"}
    assert N.classes("de B-D") == {"B", "D"}
    assert N.classes("B-A1-A44-A5") == {"B", "A1", "A5"}


def test_json_value_with_paths():
    """Rutas de campo dentro del JSON oficial: objetos anidados con "/" y listas con "[i]"."""
    js = {"Exámenes": {"Conocim. General": "82%"}, "Evaluadores": ["Ana", "Luis"]}
    assert K.json_value(js, "Exámenes/Conocim. General") == "82%"
    assert K.json_value(js, "Evaluadores[1]") == "Luis"
    assert K.json_value(js, "Evaluadores[5]") == ""


def test_native_any_difference_flags_forgery():
    """En un PDF nativo no hay error de lectura: cualquier diferencia es real."""
    assert K.compare_value("identidad", "JUAN PEREZ SOTO", "JUAN PERES SOTO", False, 1.0)[0] == K.DIFFERENT


def test_ocr_other_valid_value_flags_forgery():
    """Con OCR y confianza alta, leer otro valor válido (otro RUT, otro estado, otra clase, otra fecha) acusa."""
    assert K.compare_value("rut", VALID_ID, OTHER_ID, True, 0.99)[0] == K.DIFFERENT
    assert K.compare_value("categorico", "APROBADO", "REPROBADO", True, 0.99, "estado_evaluacion")[0] == K.DIFFERENT
    assert K.compare_value("clases", "B-D", "B-D-A4", True, 0.99)[0] == K.DIFFERENT
    assert K.compare_value("fecha", "05-09-2026", "05-09-2027", True, 0.99)[0] == K.DIFFERENT


def test_ocr_reading_errors_do_not_accuse():
    """Lo que parece error de lectura queda en duda, no acusa: RUT con dígito verificador inválido, 1-2 caracteres
    distintos, valor imposible, texto cortado o corrido, o confianza baja."""
    assert K.compare_value("rut", VALID_ID, "12.345.678-4", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("identidad", "JUAN PEREZ SOTO", "JUAN PERFZ SOTO", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("categorico", "APROBADO", "APROBAD0X", True, 0.99, "estado_evaluacion")[0] == K.DOUBTFUL
    assert K.compare_value("codigo", "AMCH-52330", "AMCH-52", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("categorico", "VALDIVIA", "VALDIVIA RENCA", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("fecha", "05-09-2026", "05 ?? 2026", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("rut", VALID_ID, OTHER_ID, True, 0.30)[0] == K.DOUBTFUL


def test_contaminated_readings_do_not_accuse():
    """Una nota sin ningún dígito leído no se convierte en número (las letras O y B no son 0 y 8), y un valor de
    otro campo (APROBADO) dentro de un nombre o domicilio es una lectura contaminada. Con OCR quedan en duda; en
    un PDF nativo siguen acusando."""
    assert K.compare_value("numero", "78%", "APROBADO", True, 0.99, "nota_refuerzo_brechas")[0] == K.DOUBTFUL
    assert K.compare_value("identidad", "Jorge Vidal Acuña", "Jorge APROBADO Vidal Soto", True, 0.99,
                            "evaluador2_nombre")[0] == K.DOUBTFUL
    assert K.compare_value("descriptivo", "AVENIDA COLON 383", "AVENIDA COLON 383 100% APROBADO", True, 0.99,
                            "domicilio")[0] == K.DOUBTFUL
    assert K.compare_value("identidad", "Jorge Vidal Acuña", "Jorge APROBADO Vidal Acuña", False, 1.0,
                            "evaluador2_nombre")[0] == K.DIFFERENT
    assert K.compare_value("numero", "78%", "98%", True, 0.99, "nota_refuerzo_brechas")[0] == K.DIFFERENT


def test_noise_that_keeps_the_value_is_tolerated():
    """Palabras de enlace junto al valor oficial completo, texto de otro campo (APROBADO...) dentro de la zona y un
    nombre de evaluador casi idéntico se toleran. Otro valor, otro evaluador o un PDF nativo siguen igual."""
    assert K.compare_value("categorico", "VALDIVIA", "de VALDIVIA", True, 0.99)[0] == K.TOLERATED
    assert K.compare_value("categorico", "MEJILLONES", "N° MEJILLONES", True, 0.95)[0] == K.TOLERATED
    assert K.compare_value("identidad", "Jorge Vidal Acuña", "Jorge APROBADcionamientoROBADO Vidal Acuña", True, 0.99,
                            "evaluador2_nombre")[0] == K.TOLERATED
    assert K.compare_value("identidad", "Alexis Gutiérrez Peralta", "Alexis Gutiérrez Peraka", True, 0.40,
                            "evaluador1_nombre")[0] == K.TOLERATED
    assert K.compare_value("identidad", "Alexis Gutiérrez Peralta", "Jorge Vidal Acuña", True, 0.99,
                            "evaluador1_nombre")[0] == K.DIFFERENT
    assert K.compare_value("identidad", "JUAN PEREZ SOTO", "JUAN PERFZ SOTO", True, 0.99, "nombre")[0] == K.DOUBTFUL
    assert K.compare_value("categorico", "VALDIVIA", "de VALDIVIA", False, 1.0)[0] == K.DIFFERENT


def test_page_2_repetition_confirmed_by_page_1():
    """Una lectura dudosa del nombre repetido en la página 2 no manda a revisión si la página 1 lo confirma; un nombre
    distinto leído con confianza sigue acusando."""
    zones = {"nombre": {"tipo": "identidad", "clave": "Nombre"}, "nombre (pág. 2)": {"tipo": "identidad", "clave": "Nombre"}}
    doc = _doc({"nombre": "JUAN PEREZ SOTO"})
    doc["valores"]["nombre (pág. 2)"] = {"texto": "JUAN PEREZ SOTA", "conf": 0.30}
    r = K.compare_document(doc, JS, zones, {}, date(2026, 10, 18))
    assert r["veredicto"] == K.AUTHENTIC
    doc["valores"]["nombre (pág. 2)"] = {"texto": "PEDRO ROJAS DIAZ", "conf": 0.99}
    assert K.compare_document(doc, JS, zones, {}, date(2026, 10, 18))["veredicto"] == K.FORGED


def test_license_renewed_after_issue_without_read_class():
    """La fila previa a una renovación posterior a la emisión deja el documento desactualizado aunque no se haya leído
    la clase. Una fecha impresa a menos de un año de la renovación oficial (licencia "renovada" en el documento) va a
    revisión."""
    js = dict(RECORD_JSON, Anotaciones=[], **{"Licencias registradas": [
        {"Registro": "ULTIMA", "Clase": "D", "Fecha otorgamiento": "07-10-2026", "Municipalidad": "CONCEPCION"}]})
    def row(date_value, lclass=""):
        """Filas leídas con una sola licencia ÚLTIMA de la fecha y clase dadas."""
        return {"licencias": [{"Registro": "ULTIMA", "Clase": lclass, "fecha": date_value, "fecha_txt": str(date_value),
                               "Municipalidad": "CONCEPCION", "conf": 0.9}], "anotaciones": [], "conf_min": 0.9,
                "antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"}}

    def states(f):
        """Estados de los campos comparados, con la Hoja de Vida emitida el 26-09-2026."""
        return {c["estado"] for c in K._compare_rows(f, js, date(2026, 9, 26), True)}

    assert states(row(date(2020, 9, 22))) == {K.OUTDATED_C, K.EQUAL}
    assert K.DOUBTFUL in states(row(date(2026, 7, 11)))
    assert K.DOUBTFUL in states(row(date(2026, 7, 11), "D"))


def test_changed_date_in_license_without_read_class():
    """Si la clase de una fila no se leyó, pero es la única fila que puede ser la licencia oficial que falta (mismo
    registro y municipalidad), su fecha distinta leída con confianza alta acusa; con dos filas posibles no se elige."""
    lic = [{"Registro": "ULTIMA", "Clase": "B", "Fecha otorgamiento": "07-01-2022", "Municipalidad": "PUERTO MONTT"},
           {"Registro": "ULTIMA", "Clase": "D", "Fecha otorgamiento": "04-01-2021", "Municipalidad": "PUERTO MONTT"}]
    js = dict(RECORD_JSON, Anotaciones=[], **{"Licencias registradas": lic})

    def rows(*without_class):
        """La licencia B leída tal cual y filas sin clase con las fechas dadas."""
        def f(lclass, date_value):
            """Una fila ÚLTIMA leída con confianza alta."""
            return {"Registro": "ULTIMA", "Clase": lclass, "fecha": date_value, "fecha_txt": f"{date_value:%d-%m-%Y}",
                    "Municipalidad": "PUERTO MONTT", "conf": 0.95}
        return {"licencias": [f("B", date(2022, 1, 7))] + [f("", x) for x in without_class], "anotaciones": [],
                "conf_min": 0.95, "antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"}}

    def date_d(f):
        """Estado de la fecha de la licencia D emparejada (None si no se emparejó), con emisión el 26-09-2026."""
        return next((c["estado"] for c in K._compare_rows(f, js, date(2026, 9, 26), True)
                     if c["zona"] == "licencia ultima clase D - fecha"), None)

    assert date_d(rows(date(2026, 3, 7))) == K.DIFFERENT
    assert date_d(rows(date(2026, 3, 7), date(2025, 2, 1))) is None
    assert date_d(rows(date(2021, 1, 4))) == K.EQUAL


def test_nonexistent_class_is_not_equal():
    """"A44" puede ser una clase A4 agregada y mal leída: se revisa, no se da por auténtica."""
    assert K.compare_value("clases", "B-A1-A5", "B-A1-A44-A5", True, 0.99)[0] == K.DOUBTFUL
    assert K.compare_value("clases", "B-D", "de B-D", True, 0.99)[0] == K.TOLERATED


def test_same_value_in_other_format():
    """El mismo valor escrito en otro formato se tolera."""
    assert K.compare_value("fecha", "31-08-2024", "31 de agosto de 2024", True, 0.99)[0] == K.TOLERATED
    assert K.compare_value("rut", "12345678-5", VALID_ID, False, 1.0)[0] == K.TOLERATED


def _field(state):
    """Campo comparado mínimo con el estado dado."""
    return {"zona": "x", "oficial": "a", "postulante": "b", "estado": state, "motivo": "", "conf": 1.0}


def test_verdict_priority():
    """NO LEGIBLE > FALSIFICADO > NO VÁLIDO (copia) > OBSERVADO > DESACTUALIZADO > AUTÉNTICO. Una copia se rechaza
    aunque solo tenga dudas, pero una diferencia real la marca como falsificada."""
    assert K.verdict([_field(K.DIFFERENT)], [], illegible=True)[0] == K.ILLEGIBLE
    assert K.verdict([_field(K.DIFFERENT), _field(K.DOUBTFUL)], [])[0] == K.FORGED
    assert K.verdict([_field(K.DOUBTFUL)], [], copy_doc=True)[0] == K.NOT_VALID
    assert K.verdict([_field(K.DIFFERENT)], [], copy_doc=True)[0] == K.FORGED
    assert K.verdict([_field(K.DOUBTFUL)], [])[0] == K.FLAGGED
    assert K.verdict([_field(K.EQUAL)], [], copy_doc=True, outdated=True)[0] == K.NOT_VALID
    assert K.verdict([_field(K.EQUAL)], [], outdated=True)[0] == K.OUTDATED
    assert K.verdict([_field(K.EQUAL)], [])[0] == K.AUTHENTIC


def _doc(values, **extra):
    """Documento leído por OCR con los valores dados (confianza 0,99)."""
    return {"es_ocr": True, "valores": {k: {"texto": v, "conf": 0.99} for k, v in values.items()},
            "tipo_documento": "ameco", "senales": []} | extra


ZONES = {"nombre": {"tipo": "identidad", "clave": "Nombre"}, "domicilio": {"tipo": "descriptivo", "clave": "Domicilio"}}
JS = {"Nombre": "JUAN PEREZ SOTO", "Domicilio": "COLON 1560"}


def test_illegible_document_is_not_compared():
    """Un documento ilegible no se compara campo a campo: se informa el motivo."""
    r = K.compare_document(_doc({}, ilegible=True, motivo_ilegible="imagen ilegible"), JS, ZONES, {}, date(2026, 10, 18))
    assert r["veredicto"] == K.ILLEGIBLE and not r["campos"] and r["motivos"] == ["imagen ilegible"]


def test_shifted_zone_with_neighbor_label_does_not_accuse():
    """Si la zona leída contiene la etiqueta de un campo vecino (zona corrida), queda en observación; sin esa señal,
    la misma diferencia acusa."""
    doc = _doc({"nombre": "JUAN PEREZ SOTO", "domicilio": "Fecha nacimiento: 24 Abril 1992"},
               formato_ajeno={"domicilio": {"FECHA", "NACIMIENTO"}})
    r = K.compare_document(doc, JS, ZONES, {}, date(2026, 10, 18))
    assert r["veredicto"] == K.FLAGGED
    doc["formato_ajeno"] = {}
    assert K.compare_document(doc, JS, ZONES, {}, date(2026, 10, 18))["veredicto"] == K.FORGED


def test_document_with_another_id():
    """Un documento auténtico pero de otra persona (otro RUT que el declarado) es una señal de fraude de identidad."""
    zones = {"rut": {"tipo": "rut", "clave": "RUT"}}
    doc = _doc({"rut": OTHER_ID}, es_ocr=False)
    r = K.compare_document(doc, {"RUT": OTHER_ID}, zones, {"rut": VALID_ID}, date(2026, 10, 18))
    assert r["veredicto"] == K.FORGED and any(s["tipo"] == "identidad" for s in r["senales"])


def _lines(*texts, x0=211):
    """Líneas de texto leídas del cuerpo de la Hoja de Vida, todas en la misma columna."""
    return [{"x0": x0, "texto": t, "conf": 0.99} for t in texts]


def test_parse_licenses_and_municipality_with_date_leftovers():
    """Licencias de la Hoja de Vida: la municipalidad se limpia de restos de la fecha (meses sueltos)."""
    f = HV.parse([_lines("PRIMERA CLASE B FECHA OTORGAMIENTO 5 FEBRERO 2019", "MUNICIPALIDAD FEB FEB ALTO HOSPICIO",
                            "ULTIMA CLASE B FECHA OTORGAMIENTO 1 JULIO 2024", "MUNICIPALIDAD LOTA")])
    assert [l["Municipalidad"] for l in f["licencias"]] == ["ALTO HOSPICIO", "LOTA"]
    assert f["licencias"][0]["fecha"] == date(2019, 2, 5) and f["licencias"][1]["Clase"] == "B"


def test_background_by_structure():
    """Los antecedentes (PRN, RNC) se reconocen por la estructura de la línea, no por su posición."""
    f = HV.parse([_lines("12345678-5 SIN ANTECEDENTES PRN CONDUCTOR", "12345678-5 CON ANTECEDENTES RNC")])
    assert f["antecedentes"] == {"PRN": "SIN ANTECEDENTES", "RNC": "CON ANTECEDENTES"}


RECORD_JSON = {"Licencias registradas": [], "Antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"},
         "Anotaciones": [{"Tipo": "INFRACCION GRAVE", "Fecha anotación": "14-06-1994",
                          "Detalle": "CONDENADO POR EL JUZGADO DE POLICIA LOCAL DE LA SERENA ROL N 44055 POR INFRACCION"}]}


def test_without_page_2_missing_annotations_do_not_accuse():
    """Si no se entregó la página 2, las anotaciones que no aparecen no acusan; con la página 2, sí."""
    empty_value = {"licencias": [], "anotaciones": [], "antecedentes": {}, "conf_min": 0.99}
    fields = K._compare_rows(empty_value, RECORD_JSON, date(2026, 10, 1), True, has_p2=False)
    assert {c["estado"] for c in fields} == {K.NOT_APPLICABLE}
    fields = K._compare_rows(empty_value, RECORD_JSON, date(2026, 10, 1), True, has_p2=True)
    assert any(c["estado"] == K.DIFFERENT for c in fields)


def test_annotation_cut_at_end_of_page_1():
    """Una anotación cortada al final de la página 1 se tolera si su comienzo coincide con la oficial."""
    cut = {"Tipo": "INFRACCION GRAVE", "fecha": date(1994, 6, 14), "conf": 0.99,
               "Detalle": "CONDENADO POR EL JUZGADO DE POLICIA LOCAL DE LA SERENA"}
    doc = {"licencias": [], "anotaciones": [cut], "antecedentes": {}, "conf_min": 0.99}
    fields = K._compare_rows(doc, RECORD_JSON, date(2026, 10, 1), False, has_p2=False)
    assert [c["estado"] for c in fields if c["zona"].startswith("anotación")] == [K.TOLERATED]


def test_cut_annotation_without_date_and_license_on_page_2():
    """Anotación cortada sin fecha y licencia que quedó en la página 2 (no entregada): nada acusa."""
    js = dict(RECORD_JSON, **{"Licencias registradas": [{"Registro": "ULTIMA", "Clase": "A5",
                                                         "Fecha otorgamiento": "25 Enero 2023",
                                                         "Municipalidad": "POZO ALMONTE"}]})
    undated = {"Tipo": "INFRACCION GRAVE", "fecha": None, "conf": 1.0, "Detalle": "CONDENADO POR EL JUZGADO"}
    doc = {"licencias": [], "anotaciones": [undated], "antecedentes": {}, "conf_min": 1.0}
    fields = K._compare_rows(doc, js, date(2026, 10, 1), False, has_p2=False)
    assert K.DIFFERENT not in {c["estado"] for c in fields}


def test_background_changing_after_issue():
    """Antecedentes que cambiaron después de emitido el documento lo dejan desactualizado; si el documento se emitió
    después del cambio, la diferencia sí acusa."""
    js = dict(RECORD_JSON, Antecedentes={"PRN": "CON ANTECEDENTES", "RNC": "CON ANTECEDENTES"},
              Anotaciones=[{"Tipo": "INFRACCION GRAVE", "Fecha anotación": "09-09-2026", "Detalle": "CONDENADO"}])
    doc = {"licencias": [], "anotaciones": [], "conf_min": 1.0,
           "antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"}}
    fields = K._compare_rows(doc, js, date(2026, 8, 1), False)
    assert {c["estado"] for c in fields} == {K.OUTDATED_C}
    fields = K._compare_rows(doc, js, date(2026, 9, 30), False)
    assert K.DIFFERENT in {c["estado"] for c in fields}


def test_confidence_levels():
    """Niveles ALTA / MEDIA / BAJA / SIN LECTURA según la confianza; un PDF original (sin OCR) es siempre ALTA. La
    vista plana recorre objetos anidados y listas en el orden del JSON."""
    from verification.profiles.operator_certification import extraction as EX
    assert EX._level(0.93, "x", True)["nivel"] == "ALTA"
    assert EX._level(0.62, "x", True)["nivel"] == "MEDIA"
    assert EX._level(0.21, "x", True)["nivel"] == "BAJA"
    assert EX._level(0.21, "", True)["nivel"] == "SIN LECTURA"
    assert EX._level(0.21, "x", False) == {"confianza": 1.0, "nivel": "ALTA"}
    js = {"Nombre": "ANA", "Exámenes": {"Nota": "90%"}, "Evaluadores": ["A", "B"], "Lic": [{"Clase": "B"}]}
    conf = {"Nombre": {"confianza": 0.9, "nivel": "ALTA"}, "Exámenes": {"Nota": {"confianza": 0.3, "nivel": "BAJA"}},
            "Evaluadores": [{"nivel": "ALTA"}, {"nivel": "MEDIA"}], "Lic": [{"nivel": "MEDIA"}]}
    assert [(x["campo"], x["nivel"]) for x in EX.flat(js, conf)] == [
        ("Nombre", "ALTA"), ("Exámenes / Nota", "BAJA"), ("Evaluadores 1", "ALTA"), ("Evaluadores 2", "MEDIA"),
        ("Lic 1 / Clase", "MEDIA")]


REQ = Q.load()
TODAY = date(2026, 10, 18)


def _record(class_dates, rnc="SIN ANTECEDENTES", annotations=()):
    """JSON oficial de Hoja de Vida con las licencias (clase, fecha) dadas."""
    return {"Licencias registradas": [{"Registro": "ULTIMA", "Clase": c, "Fecha otorgamiento": f, "Municipalidad": "X"}
                                      for c, f in class_dates],
            "Antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": rnc}, "Anotaciones": list(annotations)}


def _certificate(state="APROBADO", expires="01-01-2028", equipment="Grúa Móvil Terreno Aspero"):
    """JSON oficial de un certificado AMECO."""
    return {"N°": "AMCH-1", "Estado evaluación": state, "Fecha Expiración": expires, "Operador Equipo": equipment}


def _docs(rec="AUTÉNTICO", cert="AUTÉNTICO"):
    """Dictámenes de la Hoja de Vida y del AMECO entregados."""
    return [{"tipo_documento": "hoja_vida_p1", "veredicto": rec},
            {"tipo_documento": "ameco_rev03", "veredicto": cert, "archivo_oficial_clave": "AMCH-1"}]


def _evaluate(rec_js, cert_js, docs=None, position="Operador de Grúa Móvil"):
    """Aptitud para el cargo con los requisitos de `verification/profiles/operator_certification/requirements.yaml`."""
    return Q.evaluate_person(position, docs or _docs(), {"hoja_vida": rec_js, "ameco": {"AMCH-1": cert_js}}, REQ, TODAY)


def test_fit():
    """Licencias D y A4 vigentes y AMECO aprobado del equipo del cargo: apto."""
    assert _evaluate(_record([("D", "01-01-2024"), ("A4", "01-01-2025")]), _certificate())[0] == Q.FIT


def test_not_fit_by_license_certificate_and_background():
    """No apto: sin clase A2/A4, control de licencia vencido, AMECO expirado o reprobado, antecedentes en el RNC o
    licencia suspendida."""
    assert _evaluate(_record([("D", "01-01-2024")]), _certificate())[0] == Q.NOT_FIT
    assert _evaluate(_record([("D", "01-01-2024"), ("A2", "01-01-2020")]), _certificate())[0] == Q.NOT_FIT
    assert _evaluate(_record([("D", "01-01-2024"), ("A2", "01-01-2025")]), _certificate(expires="01-01-2026"))[0] == Q.NOT_FIT
    assert _evaluate(_record([("D", "01-01-2024"), ("A2", "01-01-2025")]), _certificate(state="REPROBADO"))[0] == Q.NOT_FIT
    assert _evaluate(_record([("D", "01-01-2024"), ("A2", "01-01-2025")], rnc="CON ANTECEDENTES"),
                     _certificate())[0] == Q.NOT_FIT
    susp = {"Tipo": "SUSPENSION DE LICENCIA DE CONDUCTOR", "Fecha anotación": "01-03-2026"}
    assert _evaluate(_record([("D", "01-01-2024"), ("A2", "01-01-2025")], annotations=[susp]), _certificate())[0] == Q.NOT_FIT


def test_incomplete_by_copy_or_position():
    """Incompleto: Hoja de Vida en copia o desactualizada, o AMECO de otro equipo que el del cargo."""
    rec = _record([("D", "01-01-2024"), ("A4", "01-01-2025")])
    assert _evaluate(rec, _certificate(), _docs(rec="NO VÁLIDO"))[0] == Q.INCOMPLETE
    assert _evaluate(rec, _certificate(), _docs(rec="DESACTUALIZADO"))[0] == Q.INCOMPLETE
    assert _evaluate(rec, _certificate(equipment="Grúa Torre"))[0] == Q.INCOMPLETE


def test_position_uppercase_and_unconfigured_position():
    """El cargo se reconoce sin importar mayúsculas ni tildes; un cargo sin requisitos configurados se informa."""
    rec = _record([("D", "01-01-2024"), ("A4", "01-01-2025")])
    assert _evaluate(rec, _certificate(), position="OPERADOR DE GRUA MOVIL")[0] == Q.FIT
    fit, reasons = _evaluate(rec, _certificate(equipment="Grúa Torre"), position="SUPERVISOR DE FAENA")
    assert fit == Q.FIT and any("no tiene requisitos configurados" in m for m in reasons)


def test_without_official_data_is_not_evaluated():
    """Si falta el dato oficial de un documento, la aptitud no se puede evaluar."""
    rec = _record([("D", "01-01-2024"), ("A4", "01-01-2025")])
    docs = [{"tipo_documento": "hoja_vida_p1", "veredicto": "AUTÉNTICO"},
            {"tipo_documento": "ameco_rev03", "veredicto": K.NO_OFFICIAL_DATA, "archivo_oficial_clave": None}]
    assert Q.evaluate_person("Operador de Grúa Móvil", docs, {"hoja_vida": rec, "ameco": {}}, REQ, TODAY)[0] == Q.NOT_EVALUABLE
