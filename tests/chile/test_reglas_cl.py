"""Reglas del flujo chileno que no necesitan plantillas ni OCR: normalización, comparación, dictamen y aptitud."""

from datetime import date

from verificacion.chile import comparar_cl as K
from verificacion.chile import hvid_cuerpo as HV
from verificacion.chile import normalizar_cl as N
from verificacion.chile import requisitos_cl as Q

RUT_OK = "12.345.678-5"
RUT_OTRO = "11.111.111-1"


def test_fechas_en_todos_los_formatos():
    """Fechas numéricas, con mes escrito o abreviado, con el separador perdido por el OCR ("29-dic2025") o con el
    mes mal leído ("Novienbre"). Si el día es ilegible o falta, no se inventa una fecha."""
    assert N.fecha("05-01-2028") == date(2028, 1, 5)
    assert N.fecha("31 de agosto de 2024") == date(2024, 8, 31)
    assert N.fecha("22-jul-2026") == date(2026, 7, 22)
    assert N.fecha("29-dic2025") == date(2025, 12, 29)
    assert N.fecha("5 Novienbre 2019") == date(2019, 11, 5)
    assert N.fecha("8. de diciembre de 2025") == date(2025, 12, 8)
    assert N.fecha("2: de diciembre del 2027") == date(2027, 12, 2)
    assert N.fecha("100% 18290 NOVIEMBRE 2012") is None
    assert N.fecha("MARZO 2023") is None


def test_rut_y_digito_verificador():
    """El RUT se normaliza sin puntos; una lectura parcial o con dígito verificador incorrecto no es válida."""
    assert N.rut("12.345.678-5") == N.rut("12345678-5") == "12345678-5"
    assert N.rut_valido("12345678-5") and N.rut_valido(N.rut(RUT_OTRO))
    assert not N.rut_valido("12345678-4")
    assert not N.rut_valido("345678-5")
    assert N.rut_ocr("l2.345.678-5") == "12345678-5"


def test_clases_ignoran_restos_de_lectura():
    """Las clases de licencia ignoran palabras vecinas y clases inexistentes como "A44"."""
    assert N.clases("B-D-A2") == {"B", "D", "A2"}
    assert N.clases("de B-D") == {"B", "D"}
    assert N.clases("B-A1-A44-A5") == {"B", "A1", "A5"}


def test_valor_json_con_rutas():
    """Rutas de campo dentro del JSON oficial: objetos anidados con "/" y listas con "[i]"."""
    js = {"Exámenes": {"Conocim. General": "82%"}, "Evaluadores": ["Ana", "Luis"]}
    assert K.valor_json(js, "Exámenes/Conocim. General") == "82%"
    assert K.valor_json(js, "Evaluadores[1]") == "Luis"
    assert K.valor_json(js, "Evaluadores[5]") == ""


def test_nativo_cualquier_diferencia_acusa():
    """En un PDF nativo no hay error de lectura: cualquier diferencia es real."""
    assert K.comparar_valor("identidad", "JUAN PEREZ SOTO", "JUAN PERES SOTO", False, 1.0)[0] == K.DIFERENTE


def test_ocr_otro_valor_valido_acusa():
    """Con OCR y confianza alta, leer otro valor válido (otro RUT, otro estado, otra clase, otra fecha) acusa."""
    assert K.comparar_valor("rut", RUT_OK, RUT_OTRO, True, 0.99)[0] == K.DIFERENTE
    assert K.comparar_valor("categorico", "APROBADO", "REPROBADO", True, 0.99, "estado_evaluacion")[0] == K.DIFERENTE
    assert K.comparar_valor("clases", "B-D", "B-D-A4", True, 0.99)[0] == K.DIFERENTE
    assert K.comparar_valor("fecha", "05-09-2026", "05-09-2027", True, 0.99)[0] == K.DIFERENTE


def test_ocr_errores_de_lectura_no_acusan():
    """Lo que parece error de lectura queda en duda, no acusa: RUT con dígito verificador inválido, 1-2 caracteres
    distintos, valor imposible, texto cortado o corrido, o confianza baja."""
    assert K.comparar_valor("rut", RUT_OK, "12.345.678-4", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("identidad", "JUAN PEREZ SOTO", "JUAN PERFZ SOTO", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("categorico", "APROBADO", "APROBAD0X", True, 0.99, "estado_evaluacion")[0] == K.DUDOSO
    assert K.comparar_valor("codigo", "AMCH-52330", "AMCH-52", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("categorico", "VALDIVIA", "de VALDIVIA", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("fecha", "05-09-2026", "05 ?? 2026", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("rut", RUT_OK, RUT_OTRO, True, 0.30)[0] == K.DUDOSO


def test_lecturas_contaminadas_no_acusan():
    """Una nota sin ningún dígito leído no se convierte en número (las letras O y B no son 0 y 8), y un valor de
    otro campo (APROBADO) dentro de un nombre o domicilio es una lectura contaminada. Con OCR quedan en duda; en
    un PDF nativo siguen acusando."""
    assert K.comparar_valor("numero", "78%", "APROBADO", True, 0.99, "nota_refuerzo_brechas")[0] == K.DUDOSO
    assert K.comparar_valor("identidad", "Jorge Vidal Acuña", "Jorge APROBADO Vidal Acuña", True, 0.99,
                            "evaluador2_nombre")[0] == K.DUDOSO
    assert K.comparar_valor("descriptivo", "AVENIDA COLON 383", "AVENIDA COLON 383 100% APROBADO", True, 0.99,
                            "domicilio")[0] == K.DUDOSO
    assert K.comparar_valor("identidad", "Jorge Vidal Acuña", "Jorge APROBADO Vidal Acuña", False, 1.0,
                            "evaluador2_nombre")[0] == K.DIFERENTE
    assert K.comparar_valor("numero", "78%", "98%", True, 0.99, "nota_refuerzo_brechas")[0] == K.DIFERENTE


def test_clase_inexistente_no_pasa_como_igual():
    """"A44" puede ser una clase A4 agregada y mal leída: se revisa, no se da por auténtica."""
    assert K.comparar_valor("clases", "B-A1-A5", "B-A1-A44-A5", True, 0.99)[0] == K.DUDOSO
    assert K.comparar_valor("clases", "B-D", "de B-D", True, 0.99)[0] == K.TOLERADO


def test_mismo_valor_en_otro_formato():
    """El mismo valor escrito en otro formato se tolera."""
    assert K.comparar_valor("fecha", "31-08-2024", "31 de agosto de 2024", True, 0.99)[0] == K.TOLERADO
    assert K.comparar_valor("rut", "12345678-5", RUT_OK, False, 1.0)[0] == K.TOLERADO


def _campo(estado):
    """Campo comparado mínimo con el estado dado."""
    return {"zona": "x", "oficial": "a", "postulante": "b", "estado": estado, "motivo": "", "conf": 1.0}


def test_prioridad_de_veredictos():
    """NO LEGIBLE > FALSIFICADO > NO VÁLIDO (copia) > OBSERVADO > DESACTUALIZADO > AUTÉNTICO. Una copia se rechaza
    aunque solo tenga dudas, pero una diferencia real la marca como falsificada."""
    assert K.veredicto([_campo(K.DIFERENTE)], [], ilegible=True)[0] == K.NO_LEGIBLE
    assert K.veredicto([_campo(K.DIFERENTE), _campo(K.DUDOSO)], [])[0] == K.FALSIFICADO
    assert K.veredicto([_campo(K.DUDOSO)], [], copia=True)[0] == K.NO_VALIDO
    assert K.veredicto([_campo(K.DIFERENTE)], [], copia=True)[0] == K.FALSIFICADO
    assert K.veredicto([_campo(K.DUDOSO)], [])[0] == K.OBSERVADO
    assert K.veredicto([_campo(K.IGUAL)], [], copia=True, desactualizado=True)[0] == K.NO_VALIDO
    assert K.veredicto([_campo(K.IGUAL)], [], desactualizado=True)[0] == K.DESACTUALIZADO
    assert K.veredicto([_campo(K.IGUAL)], [])[0] == K.AUTENTICO


def _doc(valores, **extra):
    """Documento leído por OCR con los valores dados (confianza 0,99)."""
    return {"es_ocr": True, "valores": {k: {"texto": v, "conf": 0.99} for k, v in valores.items()},
            "tipo_documento": "ameco", "senales": []} | extra


ZONAS = {"nombre": {"tipo": "identidad", "clave": "Nombre"}, "domicilio": {"tipo": "descriptivo", "clave": "Domicilio"}}
JS = {"Nombre": "JUAN PEREZ SOTO", "Domicilio": "COLON 1560"}


def test_documento_ilegible_no_se_compara():
    """Un documento ilegible no se compara campo a campo: se informa el motivo."""
    r = K.comparar_documento(_doc({}, ilegible=True, motivo_ilegible="imagen ilegible"), JS, ZONAS, {}, date(2026, 10, 18))
    assert r["veredicto"] == K.NO_LEGIBLE and r["campos"] == [] and r["motivos"] == ["imagen ilegible"]


def test_zona_corrida_con_etiqueta_vecina_no_acusa():
    """Si la zona leída contiene la etiqueta de un campo vecino (zona corrida), queda en observación; sin esa señal,
    la misma diferencia acusa."""
    doc = _doc({"nombre": "JUAN PEREZ SOTO", "domicilio": "Fecha nacimiento: 24 Abril 1992"},
               formato_ajeno={"domicilio": {"FECHA", "NACIMIENTO"}})
    r = K.comparar_documento(doc, JS, ZONAS, {}, date(2026, 10, 18))
    assert r["veredicto"] == K.OBSERVADO
    doc["formato_ajeno"] = {}
    assert K.comparar_documento(doc, JS, ZONAS, {}, date(2026, 10, 18))["veredicto"] == K.FALSIFICADO


def test_documento_a_nombre_de_otro_rut():
    """Un documento auténtico pero de otra persona (otro RUT que el declarado) es una señal de fraude de identidad."""
    zonas = {"rut": {"tipo": "rut", "clave": "RUT"}}
    doc = _doc({"rut": RUT_OTRO}, es_ocr=False)
    r = K.comparar_documento(doc, {"RUT": RUT_OTRO}, zonas, {"rut": RUT_OK}, date(2026, 10, 18))
    assert r["veredicto"] == K.FALSIFICADO and any(s["tipo"] == "identidad" for s in r["senales"])


def _lineas(*textos, x0=211):
    """Líneas de texto leídas del cuerpo de la Hoja de Vida, todas en la misma columna."""
    return [{"x0": x0, "texto": t, "conf": 0.99} for t in textos]


def test_parsear_licencias_y_municipalidad_con_restos_de_fecha():
    """Licencias de la Hoja de Vida: la municipalidad se limpia de restos de la fecha (meses sueltos)."""
    f = HV.parsear([_lineas("PRIMERA CLASE B FECHA OTORGAMIENTO 5 FEBRERO 2019", "MUNICIPALIDAD FEB FEB ALTO HOSPICIO",
                            "ULTIMA CLASE B FECHA OTORGAMIENTO 1 JULIO 2024", "MUNICIPALIDAD LOTA")])
    assert [l["Municipalidad"] for l in f["licencias"]] == ["ALTO HOSPICIO", "LOTA"]
    assert f["licencias"][0]["fecha"] == date(2019, 2, 5) and f["licencias"][1]["Clase"] == "B"


def test_antecedentes_por_estructura():
    """Los antecedentes (PRN, RNC) se reconocen por la estructura de la línea, no por su posición."""
    f = HV.parsear([_lineas("12345678-5 SIN ANTECEDENTES PRN CONDUCTOR", "12345678-5 CON ANTECEDENTES RNC")])
    assert f["antecedentes"] == {"PRN": "SIN ANTECEDENTES", "RNC": "CON ANTECEDENTES"}


JS_HV = {"Licencias registradas": [], "Antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"},
         "Anotaciones": [{"Tipo": "INFRACCION GRAVE", "Fecha anotación": "14-06-1994",
                          "Detalle": "CONDENADO POR EL JUZGADO DE POLICIA LOCAL DE LA SERENA ROL N 44055 POR INFRACCION"}]}


def test_sin_pagina_2_las_anotaciones_faltantes_no_acusan():
    """Si no se entregó la página 2, las anotaciones que no aparecen no acusan; con la página 2, sí."""
    vacio = {"licencias": [], "anotaciones": [], "antecedentes": {}, "conf_min": 0.99}
    campos = K._comparar_filas(vacio, JS_HV, date(2026, 10, 1), True, tiene_p2=False)
    assert {c["estado"] for c in campos} == {K.NO_APLICA}
    campos = K._comparar_filas(vacio, JS_HV, date(2026, 10, 1), True, tiene_p2=True)
    assert any(c["estado"] == K.DIFERENTE for c in campos)


def test_anotacion_cortada_al_final_de_la_pagina_1():
    """Una anotación cortada al final de la página 1 se tolera si su comienzo coincide con la oficial."""
    cortada = {"Tipo": "INFRACCION GRAVE", "fecha": date(1994, 6, 14), "conf": 0.99,
               "Detalle": "CONDENADO POR EL JUZGADO DE POLICIA LOCAL DE LA SERENA"}
    doc = {"licencias": [], "anotaciones": [cortada], "antecedentes": {}, "conf_min": 0.99}
    campos = K._comparar_filas(doc, JS_HV, date(2026, 10, 1), False, tiene_p2=False)
    assert [c["estado"] for c in campos if c["zona"].startswith("anotación")] == [K.TOLERADO]


def test_anotacion_cortada_sin_fecha_y_licencia_en_pagina_2():
    """Anotación cortada sin fecha y licencia que quedó en la página 2 (no entregada): nada acusa."""
    js = dict(JS_HV, **{"Licencias registradas": [{"Registro": "ULTIMA", "Clase": "A5", "Fecha otorgamiento": "25 Enero 2023",
                                                   "Municipalidad": "POZO ALMONTE"}]})
    sin_fecha = {"Tipo": "INFRACCION GRAVE", "fecha": None, "conf": 1.0, "Detalle": "CONDENADO POR EL JUZGADO"}
    doc = {"licencias": [], "anotaciones": [sin_fecha], "antecedentes": {}, "conf_min": 1.0}
    campos = K._comparar_filas(doc, js, date(2026, 10, 1), False, tiene_p2=False)
    assert K.DIFERENTE not in {c["estado"] for c in campos}


def test_antecedentes_que_cambian_despues_de_la_emision():
    """Antecedentes que cambiaron después de emitido el documento lo dejan desactualizado; si el documento se emitió
    después del cambio, la diferencia sí acusa."""
    js = dict(JS_HV, Antecedentes={"PRN": "CON ANTECEDENTES", "RNC": "CON ANTECEDENTES"},
              Anotaciones=[{"Tipo": "INFRACCION GRAVE", "Fecha anotación": "09-09-2026", "Detalle": "CONDENADO"}])
    doc = {"licencias": [], "anotaciones": [], "conf_min": 1.0,
           "antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": "SIN ANTECEDENTES"}}
    campos = K._comparar_filas(doc, js, date(2026, 8, 1), False)
    assert {c["estado"] for c in campos} == {K.DESACT_C}
    campos = K._comparar_filas(doc, js, date(2026, 9, 30), False)
    assert K.DIFERENTE in {c["estado"] for c in campos}


def test_niveles_de_confianza():
    """Niveles ALTA / MEDIA / BAJA / SIN LECTURA según la confianza; un PDF original (sin OCR) es siempre ALTA. La
    vista plana recorre objetos anidados y listas en el orden del JSON."""
    from verificacion.chile import extraccion_cl as EX
    assert EX._nivel(0.93, "x", True)["nivel"] == "ALTA"
    assert EX._nivel(0.62, "x", True)["nivel"] == "MEDIA"
    assert EX._nivel(0.21, "x", True)["nivel"] == "BAJA"
    assert EX._nivel(0.21, "", True)["nivel"] == "SIN LECTURA"
    assert EX._nivel(0.21, "x", False) == {"confianza": 1.0, "nivel": "ALTA"}
    js = {"Nombre": "ANA", "Exámenes": {"Nota": "90%"}, "Evaluadores": ["A", "B"], "Lic": [{"Clase": "B"}]}
    conf = {"Nombre": {"confianza": 0.9, "nivel": "ALTA"}, "Exámenes": {"Nota": {"confianza": 0.3, "nivel": "BAJA"}},
            "Evaluadores": [{"nivel": "ALTA"}, {"nivel": "MEDIA"}], "Lic": [{"nivel": "MEDIA"}]}
    assert [(x["campo"], x["nivel"]) for x in EX.plano(js, conf)] == [
        ("Nombre", "ALTA"), ("Exámenes / Nota", "BAJA"), ("Evaluadores 1", "ALTA"), ("Evaluadores 2", "MEDIA"),
        ("Lic 1 / Clase", "MEDIA")]


REQ = Q.cargar()
HOY = date(2026, 10, 18)


def _hv(clases_fechas, rnc="SIN ANTECEDENTES", anotaciones=()):
    """JSON oficial de Hoja de Vida con las licencias (clase, fecha) dadas."""
    return {"Licencias registradas": [{"Registro": "ULTIMA", "Clase": c, "Fecha otorgamiento": f, "Municipalidad": "X"}
                                      for c, f in clases_fechas],
            "Antecedentes": {"PRN": "SIN ANTECEDENTES", "RNC": rnc}, "Anotaciones": list(anotaciones)}


def _ameco(estado="APROBADO", expira="01-01-2028", equipo="Grúa Móvil Terreno Aspero"):
    """JSON oficial de un certificado AMECO."""
    return {"N°": "AMCH-1", "Estado evaluación": estado, "Fecha Expiración": expira, "Operador Equipo": equipo}


def _docs(hv="AUTÉNTICO", am="AUTÉNTICO"):
    """Dictámenes de la Hoja de Vida y del AMECO entregados."""
    return [{"tipo_documento": "hoja_vida_p1", "veredicto": hv},
            {"tipo_documento": "ameco_rev03", "veredicto": am, "archivo_oficial_clave": "AMCH-1"}]


def _evaluar(hv_js, am_js, docs=None, puesto="Operador de Grúa Móvil"):
    """Aptitud para el cargo con los requisitos de `config/requisitos_cl.yaml`."""
    return Q.evaluar_persona(puesto, docs or _docs(), {"hoja_vida": hv_js, "ameco": {"AMCH-1": am_js}}, REQ, HOY)


def test_apto():
    """Licencias D y A4 vigentes y AMECO aprobado del equipo del cargo: apto."""
    assert _evaluar(_hv([("D", "01-01-2024"), ("A4", "01-01-2025")]), _ameco())[0] == Q.APTO


def test_no_apto_por_licencia_ameco_y_antecedentes():
    """No apto: sin clase A2/A4, control de licencia vencido, AMECO expirado o reprobado, antecedentes en el RNC o
    licencia suspendida."""
    assert _evaluar(_hv([("D", "01-01-2024")]), _ameco())[0] == Q.NO_APTO
    assert _evaluar(_hv([("D", "01-01-2024"), ("A2", "01-01-2020")]), _ameco())[0] == Q.NO_APTO
    assert _evaluar(_hv([("D", "01-01-2024"), ("A2", "01-01-2025")]), _ameco(expira="01-01-2026"))[0] == Q.NO_APTO
    assert _evaluar(_hv([("D", "01-01-2024"), ("A2", "01-01-2025")]), _ameco(estado="REPROBADO"))[0] == Q.NO_APTO
    assert _evaluar(_hv([("D", "01-01-2024"), ("A2", "01-01-2025")], rnc="CON ANTECEDENTES"), _ameco())[0] == Q.NO_APTO
    susp = {"Tipo": "SUSPENSION DE LICENCIA DE CONDUCTOR", "Fecha anotación": "01-03-2026"}
    assert _evaluar(_hv([("D", "01-01-2024"), ("A2", "01-01-2025")], anotaciones=[susp]), _ameco())[0] == Q.NO_APTO


def test_incompleto_por_copia_o_cargo():
    """Incompleto: Hoja de Vida en copia o desactualizada, o AMECO de otro equipo que el del cargo."""
    hv = _hv([("D", "01-01-2024"), ("A4", "01-01-2025")])
    assert _evaluar(hv, _ameco(), _docs(hv="NO VÁLIDO"))[0] == Q.INCOMPLETO
    assert _evaluar(hv, _ameco(), _docs(hv="DESACTUALIZADO"))[0] == Q.INCOMPLETO
    assert _evaluar(hv, _ameco(equipo="Grúa Torre"))[0] == Q.INCOMPLETO


def test_cargo_en_mayusculas_y_cargo_no_configurado():
    """El cargo se reconoce sin importar mayúsculas ni tildes; un cargo sin requisitos configurados se informa."""
    hv = _hv([("D", "01-01-2024"), ("A4", "01-01-2025")])
    assert _evaluar(hv, _ameco(), puesto="OPERADOR DE GRUA MOVIL")[0] == Q.APTO
    apt, motivos = _evaluar(hv, _ameco(equipo="Grúa Torre"), puesto="SUPERVISOR DE FAENA")
    assert apt == Q.APTO and any("no tiene requisitos configurados" in m for m in motivos)


def test_sin_dato_oficial_no_se_evalua():
    """Si falta el dato oficial de un documento, la aptitud no se puede evaluar."""
    hv = _hv([("D", "01-01-2024"), ("A4", "01-01-2025")])
    docs = [{"tipo_documento": "hoja_vida_p1", "veredicto": "AUTÉNTICO"},
            {"tipo_documento": "ameco_rev03", "veredicto": K.SIN_DATO_OFICIAL, "archivo_oficial_clave": None}]
    assert Q.evaluar_persona("Operador de Grúa Móvil", docs, {"hoja_vida": hv, "ameco": {}}, REQ, HOY)[0] == Q.NO_EVALUABLE
