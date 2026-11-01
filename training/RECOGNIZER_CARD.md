# Ficha del reconocedor OCR: reentrenamiento "entender, no memorizar"

**Fecha:** 2026-10-27
**Modelo adoptado (2026-10-28):** `models/parseq_v4.pt`, con su calibración `confidence_calibration_v4.json`.
**Modelo anterior:** `models/parseq_puro.pt`, con su calibración `confidence_calibration_puro.json`. Se conservó como respaldo hasta el 2026-10-31 (ver sección 9).

## 1. Problema que se corrige

El reconocedor lee **palabra por palabra**. El detector separa cada campo en palabras y PARSeq lee cada recorte. En los datos con que se entrenó `parseq_puro`, los nombres y apellidos salían de una lista corta: **230 palabras**, con un promedio de **1 230 recortes por palabra** (FELIPE aparece 3 042 veces). PARSeq tiene un modelo de lenguaje interno, así que aprendió esas palabras de memoria.

| Prueba | `parseq_puro` |
|---|---|
| Nombres que vio en entrenamiento | 99,2 % |
| Nombres nunca vistos | **76,6 %** |
| Texto aleatorio (letras sin sentido) | **62,5 %** |

Con recortes borrosos, el modelo "completaba" un nombre que se sabía de memoria en lugar de leer lo que había. Un postulante real con un apellido fuera de esa lista quedaba expuesto a lecturas erróneas.

## 2. Datos nuevos

**Vocabulario amplio** (`scripts/synthetic/broad_vocabulary.py` → `training/vocabulario_v2.json`):

- 4 000 nombres: 1 270 reales de unos 25 orígenes y 2 730 generados por sílabas y sufijos.
- 12 000 apellidos: 2 074 reales y 9 926 generados.
- Reparto **80/20 con palabras separadas**: un 20 % de nombres y apellidos queda reservado solo para prueba y nunca aparece en entrenamiento. Las palabras reservadas excluyen además todo el vocabulario anterior.
- Conserva tildes, Ñ, Ü, apóstrofo y guion. Las letras extranjeras se transliteran (ß → SS, Ø → O).

**Generación** (`scripts/synthetic/ocr_training_v2.py`, generador local que no se publica porque
rellena las plantillas oficiales). Tomó 2,67 h, ocupa 13,3 GB y terminó con 0 errores.

| Conjunto | Documentos completos | Nombres en plantilla | Texto aleatorio |
|---|---|---|---|
| `training_v2` | 25 000 docs → 1 056 717 recortes | 1 141 890 | 240 719 |
| `testing_v2` | 5 000 docs → 210 224 recortes | 34 474 | 24 101 |

- Nombres: de 1 a 3 nombres y de 1 a 2 apellidos, sin especializarse en una composición fija.
- Texto aleatorio: cadenas sin sentido, que obligan a leer letra por letra.
- Degradaciones: escaneo, foto, foto mala, WhatsApp y captura de pantalla, con desenfoque, ruido, JPEG, deformación, resaltador y lápiz.

**Datos anteriores redistribuidos** (`--review-v1`). Se reutilizan los 2,17 M de recortes de `training` con dos ajustes:

- Se reserva el 20 % de las palabras de nombres anteriores: 47 de 235, en `models/reserva_vocabulario_v1.json`. Sus recortes salen del entrenamiento y se usan solo para medir.
- Se pone un tope de 300 recortes por palabra de nombre anterior. Fechas, RUT, notas y textos fijos se mantienen completos.
- Quedan 1 940 017 recortes.

**Total de entrenamiento: 4 378 815 recortes.**

## 3. Entrenamiento

- **Punto de partida:** el PARSeq **genérico** de docTR, no `parseq_puro`. Así las palabras reservadas son realmente nuevas para el modelo.
- **Barrido de tasa de aprendizaje** (3 000 pasos cada una): 1e-3 → 96,10 %, **2,5e-3 → 96,21 %**, 4e-3 → 96,12 %.
- **Configuración:** 4 épocas (68 416 pasos), lotes de 256, AdamW con OneCycle y bf16. Unas 4 h en una RTX 5070 Ti.
- **Criterio para guardar el mejor punto:** promedio por igual de cuatro validaciones: nombres no vistos, texto aleatorio, documentos nuevos y conjunto anterior. Un modelo que memoriza no puede ganar con este criterio aunque lea bien los documentos.

| Época | Nombres no vistos | Texto aleatorio | Documentos nuevos | Conjunto anterior | Nombres anteriores reservados* |
|---|---|---|---|---|---|
| 1 | 98,2 % | 87,8 % | 97,0 % | 97,0 % | 92,7 % |
| 2 | 98,9 % | 91,9 % | 98,0 % | 98,0 % | 94,0 % |
| 3 | 99,4 % | 95,1 % | 98,7 % | 98,7 % | 95,8 % |
| 4 | **99,5 %** | **96,2 %** | **98,8 %** | **98,9 %** | **96,6 %** |

\* Solo se informa; no entra en el criterio.

## 4. Resultados: `parseq_puro` frente a `parseq_v4`

Cada fila usa el conjunto completo y los mismos recortes para ambos modelos.

| Prueba | Recortes | puro | **v4** |
|---|---|---|---|
| `testing_v2` total | 268 702 | 89,4 % | **98,7 %** |
| · nombres nunca vistos | 34 474 | 76,6 % | **99,5 %** |
| · texto aleatorio | 24 093 | 62,5 % | **96,2 %** |
| · fotos malas | 5 782 | 60,8 % | **95,0 %** |
| · palabras con tilde o Ñ | 26 970 | 77,6 % | **97,1 %** |
| `testing` (conjunto anterior) | 850 063 | **99,05 %** | 98,88 % |
| Nombres anteriores reservados | 22 771 | 99,2 %† | 96,6 % |
| PDFs reales (`DATOS REALES`), palabras, 100 DPI | 1 096 | 99,09 % | **99,82 %** |
| PDFs reales, palabras, 150 DPI | 1 096 | 99,45 % | **99,82 %** |
| PDFs reales, palabras, 200 DPI | 1 096 | 99,18 % | **99,91 %** |
| PDFs reales, signos sueltos | 78 | 77–79 % | **90–92 %** |
| Plantillas oficiales, palabras, 200 DPI | 464 | 99,57 % | **100 %** |
| Campos de una palabra de `postulant_files` | 6 000 | 99,68 % | 99,68 % |

† `puro` vio esas palabras en su entrenamiento, así que no es una medida de generalización para él.

**Lectura de los resultados:**

- Con lo nunca visto (nombres, texto aleatorio, degradaciones nuevas) v4 supera ampliamente a puro.
- En PDFs reales, que ninguno de los dos vio, v4 comete unas 5 veces menos errores por palabra.
- En los campos de postulantes empatan.
- La única baja es de **0,17 puntos en el conjunto anterior**. Sale de los nombres que `puro` vio miles de veces: con recortes ilegibles, `puro` acierta "FRANCISCO" porque se lo sabe de memoria. Esa ventaja no se traslada a postulantes reales con otros nombres.

**Evidencia de que v4 no memoriza:**

- Lee nombres que nunca vio (99,5 %) tan bien como los documentos.
- Durante el entrenamiento, los grupos vistos y no vistos subieron juntos en cada época.

## 5. Cómo reproducir

La generación de recortes (segunda línea) requiere el generador local y las plantillas oficiales, que no están en el
repositorio público. Lo demás funciona con cualquier conjunto de recortes con el mismo formato (ZIP con
`labels.json` y `meta.jsonl`).

```powershell
python -u -m scripts.synthetic.broad_vocabulary
python -u -m scripts.synthetic.ocr_training_v2
python -u -m training.train_recognizer --mode barrido --weights generico --lrs 1e-3 2.5e-3 4e-3 --steps 3000 `
  --train-splits training_v2 --val-splits testing_v2:nombres testing_v2:aleatorio testing_v2:docs testing --review-v1
python -u -m training.train_recognizer --mode entrenar --weights generico --lr 2.5e-3 --epochs 4 `
  --train-splits training_v2 --val-splits testing_v2:nombres testing_v2:aleatorio testing_v2:docs testing --review-v1 `
  --output models/parseq_v4.pt
python -u -m training.train_recognizer --mode evaluar --weights models/parseq_v4.pt --split-eval testing_v2 --label _testing_v2
python -u -m training.train_recognizer --mode evaluar --weights models/parseq_v4.pt --split-eval testing --only-reserved-v1 --label _reservadas_v1
python -u -m training.evaluate_independent --recos actual=models/parseq_puro.pt nuevo=models/parseq_v4.pt --applicants
```

**Archivos de resultados** en `models/`:

- `eval_reconocedor_parseq_{puro,v4}_{testing,testing_v2,reservadas_v1}.json`
- `eval_independiente_v4.json`
- `parseq_v4_metricas.json`
- `barrido_reconocedor.json`

**Registros** en `training/`: `generacion_v2.log`, `barrido_v2.log`, `entrenamiento_v4.log`, `eval_v4.log` y `eval_independiente_v4.log`.

## 6. Afinado adicional (v5) y adopción

**Afinado v5.** Partió de v4 y agregó 2 épocas con tasa baja (3e-4), con los mismos datos y criterio. **No mejoró**:
el promedio quedó en 98,34 % contra 98,37 % del v4, con diferencias de centésimas en cada grupo. El v4 ya había
convergido con estos datos, así que se descartó el v5.

**Calibración del v4** (200 000 recortes de `testing` y `testing_v2`):

- error de calibración 1,03 % → 0,07 %;
- las lecturas con confianza ALTA (≥ 0,80) son el 98 % del total y aciertan el 99,76 %.

**Verificador completo con 600 postulantes**, mismos documentos y reglas para ambos modelos (`--limit 300` de cada
subconjunto):

| | Actual (puro) ajuste | **v4 ajuste** | Actual reservado | **v4 reservado** |
|---|---|---|---|---|
| Falsificaciones detectadas | 232/267 | **234/267** | **224/253** | 221/253 |
| Falsificaciones escapadas (AUTÉNTICO) | 1 | 1 | 0 | 0 |
| Genuinos acusados (FALSIFICADO) | 2/626 | **0/626** | 6/634 | **2/634** |
| Genuinos a revisión (OBSERVADO) | 75 | 76 | 71 | 76 |

- **Resultado total:** misma detección (455 contra 456 de 520), mismas falsificaciones escapadas (1) y genuinos
  acusados de 8 a 2.
- **Las 3 detecciones menos del grupo reservado no escapan:** quedan en OBSERVADO, porque la calibración nueva da
  confianza baja a algunas filas de licencias leídas en fotos.
- **Dos reglas endurecidas con el subconjunto de ajuste**
  (`verification/profiles/operator_certification/rules.py`, `test_lecturas_contaminadas_no_acusan`), confirmadas en el reservado:
  - una nota sin ningún dígito leído ya no se convierte en número (las letras O y B no son 0 y 8);
  - "APROBADO" o "REPROBADO" dentro de un nombre, domicilio o código es una lectura contaminada (firma, timbre o
    interfaz de una captura).

**Adoptado el 2026-10-28:** `parseq_v4.pt` + `confidence_calibration_v4.json` en `verification/config.py`.

**Pasos de adopción** del modelo ganador (`<m>` = v4 o v5):

1. **Calibrar la confianza**, también con nombres no vistos:
   `python -u -m training.calibrate_recognizer --weights models/parseq_<m>.pt --splits testing testing_v2 --output models/confidence_calibration_<m>.json`
2. **Medir el verificador completo con el candidato, sin cambiar lo adoptado.** Antes de lanzarlo, `$env:OCR_PARSEQ_MODEL` y `$env:OCR_CALIBRATION` deben apuntar a los archivos del candidato:
   - `python -u -m verification.pipeline --subset ajuste --output results/<m>`
   - `python -m audit.measure_results --results results/<m> --subset ajuste`

   Se compara contra la misma medición hecha con `puro`.
3. **Si no empeora:** actualizar `MODELO_PARSEQ` y `CALIBRACION_CONF` en `verification/config.py` y correr `pytest tests`.

**Respaldo.** `parseq_puro.pt` se conservó hasta el 2026-10-31 (ver sección 9).

## 7. Menos genuinos a revisión (reglas, 2026-10-28)

Con el v4 adoptado, el 12 % de los genuinos seguía en OBSERVADO. Se analizaron las causas en el subconjunto de
ajuste y se agregaron reglas en `verification/profiles/operator_certification/rules.py` (tests en `tests/test_rules.py`):

- **Repetición de la página 2:** si la página 1 ya confirmó el nombre, el RUN o la fecha de nacimiento, una lectura
  dudosa de la cabecera repetida en la página 2 no manda a revisión. En las falsificaciones el dato cambia en ambas
  páginas y se lee con confianza alta, así que sigue acusando.
- **Ruido que no cambia el valor:** palabras de enlace junto al valor oficial completo ("de TALCAHUANO",
  "N° MEJILLONES") y texto de otro campo dentro de la zona ("Jorge APROBADO Vidal Acuña").
- **Nombres de evaluador casi idénticos** al oficial (1-2 caracteres). Otro evaluador sigue acusando.
- **Licencia renovada después de la emisión con la clase sin leer:** la fila anterior a la renovación deja el
  documento DESACTUALIZADO, como cuando la clase sí se lee.
- **Renovación demasiado reciente:** una fecha impresa a menos de un año de la renovación oficial no se da por
  "desactualizada" y va a revisión. Es la técnica de quien "renueva" una licencia en el documento; sin esta condición,
  una de esas falsificaciones habría pasado de OBSERVADO a DESACTUALIZADO.

| | v4 ajuste antes | **v4 ajuste ahora** | v4 reservado antes | **v4 reservado ahora** |
|---|---|---|---|---|
| Falsificaciones detectadas | 234/267 | 234/267 | 221/253 | 221/253 |
| Falsificaciones escapadas (AUTÉNTICO) | 1 | 1 | 0 | 0 |
| Genuinos acusados (FALSIFICADO) | 0/626 | 0/626 | 2/634 | 2/634 |
| Genuinos a revisión (OBSERVADO) | 76 | **51** | 76 | **55** |

En ambos subconjuntos solo cambiaron documentos genuinos (de OBSERVADO a AUTÉNTICO o DESACTUALIZADO): ninguna
falsificación cambió de dictamen. Lo que sigue en revisión es, sobre todo, la Hoja de Vida sin página 2 (correcto),
fechas a las que el detector no les vio el día (un "9" suelto) y fotos con texto de otro documento superpuesto.

## 8. Revisión integral: limpieza, rendimiento y reglas (2026-10-31)

**Sin cambio de dictámenes (verificado):** seguridad del visor de pruebas, optimizaciones de lectura (cada página de
PDF se leía y extraía varias veces) y limpieza del código del flujo anterior (−1 580 líneas). En el subconjunto de
ajuste los 950 documentos dieron el mismo dictamen, el mismo estado y el mismo texto leído en cada campo, y la
medición pasó de 880 s a 678 s.

**Mejoras de lectura y reglas** (diseñadas en ajuste, confirmadas en reservado):

- **Caracteres sueltos** (`verification/ocr.py`): en la sección variable de la Hoja de Vida, la tinta del tamaño de
  1 a 3 caracteres que el detector dejó fuera de toda caja se lee aparte (la clase "D" de una licencia, el día de una
  fecha). Se conserva solo con confianza ≥ 0,5.
- **Licencia sin clase leída con la fecha cambiada** (`rules._single_row_without_class`): si una fila sin clase
  es la única que puede ser la licencia oficial que falta (mismo registro y municipalidad), se comparan sus fechas.
  Era la técnica "renovar licencia" que quedaba en revisión.
- **Timbre electrónico:** en PDF nativos se reintenta a 400 DPI (a 300 DPI se leía en un tercio de las Hojas de Vida
  nativas; a 400, en todas las probadas); en fotos de Hoja de Vida se prueban umbral adaptativo, Otsu y ampliación
  (recupera ~15 % de los timbres que no se leían).
- **Páginas de la Hoja de Vida en fotos sueltas** se juntan por folio, código y fecha de emisión, no solo por orden.
- **Prioridad de revisión** (ALTA / BAJA) en los documentos OBSERVADO: ALTA si hay un indicio técnico o un valor
  distinto leído con confianza ≥ 0,5. No cambia dictámenes; ordena la revisión.

| | ajuste antes | **ajuste ahora** | reservado antes | **reservado ahora** |
|---|---|---|---|---|
| Falsificaciones detectadas | 234/267 | **239/267** | 221/253 | **223/253** |
| Falsificaciones escapadas (AUTÉNTICO) | 1 | 1 | 0 | 0 |
| Genuinos acusados (FALSIFICADO) | 0/626 | 0/626 | 2/634 | 4/634 (ver nota) |
| Genuinos a revisión (OBSERVADO) | 52 | **51** | 55 | **51** |
| OBSERVADO con prioridad ALTA: falsos / genuinos | — | 8 / 6 | — | 10 / 7 |

"Antes" en ajuste es el entorno actual (Python 3.12, Pillow 13); en reservado es la última medición disponible, con
Python 3.10 y Pillow 11. **Nota:** los 2 genuinos acusados nuevos de reservado (un AMECO en foto y una Hoja de Vida
COPIA por WhatsApp) se acusan igual sin las reglas nuevas: vienen del cambio a Pillow 13, que decodifica JPEG con
píxeles levemente distintos y cuyo efecto solo se había medido en ajuste. Quedan por analizar.

No se adoptó: tratar como falsificada una anotación oficial que no aparece cuando no queda ninguna anotación sin
emparejar (técnica "borrar anotaciones negativas"); en ajuste también aparecía en 2 genuinos.

## 9. Limpieza (2026-10-31)

Se borraron para liberar unos 8,6 GB:

- `parseq_puro.pt` y su checkpoint: v4 lleva tres días adoptado y medido en ajuste y reservado;
- `parseq_v5.pt` y su checkpoint: el afinado v5 no mejoró (sección 6);
- `parseq_v4_checkpoint.pt`: estado del optimizador; para usar el modelo basta `parseq_v4.pt`;
- los conjuntos v1 `training/training` y `training/testing` (7,3 GB).

Se conservan las métricas (`*_metricas.json`, `eval_*.json`) y las calibraciones.

Para repetir el entrenamiento del v4 (`--review-v1` y la validación `testing`) o recalibrar con `--splits testing`,
primero hay que regenerar los conjuntos v1 con la misma semilla. Esto necesita el generador local y las plantillas
oficiales:

```powershell
python -u -m scripts.synthetic.ocr_training --training 50000 --testing 20000 --seed 20261019
```
