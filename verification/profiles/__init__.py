"""Perfiles de documentos. Un perfil reúne lo propio de un conjunto de documentos (plantillas y zonas,
normalización de valores, reglas de comparación, requisitos de aptitud y datos oficiales); el núcleo de
verification/ es el mismo para todos.

Un perfil es un paquete con los módulos documents, extraction, normalize, requirements, rules y templates. Se elige
con config.PROFILE (variable de entorno VERIFICATION_PROFILE).
"""

import importlib

from .. import config as C

ACTIVE = importlib.import_module(f"{__name__}.{C.PROFILE}")
"""Perfil en uso."""
