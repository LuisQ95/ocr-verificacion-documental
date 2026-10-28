"""Configuración de pytest: agrega la raíz del proyecto al path para importar `verificacion`, `visor_pruebas`, etc."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
