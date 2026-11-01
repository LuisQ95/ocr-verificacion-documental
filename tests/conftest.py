"""Configuración de pytest: agrega la raíz del proyecto al path para importar `verificacion`, `evaluation_viewer`, etc."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
