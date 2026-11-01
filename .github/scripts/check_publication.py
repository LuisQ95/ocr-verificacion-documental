"""Revisiones de lo que se publica en el repositorio (lo usan los workflows y el gancho local pre-push).

datos        Datos sensibles en una versión (por defecto HEAD): archivos que no deben publicarse (datos reales,
             plantillas oficiales, generador de documentos, modelos, documentos escaneados), RUT válidos de personas
             reales y términos prohibidos (nombre del cliente, nombres de personas reales).
comentarios  Comentarios con # en el código Python: el estilo del proyecto es docstring, no comentarios.

Los términos prohibidos no se escriben en el repositorio (publicarlos sería la filtración que se quiere evitar): se
leen de la variable de entorno TERMINOS_PROHIBIDOS (secreto del repositorio en GitHub) o, en local, de
"DATOS REALES/terminos_prohibidos.txt". Un término por línea.

Los mensajes nunca muestran el dato encontrado, solo el archivo y la línea: los registros de GitHub Actions de un
repositorio público son públicos.

Uso: python .github/scripts/check_publication.py datos [--rev SHA]
     python .github/scripts/check_publication.py comentarios
"""

import argparse
import fnmatch
import io
import os
import re
import subprocess
import sys
import tokenize
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_PATHS = [
    "DATOS REALES/*", "kit_evaluador*", "official_file/*", "postulant_files/*", "webscrap_files/*",
    "ground_truth*", "etiquetas_ocr*", "models/*", "results/*", "herramientas/*",
    "evaluation_viewer/access.json", "evaluation_viewer/kit_answers.md", "evaluation_viewer/ACCESS_GUIDE.md",
    "evaluation_viewer/allow_firewall.ps1", "PROMPT_ADAPTACION_DATOS.md",
    "training/training*", "training/testing*", "training/vocabulario_v2.json",
    "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.db", "*.xlsx", "*.bat", "*.log",
]
"""Archivos que no se publican (ver .gitignore): datos de personas, documentos oficiales, pesos, resultados."""
GENERATOR = "scripts/synthetic/"
PUBLIC_GENERATOR = {"scripts/synthetic/__init__.py", "scripts/synthetic/broad_vocabulary.py"}
"""Del generador de documentos solo se publica el vocabulario: el resto rellena documentos oficiales con timbres y
firmas y sería un kit de falsificación."""
DOCUMENT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff", ".bmp"}
"""Documentos e imágenes: pueden ser documentos de personas. Solo se admiten dentro de PUBLIC_IMAGES_FOLDER."""
PUBLIC_IMAGES_FOLDER = "docs/"
SAMPLE_ID = {11111111, 12345678}
"""RUT de ejemplo usados en documentación y pruebas."""
MAX_REAL_ID = 30_000_000
"""Los RUT de personas reales están por debajo; los datos sintéticos usan RUT desde aquí."""
NATIONAL_ID = re.compile(r"(?<![\d.])(\d{1,2}\.?\d{3}\.?\d{3})-?([\dkK])(?!\d)")
TERMS_FILE = REPO_ROOT / "DATOS REALES" / "terminos_prohibidos.txt"


def git(*args):
    """Salida de un comando git en la raíz del repositorio."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, check=True).stdout


def files(rev):
    """Archivos versionados en `rev`."""
    return [a for a in git("ls-tree", "-r", "-z", "--name-only", rev).decode("utf-8").split("\0") if a]


def content(rev, path):
    """Texto de un archivo en `rev` (None si es binario)."""
    data = git("show", f"{rev}:{path}")
    if b"\0" in data[:8000]:
        return None
    return data.decode("utf-8", errors="replace")


def check_digit(number):
    """Dígito verificador de un número de identidad con módulo 11."""
    total_sum, factor = 0, 2
    for d in reversed(str(number)):
        total_sum += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    rest = 11 - total_sum % 11
    return {11: "0", 10: "K"}.get(rest, str(rest))


def normalize_text(text):
    """Mayúsculas sin tildes y con la puntuación como espacios, rodeado de espacios (para buscar palabras completas)."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return " " + re.sub(r"[^A-Z0-9]+", " ", t).strip() + " "


def forbidden_terms():
    """Términos normalizados desde el secreto o el archivo local; lista vacía si no hay ninguno."""
    text = os.environ.get("TERMINOS_PROHIBIDOS", "")
    if not text.strip() and TERMS_FILE.exists():
        text = TERMS_FILE.read_text(encoding="utf-8")
    return [normalize_text(t) for t in text.splitlines() if t.strip()]


def forbidden_path(path):
    """Motivo por el que un archivo no debe publicarse; "" si puede publicarse."""
    if any(fnmatch.fnmatch(path, p) for p in FORBIDDEN_PATHS):
        return "archivo que no se publica (ver .gitignore)"
    if path.startswith(GENERATOR) and path not in PUBLIC_GENERATOR:
        return "parte del generador de documentos, que no se publica"
    if Path(path).suffix.lower() in DOCUMENT_EXTENSIONS and not path.startswith(PUBLIC_IMAGES_FOLDER):
        return f"documento o imagen fuera de {PUBLIC_IMAGES_FOLDER} (puede ser de una persona real)"
    return ""


def check_data(rev):
    """Hallazgos (ruta, línea, motivo) de datos sensibles en la versión `rev`."""
    terms = forbidden_terms()
    if not terms:
        notice("no hay términos prohibidos configurados (secreto TERMINOS_PROHIBIDOS o "
              "DATOS REALES/terminos_prohibidos.txt): solo se revisan archivos y RUT")
    findings = []
    for path in files(rev):
        reason = forbidden_path(path)
        if reason:
            findings.append((path, 0, reason))
            continue
        if any(t in normalize_text(path) for t in terms):
            findings.append((path, 0, "el nombre del archivo contiene un término prohibido"))
        text = content(rev, path)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in NATIONAL_ID.finditer(line):
                number = int(m.group(1).replace(".", ""))
                if number < MAX_REAL_ID and number not in SAMPLE_ID \
                        and check_digit(number) == m.group(2).upper():
                    findings.append((path, n, "RUT válido de una persona real (usar 12.345.678-5 como ejemplo)"))
            if terms and any(t in normalize_text(line) for t in terms):
                findings.append((path, n, "término prohibido (nombre del cliente o de una persona real)"))
    return findings


def check_comments():
    """Hallazgos (ruta, línea, motivo) de comentarios # en los archivos Python versionados y en los nuevos aún sin
    versionar (los borrados se omiten): revisa el estado actual de la carpeta."""
    findings = []
    current_ones = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")
    for path in current_ones:
        if not path.endswith(".py") or not (REPO_ROOT / path).exists():
            continue
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                findings.append((path, tok.start[0], "comentario con #: el estilo del proyecto es docstring"))
    return findings


def notice(message):
    """Aviso visible en GitHub Actions y en consola."""
    print(f"::warning::{message}" if os.environ.get("GITHUB_ACTIONS") else f"AVISO: {message}")


def inform(findings, what):
    """Imprime los hallazgos (anotaciones de GitHub Actions si corre allí) y devuelve el código de salida."""
    for path, line, reason in findings:
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error file={path},line={max(line, 1)}::{reason}")
        else:
            print(f"{path}:{line}: {reason}")
    print(f"{what}: {len(findings)} hallazgo(s)" if findings else f"{what}: sin hallazgos")
    return 1 if findings else 0


def main():
    """Línea de comandos: `datos` o `comentarios`; sale con 1 si hay hallazgos."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("check", choices=["datos", "comentarios"])
    ap.add_argument("--rev", default="HEAD", help="versión a revisar (solo para datos)")
    a = ap.parse_args()
    if a.check == "datos":
        return inform(check_data(a.rev), f"Datos sensibles en {a.rev}")
    return inform(check_comments(), "Comentarios")


if __name__ == "__main__":
    sys.exit(main())
