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

Uso: python .github/scripts/revisar_publicacion.py datos [--rev SHA]
     python .github/scripts/revisar_publicacion.py comentarios
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

RAIZ = Path(__file__).resolve().parents[2]

RUTAS_PROHIBIDAS = [
    "DATOS REALES/*", "kit_evaluador*", "official_file/*", "postulant_files/*", "webscrap_files/*",
    "ground_truth*", "etiquetas_ocr*", "modelos/*", "resultados/*", "herramientas/*",
    "visor_pruebas/acceso.json", "visor_pruebas/kit_respuestas.md", "visor_pruebas/GUIA_ACCESO.md",
    "visor_pruebas/permitir_firewall.ps1", "PROMPT_ADAPTACION_DATOS.md",
    "entrenamiento/training*", "entrenamiento/testing*", "entrenamiento/vocabulario_v2.json",
    "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.db", "*.xlsx", "*.bat", "*.log",
]
"""Archivos que no se publican (ver .gitignore): datos de personas, documentos oficiales, pesos, resultados."""
GENERADOR = "scripts/sinteticos_reales/"
GENERADOR_PUBLICO = {"scripts/sinteticos_reales/__init__.py", "scripts/sinteticos_reales/vocabulario_amplio.py"}
"""Del generador de documentos solo se publica el vocabulario: el resto rellena documentos oficiales con timbres y
firmas y sería un kit de falsificación."""
EXTENSIONES_DOCUMENTO = {".pdf", ".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff", ".bmp"}
"""Documentos e imágenes: pueden ser documentos de personas. Solo se admiten dentro de CARPETA_IMAGENES_PUBLICAS."""
CARPETA_IMAGENES_PUBLICAS = "docs/"
RUT_EJEMPLO = {11111111, 12345678}
"""RUT de ejemplo usados en documentación y pruebas."""
RUT_MAXIMO_REAL = 30_000_000
"""Los RUT de personas reales están por debajo; los datos sintéticos usan RUT desde aquí."""
RUT = re.compile(r"(?<![\d.])(\d{1,2}\.?\d{3}\.?\d{3})-?([\dkK])(?!\d)")
ARCHIVO_TERMINOS = RAIZ / "DATOS REALES" / "terminos_prohibidos.txt"


def git(*args):
    """Salida de un comando git en la raíz del repositorio."""
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, check=True).stdout


def archivos(rev):
    """Archivos versionados en `rev`."""
    return [a for a in git("ls-tree", "-r", "-z", "--name-only", rev).decode("utf-8").split("\0") if a]


def contenido(rev, ruta):
    """Texto de un archivo en `rev` (None si es binario)."""
    datos = git("show", f"{rev}:{ruta}")
    if b"\0" in datos[:8000]:
        return None
    return datos.decode("utf-8", errors="replace")


def digito_verificador(numero):
    """Dígito verificador de un RUT chileno (módulo 11)."""
    suma, factor = 0, 2
    for d in reversed(str(numero)):
        suma += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - suma % 11
    return {11: "0", 10: "K"}.get(resto, str(resto))


def normalizar(texto):
    """Mayúsculas sin tildes y con la puntuación como espacios, rodeado de espacios (para buscar palabras completas)."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()
    return " " + re.sub(r"[^A-Z0-9]+", " ", t).strip() + " "


def terminos_prohibidos():
    """Términos normalizados desde el secreto o el archivo local; lista vacía si no hay ninguno."""
    texto = os.environ.get("TERMINOS_PROHIBIDOS", "")
    if not texto.strip() and ARCHIVO_TERMINOS.exists():
        texto = ARCHIVO_TERMINOS.read_text(encoding="utf-8")
    return [normalizar(t) for t in texto.splitlines() if t.strip()]


def ruta_prohibida(ruta):
    """Motivo por el que un archivo no debe publicarse; "" si puede publicarse."""
    if any(fnmatch.fnmatch(ruta, p) for p in RUTAS_PROHIBIDAS):
        return "archivo que no se publica (ver .gitignore)"
    if ruta.startswith(GENERADOR) and ruta not in GENERADOR_PUBLICO:
        return "parte del generador de documentos, que no se publica"
    if Path(ruta).suffix.lower() in EXTENSIONES_DOCUMENTO and not ruta.startswith(CARPETA_IMAGENES_PUBLICAS):
        return f"documento o imagen fuera de {CARPETA_IMAGENES_PUBLICAS} (puede ser de una persona real)"
    return ""


def revisar_datos(rev):
    """Hallazgos (ruta, línea, motivo) de datos sensibles en la versión `rev`."""
    terminos = terminos_prohibidos()
    if not terminos:
        aviso("no hay términos prohibidos configurados (secreto TERMINOS_PROHIBIDOS o "
              "DATOS REALES/terminos_prohibidos.txt): solo se revisan archivos y RUT")
    hallazgos = []
    for ruta in archivos(rev):
        motivo = ruta_prohibida(ruta)
        if motivo:
            hallazgos.append((ruta, 0, motivo))
            continue
        if any(t in normalizar(ruta) for t in terminos):
            hallazgos.append((ruta, 0, "el nombre del archivo contiene un término prohibido"))
        texto = contenido(rev, ruta)
        if texto is None:
            continue
        for n, linea in enumerate(texto.splitlines(), 1):
            for m in RUT.finditer(linea):
                numero = int(m.group(1).replace(".", ""))
                if numero < RUT_MAXIMO_REAL and numero not in RUT_EJEMPLO \
                        and digito_verificador(numero) == m.group(2).upper():
                    hallazgos.append((ruta, n, "RUT válido de una persona real (usar 12.345.678-5 como ejemplo)"))
            if terminos and any(t in normalizar(linea) for t in terminos):
                hallazgos.append((ruta, n, "término prohibido (nombre del cliente o de una persona real)"))
    return hallazgos


def revisar_comentarios():
    """Hallazgos (ruta, línea, motivo) de comentarios # en los archivos Python versionados."""
    hallazgos = []
    for ruta in archivos("HEAD"):
        if not ruta.endswith(".py"):
            continue
        texto = (RAIZ / ruta).read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(texto).readline):
            if tok.type == tokenize.COMMENT:
                hallazgos.append((ruta, tok.start[0], "comentario con #: el estilo del proyecto es docstring"))
    return hallazgos


def aviso(mensaje):
    """Aviso visible en GitHub Actions y en consola."""
    print(f"::warning::{mensaje}" if os.environ.get("GITHUB_ACTIONS") else f"AVISO: {mensaje}")


def informar(hallazgos, que):
    """Imprime los hallazgos (anotaciones de GitHub Actions si corre allí) y devuelve el código de salida."""
    for ruta, linea, motivo in hallazgos:
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error file={ruta},line={max(linea, 1)}::{motivo}")
        else:
            print(f"{ruta}:{linea}: {motivo}")
    print(f"{que}: {len(hallazgos)} hallazgo(s)" if hallazgos else f"{que}: sin hallazgos")
    return 1 if hallazgos else 0


def main():
    """Línea de comandos: `datos` o `comentarios`; sale con 1 si hay hallazgos."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("revision", choices=["datos", "comentarios"])
    ap.add_argument("--rev", default="HEAD", help="versión a revisar (solo para datos)")
    a = ap.parse_args()
    if a.revision == "datos":
        return informar(revisar_datos(a.rev), f"Datos sensibles en {a.rev}")
    return informar(revisar_comentarios(), "Comentarios")


if __name__ == "__main__":
    sys.exit(main())
