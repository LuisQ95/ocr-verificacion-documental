"""Protección de recursos para convivir con otros procesos que usan la GPU y la memoria.

  - verificar(): antes de lanzar, observa la GPU y la memoria comprometida durante una ventana
    (para captar los picos de otros procesos) y aborta si no hay presupuesto + margen.
  - Candado: impide lanzar dos veces el mismo trabajo (evita ejecuciones duplicadas).
  - Vigia: durante la ejecución, si la memoria comprometida libre del sistema baja de un umbral
    crítico, levanta una alerta para que mi proceso se detenga de forma ordenada.
  - limit_gpu(): límite duro de VRAM en PyTorch; si se excede, falla solo este proceso.

Solo lee información del sistema; nunca toca procesos ajenos.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

from . import config as C


class InsufficientResources(RuntimeError):
    """No hay GPU o memoria suficientes para lanzar sin afectar a otros procesos, o ya hay una ejecución en curso."""


class _MEMORYSTATUSEX(ctypes.Structure):
    """Estructura MEMORYSTATUSEX de la API de Windows (GlobalMemoryStatusEx)."""

    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory():
    """RAM física libre y memoria comprometida libre (RAM + archivo de paginación), en GB."""
    if sys.platform == "win32":
        st = _MEMORYSTATUSEX(dwLength=ctypes.sizeof(_MEMORYSTATUSEX))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return {"ram_libre_gb": st.ullAvailPhys / 2 ** 30,
                "comprometida_libre_gb": st.ullAvailPageFile / 2 ** 30,
                "comprometida_limite_gb": st.ullTotalPageFile / 2 ** 30}
    pages = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    return {"ram_libre_gb": pages, "comprometida_libre_gb": pages, "comprometida_limite_gb": None}


def gpu():
    """VRAM usada/total en GB según nvidia-smi (incluye lo que usan otros procesos)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False).stdout.strip().splitlines()[0]
        was_used, total = (float(x) / 1024 for x in out.split(","))
        return {"vram_usada_gb": was_used, "vram_total_gb": total, "vram_libre_gb": total - was_used}
    except Exception:
        return None


def verify(vram_gb, ram_gb, window_s=C.OBSERVATION_WINDOW_S, interval_s=2.0, log=print):
    """Observa durante `window_s` y usa el PEOR momento (pico de otros procesos) para decidir."""
    worst_vram, worst_commit = float("inf"), float("inf")
    end = time.time() + window_s
    log(f"Chequeo de recursos: observando {window_s}s para captar picos de otros procesos...")
    while True:
        m = memory()
        worst_commit = min(worst_commit, m["comprometida_libre_gb"])
        if vram_gb > 0:
            g = gpu()
            if g is None:
                raise InsufficientResources("no se pudo consultar la GPU con nvidia-smi")
            worst_vram = min(worst_vram, g["vram_libre_gb"])
        if time.time() >= end:
            break
        time.sleep(interval_s)

    problems = []
    if vram_gb > 0 and worst_vram < vram_gb + C.VRAM_MARGIN_GB:
        problems.append(f"VRAM libre en el peor momento {worst_vram:.1f} GB < necesaria {vram_gb:.1f} + "
                         f"margen {C.VRAM_MARGIN_GB:.1f} GB")
    if worst_commit < ram_gb + C.COMMIT_MARGIN_GB:
        problems.append(f"memoria comprometida libre en el peor momento {worst_commit:.1f} GB < necesaria "
                         f"{ram_gb:.1f} + margen {C.COMMIT_MARGIN_GB:.1f} GB")
    if problems:
        raise InsufficientResources("No se lanza para no afectar otros procesos: " + "; ".join(problems))
    log(f"Recursos OK (peor momento: VRAM libre {worst_vram:.1f} GB, comprometida libre {worst_commit:.1f} GB)")


def limit_gpu(gb):
    """Límite duro de VRAM para este proceso: si se excede, PyTorch falla aquí y no afecta a otros."""
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total), 0)


def _process_alive(pid):
    """True si el proceso `pid` sigue en ejecución.

    En Windows `os.kill(pid, 0)` enviaría CTRL_C al proceso: se consulta su código de salida con
    PROCESS_QUERY_LIMITED_INFORMATION (0x1000) sin tocarlo; 259 (STILL_ACTIVE) significa que sigue vivo.
    """
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259


class RunLock:
    """Ejecución única de un trabajo: un archivo `models/.candado_<nombre>` con el PID de quien lo tiene.

    Si el archivo existe y su proceso sigue vivo, no se lanza otra ejecución; si el proceso murió, el candado se
    considera abandonado y se toma.
    """

    def __init__(self, name):
        """Candado para el trabajo `nombre` (p. ej. "reconocedor", "pipeline")."""
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = C.MODELS_DIR / f".candado_{name}"

    def __enter__(self):
        """Toma el candado o levanta RecursosInsuficientes si otra ejecución viva lo tiene."""
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and _process_alive(pid):
                raise InsufficientResources(f"ya hay una ejecución en curso (PID {pid}); no se lanza otra")
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        """Libera el candado solo si sigue siendo de este proceso."""
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass


class Watchdog:
    """Hilo que revisa la memoria del sistema; si se vuelve crítica, deja un mensaje en `alerta`.

    Quien lo usa consulta `alerta` en su bucle y decide qué hacer (guardar y pausar, o detenerse).
    """

    def __init__(self, interval_s=C.WATCH_INTERVAL_S):
        """Vigía que revisa la memoria cada `interval_s` segundos."""
        self.interval_s = interval_s
        self.alert = None
        self._stop = threading.Event()

    def start(self):
        """Arranca el hilo de vigilancia y devuelve el vigía."""
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        """Detiene el hilo de vigilancia."""
        self._stop.set()

    def _loop(self):
        """Revisa la memoria comprometida libre hasta que se detiene o baja de CRITICAL_COMMIT_GB."""
        while not self._stop.wait(self.interval_s):
            free = memory()["comprometida_libre_gb"]
            if free < C.CRITICAL_COMMIT_GB:
                self.alert = (f"memoria comprometida libre del sistema en {free:.2f} GB "
                               f"(< {C.CRITICAL_COMMIT_GB} GB): se detiene este proceso para no afectar a otros")
                return
