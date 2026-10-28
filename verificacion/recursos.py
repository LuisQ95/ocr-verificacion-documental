"""Protección de recursos para convivir con otros procesos que usan la GPU y la memoria.

  - verificar(): antes de lanzar, observa la GPU y la memoria comprometida durante una ventana
    (para captar los picos de otros procesos) y aborta si no hay presupuesto + margen.
  - Candado: impide lanzar dos veces el mismo trabajo (evita ejecuciones duplicadas).
  - Vigia: durante la ejecución, si la memoria comprometida libre del sistema baja de un umbral
    crítico, levanta una alerta para que mi proceso se detenga de forma ordenada.
  - limitar_gpu(): límite duro de VRAM en PyTorch; si se excede, falla solo este proceso.

Solo lee información del sistema; nunca toca procesos ajenos.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

from . import config as C


class RecursosInsuficientes(RuntimeError):
    """No hay GPU o memoria suficientes para lanzar sin afectar a otros procesos, o ya hay una ejecución en curso."""


class _MEMORYSTATUSEX(ctypes.Structure):
    """Estructura MEMORYSTATUSEX de la API de Windows (GlobalMemoryStatusEx)."""

    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memoria():
    """RAM física libre y memoria comprometida libre (RAM + archivo de paginación), en GB."""
    if sys.platform == "win32":
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return {"ram_libre_gb": st.ullAvailPhys / 2 ** 30,
                "comprometida_libre_gb": st.ullAvailPageFile / 2 ** 30,
                "comprometida_limite_gb": st.ullTotalPageFile / 2 ** 30}
    paginas = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    return {"ram_libre_gb": paginas, "comprometida_libre_gb": paginas, "comprometida_limite_gb": None}


def gpu():
    """VRAM usada/total en GB según nvidia-smi (incluye lo que usan otros procesos)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        usada, total = (float(x) / 1024 for x in out.split(","))
        return {"vram_usada_gb": usada, "vram_total_gb": total, "vram_libre_gb": total - usada}
    except Exception:
        return None


def verificar(vram_gb, ram_gb, ventana_s=C.VENTANA_OBSERVACION_S, intervalo_s=2.0, log=print):
    """Observa durante `ventana_s` y usa el PEOR momento (pico de otros procesos) para decidir."""
    peor_vram, peor_comp = float("inf"), float("inf")
    fin = time.time() + ventana_s
    log(f"Chequeo de recursos: observando {ventana_s}s para captar picos de otros procesos...")
    while True:
        m = memoria()
        peor_comp = min(peor_comp, m["comprometida_libre_gb"])
        if vram_gb > 0:
            g = gpu()
            if g is None:
                raise RecursosInsuficientes("no se pudo consultar la GPU con nvidia-smi")
            peor_vram = min(peor_vram, g["vram_libre_gb"])
        if time.time() >= fin:
            break
        time.sleep(intervalo_s)

    problemas = []
    if vram_gb > 0 and peor_vram < vram_gb + C.MARGEN_VRAM_GB:
        problemas.append(f"VRAM libre en el peor momento {peor_vram:.1f} GB < necesaria {vram_gb:.1f} + "
                         f"margen {C.MARGEN_VRAM_GB:.1f} GB")
    if peor_comp < ram_gb + C.MARGEN_COMPROMETIDA_GB:
        problemas.append(f"memoria comprometida libre en el peor momento {peor_comp:.1f} GB < necesaria "
                         f"{ram_gb:.1f} + margen {C.MARGEN_COMPROMETIDA_GB:.1f} GB")
    if problemas:
        raise RecursosInsuficientes("No se lanza para no afectar otros procesos: " + "; ".join(problemas))
    log(f"Recursos OK (peor momento: VRAM libre {peor_vram:.1f} GB, comprometida libre {peor_comp:.1f} GB)")


def limitar_gpu(gb):
    """Límite duro de VRAM para este proceso: si se excede, PyTorch falla aquí y no afecta a otros."""
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total), 0)


def _proceso_vivo(pid):
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
    codigo = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(codigo))
    k32.CloseHandle(h)
    return codigo.value == 259


class Candado:
    """Ejecución única de un trabajo: un archivo `modelos/.candado_<nombre>` con el PID de quien lo tiene.

    Si el archivo existe y su proceso sigue vivo, no se lanza otra ejecución; si el proceso murió, el candado se
    considera abandonado y se toma.
    """

    def __init__(self, nombre):
        """Candado para el trabajo `nombre` (p. ej. "reconocedor", "pipeline")."""
        C.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
        self.ruta = C.DIR_MODELOS / f".candado_{nombre}"

    def __enter__(self):
        """Toma el candado o levanta RecursosInsuficientes si otra ejecución viva lo tiene."""
        if self.ruta.exists():
            try:
                pid = int(self.ruta.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and _proceso_vivo(pid):
                raise RecursosInsuficientes(f"ya hay una ejecución en curso (PID {pid}); no se lanza otra")
        self.ruta.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        """Libera el candado solo si sigue siendo de este proceso."""
        try:
            if self.ruta.read_text().strip() == str(os.getpid()):
                self.ruta.unlink()
        except OSError:
            pass


class Vigia:
    """Hilo que revisa la memoria del sistema; si se vuelve crítica, deja un mensaje en `alerta`.

    Quien lo usa consulta `alerta` en su bucle y decide qué hacer (guardar y pausar, o detenerse).
    """

    def __init__(self, intervalo_s=C.INTERVALO_VIGIA_S):
        """Vigía que revisa la memoria cada `intervalo_s` segundos."""
        self.intervalo_s = intervalo_s
        self.alerta = None
        self._parar = threading.Event()

    def iniciar(self):
        """Arranca el hilo de vigilancia y devuelve el vigía."""
        threading.Thread(target=self._bucle, daemon=True).start()
        return self

    def detener(self):
        """Detiene el hilo de vigilancia."""
        self._parar.set()

    def _bucle(self):
        """Revisa la memoria comprometida libre hasta que se detiene o baja de COMPROMETIDA_CRITICA_GB."""
        while not self._parar.wait(self.intervalo_s):
            libre = memoria()["comprometida_libre_gb"]
            if libre < C.COMPROMETIDA_CRITICA_GB:
                self.alerta = (f"memoria comprometida libre del sistema en {libre:.2f} GB "
                               f"(< {C.COMPROMETIDA_CRITICA_GB} GB): se detiene este proceso para no afectar a otros")
                return
