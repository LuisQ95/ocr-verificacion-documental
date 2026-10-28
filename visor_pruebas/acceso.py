"""Configuración de acceso del visor de pruebas (visor_pruebas/acceso.json): dirección donde escucha, IP permitidas
(la PC del evaluador en Tailscale) y la contraseña (solo se guarda su hash PBKDF2, nunca la contraseña)."""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

RUTA = Path(__file__).parent / "acceso.json"
LOCALES = {"127.0.0.1", "::1"}
"""Esta misma PC (para probar antes de compartir)."""
ITERACIONES = 200_000


def cargar(ruta=RUTA):
    """Configuración guardada; sin archivo, solo local en el puerto 8010 y sin contraseña."""
    if not ruta.exists():
        return {"host": "127.0.0.1", "puerto": 8010, "ips_permitidas": [], "clave": None}
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar(cfg, ruta=RUTA):
    """Guarda la configuración en acceso.json."""
    ruta.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")


def hash_clave(clave, sal=None):
    """Hash PBKDF2-SHA256 de la contraseña con una sal aleatoria de 16 bytes (o la dada)."""
    sal = sal or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), bytes.fromhex(sal), ITERACIONES).hex()
    return {"sal": sal, "hash": h, "iteraciones": ITERACIONES}


def clave_correcta(clave, guardada):
    """True si la contraseña coincide con el hash guardado (comparación en tiempo constante)."""
    if not guardada or not clave:
        return False
    h = hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), bytes.fromhex(guardada["sal"]),
                            guardada.get("iteraciones", ITERACIONES)).hex()
    return hmac.compare_digest(h, guardada["hash"])


def ip_permitida(ip, cfg):
    """True para esta PC (también por su propia IP de Tailscale, cuando el visor escucha en ella) o las IP
    registradas."""
    return ip in LOCALES or ip == cfg.get("host") or ip in set(cfg.get("ips_permitidas", []))
