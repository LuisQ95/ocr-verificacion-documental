"""Configuración de acceso del visor de pruebas (evaluation_viewer/access.json): dirección donde escucha, IP permitidas
(la PC del evaluador en Tailscale) y la contraseña (solo se guarda su hash PBKDF2, nunca la contraseña)."""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "access.json"
LOCAL_ONLY = {"127.0.0.1", "::1"}
"""Esta misma PC (para probar antes de compartir)."""
ITERATIONS = 200_000


def load(path=CONFIG_PATH):
    """Configuración guardada; sin archivo, solo local en el puerto 8010 y sin contraseña."""
    if not path.exists():
        return {"host": "127.0.0.1", "puerto": 8010, "ips_permitidas": [], "clave": None}
    return json.loads(path.read_text(encoding="utf-8"))


def version(path=CONFIG_PATH):
    """Fecha de modificación de access.json en nanosegundos (None si no existe): cambia cada vez que se guarda."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def save(cfg, path=CONFIG_PATH):
    """Guarda la configuración en access.json."""
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")


def key_hash(key, out_f=None):
    """Hash PBKDF2-SHA256 de la contraseña con una sal aleatoria de 16 bytes (o la dada)."""
    out_f = out_f or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), bytes.fromhex(out_f), ITERATIONS).hex()
    return {"sal": out_f, "hash": h, "iteraciones": ITERATIONS}


def correct_key(key, saved):
    """True si la contraseña coincide con el hash guardado (comparación en tiempo constante)."""
    if not saved or not key:
        return False
    h = hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), bytes.fromhex(saved["sal"]),
                            saved.get("iteraciones", ITERATIONS)).hex()
    return hmac.compare_digest(h, saved["hash"])


def ip_allowed(ip, cfg):
    """True para esta PC (también por su propia IP de Tailscale, cuando el visor escucha en ella) o las IP
    registradas."""
    return ip in LOCAL_ONLY or ip == cfg.get("host") or ip in set(cfg.get("ips_permitidas", []))
