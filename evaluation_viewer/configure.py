"""Configura el acceso al visor de pruebas.

  python -m evaluation_viewer.configurar --clave                 # pide la contraseña (no se muestra al escribirla)
  python -m evaluation_viewer.configurar --ip 100.101.102.103    # la PC del evaluador (su IP de Tailscale)
  python -m evaluation_viewer.configurar --quitar-ip 100.101.102.103
  python -m evaluation_viewer.configurar --host 100.64.0.5       # la IP de Tailscale de ESTA PC (tailscale ip -4)
  python -m evaluation_viewer.configurar --ver
"""

import argparse
import getpass
import ipaddress

from . import access as A


def main():
    """Aplica las opciones dadas y muestra la configuración resultante. Las IP se validan antes de guardarse."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="store_true", help="definir o cambiar la contraseña")
    ap.add_argument("--ip", help="agregar la IP permitida (PC del evaluador)")
    ap.add_argument("--remove-ip")
    ap.add_argument("--host", help="IP donde escucha el visor (la de Tailscale de esta PC; 127.0.0.1 = solo esta PC)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    cfg = A.load()
    if a.key:
        c1 = getpass.getpass("Nueva contraseña (mínimo 10 caracteres): ")
        if len(c1) < 10:
            raise SystemExit("Muy corta: use al menos 10 caracteres.")
        if getpass.getpass("Repítala: ") != c1:
            raise SystemExit("No coinciden.")
        cfg["clave"] = A.key_hash(c1)
        print("Contraseña guardada (solo su hash).")
    for value in (a.ip, a.host):
        if value:
            ipaddress.ip_address(value)
    if a.ip and a.ip not in cfg["ips_permitidas"]:
        cfg["ips_permitidas"].append(a.ip)
    if a.remove_ip and a.remove_ip in cfg["ips_permitidas"]:
        cfg["ips_permitidas"].remove(a.remove_ip)
    if a.host:
        cfg["host"] = a.host
    if a.port:
        cfg["puerto"] = a.port
    A.save(cfg)
    print(f"Escucha en: {cfg['host']}:{cfg['puerto']}")
    print(f"IP permitidas: {', '.join(cfg['ips_permitidas']) or '(ninguna; solo esta PC)'}")
    print(f"Contraseña: {'definida' if cfg.get('clave') else 'NO definida (el visor no arranca sin contraseña)'}")


if __name__ == "__main__":
    main()
