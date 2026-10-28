"""Configura el acceso al visor de pruebas.

  python -m visor_pruebas.configurar --clave                 # pide la contraseña (no se muestra al escribirla)
  python -m visor_pruebas.configurar --ip 100.101.102.103    # la PC del evaluador (su IP de Tailscale)
  python -m visor_pruebas.configurar --quitar-ip 100.101.102.103
  python -m visor_pruebas.configurar --host 100.64.0.5       # la IP de Tailscale de ESTA PC (tailscale ip -4)
  python -m visor_pruebas.configurar --ver
"""

import argparse
import getpass
import ipaddress

from . import acceso as A


def main():
    """Aplica las opciones dadas y muestra la configuración resultante. Las IP se validan antes de guardarse."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--clave", action="store_true", help="definir o cambiar la contraseña")
    ap.add_argument("--ip", help="agregar la IP permitida (PC del evaluador)")
    ap.add_argument("--quitar-ip")
    ap.add_argument("--host", help="IP donde escucha el visor (la de Tailscale de esta PC; 127.0.0.1 = solo esta PC)")
    ap.add_argument("--puerto", type=int)
    ap.add_argument("--ver", action="store_true")
    a = ap.parse_args()
    cfg = A.cargar()
    if a.clave:
        c1 = getpass.getpass("Nueva contraseña (mínimo 10 caracteres): ")
        if len(c1) < 10:
            raise SystemExit("Muy corta: use al menos 10 caracteres.")
        if getpass.getpass("Repítala: ") != c1:
            raise SystemExit("No coinciden.")
        cfg["clave"] = A.hash_clave(c1)
        print("Contraseña guardada (solo su hash).")
    for valor in (a.ip, a.host):
        if valor:
            ipaddress.ip_address(valor)
    if a.ip and a.ip not in cfg["ips_permitidas"]:
        cfg["ips_permitidas"].append(a.ip)
    if a.quitar_ip and a.quitar_ip in cfg["ips_permitidas"]:
        cfg["ips_permitidas"].remove(a.quitar_ip)
    if a.host:
        cfg["host"] = a.host
    if a.puerto:
        cfg["puerto"] = a.puerto
    A.guardar(cfg)
    print(f"Escucha en: {cfg['host']}:{cfg['puerto']}")
    print(f"IP permitidas: {', '.join(cfg['ips_permitidas']) or '(ninguna; solo esta PC)'}")
    print(f"Contraseña: {'definida' if cfg.get('clave') else 'NO definida (el visor no arranca sin contraseña)'}")


if __name__ == "__main__":
    main()
