"""Operaciones con cajas y rectángulos (x0, y0, x1, y1, ...) en cualquier unidad: los elementos después del
cuarto (puntaje, texto) se ignoran."""


def center(b):
    """Centro (x, y) de una caja."""
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inside(c, r):
    """True si el punto `c` está dentro del rectángulo `r` (bordes incluidos)."""
    return r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]


def intersects(a, b):
    """True si los rectángulos `a` y `b` se superponen (tocarse en un borde no cuenta)."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def iou(a, b):
    """Intersección sobre unión de dos cajas (0 si no se tocan o si su unión no tiene área)."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
