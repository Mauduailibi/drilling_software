"""Entradas padrão da GUI para correção de trajetória 3D.

Estes vetores são o contrato entre a aba Well Path, ``scripts/capture_goldens.py``
e os snapshots em ``tests/goldens/well_path_defaults.json``. Alterar um número
aqui deve falhar o pytest da Fase 0.
"""

from __future__ import annotations

from drilling.core import Point3D, Vector3, WellPathInput

DEFAULT_WELL_PATH_INPUT = WellPathInput(
    pin=Point3D(0.0, 0.0, 0.0),
    pbd=Point3D(0.0, 0.0, -2000.0),
    p1=Point3D(50.0, 100.0, -1800.0),
    pt=Point3D(1000.0, 0.0, -3000.0),
    v=Vector3(0.2, 0.4, -1.0),
)
"""Cinco vetores que a aba Well Path mostra ao abrir."""

CASE3_FEASIBLE_DIRECTION = Vector3(0.05, 0.05, -1.0)
"""Direção viável do Caso 3 (o ``v`` padrão da GUI estoura o limite de 20°)."""
