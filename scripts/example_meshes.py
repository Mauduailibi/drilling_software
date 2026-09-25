"""Modelos geológicos de exemplo compartilhados pelos scripts de pesquisa.

Uma coluna estratigráfica e um conjunto de propriedades de rocha, em três
variantes geométricas, de modo que trocar entre elas muda *apenas* a geometria:

``flat``
    Camadas horizontais, o "bolo em camadas" da antiga implementação em blocos.
    Um modelo plano é só um caso particular do modelo 3D (superfícies
    constantes), não um "modo 2D" separado.
``dipping``
    A mesma coluna inclinada, dobrada sobre um anticlinal e cortada por uma
    falha normal.
``facies``
    A coluna inclinada, mas a dolomita dura passa lateralmente a arenito perto
    da cabeça do poço: duas litologias na mesma profundidade, o que uma pilha
    de paralelepípedos não representava.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import drilling.features.minimization.geology as geo


ROP_VALUES = {"Sandstone": 18.0, "Shale": 15.0, "Limestone": 11.0, "Dolomite": 3.2, "Evaporite": 24.0}
MU_VALUES = {"Sandstone": 0.25, "Shale": 0.20, "Limestone": 0.28, "Dolomite": 0.40, "Evaporite": 0.18}

# Base of each unit at the reference point (x = 0), top to bottom.
COLUMN = [
    ("Sandstone", 600.0),
    ("Limestone", 1200.0),
    ("Dolomite", 1800.0),   # the slow, abrasive unit the trajectories try to avoid
    ("Sandstone", 2400.0),
    ("Evaporite", 2700.0),
    ("Sandstone", 3400.0),
]

# Horizontal offset beyond which the dolomite exists in the 'facies' model.
FACIES_BOUNDARY_X = 300.0

DIP_X = 0.085
DIP_Y = 0.020


def build_grid() -> geo.GeoGrid2D:
    return geo.GeoGrid2D.from_bounds((-300, 1600), (-300, 700), nx=97, ny=51)


def build_model(kind: str = "facies") -> geo.HorizonModel:
    """Cria uma das três variantes: ``flat``, ``dipping`` ou ``facies``."""
    if kind not in ("flat", "dipping", "facies"):
        raise ValueError("'kind' must be one of 'flat', 'dipping' or 'facies'.")

    grid = build_grid()
    dip_x = 0.0 if kind == "flat" else DIP_X
    dip_y = 0.0 if kind == "flat" else DIP_Y

    layers = []
    for index, (lithology, base_depth) in enumerate(COLUMN):
        if kind == "flat":
            surface = geo.flat_surface(grid, base_depth)
        else:
            surface = geo.dipping_surface(grid, base_depth, dip_x=dip_x, dip_y=dip_y)
            # A broad anticline over the middle of the field, dying out downwards.
            relief = 180.0 * max(0.0, 1.0 - base_depth / 3400.0)
            surface = geo.anticline(grid, surface, 700.0, 250.0, relief, 450.0, 320.0)
            # A normal fault that offsets everything above the evaporite seal.
            if base_depth <= 2400.0:
                surface = geo.fault(grid, surface, (850.0, 0.0), strike_deg=72.0, throw=65.0, smooth_m=70.0)

        facies = lithology
        # The hard dolomite is only present away from the well head.
        if kind == "facies" and lithology == "Dolomite":
            facies = geo.facies_linear_boundary(
                grid, "Sandstone", "Dolomite", (FACIES_BOUNDARY_X, 0.0), strike_deg=78.0
            )
        layers.append({"lithology": facies, "base": surface, "name": f"{index}-{lithology}"})

    return geo.HorizonModel.from_layer_stack(
        grid, geo.flat_surface(grid, 0.0), layers,
        rop_values=ROP_VALUES, mu_values=MU_VALUES, name=kind,
    )
