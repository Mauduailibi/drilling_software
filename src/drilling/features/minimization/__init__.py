"""Otimização de trajetória Tipo 1 (força, torque, tempo de broca, tempo total).

A geometria é um trecho vertical L1, uma curva de build-up de raio R e um
trecho tangente até o alvo ``P3``. A profundidade é o eixo **Y positivo**,
convenção diferente da usada em ``drilling.features.well_path``.

Orquestração pública
--------------------
``optimize.calculate_minimization`` é a função chamada pelo worker da GUI.
``defaults.build_default_data`` e ``defaults.build_default_mesh`` são as
entradas congeladas pelos testes golden da Fase 0.

Geologia
--------
A GUI recebe a malha pronta em GRDECL: ``grid.read_grdecl`` lê o arquivo e
``grid.GridGeology`` posiciona cabeça do poço e alvo nela. A ``mesh`` de
intervalos de profundidade continua sendo a geologia dos testes golden.
"""

from .defaults import (
    LITHOLOGIES,
    build_default_data,
    build_default_mesh,
    build_default_operational_parameters,
)
from .grid import CornerPointGrid, GridGeology, read_grdecl
from .optimize import calculate_minimization

__all__ = [
    "CornerPointGrid",
    "GridGeology",
    "LITHOLOGIES",
    "build_default_data",
    "build_default_mesh",
    "build_default_operational_parameters",
    "calculate_minimization",
    "read_grdecl",
]
