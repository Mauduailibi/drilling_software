"""Otimização de trajetória Tipo 1 (força, torque, tempo de broca, tempo total).

A geometria é um trecho vertical L1, uma curva de build-up de raio R e um
trecho tangente até o alvo ``P3``. A profundidade é o eixo **Y positivo**,
convenção diferente da usada em ``drilling.features.well_path``.

Orquestração pública
--------------------
``optimize.calculate_minimization`` é a função chamada pelo worker da GUI.
``defaults.build_default_data`` e ``defaults.build_default_mesh`` são as
entradas congeladas pelos testes golden da Fase 0.
"""

from .defaults import (
    LITHOLOGIES,
    build_default_data,
    build_default_mesh,
    build_default_operational_parameters,
)
from .optimize import calculate_minimization

__all__ = [
    "LITHOLOGIES",
    "build_default_data",
    "build_default_mesh",
    "build_default_operational_parameters",
    "calculate_minimization",
]
