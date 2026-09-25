"""Otimização de trajetória Tipo 1 (força, torque, tempo de broca, tempo total).

A geometria é um trecho vertical L1, uma curva de build-up de raio R e um
trecho tangente até o alvo ``P3``. Os pontos são ``(x, y, z)`` com ``z``
(profundidade) **positivo para baixo**, convenção diferente da usada em
``drilling.features.well_path``. O poço fica no plano vertical que liga ``P0``
ao alvo.

A geologia é um ``HorizonModel`` (``geology``): horizontes ``z(x, y)`` e mapas
de fácies permitem litologias diferentes na mesma profundidade. ``mesh(...)``
continua criando o caso particular de camadas planas.

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
from .geology import GeoGrid2D, HorizonModel
from .data_base import DataSet, mesh
from .optimize import calculate_minimization

__all__ = [
    "DataSet",
    "GeoGrid2D",
    "HorizonModel",
    "LITHOLOGIES",
    "build_default_data",
    "build_default_mesh",
    "build_default_operational_parameters",
    "calculate_minimization",
    "mesh",
]
