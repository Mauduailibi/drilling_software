"""Tipos, parse e convenções compartilhados pelos dois módulos de produto.

Este pacote não conhece Qt, PyVista nem as fórmulas dos solvers. Well Path e
Minimization continuam independentes: aqui só há pontos, parse de texto,
``ConstraintCheck`` e envelopes em volta dos dicts que os kernels já devolvem.

Coordenadas
-----------
* Well Path: XYZ em metros, profundidade **Z negativa**.
* Minimization: XY em metros, profundidade **Y positiva** para baixo.

Não existe conversão automática 3D ↔ 2D.
"""

from .fields import FieldSpec
from .geometry import Point2D, Point3D, Vector3, format_pair, parse_pair, parse_vector3
from .types import (
    BuildupTrajectory,
    CandidateMetrics,
    ConstraintCheck,
    CorrectionResult,
    OptimizationResult,
    WellPathInput,
)

__all__ = [
    "BuildupTrajectory",
    "CandidateMetrics",
    "ConstraintCheck",
    "CorrectionResult",
    "FieldSpec",
    "OptimizationResult",
    "Point2D",
    "Point3D",
    "Vector3",
    "WellPathInput",
    "format_pair",
    "parse_pair",
    "parse_vector3",
]
