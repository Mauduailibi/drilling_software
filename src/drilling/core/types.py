"""Envelopes de dados em volta dos dicts e tuples que os kernels já devolvem.

Nenhuma fórmula vive aqui. Os solvers continuam recebendo arrays e devolvendo
dicts; estas dataclasses só dão nome, unidade implícita e autocomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .geometry import Point2D, Point3D, Vector3, parse_vector3


@dataclass(frozen=True)
class ConstraintCheck:
    """Uma linha da tabela de restrições do Well Path.

    O solver ainda devolve ``(name, ok, value, limit)``. Use ``from_item``
    na GUI e ``as_tuple`` quando um teste golden precisar do formato antigo.
    """

    name: str
    ok: bool
    value: str
    limit: str

    def as_tuple(self) -> tuple[str, bool, str, str]:
        """Formato histórico consumido pelos snapshots golden."""
        return (self.name, self.ok, self.value, self.limit)

    @classmethod
    def from_item(cls, item: ConstraintCheck | tuple) -> ConstraintCheck:
        """Aceita a dataclass ou a tuple ``(name, ok, value, limit)``."""
        if isinstance(item, cls):
            return item
        name, ok, value, limit = item
        return cls(name=str(name), ok=bool(ok), value=str(value), limit=str(limit))

    @classmethod
    def from_items(cls, items: Iterable) -> tuple[ConstraintCheck, ...]:
        """Converte a lista de status do solver."""
        return tuple(cls.from_item(item) for item in items)


@dataclass(frozen=True)
class WellPathInput:
    """Os cinco vetores da aba Well Path Correction.

    ``pin`` e ``pbd`` descrevem o poço planejado; ``p1``, ``pt`` e ``v`` entram
    no solver. Coordenadas em metros, Z negativo.
    """

    pin: Point3D
    pbd: Point3D
    p1: Point3D
    pt: Point3D
    v: Vector3

    def solver_kwargs(self) -> dict[str, Any]:
        """Keyword-args aceitos por ``solve_case1`` / ``2`` / ``3``."""
        return {
            "Pin": self.pin.as_array(),
            "Pbd": self.pbd.as_array(),
            "p1": self.p1.as_array(),
            "pt": self.pt.as_array(),
            "v": self.v.as_array(),
        }

    @classmethod
    def from_text(
        cls,
        pin: str,
        pbd: str,
        p1: str,
        pt: str,
        v: str,
    ) -> WellPathInput:
        """Monta a entrada a partir dos campos da GUI."""
        return cls(
            pin=Point3D.from_text(pin),
            pbd=Point3D.from_text(pbd),
            p1=Point3D.from_text(p1),
            pt=Point3D.from_text(pt),
            v=Vector3.from_array(parse_vector3(v)),
        )


@dataclass(frozen=True)
class CorrectionResult:
    """Envelope do dicionário devolvido por ``solve_case*``.

    O payload bruto segue indo para o PyVista. ``status`` vira
    ``ConstraintCheck`` para a tabela da GUI.
    """

    payload: dict
    status: tuple[ConstraintCheck, ...]

    @property
    def is_valid(self) -> bool:
        """Verdadeiro se todas as restrições passaram."""
        return all(check.ok for check in self.status)

    def as_dict(self) -> dict:
        """O dict original do solver, inclusive a tabela de status em tuples."""
        return self.payload

    @classmethod
    def from_solver_dict(cls, payload: Mapping[str, Any]) -> CorrectionResult:
        """Empacota a saída de ``solve_case*`` sem alterar chaves."""
        data = dict(payload)
        return cls(payload=data, status=ConstraintCheck.from_items(data.get("status", ())))


@dataclass(frozen=True)
class BuildupTrajectory:
    """Geometria Tipo 1: origem ``P0``, alvo ``P3``, vertical L1 e raio R.

    Profundidade no eixo Y positivo. ``P0`` deve ser ``(0, 0)`` no kernel atual.
    """

    p0: Point2D
    p3: Point2D
    l1: float
    radius: float


@dataclass(frozen=True)
class CandidateMetrics:
    """Métricas de um candidato ``(L1, R)`` na malha de otimização.

    O dict do varredor permanece em ``payload``; as propriedades abaixo são
    atalhos das chaves já usadas pela GUI e pelos testes golden.
    """

    payload: dict

    @property
    def l1(self) -> float:
        return float(self.payload["l1"])

    @property
    def radius(self) -> float:
        return float(self.payload["R"])

    @property
    def l2(self) -> float:
        return float(self.payload["l2"])

    @property
    def l3(self) -> float:
        return float(self.payload["l3"])

    @property
    def angle_deg(self) -> float:
        return float(self.payload["angle_deg"])

    @property
    def top_axial_force(self) -> float:
        return float(self.payload["up_force_1"])

    @property
    def torque(self) -> float:
        return float(self.payload["torque"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateMetrics:
        """Empacota o registro de um objetivo sem copiar a matemática."""
        return cls(payload=dict(payload))


@dataclass(frozen=True)
class OptimizationResult:
    """Envelope do payload de ``calculate_minimization``.

    Os testes golden continuam lendo o dict. A GUI pode usar as propriedades.
    """

    payload: dict

    @property
    def results(self) -> dict:
        return self.payload["results"]

    @property
    def series(self) -> dict:
        return self.payload["series"]

    @property
    def data(self):
        return self.payload["data"]

    @property
    def mesh(self):
        return self.payload["mesh"]

    def candidate(self, objective: str) -> CandidateMetrics:
        """Atalho para um dos quatro objetivos (``force``, ``torque``, ``time``, ``total``)."""
        return CandidateMetrics.from_dict(self.results[objective])

    def as_dict(self) -> dict:
        """O payload original, bit a bit."""
        return self.payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OptimizationResult:
        """Empacota a saída de ``calculate_minimization``."""
        return cls(payload=dict(payload))
