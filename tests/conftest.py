"""Fixtures pytest dos testes golden da Fase 0."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.numeric import GOLDENS_DIR, load_golden

# Reexporta para que importações históricas continuem funcionando, se necessário.
from tests.numeric import assert_array_fingerprint, assert_close  # noqa: F401


@pytest.fixture(scope="session")
def well_path_golden() -> dict:
    """Snapshots commitados dos três casos de correção de trajetória."""
    return load_golden("well_path_defaults.json")


@pytest.fixture(scope="session")
def minimization_point_golden() -> dict:
    """Snapshot commitado da configuração de referência L1=1200 m, R=500 m."""
    return load_golden("minimization_point_l1_1200_r_500.json")


@pytest.fixture(scope="session")
def minimization_mechanical_golden() -> dict:
    """Snapshots commitados da força mínima no topo e do torque mínimo."""
    return load_golden("minimization_mechanical_optima.json")


@pytest.fixture(scope="session")
def minimization_optima_golden() -> dict:
    """Snapshots commitados dos quatro objetivos padrão de otimização."""
    path = GOLDENS_DIR / "minimization_optima.json"
    if not path.exists():
        pytest.skip("Missing tests/goldens/minimization_optima.json. Run: python scripts/capture_goldens.py --slow")
    return load_golden("minimization_optima.json")
