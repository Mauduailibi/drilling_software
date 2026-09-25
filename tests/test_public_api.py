"""Checagens de sanidade: pontos de entrada públicos continuam documentados e importáveis."""

from __future__ import annotations

from drilling.core import parse_vector3
from drilling.features.minimization import calculate_minimization
from drilling.features.minimization.auxiliaries import lenght, theta, validate_configuration
from drilling.features.minimization.defaults import build_default_data, build_default_mesh
from drilling.features.well_path import solve_case1, solve_case2, solve_case3
from drilling.features.well_path.defaults import DEFAULT_WELL_PATH_INPUT


def test_public_solvers_have_numpy_docstrings() -> None:
    """A Fase 0 documenta a API pública para que novos contribuidores usem help() e pdoc."""
    for func in (
        solve_case1,
        solve_case2,
        solve_case3,
        calculate_minimization,
        build_default_data,
        build_default_mesh,
        theta,
        lenght,
        validate_configuration,
        parse_vector3,
    ):
        assert func.__doc__, f"{func.__name__} is missing a docstring"
        assert "Returns" in func.__doc__ or "Parameters" in func.__doc__, (
            f"{func.__name__} docstring is not NumPy-style"
        )


def test_package_exports() -> None:
    """Os dois pacotes de funcionalidade exportam os nomes usados pelos testes e pela GUI."""
    from drilling.features import well_path, minimization

    assert well_path.solve_case1 is solve_case1
    assert well_path.DEFAULT_WELL_PATH_INPUT is DEFAULT_WELL_PATH_INPUT
    assert minimization.calculate_minimization is calculate_minimization


def test_desktop_entrypoint_importable() -> None:
    """A Fase 1 entrega ``python -m drilling.app`` como caminho de inicialização suportado."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from drilling.app import main
    from drilling import __version__

    assert callable(main)
    assert __version__
