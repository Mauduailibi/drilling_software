"""Casca desktop em Qt.

``MainWindow`` hospeda uma aba por módulo de produto. Não deve importar
internos dos solvers; cada aba é dona do próprio pacote de funcionalidade.

Os nomes públicos são carregados sob demanda para as views poderem importar
``param_form`` sem ciclo com ``main_window``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MainWindow", "ParamForm"]


def __getattr__(name: str) -> Any:
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    if name == "ParamForm":
        from .param_form import ParamForm

        return ParamForm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
