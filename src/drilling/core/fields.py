"""Especificação de um parâmetro conhecido da GUI (substitui tabela chave-valor)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """Descreve um campo tipado: chave estável, rótulo, unidade e widget.

    Parameters
    ----------
    key : str
        Nome da chave no dict consumido pelos kernels.
    label : str
        Texto exibido na GUI.
    unit : str, optional
        Unidade mostrada ao lado do rótulo.
    kind : {'float', 'optional_float', 'bool'}
        Tipo do widget. ``optional_float`` trata campo vazio como ``None``.
    minimum, maximum, step : float
        Faixa do ``QDoubleSpinBox`` quando ``kind='float'``.
    decimals : int
        Casas decimais do spin box.
    """

    key: str
    label: str
    unit: str = ""
    kind: str = "float"
    minimum: float = 0.0
    maximum: float = 1e12
    step: float = 1.0
    decimals: int = 4
