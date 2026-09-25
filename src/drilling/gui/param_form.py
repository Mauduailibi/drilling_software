"""Formulário Qt gerado a partir de ``FieldSpec`` — sem tabela chave-valor."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)

from drilling.core import FieldSpec


def _spin(value: float, spec: FieldSpec) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setDecimals(spec.decimals)
    widget.setRange(float(spec.minimum), float(spec.maximum))
    widget.setSingleStep(float(spec.step))
    widget.setValue(float(value))
    return widget


def spec_label(spec: FieldSpec) -> str:
    """Rótulo com unidade, se houver."""
    if spec.unit:
        return f"{spec.label} ({spec.unit})"
    return spec.label


class ParamForm:
    """Monta widgets tipados para um dict de parâmetros conhecidos.

    Parameters
    ----------
    specs : list of FieldSpec
        Schema estável (chave, tipo, unidade).
    values : dict
        Valores iniciais, indexados por ``spec.key``.
    """

    def __init__(self, specs: list[FieldSpec], values: dict):
        self.specs = list(specs)
        self.widgets: dict[str, QWidget] = {}
        self.layout = QFormLayout()
        for spec in self.specs:
            widget = self._make_widget(spec, values.get(spec.key))
            self.widgets[spec.key] = widget
            self.layout.addRow(spec_label(spec), widget)

    def _make_widget(self, spec: FieldSpec, value):
        if spec.kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            return widget
        if spec.kind == "optional_float":
            widget = QLineEdit("" if value is None else str(value))
            widget.setPlaceholderText("Optional")
            return widget
        if value is None:
            value = 0.0
        return _spin(value, spec)

    def to_dict(self) -> dict:
        """Lê os widgets de volta para o dict consumido pelos kernels."""
        params: dict = {}
        for spec in self.specs:
            widget = self.widgets[spec.key]
            if spec.kind == "bool":
                params[spec.key] = widget.isChecked()
            elif spec.kind == "optional_float":
                text = widget.text().strip()
                params[spec.key] = None if text == "" else float(text)
            else:
                params[spec.key] = float(widget.value())
        return params
