"""Pontos, vetores e parse de campos de texto ``x, y[, z]``.

Os dois módulos de produto usam metros. As convenções de profundidade são
opostas e não são convertidas aqui.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parse_vector3(text: str) -> np.ndarray:
    """Interpreta um texto ``x, y, z`` em um vetor NumPy de comprimento 3.

    Parameters
    ----------
    text : str
        Três flutuantes separados por vírgula.

    Returns
    -------
    numpy.ndarray
        Array ``(3,)`` em ``float``.

    Raises
    ------
    ValueError
        Se o texto não for exatamente três números.
    """
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid format: {text}. Use 'x, y, z'.")
    try:
        return np.array([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Invalid format: {text}. Use 'x, y, z'.") from exc


def parse_pair(text: str) -> tuple[float, float]:
    """Interpreta um texto ``x, y`` em um par de flutuantes.

    Parameters
    ----------
    text : str
        Dois flutuantes separados por vírgula.

    Returns
    -------
    tuple of float
        ``(x, y)``.

    Raises
    ------
    ValueError
        Se o texto não for exatamente dois números.
    """
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Use two comma-separated values.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("Use two comma-separated values.") from exc


def format_pair(values: tuple[float, float] | list[float]) -> str:
    """Formata um par para um campo da GUI."""
    return f"{values[0]}, {values[1]}"


@dataclass(frozen=True)
class Point3D:
    """Posição XYZ em metros.

    No módulo Well Path a profundidade é **Z negativo**.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        """Devolve ``(x, y, z)`` como array NumPy."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_text(self) -> str:
        """Texto ``x, y, z`` no formato dos campos da GUI."""
        return f"{self.x}, {self.y}, {self.z}"

    @classmethod
    def from_array(cls, values) -> Point3D:
        """Constrói a partir de um array-like de comprimento 3."""
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != 3:
            raise ValueError("Point3D requires three coordinates.")
        return cls(float(array[0]), float(array[1]), float(array[2]))

    @classmethod
    def from_text(cls, text: str) -> Point3D:
        """Constrói a partir de um campo ``x, y, z``."""
        return cls.from_array(parse_vector3(text))


@dataclass(frozen=True)
class Vector3:
    """Vetor 3D (direção da broca ``v``, não uma posição)."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        """Devolve ``(x, y, z)`` como array NumPy."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_text(self) -> str:
        """Texto ``x, y, z`` no formato dos campos da GUI."""
        return f"{self.x}, {self.y}, {self.z}"

    @classmethod
    def from_array(cls, values) -> Vector3:
        """Constrói a partir de um array-like de comprimento 3."""
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != 3:
            raise ValueError("Vector3 requires three components.")
        return cls(float(array[0]), float(array[1]), float(array[2]))

    @classmethod
    def from_text(cls, text: str) -> Vector3:
        """Constrói a partir de um campo ``x, y, z``."""
        return cls.from_array(parse_vector3(text))


@dataclass(frozen=True)
class Point2D:
    """Posição XY em metros.

    No módulo Minimization a profundidade é **Y positivo** para baixo.
    """

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        """Devolve ``(x, y)``."""
        return (self.x, self.y)

    def as_text(self) -> str:
        """Texto ``x, y`` no formato dos campos da GUI."""
        return format_pair(self.as_tuple())

    @classmethod
    def from_pair(cls, values) -> Point2D:
        """Constrói a partir de um par ``(x, y)``."""
        x, y = values
        return cls(float(x), float(y))

    @classmethod
    def from_text(cls, text: str) -> Point2D:
        """Constrói a partir de um campo ``x, y``."""
        return cls.from_pair(parse_pair(text))
