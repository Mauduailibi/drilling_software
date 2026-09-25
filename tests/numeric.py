"""Auxiliares de comparação numérica compartilhados pelos testes golden."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

RTOL = 1e-10
ATOL = 1e-10


def load_golden(name: str) -> dict:
    """Carrega um snapshot JSON commitado de ``tests/goldens``.

    Parameters
    ----------
    name : str
        Nome do arquivo, por exemplo ``well_path_defaults.json``.

    Returns
    -------
    dict
        Payload JSON interpretado.
    """
    path = GOLDENS_DIR / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_close(actual, expected, *, rtol: float = RTOL, atol: float = ATOL, name: str = "value") -> None:
    """Compara escalares ou sequências 1D com tolerância relativa/absoluta estrita.

    Parameters
    ----------
    actual : array_like
        Valor produzido pelo código atual.
    expected : array_like
        Valor golden congelado.
    rtol, atol : float
        Tolerâncias de comparação do NumPy.
    name : str
        Rótulo incluído na mensagem da asserção.
    """
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=atol,
        err_msg=f"Golden mismatch in {name}",
    )


def assert_array_fingerprint(actual: np.ndarray, fingerprint: dict, *, name: str) -> None:
    """Compara um array numérico com uma impressão digital golden compacta.

    Parameters
    ----------
    actual : numpy.ndarray
        Array produzido pelo código atual.
    fingerprint : dict
        Mapeamento com as chaves ``shape``, ``first``, ``last``, ``sum`` e ``n``.
    name : str
        Rótulo incluído nas mensagens de asserção.
    """
    array = np.asarray(actual, dtype=float)
    assert list(array.shape) == fingerprint["shape"], f"{name}.shape"
    assert int(array.size) == int(fingerprint["n"]), f"{name}.n"
    flat = array.reshape(-1)
    n_first = len(fingerprint["first"])
    n_last = len(fingerprint["last"])
    assert_close(flat[:n_first], fingerprint["first"], name=f"{name}.first")
    assert_close(flat[-n_last:], fingerprint["last"], name=f"{name}.last")
    assert_close(float(np.sum(array)), fingerprint["sum"], name=f"{name}.sum")
