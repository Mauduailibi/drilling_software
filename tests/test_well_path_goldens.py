"""Testes golden dos solvers de correção de trajetória.

Os snapshots congelam o comportamento numérico dos Casos 1–3 para as
entradas padrão da GUI, mais um vetor viável do Caso 3. Qualquer mudança
na geometria, no validador ou nesses defaults deve falhar aqui.
"""

from __future__ import annotations

import numpy as np
import pytest

from drilling.features.well_path.logic import solve_case1, solve_case2, solve_case3

from tests.numeric import assert_array_fingerprint, assert_close


def _vectors(golden_block: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstrói os cinco vetores da GUI gravados em um bloco golden."""
    return (
        np.asarray(golden_block["Pin"], dtype=float),
        np.asarray(golden_block["Pbd"], dtype=float),
        np.asarray(golden_block["p1"], dtype=float),
        np.asarray(golden_block["pt"], dtype=float),
        np.asarray(golden_block["v"], dtype=float),
    )


def _assert_status(actual_status, expected_status) -> None:
    """Compara tabelas de restrições, inclusive as strings formatadas de exibição."""
    assert len(actual_status) == len(expected_status)
    for actual, expected in zip(actual_status, expected_status):
        name, ok, value, limit = actual
        assert name == expected["name"]
        assert bool(ok) is bool(expected["ok"])
        assert value == expected["value"]
        assert limit == expected["limit"]


def _assert_result(actual: dict, expected: dict) -> None:
    """Compara um dicionário do solver com o payload golden."""
    for key, frozen in expected.items():
        assert key in actual, f"missing key {key}"
        current = actual[key]
        if key == "status":
            _assert_status(current, frozen)
        elif isinstance(frozen, dict) and "shape" in frozen:
            assert_array_fingerprint(current, frozen, name=key)
        elif isinstance(frozen, list):
            assert_close(current, frozen, name=key)
        else:
            assert_close(current, frozen, name=key)


def test_case1_gui_defaults_match_golden(well_path_golden: dict) -> None:
    """O Caso 1 com os defaults da GUI deve reproduzir a geometria e o status congelados."""
    block = well_path_golden["gui_defaults"]
    Pin, Pbd, p1, pt, v = _vectors(block)
    result = solve_case1(Pin, Pbd, p1, pt, v)
    _assert_result(result, block["case1"])


def test_case2_gui_defaults_match_golden(well_path_golden: dict) -> None:
    """O Caso 2 com os defaults da GUI deve reproduzir a geometria e o status congelados."""
    block = well_path_golden["gui_defaults"]
    Pin, Pbd, p1, pt, v = _vectors(block)
    result = solve_case2(Pin, Pbd, p1, pt, v)
    _assert_result(result, block["case2"])


def test_case3_gui_defaults_still_reject_alignment(well_path_golden: dict) -> None:
    """O vetor padrão da GUI ultrapassa o limite de alinhamento de 20° — esse erro está congelado."""
    block = well_path_golden["gui_defaults"]
    Pin, Pbd, p1, pt, v = _vectors(block)
    expected = block["case3_error"]
    with pytest.raises(ValueError, match=expected["message"]):
        solve_case3(Pin, Pbd, p1, pt, v)


def test_case3_feasible_reference_matches_golden(well_path_golden: dict) -> None:
    """Uma entrada viável documentada do Caso 3 deve manter os mesmos comprimentos e ângulos."""
    block = well_path_golden["case3_feasible"]
    Pin, Pbd, p1, pt, v = _vectors(block)
    result = solve_case3(Pin, Pbd, p1, pt, v)
    _assert_result(result, block["result"])
