"""Testes golden dos núcleos de otimização de trajetória Tipo 1.

Os testes rápidos congelam uma única configuração (L1, R) e os dois ótimos
mecânicos. A malha completa dos quatro objetivos (incluindo tempo de broca
e tempo operacional) está marcada como ``slow`` porque varre o domínio
padrão L1 × R.
"""

from __future__ import annotations

import pytest

from drilling.features.minimization.auxiliaries import (
    Nl,
    buckling,
    down_tension,
    lenght,
    theta,
    up_tension,
    validate_configuration,
)
from drilling.features.minimization.minimal import drilling_time_breakdown
from drilling.features.minimization.operational import (
    DEFAULT_MECHANICAL_LIMITS,
    _best_mechanical_candidates,
    operational_time_breakdown,
)
from drilling.features.minimization.defaults import build_default_data, build_default_mesh
from drilling.features.minimization.optimize import calculate_minimization

from tests.numeric import assert_close


def _default_problem():
    """Reconstrói o DataSet, a malha e os parâmetros operacionais padrão da GUI."""
    data, operational_parameters = build_default_data()
    geological_mesh = build_default_mesh()
    return data, geological_mesh, operational_parameters


def test_reference_point_l1_1200_r_500_matches_golden(minimization_point_golden: dict) -> None:
    """As saídas do núcleo para L1=1200 m, R=500 m devem permanecer estáveis bit a bit."""
    data, geological_mesh, operational_parameters = _default_problem()
    frozen = minimization_point_golden
    l1 = float(frozen["l1"])
    radius = float(frozen["R"])

    config = validate_configuration(data, l1, radius)
    for key, value in frozen["config"].items():
        assert_close(config[key], value, name=f"config.{key}")

    assert_close(theta(data, l1, radius), frozen["theta_rad"], name="theta")
    assert_close(lenght(data, l1, radius), frozen["lengths"], name="lenght")
    assert_close(buckling(data, l1, radius), frozen["lc"], name="buckling")
    assert_close(Nl(data, l1, radius), frozen["neutral_line"], name="neutral_line")
    assert_close(up_tension(data, l1, radius), frozen["up_tension"], name="up_tension")
    assert_close(down_tension(data, l1, radius), frozen["down_tension"], name="down_tension")

    timing = drilling_time_breakdown(data, geological_mesh, l1, radius)
    assert_close(timing["total_time_h"], frozen["drilling_time_h"], name="drilling_time_h")
    assert_close(timing["average_rop_mph"], frozen["average_rop_mph"], name="average_rop_mph")
    assert_close(timing["total_length_m"], frozen["total_length_m"], name="total_length_m")
    assert len(timing["elements"]) == frozen["n_elements"]

    for lithology, values in frozen["by_lithology"].items():
        assert_close(timing["by_lithology"][lithology]["length_m"], values["length_m"], name=f"{lithology}.length_m")
        assert_close(timing["by_lithology"][lithology]["time_h"], values["time_h"], name=f"{lithology}.time_h")
    for section, values in frozen["by_section"].items():
        assert_close(timing["by_section"][section]["length_m"], values["length_m"], name=f"{section}.length_m")
        assert_close(timing["by_section"][section]["time_h"], values["time_h"], name=f"{section}.time_h")

    operational = operational_time_breakdown(
        data,
        geological_mesh,
        l1,
        radius,
        drilling_timing=timing,
        operational_parameters=operational_parameters,
    )
    assert_close(operational["total_operational_time_h"], frozen["operational_time_h"], name="operational_time_h")
    assert_close(operational["total_time_h"], frozen["total_time_h"], name="total_time_h")
    for category, values in frozen["by_category"].items():
        assert_close(operational["by_category"][category]["time_h"], values["time_h"], name=f"{category}.time_h")
        assert operational["by_category"][category]["count"] == values["count"]


def test_mechanical_optima_match_golden(minimization_mechanical_golden: dict) -> None:
    """A força mínima no topo e o torque mínimo permanecem nos pontos congelados da malha."""
    data, _, _ = _default_problem()
    best_force, best_torque = _best_mechanical_candidates(data)
    frozen_force = minimization_mechanical_golden["force"]
    frozen_torque = minimization_mechanical_golden["torque"]
    assert_close(best_force["l1"], frozen_force["l1"], name="force.l1")
    assert_close(best_force["R"], frozen_force["R"], name="force.R")
    assert_close(best_force["up_force_1"], frozen_force["up_force_1"], name="force.up_force_1")
    assert_close(best_force["torque"], frozen_force["torque"], name="force.torque")
    assert_close(best_torque["l1"], frozen_torque["l1"], name="torque.l1")
    assert_close(best_torque["R"], frozen_torque["R"], name="torque.R")
    assert_close(best_torque["up_force_1"], frozen_torque["up_force_1"], name="torque.up_force_1")
    assert_close(best_torque["torque"], frozen_torque["torque"], name="torque.torque")


def _slim_result(result: dict) -> dict:
    """Mantém os mesmos campos gravados em ``minimization_optima.json``."""
    timing = result["timing"]
    operational = result["operational"]
    mechanical = result["mechanical"]
    return {
        "l1": float(result["l1"]),
        "R": float(result["R"]),
        "l2": float(result["l2"]),
        "l3": float(result["l3"]),
        "angle_deg": float(result["angle_deg"]),
        "lc": float(result["lc"]),
        "up_force_1": float(result["up_force_1"]),
        "up_force_2": float(result["up_force_2"]),
        "up_force_3": float(result["up_force_3"]),
        "down_force_1": float(result["down_force_1"]),
        "down_force_2": float(result["down_force_2"]),
        "down_force_3": float(result["down_force_3"]),
        "torque": float(result["torque"]),
        "neutral_line": float(result["neutral_line"]),
        "drilling_time_h": float(result["drilling_time_h"]),
        "operational_time_h": float(result["operational_time_h"]),
        "total_time_h": float(result["total_time_h"]),
        "average_rop_mph": float(timing["average_rop_mph"]),
        "total_length_m": float(timing["total_length_m"]),
        "is_valid": bool(mechanical["is_valid"]),
        "by_lithology": {
            name: {"length_m": float(values["length_m"]), "time_h": float(values["time_h"])}
            for name, values in timing["by_lithology"].items()
        },
        "by_section": {
            name: {"length_m": float(values["length_m"]), "time_h": float(values["time_h"])}
            for name, values in timing["by_section"].items()
        },
        "by_category": {
            name: {"time_h": float(values["time_h"]), "count": int(values["count"])}
            for name, values in operational["by_category"].items()
        },
    }


def _assert_optima_result(actual: dict, expected: dict, label: str) -> None:
    """Compara um objetivo de otimização reduzido com as métricas congeladas."""
    scalar_keys = [
        "l1",
        "R",
        "l2",
        "l3",
        "angle_deg",
        "lc",
        "up_force_1",
        "up_force_2",
        "up_force_3",
        "down_force_1",
        "down_force_2",
        "down_force_3",
        "torque",
        "neutral_line",
        "drilling_time_h",
        "operational_time_h",
        "total_time_h",
        "average_rop_mph",
        "total_length_m",
        "is_valid",
    ]
    for key in scalar_keys:
        if key == "is_valid":
            assert bool(actual[key]) is bool(expected[key]), f"{label}.is_valid"
            continue
        assert_close(actual[key], expected[key], name=f"{label}.{key}")
    for group_name in ("by_lithology", "by_section"):
        for item_name, values in expected[group_name].items():
            assert_close(
                actual[group_name][item_name]["length_m"],
                values["length_m"],
                name=f"{label}.{item_name}.length_m",
            )
            assert_close(
                actual[group_name][item_name]["time_h"],
                values["time_h"],
                name=f"{label}.{item_name}.time_h",
            )
    for category, values in expected["by_category"].items():
        assert_close(
            actual["by_category"][category]["time_h"],
            values["time_h"],
            name=f"{label}.{category}.time_h",
        )
        assert actual["by_category"][category]["count"] == values["count"]


@pytest.mark.slow
def test_four_objectives_match_golden(minimization_optima_golden: dict) -> None:
    """Ótimos da malha padrão para força, torque, tempo de broca e tempo total."""
    data, geological_mesh, operational_parameters = _default_problem()
    payload = calculate_minimization(
        data,
        geological_mesh,
        operational_parameters,
        DEFAULT_MECHANICAL_LIMITS,
    )
    frozen_results = minimization_optima_golden["results"]
    assert set(payload["results"]) == set(frozen_results)
    for key, expected in frozen_results.items():
        _assert_optima_result(_slim_result(payload["results"][key]), expected, key)

    frozen_series = minimization_optima_golden["series"]
    for scope, family in frozen_series.items():
        for objective, expected in family.items():
            item = payload["series"][scope][objective]
            x = [float(v) for v in item["x"]]
            y = [float(v) for v in item["y"]]
            assert len(x) == expected["n"]
            assert_close(x[0], expected["x_first"], name=f"{scope}.{objective}.x_first")
            assert_close(x[-1], expected["x_last"], name=f"{scope}.{objective}.x_last")
            assert_close(sum(x), expected["x_sum"], name=f"{scope}.{objective}.x_sum", rtol=1e-8, atol=1e-6)
            assert_close(y[0], expected["y_first"], name=f"{scope}.{objective}.y_first")
            assert_close(y[-1], expected["y_last"], name=f"{scope}.{objective}.y_last")
            assert_close(sum(y), expected["y_sum"], name=f"{scope}.{objective}.y_sum", rtol=1e-8, atol=1e-6)
            assert_close(min(y), expected["y_min"], name=f"{scope}.{objective}.y_min")
            assert_close(max(y), expected["y_max"], name=f"{scope}.{objective}.y_max")
            if expected["best_x"] is not None:
                assert_close(item["best_x"], expected["best_x"], name=f"{scope}.{objective}.best_x")
