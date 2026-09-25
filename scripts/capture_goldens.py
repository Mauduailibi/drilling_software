"""Regenera os snapshots golden commitados a partir dos solvers atuais.

Este script não altera a matemática. Ele registra o que o código produz
hoje para que o pytest detecte uma deriva numérica posterior.

Usage
-----
    .venv/bin/python scripts/capture_goldens.py
    .venv/bin/python scripts/capture_goldens.py --slow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drilling.features.minimization.auxiliaries import (  # noqa: E402
    Nl,
    buckling,
    down_tension,
    lenght,
    theta,
    up_tension,
    validate_configuration,
)
from drilling.features.minimization.minimal import drilling_time_breakdown  # noqa: E402
from drilling.features.minimization.operational import (  # noqa: E402
    DEFAULT_MECHANICAL_LIMITS,
    _best_mechanical_candidates,
    operational_time_breakdown,
)
from drilling.features.minimization.defaults import build_default_data, build_default_mesh  # noqa: E402
from drilling.features.minimization.optimize import calculate_minimization  # noqa: E402
from drilling.features.well_path import solve_case1, solve_case2, solve_case3  # noqa: E402
from drilling.features.well_path.defaults import CASE3_FEASIBLE_DIRECTION, DEFAULT_WELL_PATH_INPUT  # noqa: E402

GOLDENS = ROOT / "tests" / "goldens"


def pack_status(status):
    """Serializa uma tabela de restrições em dicionários amigáveis a JSON."""
    return [{"name": name, "ok": bool(ok), "value": value, "limit": limit} for name, ok, value, limit in status]


def pack_arr(array):
    """Impressão digital compacta de um array numérico."""
    array = np.asarray(array, dtype=float)
    return {
        "shape": list(array.shape),
        "first": array.reshape(-1)[: min(6, array.size)].tolist(),
        "last": array.reshape(-1)[-min(6, array.size) :].tolist(),
        "sum": float(np.sum(array)),
        "n": int(array.size),
    }


def pack_case(result, skip=()):
    """Serializa um dicionário do solver de trajetória, omitindo listas volumosas de candidatos."""
    out = {}
    for key, val in result.items():
        if key in skip:
            continue
        if key == "status":
            out["status"] = pack_status(val)
        elif isinstance(val, np.ndarray):
            if val.ndim == 1 and val.size <= 3:
                out[key] = [float(x) for x in val]
            else:
                out[key] = pack_arr(val)
        elif isinstance(val, (float, np.floating)):
            out[key] = float(val)
        elif isinstance(val, (bool, np.bool_)):
            out[key] = bool(val)
        elif isinstance(val, (int, np.integer)):
            out[key] = int(val)
        else:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def capture_well_path() -> None:
    """Grava ``well_path_defaults.json``."""
    defaults = DEFAULT_WELL_PATH_INPUT
    Pin = defaults.pin.as_array()
    Pbd = defaults.pbd.as_array()
    p1 = defaults.p1.as_array()
    pt = defaults.pt.as_array()
    v = defaults.v.as_array()
    feasible_v = CASE3_FEASIBLE_DIRECTION.as_array()

    case3_error = None
    try:
        solve_case3(Pin, Pbd, p1, pt, v)
    except Exception as exc:
        case3_error = {"type": type(exc).__name__, "message": str(exc)}

    feasible = solve_case3(Pin, Pbd, p1, pt, feasible_v)
    payload = {
        "gui_defaults": {
            "Pin": defaults.pin.as_array().tolist(),
            "Pbd": defaults.pbd.as_array().tolist(),
            "p1": defaults.p1.as_array().tolist(),
            "pt": defaults.pt.as_array().tolist(),
            "v": defaults.v.as_array().tolist(),
            "case1": pack_case(solve_case1(Pin, Pbd, p1, pt, v)),
            "case2": pack_case(solve_case2(Pin, Pbd, p1, pt, v)),
            "case3_error": case3_error,
        },
        "case3_feasible": {
            "Pin": defaults.pin.as_array().tolist(),
            "Pbd": defaults.pbd.as_array().tolist(),
            "p1": defaults.p1.as_array().tolist(),
            "pt": defaults.pt.as_array().tolist(),
            "v": feasible_v.tolist(),
            "result": pack_case(feasible, skip=("tested_candidates",)),
        },
    }
    (GOLDENS / "well_path_defaults.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", GOLDENS / "well_path_defaults.json", flush=True)


def capture_minimization_point() -> None:
    """Grava o snapshot do núcleo L1=1200 m, R=500 m e os ótimos mecânicos."""
    data, ops = build_default_data()
    geological_mesh = build_default_mesh()
    l1, radius = 1200.0, 500.0
    config = validate_configuration(data, l1, radius)
    timing = drilling_time_breakdown(data, geological_mesh, l1, radius)
    operational = operational_time_breakdown(
        data,
        geological_mesh,
        l1,
        radius,
        drilling_timing=timing,
        operational_parameters=ops,
    )
    point = {
        "l1": l1,
        "R": radius,
        "config": {key: float(value) for key, value in config.items()},
        "theta_rad": float(theta(data, l1, radius)),
        "lengths": [float(value) for value in lenght(data, l1, radius)],
        "lc": float(buckling(data, l1, radius)),
        "neutral_line": float(Nl(data, l1, radius)),
        "up_tension": [float(value) for value in up_tension(data, l1, radius)],
        "down_tension": [float(value) for value in down_tension(data, l1, radius)],
        "drilling_time_h": float(timing["total_time_h"]),
        "average_rop_mph": float(timing["average_rop_mph"]),
        "total_length_m": float(timing["total_length_m"]),
        "n_elements": int(len(timing["elements"])),
        "operational_time_h": float(operational["total_operational_time_h"]),
        "total_time_h": float(operational["total_time_h"]),
        "by_lithology": {
            key: {"length_m": float(val["length_m"]), "time_h": float(val["time_h"])}
            for key, val in timing["by_lithology"].items()
        },
        "by_section": {
            key: {"length_m": float(val["length_m"]), "time_h": float(val["time_h"])}
            for key, val in timing["by_section"].items()
        },
        "by_category": {
            key: {"time_h": float(val["time_h"]), "count": int(val["count"])}
            for key, val in operational["by_category"].items()
        },
    }
    (GOLDENS / "minimization_point_l1_1200_r_500.json").write_text(
        json.dumps(point, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", GOLDENS / "minimization_point_l1_1200_r_500.json", flush=True)

    best_force, best_torque = _best_mechanical_candidates(data, geological_mesh)
    mechanical = {
        "force": {
            "l1": float(best_force["l1"]),
            "R": float(best_force["R"]),
            "up_force_1": float(best_force["up_force_1"]),
            "torque": float(best_force["torque"]),
            "angle_deg": float(best_force["angle_deg"]),
        },
        "torque": {
            "l1": float(best_torque["l1"]),
            "R": float(best_torque["R"]),
            "up_force_1": float(best_torque["up_force_1"]),
            "torque": float(best_torque["torque"]),
            "angle_deg": float(best_torque["angle_deg"]),
        },
    }
    (GOLDENS / "minimization_mechanical_optima.json").write_text(
        json.dumps(mechanical, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", GOLDENS / "minimization_mechanical_optima.json", flush=True)


def slim_result(result):
    """Mantém os campos numéricos comparados pelo teste lento dos quatro objetivos."""
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
            key: {"length_m": float(val["length_m"]), "time_h": float(val["time_h"])}
            for key, val in timing["by_lithology"].items()
        },
        "by_section": {
            key: {"length_m": float(val["length_m"]), "time_h": float(val["time_h"])}
            for key, val in timing["by_section"].items()
        },
        "by_category": {
            key: {"time_h": float(val["time_h"]), "count": int(val["count"])}
            for key, val in operational["by_category"].items()
        },
    }


def capture_minimization_optima() -> None:
    """Grava o snapshot da malha dos quatro objetivos. Pode levar vários minutos."""
    data, ops = build_default_data()
    geological_mesh = build_default_mesh()
    payload = calculate_minimization(data, geological_mesh, ops, DEFAULT_MECHANICAL_LIMITS)
    results = {key: slim_result(val) for key, val in payload["results"].items()}
    series = {}
    for scope, family in payload["series"].items():
        series[scope] = {}
        for objective, item in family.items():
            x_vals = [float(v) for v in item["x"]]
            y_vals = [float(v) for v in item["y"]]
            series[scope][objective] = {
                "n": len(x_vals),
                "x_first": x_vals[0],
                "x_last": x_vals[-1],
                "x_sum": float(sum(x_vals)),
                "y_first": y_vals[0],
                "y_last": y_vals[-1],
                "y_sum": float(sum(y_vals)),
                "y_min": float(min(y_vals)),
                "y_max": float(max(y_vals)),
                "best_x": float(item["best_x"]) if "best_x" in item else None,
            }
    (GOLDENS / "minimization_optima.json").write_text(
        json.dumps({"results": results, "series": series}, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", GOLDENS / "minimization_optima.json", flush=True)


def main() -> None:
    """Ponto de entrada da CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Also capture the full four-objective grid (minutes).",
    )
    args = parser.parse_args()
    GOLDENS.mkdir(parents=True, exist_ok=True)
    capture_well_path()
    capture_minimization_point()
    if args.slow:
        capture_minimization_optima()


if __name__ == "__main__":
    main()
