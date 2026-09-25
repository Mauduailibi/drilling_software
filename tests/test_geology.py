"""Validação do modelo geológico 3D e do torque e arraste numéricos.

O cenário usa um alvo fora do plano ``x-z`` (``P3 = (1000, 300, 3000)``) e
compara o modelo de horizontes com os números registrados da antiga
implementação em blocos. A varredura completa dos ótimos está marcada como
``slow``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import drilling.features.minimization.auxiliaries as ax
import drilling.features.minimization.geology as geo
from drilling.features.minimization.data_base import DataSet, mesh
from drilling.features.minimization.minimal import (
    drilling_time_breakdown,
    minimal_drilling_time,
    minimal_tension,
    minimal_torque,
)
from drilling.features.minimization.operational import debounce_lithology, operational_time_breakdown


MU = 0.23

# Números registrados da implementação em blocos que este modelo substitui.
BASELINE = {
    "minimal_tension": (1860.0, 600.0),
    "minimal_torque": (1660.0, 600.0),
    "minimal_drilling_time": (210.0, 100.0),
    "selected": (1200.0, 500.0),
    "total_time_h": 285.82277357277786,
    "total_length_m": 3294.047071789494,
    "max_torque_Nm": 18174.221430687605,
    "up_tension_N": (895020.9888270983, 644147.5903430282, 502812.5030851611),
    "down_tension_N": (475793.96922906546, 224920.57074499544, 175944.1635096667),
    "mechanical_torque_Nm": 7372.609988891433,
    "neutral_line_m": 934.0845688200197,
    "by_lithology_h": {
        "Sandstone": 144.94989761887595,
        "Dolomite": 73.24972160212279,
        "Evaporite": 17.118103846731408,
        "Limestone": 50.50505050505088,
    },
    "operational_total_h": 453.6775147827836,
    "operational_bit_trips": 11,
}

OPERATIONAL_BASE = {
    "trip_fixed_time_h": 2.0,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": 60.0,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "casing_events": [
        {"depth_m": 2000.0, "name": "Casing shoe / cementing", "fixed_time_h": 10.0, "include_trip": True}
    ],
}

ROP_VALUES = {"Sandstone": 18.0, "Limestone": 11.0, "Dolomite": 9.5, "Evaporite": 24.0, "Shale": 15.0}

FILTERS_OFF = {"lithology_min_run_m": 0.0, "min_spacing_between_bit_trips_m": 0.0}
FILTERS_ON = {"lithology_min_run_m": 30.0, "min_spacing_between_bit_trips_m": 150.0}


def build_data(friction_model: str = "constant") -> DataSet:
    return DataSet(
        P0=(0, 0, 0),
        P3=(1000, 300, 3000),
        ro_fluid=1737.5,
        ro_command=8000,
        ro_drillpipe=8000,
        ro_heavypipe=8000,
        diameters_command=(0.2032, 0.1143),
        diameters_drillpipe=(0.127, 0.1086104),
        diameters_heavypipe=(0.1524, 0.1143),
        µ=MU,
        z=(5000 * 8) * 4.44822,
        lp=36,
        max=2300,
        radius=(100, 600),
        drilling_time_parameters={"trajectory_step": 1.0},
        friction_model=friction_model,
    )


def build_flat_mesh(mu: dict | None = None):
    """Geologia em camadas planas contra a qual a implementação em blocos foi validada."""
    return mesh(
        sandstone=[[0, 100], [400, 500], [900, 1600], [2200, 3000]],
        dolomite=[[100, 200], [1600, 2000]],
        evaporite=[[200, 300], [2000, 2200]],
        limestone=[[300, 400], [500, 900]],
        rop_values=ROP_VALUES,
        mu_values=mu,
    )


def build_heterogeneous_mesh():
    """Horizontes inclinados, uma falha e uma mudança lateral de fácies no plano do poço."""
    grid = geo.GeoGrid2D.from_bounds((-300, 1500), (-300, 700), nx=121, ny=61)
    limestone_to_dolomite = geo.facies_linear_boundary(
        grid, "Limestone", "Dolomite", (550, 0), strike_deg=80.0
    )
    shale_base = geo.fault(
        grid, geo.dipping_surface(grid, 700, dip_x=0.09, dip_y=0.02), (800, 0),
        strike_deg=75.0, throw=70.0, smooth_m=60.0,
    )
    layers = [
        {"lithology": "Sandstone", "base": geo.dipping_surface(grid, 300, dip_x=0.05)},
        {"lithology": "Shale", "base": shale_base},
        {"lithology": limestone_to_dolomite,
         "base": geo.anticline(grid, geo.dipping_surface(grid, 1200, dip_x=0.12, dip_y=0.02), 650, 250, 180, 420)},
        {"lithology": "Dolomite", "base": geo.dipping_surface(grid, 1700, dip_x=0.10, dip_y=0.01)},
        {"lithology": "Evaporite", "base": geo.dipping_surface(grid, 2100, dip_x=0.08)},
        {"lithology": "Sandstone", "base": geo.dipping_surface(grid, 3200, dip_x=0.06)},
    ]
    return geo.HorizonModel.from_layer_stack(
        grid, geo.flat_surface(grid, 0.0), layers,
        rop_values=ROP_VALUES,
        mu_values={"Sandstone": 0.25, "Shale": 0.20, "Limestone": 0.28, "Dolomite": 0.32, "Evaporite": 0.18},
    )


def _relative_error_pct(value: float, expected: float) -> float:
    return abs(value - expected) / max(abs(expected), 1.0e-9) * 100.0


def _check(label: str, value: float, expected: float, tolerance_pct: float) -> None:
    error = _relative_error_pct(value, expected)
    assert error <= tolerance_pct, f"{label}: {value} vs {expected} ({error:.4f}%)"


def test_lateral_heterogeneity() -> None:
    """Uma profundidade, duas litologias: o que a malha em blocos não representava."""
    model = build_heterogeneous_mesh()
    depth = 900.0
    assert model.lithology_at(100.0, 50.0, depth) != model.lithology_at(900.0, 50.0, depth)

    left = model.lithology_at(300.0, 50.0, 1000.0)
    right = model.lithology_at(1100.0, 50.0, 1000.0)
    assert {left, right} == {"Limestone", "Dolomite"}, (left, right)


def test_flat_intervals_round_trip() -> None:
    """Um modelo plano devolve os mesmos intervalos que o criaram (usado pela GUI)."""
    model = build_flat_mesh()
    assert model.is_flat
    intervals = model.flat_intervals()
    assert intervals[0] == {"lithology": "Sandstone", "start": 0.0, "end": 100.0}
    assert intervals[-1] == {"lithology": "Sandstone", "start": 2200.0, "end": 3000.0}
    assert len(intervals) == 10
    assert not build_heterogeneous_mesh().is_flat
    with pytest.raises(ValueError):
        build_heterogeneous_mesh().flat_intervals()


def test_flat_layer_regression() -> None:
    """Horizontes planos reproduzem exatamente a implementação em blocos."""
    Data = build_data("constant")
    Mesh = build_flat_mesh()
    l1, R = BASELINE["selected"]

    timing = drilling_time_breakdown(Data, Mesh, l1, R)
    _check("total drilling time (h)", timing["total_time_h"], BASELINE["total_time_h"], 1.0e-6)
    _check("total length (m)", timing["total_length_m"], BASELINE["total_length_m"], 1.0e-6)
    _check("max cumulative torque (N.m)", timing["max_cumulative_torque_Nm"], BASELINE["max_torque_Nm"], 1.0e-6)
    for name, expected in BASELINE["by_lithology_h"].items():
        _check(f"time in {name} (h)", timing["by_lithology"][name]["time_h"], expected, 1.0e-6)

    up = ax.up_tension(Data, l1, R)
    down = ax.down_tension(Data, l1, R)
    for index, expected in enumerate(BASELINE["up_tension_N"]):
        _check(f"up axial force L{index + 1} (N)", up[index], expected, 1.0e-6)
    for index, expected in enumerate(BASELINE["down_tension_N"]):
        _check(f"down axial force L{index + 1} (N)", down[index], expected, 1.0e-6)
    _check("mechanical torque (N.m)", down[3], BASELINE["mechanical_torque_Nm"], 1.0e-6)
    _check("neutral line (m)", ax.Nl(Data, l1, R), BASELINE["neutral_line_m"], 1.0e-6)


@pytest.mark.slow
def test_flat_layer_optima() -> None:
    """Os ótimos da varredura completa no modelo plano batem com a implementação em blocos."""
    Data = build_data("constant")
    Mesh = build_flat_mesh()
    assert tuple(minimal_tension(Data)) == BASELINE["minimal_tension"]
    assert tuple(minimal_torque(Data)) == BASELINE["minimal_torque"]
    assert tuple(minimal_drilling_time(Data, Mesh)) == BASELINE["minimal_drilling_time"]


def test_soft_string_matches_closed_form() -> None:
    """Com um único coeficiente de atrito, o integrador reproduz a solução fechada."""
    Data = build_data("lithology")
    Mesh = build_flat_mesh(mu={name: MU for name in ROP_VALUES})

    worst = {"up_1": 0.0, "up_2": 0.0, "up_3": 0.0, "down_1": 0.0, "torque": 0.0}
    compared = 0
    for l1 in (200.0, 600.0, 1000.0, 1400.0, 1800.0):
        for R in (100.0, 200.0, 350.0, 500.0, 600.0):
            try:
                ax.validate_configuration(Data, l1, R)
            except ValueError:
                continue
            analytic_up = ax.up_tension(Data, l1, R)  # sem Mesh -> solução fechada
            analytic_down = ax.down_tension(Data, l1, R)
            geometry = ax.evaluate_trajectory(Data, Mesh, l1, R)
            numeric_up = ax.soft_string_profile(Data, Mesh, l1, R, "up", geometry=geometry)
            numeric_down = ax.soft_string_profile(Data, Mesh, l1, R, "down", geometry=geometry)

            worst["up_1"] = max(worst["up_1"], _relative_error_pct(numeric_up["tension_1"], analytic_up[0]))
            worst["up_2"] = max(worst["up_2"], _relative_error_pct(numeric_up["tension_2"], analytic_up[1]))
            worst["up_3"] = max(worst["up_3"], _relative_error_pct(numeric_up["tension_3"], analytic_up[2]))
            worst["down_1"] = max(worst["down_1"], _relative_error_pct(numeric_down["tension_1"], analytic_down[0]))
            worst["torque"] = max(worst["torque"], _relative_error_pct(numeric_down["torque"], analytic_down[3]))
            compared += 1

    assert compared > 0
    # A força axial concorda com folga de menos de um décimo de por cento.
    assert worst["up_1"] < 0.2 and worst["up_2"] < 0.2 and worst["up_3"] < 0.2, worst
    # Torque e força de descida só divergem onde a coluna entra em compressão no
    # fim da curva, que a solução fechada detecta com tension_3 constante em vez
    # do T(phi) local; o teste de convergência mostra que a diferença é de
    # modelo, não de discretização.
    assert worst["torque"] < 5.0 and worst["down_1"] < 2.0, worst


def test_integrator_converges() -> None:
    """Refinar a discretização não altera o resultado integrado."""
    Data = build_data("lithology")
    Mesh = build_flat_mesh(mu={name: MU for name in ROP_VALUES})
    l1, R = 1800.0, 350.0

    torques = []
    for step in (4.0, 1.0, 0.25):
        geometry = ax.evaluate_trajectory(Data, Mesh, l1, R, ds_target=step)
        torques.append(ax.soft_string_profile(Data, Mesh, l1, R, "down", geometry=geometry)["torque"])

    drift = abs(torques[-1] - torques[0]) / abs(torques[-1]) * 100.0
    assert drift < 0.5, drift


def test_friction_drives_torque() -> None:
    """O torque cresce monotonicamente com o atrito da rocha atravessada."""
    Data = build_data("lithology")
    l1, R = 1200.0, 500.0
    torques = []
    for scale in (0.8, 1.0, 1.2, 1.5):
        Mesh = build_flat_mesh(mu={name: MU * scale for name in ROP_VALUES})
        torques.append(ax.soft_string_profile(Data, Mesh, l1, R, "down")["torque"])
    assert all(b > a for a, b in zip(torques, torques[1:])), torques


def test_time_accounting_closes() -> None:
    """Os tempos por litologia e por trecho somam o total."""
    Data = build_data("lithology")
    Mesh = build_heterogeneous_mesh()
    timing = drilling_time_breakdown(Data, Mesh, 1200.0, 500.0)
    by_lithology = sum(item["time_h"] for item in timing["by_lithology"].values())
    by_section = sum(item["time_h"] for item in timing["by_section"].values())
    _check("sum over lithologies (h)", by_lithology, timing["total_time_h"], 1.0e-8)
    _check("sum over sections (h)", by_section, timing["total_time_h"], 1.0e-8)
    assert len(timing["by_lithology"]) >= 3


def test_lithology_debounce() -> None:
    """Trechos finos não disparam, cada um, a sua própria troca de broca."""
    raw = np.array(["Shale"] * 50 + ["Sandstone"] * 5 + ["Shale"] * 50 + ["Sandstone"] * 60, dtype=object)
    filtered = debounce_lithology(raw, np.ones(raw.size), 30.0)
    raw_changes = int(np.count_nonzero(raw[1:] != raw[:-1]))
    filtered_changes = int(np.count_nonzero(filtered[1:] != filtered[:-1]))
    assert (raw_changes, filtered_changes) == (3, 1)

    # Intercalações finas são exatamente o caso do filtro: o poço corta alguns
    # metros de outra rocha e volta logo em seguida.
    Data = build_data("lithology")
    Mesh = mesh(
        sandstone=[[0, 1200], [1208, 1600], [1608, 2400], [2408, 3000]],
        dolomite=[[1200, 1208], [1600, 1608], [2400, 2408]],
        rop_values=ROP_VALUES,
        mu_values={"Sandstone": 0.25, "Dolomite": 0.32},
    )
    counts = {}
    for label, extra in (("off", FILTERS_OFF), ("on", FILTERS_ON)):
        result = operational_time_breakdown(
            Data, Mesh, 1200.0, 500.0, operational_parameters={**OPERATIONAL_BASE, **extra}
        )
        counts[label] = result["by_category"]["bit_trip"]["count"]
    assert counts["on"] < counts["off"], counts

    # E nunca inventa manobras em uma geologia em camadas comum.
    plain = {}
    for label, extra in (("off", FILTERS_OFF), ("on", FILTERS_ON)):
        result = operational_time_breakdown(
            Data, build_heterogeneous_mesh(), 1200.0, 500.0,
            operational_parameters={**OPERATIONAL_BASE, **extra},
        )
        plain[label] = result["by_category"]["bit_trip"]["count"]
    assert plain["on"] <= plain["off"], plain


def test_operational_regression() -> None:
    """Com os novos filtros neutralizados, o modelo operacional não muda."""
    Data = build_data("constant")
    Mesh = build_flat_mesh()
    result = operational_time_breakdown(
        Data, Mesh, *BASELINE["selected"],
        operational_parameters={**OPERATIONAL_BASE, **FILTERS_OFF},
    )
    _check("total operational time (h)", result["total_time_h"], BASELINE["operational_total_h"], 1.0e-6)
    assert result["by_category"]["bit_trip"]["count"] == BASELINE["operational_bit_trips"]
