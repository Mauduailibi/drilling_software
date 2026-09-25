"""Entradas padrão da GUI para o módulo de otimização Tipo 1.

Esses valores são o contrato entre a aba Minimization, os scripts de pesquisa
e os testes golden. Alterar um número aqui é decisão de produto e deve falhar
os snapshots da Fase 0.

``DRILLING_TIME_FIELD_SPECS`` e ``OPERATIONAL_FIELD_SPECS`` descrevem as chaves
conhecidas para a GUI tipada — não use tabela chave-valor nesses parâmetros.
"""

from __future__ import annotations

from copy import deepcopy

from drilling.core import FieldSpec
from drilling.features.minimization.data_base import DataSet, mesh
from drilling.features.minimization.operational import (
    DEFAULT_OPERATIONAL_PARAMETERS,
)

LITHOLOGIES = ["Shale", "Siltstone", "Sandstone", "Limestone", "Dolomite", "Evaporite"]
"""Nomes de litologia em ordem, usados pelo editor de malha e pelo modelo geológico."""

DRILLING_TIME_FIELD_SPECS = [
    FieldSpec("trajectory_step", "Trajectory step", "m", step=0.1),
    FieldSpec("min_inclination_factor", "Min inclination factor", step=0.01),
    FieldSpec("inclination_reduction", "Inclination reduction", step=0.01),
    FieldSpec("inclination_exponent", "Inclination exponent", step=0.01),
    FieldSpec("reference_dls_deg_per_30m", "Reference DLS", "deg/30 m", step=0.1),
    FieldSpec("min_dls_factor", "Min DLS factor", step=0.01),
    FieldSpec("dls_reduction", "DLS reduction", step=0.01),
    FieldSpec("dls_exponent", "DLS exponent", step=0.01),
    FieldSpec("surface_wob", "Surface WOB", "N", step=100.0),
    FieldSpec("optimal_wob", "Optimal WOB", "N", step=100.0),
    FieldSpec("min_wob_factor", "Min WOB factor", step=0.01),
    FieldSpec("wob_factor_exponent", "WOB factor exponent", step=0.01),
    FieldSpec("drag_inclination_coeff", "Drag inclination coeff.", step=0.01),
    FieldSpec("drag_dls_coeff", "Drag DLS coeff.", step=0.01),
    FieldSpec("wob_transfer_exponent", "WOB transfer exponent", step=0.01),
    FieldSpec("torque_limit", "Torque limit", "N·m", step=10.0),
    FieldSpec("min_torque_factor", "Min torque factor", step=0.01),
    FieldSpec("torque_reduction", "Torque reduction", step=0.01),
    FieldSpec("torque_exponent", "Torque exponent", step=0.01),
    FieldSpec("bit_radius", "Bit radius", "m", kind="optional_float"),
    FieldSpec("mesh_plot_margin_x", "Mesh plot margin X", "m", step=1.0),
    FieldSpec("mesh_plot_alpha", "Mesh plot alpha", step=0.01, maximum=1.0),
]

OPERATIONAL_FIELD_SPECS = [
    FieldSpec("trip_fixed_time_h", "Trip fixed time", "h", step=0.1),
    FieldSpec("trip_speed_drillpipe_mph", "Trip speed drillpipe", "m/h", step=10.0),
    FieldSpec("trip_speed_heavypipe_mph", "Trip speed heavypipe", "m/h", step=10.0),
    FieldSpec("trip_speed_command_mph", "Trip speed command", "m/h", step=10.0),
    FieldSpec("bit_run_length_limit_m", "Bit-run length limit", "m", kind="optional_float"),
    FieldSpec("bit_run_time_limit_h", "Bit-run time limit", "h", kind="optional_float"),
    FieldSpec("routine_stop_every_m", "Routine stop every", "m", step=10.0),
    FieldSpec("routine_stop_time_h", "Routine stop time", "h", step=0.1),
    FieldSpec("fatigue_dls_threshold_deg_per_30m", "Fatigue DLS threshold", "deg/30 m", step=0.1),
    FieldSpec("fatigue_dls_multiplier", "Fatigue DLS multiplier", step=0.01),
    FieldSpec("fatigue_torque_ratio_threshold", "Fatigue torque ratio", step=0.01, maximum=2.0),
    FieldSpec("fatigue_torque_multiplier", "Fatigue torque multiplier", step=0.01),
    FieldSpec("bit_trip_on_lithology_change", "Bit trip on lithology change", kind="bool"),
    FieldSpec("lithology_min_run_m", "Min lithology run for bit trip", "m", step=1.0),
    FieldSpec("min_spacing_between_bit_trips_m", "Min spacing between bit trips", "m", step=10.0),
    FieldSpec("operation_merge_distance_m", "Operation merge distance", "m", step=1.0),
    FieldSpec("casing_connection_length_m", "Casing connection length", "m", step=0.1),
    FieldSpec("casing_connection_time_h", "Casing connection time", "h", step=0.01, decimals=4),
    FieldSpec("casing_trip_speed_mph", "Casing trip speed", "m/h", step=10.0),
    FieldSpec("casing_logging_time_h", "Casing logging time", "h", step=0.1),
    FieldSpec("cement_pumping_time_h", "Cement pumping time", "h", step=0.1),
    FieldSpec("cement_curing_time_h", "Cement curing time", "h", step=0.1),
]


DEFAULT_CASING_EVENTS = [
    {
        "depth_m": 2000.0,
        "name": "Casing shoe / cementing",
        "fixed_time_h": 10.0,
        "include_trip": True,
    }
]


def build_default_operational_parameters() -> dict:
    """Devolve os parâmetros operacionais padrão da GUI, inclusive wear factors.

    Returns
    -------
    dict
        ``DEFAULT_OPERATIONAL_PARAMETERS`` com os eventos de revestimento da aba.
    """
    params = deepcopy(DEFAULT_OPERATIONAL_PARAMETERS)
    params["casing_events"] = deepcopy(DEFAULT_CASING_EVENTS)
    return params


def build_default_data():
    """Devolve o ``DataSet`` e o dicionário de parâmetros operacionais padrão da GUI.

    Returns
    -------
    tuple[DataSet, dict]
        Dados mecânicos/de tempo e a sobreposição operacional usada pela
        aba Minimization na abertura.
    """
    drilling_time_parameters = {
        "trajectory_step": 1.0,
        "reference_dls_deg_per_30m": 3.0,
        "surface_wob": 1.60e5,
        "optimal_wob": 1.80e5,
        "torque_limit": 1.20e4,
        "mesh_plot_alpha": 0.45,
    }
    operational_parameters = build_default_operational_parameters()
    data = DataSet(
        (0, 0),
        (1000, 3000),
        1737.5,
        8000,
        8000,
        8000,
        (0.2032, 0.1143),
        (0.127, 0.1086104),
        (0.1524, 0.1143),
        36,
        0.23,
        (5000 * 8) * 4.44822,
        2300,
        (100, 600),
        drilling_time_parameters=drilling_time_parameters,
    )
    data.l1_step = 10.0
    data.radius_step = 50.0
    data.min_l1 = 100.0
    return data, operational_parameters


def build_default_mesh():
    """Devolve a malha geológica e a tabela de ROP padrão da GUI.

    Returns
    -------
    HorizonModel
        Modelo plano (intervalos de profundidade sem sobreposição) usado pela
        aba Minimization.
    """
    return mesh(
        sandstone=[[0, 100], [400, 500], [900, 1600], [2200, 3000]],
        dolomite=[[100, 200], [1600, 2000]],
        evaporite=[[200, 300], [2000, 2200]],
        limestone=[[300, 400], [500, 900]],
        rop_values={
            "Sandstone": 18.0,
            "Limestone": 11.0,
            "Dolomite": 9.5,
            "Evaporite": 24.0,
        },
    )
