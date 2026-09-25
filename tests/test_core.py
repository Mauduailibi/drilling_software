"""Contrato da Fase 2: parse, defaults únicos e envelopes de dados."""

from __future__ import annotations

import pytest

from drilling.core import (
    ConstraintCheck,
    CorrectionResult,
    Point2D,
    Point3D,
    Vector3,
    parse_pair,
    parse_vector3,
)
from drilling.features.minimization.defaults import (
    DRILLING_TIME_FIELD_SPECS,
    OPERATIONAL_FIELD_SPECS,
    build_default_data,
    build_default_mesh,
    build_default_operational_parameters,
)
from drilling.features.minimization.operational import DEFAULT_OPERATIONAL_PARAMETERS
from drilling.features.well_path import solve_case1
from drilling.features.well_path.defaults import CASE3_FEASIBLE_DIRECTION, DEFAULT_WELL_PATH_INPUT

from tests.numeric import assert_close


def test_parse_vector3_and_pair() -> None:
    """Os campos da GUI passam pelo parse compartilhado de ``drilling.core``."""
    assert_close(parse_vector3("1, 2, 3"), [1.0, 2.0, 3.0], name="xyz")
    assert parse_pair("1000, 3000") == (1000.0, 3000.0)
    with pytest.raises(ValueError, match="x, y, z"):
        parse_vector3("1, 2")
    with pytest.raises(ValueError, match="two comma-separated"):
        parse_pair("1, 2, 3")


def test_point_types_roundtrip() -> None:
    """Point3D / Point2D / Vector3 preservam os números e o texto da GUI."""
    point = Point3D.from_text("50.0, 100.0, -1800.0")
    assert point.as_text() == "50.0, 100.0, -1800.0"
    pair = Point2D.from_text("1000, 3000")
    assert pair.as_tuple() == (1000.0, 3000.0)
    direction = Vector3.from_text("0.2, 0.4, -1.0")
    assert_close(direction.as_array(), [0.2, 0.4, -1.0], name="v")


def test_well_path_defaults_match_golden(well_path_golden: dict) -> None:
    """A aba, o script de captura e o golden leem o mesmo ``WellPathInput``."""
    block = well_path_golden["gui_defaults"]
    inputs = DEFAULT_WELL_PATH_INPUT
    assert_close(inputs.pin.as_array(), block["Pin"], name="Pin")
    assert_close(inputs.pbd.as_array(), block["Pbd"], name="Pbd")
    assert_close(inputs.p1.as_array(), block["p1"], name="p1")
    assert_close(inputs.pt.as_array(), block["pt"], name="pt")
    assert_close(inputs.v.as_array(), block["v"], name="v")
    assert_close(CASE3_FEASIBLE_DIRECTION.as_array(), well_path_golden["case3_feasible"]["v"], name="case3_v")


def test_well_path_input_feeds_solver(well_path_golden: dict) -> None:
    """``WellPathInput.solver_kwargs`` reproduz o Caso 1 dos defaults."""
    result = CorrectionResult.from_solver_dict(solve_case1(**DEFAULT_WELL_PATH_INPUT.solver_kwargs()))
    frozen = well_path_golden["gui_defaults"]["case1"]
    assert_close(result.payload["radius"], frozen["radius"], name="radius")
    assert [check.name for check in result.status] == [row["name"] for row in frozen["status"]]
    assert [check.ok for check in result.status] == [row["ok"] for row in frozen["status"]]


def test_constraint_check_roundtrip() -> None:
    """A tuple histórica e a dataclass descrevem a mesma linha de restrição."""
    raw = ("Deeper Target", True, "-3000", "≤ -1800")
    check = ConstraintCheck.from_item(raw)
    assert check.as_tuple() == raw
    assert ConstraintCheck.from_item(check) is check


def test_minimization_defaults_are_the_gui_contract() -> None:
    """GUI e scripts constroem o mesmo DataSet / malha / operações."""
    data, operational = build_default_data()
    mesh = build_default_mesh()
    merged = build_default_operational_parameters()
    assert data.P0 == (0, 0)
    assert data.P3 == (1000, 3000)
    assert data.min_l1 == 100.0
    assert data.l1_step == 10.0
    assert data.radius_step == 50.0
    assert data.drilling_time_parameters["trajectory_step"] == 1.0
    assert operational["casing_events"][0]["depth_m"] == 2000.0
    assert merged["lithology_wear_factors"]["Sandstone"] == DEFAULT_OPERATIONAL_PARAMETERS["lithology_wear_factors"]["Sandstone"]
    assert mesh.rop_values["Sandstone"] == 18.0


def test_field_specs_cover_known_keys() -> None:
    """Nenhuma chave conhecida da GUI fica de fora do schema tipado."""
    data, operational = build_default_data()
    time_keys = {spec.key for spec in DRILLING_TIME_FIELD_SPECS}
    op_keys = {spec.key for spec in OPERATIONAL_FIELD_SPECS}
    assert time_keys == set(data.drilling_time_parameters)
    leftover = set(operational) - {"casing_events", "lithology_wear_factors"}
    assert op_keys == leftover
