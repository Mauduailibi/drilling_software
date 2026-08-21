from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features.minimization.Auxiliaries as ax
from features.minimization.Minimal import (
    DEFAULT_STYLE,
    _scan_candidates,
    drilling_time_breakdown,
    minimal_tension,
    minimal_torque,
)


DEFAULT_OPERATIONAL_PARAMETERS = {
    "trip_fixed_time_h": 1.0,
    "trip_speed_drillpipe_mph": 500.0,
    "trip_speed_heavypipe_mph": 250.0,
    "trip_speed_command_mph": 150.0,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": None,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "bit_trip_on_lithology_change": True,
    "operation_merge_distance_m": 10.0,
    "casing_connection_length_m": 9.0,
    "casing_connection_time_h": 0.10,
    "casing_trip_speed_mph": 300.0,
    "casing_logging_time_h": 5.0,
    "cement_pumping_time_h": 2.5,
    "cement_curing_time_h": 12.0,
    "casing_events": [],
    "lithology_wear_factors": {
        "Shale": 0.95,
        "Siltstone": 1.00,
        "Sandstone": 1.05,
        "Limestone": 1.15,
        "Dolomite": 1.25,
        "Evaporite": 0.75,
        "Undefined": 1.00,
    },
}

DEFAULT_MECHANICAL_LIMITS = {
    "max_top_axial_force_N": None,
    "max_torque_Nm": None,
}

_OPERATIONAL_CACHE: dict[tuple, list[dict]] = {}


def _data_signature(Data) -> tuple:
    if hasattr(Data, "cache_signature"):
        return Data.cache_signature()
    return (id(Data),)


def _mesh_signature(Mesh) -> tuple:
    if hasattr(Mesh, "cache_signature"):
        return Mesh.cache_signature()
    return (id(Mesh),)


def _make_hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(v) for v in value)
    return value


def _normalize_casing_events(casing_events: list[dict]) -> list[dict]:
    events = []
    for item in casing_events:
        if "depth_m" not in item:
            raise ValueError("Each casing event must define 'depth_m'.")
        depth_m = float(item["depth_m"])
        if depth_m <= 0.0:
            raise ValueError("Each casing-event depth must be positive.")
        fixed_time_h = float(item.get("fixed_time_h", 0.0))
        if fixed_time_h < 0.0:
            raise ValueError("Each casing-event fixed time must be non-negative.")
        events.append(
            {
                "depth_m": depth_m,
                "name": str(item.get("name", f"Casing/Cement at {depth_m:.1f} m")),
                "fixed_time_h": fixed_time_h,
                "include_trip": bool(item.get("include_trip", True)),
            }
        )
    events.sort(key=lambda row: row["depth_m"])
    return events


def get_operational_parameters(Data=None, operational_parameters: dict | None = None) -> dict:
    params = deepcopy(DEFAULT_OPERATIONAL_PARAMETERS)
    if Data is not None and getattr(Data, "operational_parameters", None) is not None:
        params.update(deepcopy(Data.operational_parameters))
    if operational_parameters is not None:
        params.update(deepcopy(operational_parameters))
    wear_factors = deepcopy(DEFAULT_OPERATIONAL_PARAMETERS["lithology_wear_factors"])
    wear_factors.update(params.get("lithology_wear_factors", {}))
    params["lithology_wear_factors"] = wear_factors
    params["casing_events"] = _normalize_casing_events(params.get("casing_events", []))

    positive_keys = [
        "trip_fixed_time_h",
        "trip_speed_drillpipe_mph",
        "trip_speed_heavypipe_mph",
        "trip_speed_command_mph",
        "operation_merge_distance_m",
        "casing_connection_length_m",
        "casing_connection_time_h",
        "casing_trip_speed_mph",
        "casing_logging_time_h",
        "cement_pumping_time_h",
        "cement_curing_time_h",
    ]
    for key in positive_keys:
        if float(params[key]) <= 0.0:
            raise ValueError(f"'{key}' must be positive.")
    return params


def get_mechanical_limits(mechanical_limits: dict | None = None) -> dict:
    limits = deepcopy(DEFAULT_MECHANICAL_LIMITS)
    if mechanical_limits is not None:
        limits.update(deepcopy(mechanical_limits))
    if limits["max_top_axial_force_N"] is not None and limits["max_top_axial_force_N"] <= 0:
        raise ValueError("'max_top_axial_force_N' must be positive when provided.")
    if limits["max_torque_Nm"] is not None and limits["max_torque_Nm"] <= 0:
        raise ValueError("'max_torque_Nm' must be positive when provided.")
    return limits


def evaluate_mechanical_limits(
    top_axial_force_N: float,
    torque_Nm: float,
    mechanical_limits: dict | None = None,
) -> dict:
    limits = get_mechanical_limits(mechanical_limits)
    force_ok = True if limits["max_top_axial_force_N"] is None else float(top_axial_force_N) <= float(limits["max_top_axial_force_N"])
    torque_ok = True if limits["max_torque_Nm"] is None else float(torque_Nm) <= float(limits["max_torque_Nm"])
    return {
        "max_top_axial_force_N": limits["max_top_axial_force_N"],
        "max_torque_Nm": limits["max_torque_Nm"],
        "top_axial_force_N": float(top_axial_force_N),
        "torque_Nm": float(torque_Nm),
        "force_ok": bool(force_ok),
        "torque_ok": bool(torque_ok),
        "is_valid": bool(force_ok and torque_ok),
    }


def selected_trajectory_mechanical_table(
    Data,
    l1: float,
    R: float,
    mechanical_limits: dict | None = None,
) -> None:
    up_t1, *_ = ax.up_tension(Data, l1, R)
    *_, torque = ax.down_tension(Data, l1, R)
    evaluation = evaluate_mechanical_limits(up_t1, torque, mechanical_limits=mechanical_limits)
    limits = get_mechanical_limits(mechanical_limits)
    summary = pd.DataFrame(
        {
            "Value": [
                round(float(up_t1), 3),
                round(float(torque), 3),
                limits["max_top_axial_force_N"],
                limits["max_torque_Nm"],
                evaluation["force_ok"],
                evaluation["torque_ok"],
                evaluation["is_valid"],
            ]
        },
        index=[
            "Calculated top axial force (N)",
            "Calculated torque (N*m)",
            "Maximum allowed top axial force (N)",
            "Maximum allowed torque (N*m)",
            "Axial-force criterion satisfied",
            "Torque criterion satisfied",
            "Trajectory valid",
        ],
    )
    print("\n--- Mechanical-constraint evaluation for the selected trajectory ---")
    print(summary)
    print("")


def pipe_trip_breakdown_from_measured_depth(Data, l1: float, R: float, measured_depth_m: float, params: dict) -> dict:
    depth = max(float(measured_depth_m), 0.0)
    config = ax.validate_configuration(Data, l1, R)

    command_length_m = min(float(config["lc"]), depth)
    remaining_depth_m = max(depth - command_length_m, 0.0)
    heavypipe_length_m = min(float(Data.lp), remaining_depth_m)
    drillpipe_length_m = max(remaining_depth_m - heavypipe_length_m, 0.0)

    one_way_time_h = (
        drillpipe_length_m / float(params["trip_speed_drillpipe_mph"])
        + heavypipe_length_m / float(params["trip_speed_heavypipe_mph"])
        + command_length_m / float(params["trip_speed_command_mph"])
    )
    total_time_h = float(params["trip_fixed_time_h"] + 2.0 * one_way_time_h)

    return {
        "measured_depth_m": float(depth),
        "drillpipe_length_m": float(drillpipe_length_m),
        "heavypipe_length_m": float(heavypipe_length_m),
        "command_length_m": float(command_length_m),
        "one_way_time_h": float(one_way_time_h),
        "added_time_h": float(total_time_h),
    }


def trip_time_from_measured_depth(measured_depth_m: float, params: dict, Data=None, l1: float | None = None, R: float | None = None) -> float:
    if Data is not None and l1 is not None and R is not None:
        return pipe_trip_breakdown_from_measured_depth(Data, l1, R, measured_depth_m, params)["added_time_h"]
    depth = max(float(measured_depth_m), 0.0)
    return float(params["trip_fixed_time_h"] + 2.0 * (depth / float(params["trip_speed_drillpipe_mph"])))


def casing_cementing_breakdown_from_depth(casing_depth_m: float, params: dict) -> dict:
    depth = max(float(casing_depth_m), 0.0)
    connection_length_m = float(params["casing_connection_length_m"])
    number_of_connections = int(np.ceil(depth / connection_length_m)) if depth > 0.0 else 0
    connection_time_h = number_of_connections * float(params["casing_connection_time_h"])
    casing_trip_time_h = depth / float(params["casing_trip_speed_mph"])
    logging_time_h = float(params["casing_logging_time_h"])
    cement_pumping_time_h = float(params["cement_pumping_time_h"])
    cement_curing_time_h = float(params["cement_curing_time_h"])
    total_time_h = connection_time_h + casing_trip_time_h + logging_time_h + cement_pumping_time_h + cement_curing_time_h

    return {
        "casing_depth_m": float(depth),
        "number_of_connections": int(number_of_connections),
        "connection_time_h": float(connection_time_h),
        "casing_trip_time_h": float(casing_trip_time_h),
        "logging_time_h": float(logging_time_h),
        "cement_pumping_time_h": float(cement_pumping_time_h),
        "cement_curing_time_h": float(cement_curing_time_h),
        "added_time_h": float(total_time_h),
    }


def _bit_wear_increment(
    row: dict,
    params: dict,
    torque_reference: float,
    previous_lithology: str | None,
) -> tuple[float, bool]:
    ds = float(row["element_length_m"])
    lithology = row["lithology"]
    wear_factor = float(params["lithology_wear_factors"].get(lithology, 1.0))

    dls = float(row["dls_deg_per_30m"])
    dls_threshold = float(params["fatigue_dls_threshold_deg_per_30m"])
    dls_excess = max(0.0, (dls - dls_threshold) / dls_threshold) if dls_threshold > 0 else 0.0
    dls_multiplier = 1.0 + float(params["fatigue_dls_multiplier"]) * dls_excess

    torque_ratio = 0.0 if torque_reference <= 0.0 else float(row["cumulative_torque_Nm"]) / torque_reference
    torque_threshold = float(params["fatigue_torque_ratio_threshold"])
    if torque_ratio > torque_threshold:
        denominator = max(1.0 - torque_threshold, 1.0e-8)
        torque_excess = (torque_ratio - torque_threshold) / denominator
    else:
        torque_excess = 0.0
    torque_multiplier = 1.0 + float(params["fatigue_torque_multiplier"]) * torque_excess

    lithology_changed = previous_lithology is not None and lithology != previous_lithology
    wear = ds * wear_factor * dls_multiplier * torque_multiplier
    return float(wear), lithology_changed


def operational_time_breakdown(
    Data,
    Mesh,
    l1: float,
    R: float,
    drilling_timing: dict | None = None,
    operational_parameters: dict | None = None,
) -> dict:
    params = get_operational_parameters(Data, operational_parameters)
    base_timing = drilling_time_breakdown(Data, Mesh, l1, R) if drilling_timing is None else drilling_timing
    elements = ax.trajectory_elements(Data, l1, R, ds_target=Data.drilling_time_parameters["trajectory_step"])
    rows = base_timing["elements"]
    if len(elements) != len(rows):
        raise ValueError("The operational module expected the same number of geometric and timing elements.")

    torque_reference = float(Data.drilling_time_parameters.get("torque_limit", 1.0))
    events = []
    total_trip_time_h = 0.0
    total_routine_time_h = 0.0
    total_casing_time_h = 0.0
    total_transition_time_h = 0.0

    measured_depth_m = 0.0
    equivalent_bit_run_m = 0.0
    base_time_since_bit_trip_h = 0.0
    drilled_since_routine_m = 0.0
    drilled_since_last_bit_trip_m = 0.0
    previous_lithology = None
    previous_tvd_m = float(Data.P0[1])
    pending_casing_events = deepcopy(params["casing_events"])
    merge_distance_m = float(params.get("operation_merge_distance_m", 30.0))
    last_bit_reset_measured_depth_m = None

    for element, row in zip(elements, rows):
        ds = float(row["element_length_m"])
        depth_end_m = max(float(element["y0"]), float(element["y1"]))
        measured_depth_m += ds
        drilled_since_last_bit_trip_m += ds
        drilled_since_routine_m += ds
        base_time_since_bit_trip_h += float(row["time_h"])

        wear_increment, _ = _bit_wear_increment(row, params, torque_reference, previous_lithology)
        equivalent_bit_run_m += wear_increment
        lithology_changed = previous_lithology is not None and row["lithology"] != previous_lithology
        bit_was_reset_this_element = False

        while pending_casing_events and previous_tvd_m < pending_casing_events[0]["depth_m"] <= depth_end_m:
            casing_event = pending_casing_events.pop(0)
            casing_breakdown = casing_cementing_breakdown_from_depth(casing_event["depth_m"], params)
            pipe_trip_breakdown = {
                "added_time_h": 0.0,
                "drillpipe_length_m": 0.0,
                "heavypipe_length_m": 0.0,
                "command_length_m": 0.0,
                "one_way_time_h": 0.0,
            }
            if casing_event["include_trip"]:
                pipe_trip_breakdown = pipe_trip_breakdown_from_measured_depth(Data, l1, R, measured_depth_m, params)
            added_time_h = float(casing_event["fixed_time_h"]) + float(pipe_trip_breakdown["added_time_h"]) + float(casing_breakdown["added_time_h"])
            events.append(
                {
                    "category": "casing_cement",
                    "cause": f"{casing_event['name']} with bit change",
                    "depth_tvd_m": float(casing_event["depth_m"]),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": float(added_time_h),
                    "pipe_trip_time_h": float(pipe_trip_breakdown["added_time_h"]),
                    "drillpipe_length_m": float(pipe_trip_breakdown["drillpipe_length_m"]),
                    "heavypipe_length_m": float(pipe_trip_breakdown["heavypipe_length_m"]),
                    "command_length_m": float(pipe_trip_breakdown["command_length_m"]),
                    "casing_connections": int(casing_breakdown["number_of_connections"]),
                    "casing_connection_time_h": float(casing_breakdown["connection_time_h"]),
                    "casing_trip_time_h": float(casing_breakdown["casing_trip_time_h"]),
                    "logging_time_h": float(casing_breakdown["logging_time_h"]),
                    "cement_pumping_time_h": float(casing_breakdown["cement_pumping_time_h"]),
                    "cement_curing_time_h": float(casing_breakdown["cement_curing_time_h"]),
                }
            )
            total_casing_time_h += float(added_time_h)
            equivalent_bit_run_m = 0.0
            base_time_since_bit_trip_h = 0.0
            drilled_since_last_bit_trip_m = 0.0
            bit_was_reset_this_element = True
            last_bit_reset_measured_depth_m = float(measured_depth_m)

        while drilled_since_routine_m >= float(params["routine_stop_every_m"]):
            routine_time_h = float(params["routine_stop_time_h"])
            events.append(
                {
                    "category": "routine_stop",
                    "cause": "Programmed routine stop",
                    "depth_tvd_m": float(depth_end_m),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": routine_time_h,
                }
            )
            total_routine_time_h += routine_time_h
            drilled_since_routine_m -= float(params["routine_stop_every_m"])

        reached_run_limit = equivalent_bit_run_m >= float(params["bit_run_length_limit_m"])
        bit_run_time_limit_h = params.get("bit_run_time_limit_h")
        reached_time_limit = False if bit_run_time_limit_h is None else base_time_since_bit_trip_h >= float(bit_run_time_limit_h)

        recent_bit_reset_nearby = (
            last_bit_reset_measured_depth_m is not None
            and abs(float(measured_depth_m) - float(last_bit_reset_measured_depth_m)) <= merge_distance_m
        )
        upcoming_casing_nearby = (
            bool(pending_casing_events)
            and 0.0 <= (float(pending_casing_events[0]["depth_m"]) - float(depth_end_m)) <= merge_distance_m
        )
        lithology_trip_merged_with_nearby_operation = bool(
            lithology_changed and (recent_bit_reset_nearby or upcoming_casing_nearby)
        )

        bit_trip_cause = None
        if not bit_was_reset_this_element and not lithology_trip_merged_with_nearby_operation:
            if lithology_changed and bool(params["bit_trip_on_lithology_change"]):
                if reached_run_limit and reached_time_limit:
                    bit_trip_cause = f"Bit change due to lithology transition from {previous_lithology} to {row['lithology']} and equivalent run-length/time limits"
                elif reached_run_limit:
                    bit_trip_cause = f"Bit change due to lithology transition from {previous_lithology} to {row['lithology']} and equivalent run-length limit"
                elif reached_time_limit:
                    bit_trip_cause = f"Bit change due to lithology transition from {previous_lithology} to {row['lithology']} and run-time limit"
                else:
                    bit_trip_cause = f"Bit change due to lithology transition from {previous_lithology} to {row['lithology']}"
            elif reached_run_limit and reached_time_limit:
                bit_trip_cause = "Bit change due to equivalent run-length and run-time limits"
            elif reached_run_limit:
                bit_trip_cause = "Bit change due to equivalent run-length limit"
            elif reached_time_limit:
                bit_trip_cause = "Bit change due to run-time limit"
        if bit_trip_cause is not None:
            pipe_trip_breakdown = pipe_trip_breakdown_from_measured_depth(Data, l1, R, measured_depth_m, params)
            added_trip_h = float(pipe_trip_breakdown["added_time_h"])
            events.append(
                {
                    "category": "bit_trip",
                    "cause": bit_trip_cause,
                    "depth_tvd_m": float(depth_end_m),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": float(added_trip_h),
                    "drillpipe_length_m": float(pipe_trip_breakdown["drillpipe_length_m"]),
                    "heavypipe_length_m": float(pipe_trip_breakdown["heavypipe_length_m"]),
                    "command_length_m": float(pipe_trip_breakdown["command_length_m"]),
                    "one_way_trip_time_h": float(pipe_trip_breakdown["one_way_time_h"]),
                    "equivalent_bit_run_m": float(equivalent_bit_run_m),
                }
            )
            total_trip_time_h += float(added_trip_h)
            equivalent_bit_run_m = 0.0
            base_time_since_bit_trip_h = 0.0
            drilled_since_last_bit_trip_m = 0.0
            last_bit_reset_measured_depth_m = float(measured_depth_m)

        previous_lithology = row["lithology"]
        previous_tvd_m = depth_end_m

    total_operational_time_h = float(total_trip_time_h + total_routine_time_h + total_casing_time_h + total_transition_time_h)
    total_time_h = float(base_timing["total_time_h"] + total_operational_time_h)

    by_category = {
        "bit_trip": {"time_h": float(total_trip_time_h), "count": 0},
        "routine_stop": {"time_h": float(total_routine_time_h), "count": 0},
        "casing_cement": {"time_h": float(total_casing_time_h), "count": 0},
        "transition_stop": {"time_h": float(total_transition_time_h), "count": 0},
    }
    for event in events:
        by_category[event["category"]]["count"] += 1

    return {
        "l1": float(l1),
        "R": float(R),
        "parameters": params,
        "drilling_time_h": float(base_timing["total_time_h"]),
        "total_operational_time_h": total_operational_time_h,
        "total_time_h": total_time_h,
        "trip_time_h": float(total_trip_time_h),
        "routine_time_h": float(total_routine_time_h),
        "casing_cement_time_h": float(total_casing_time_h),
        "transition_time_h": float(total_transition_time_h),
        "events": events,
        "by_category": by_category,
        "base_timing": base_timing,
    }


def operational_time_table(Data, Mesh, l1: float, R: float, operational_parameters: dict | None = None) -> None:
    result = operational_time_breakdown(Data, Mesh, l1, R, operational_parameters=operational_parameters)
    summary = pd.DataFrame(
        {
            "Value": np.round(
                [
                    result["drilling_time_h"],
                    result["trip_time_h"],
                    result["routine_time_h"],
                    result["transition_time_h"],
                    result["casing_cement_time_h"],
                    result["total_operational_time_h"],
                    result["total_time_h"],
                    result["by_category"]["bit_trip"]["count"],
                    result["by_category"]["routine_stop"]["count"],
                    result["by_category"]["transition_stop"]["count"],
                    result["by_category"]["casing_cement"]["count"],
                ],
                3,
            )
        },
        index=[
            "Pure drilling time (h)",
            "Bit-trip time (h)",
            "Routine-stop time (h)",
            "Transition-stop time (h)",
            "Casing/cement time (h)",
            "Total operational time (h)",
            "Total well time (h)",
            "Number of bit trips",
            "Number of routine stops",
            "Number of transition stops",
            "Number of casing/cement events",
        ],
    )
    print("\n--- Operational-time summary for the selected trajectory ---")
    print(summary)
    print("")

    if result["events"]:
        event_df = pd.DataFrame(result["events"])
        event_df["depth_tvd_m"] = event_df["depth_tvd_m"].round(3)
        event_df["measured_depth_m"] = event_df["measured_depth_m"].round(3)
        event_df["added_time_h"] = event_df["added_time_h"].round(3)
        print("--- Operational events ---")
        print(event_df.to_string(index=False))
        print("")
    else:
        print("--- Operational events ---")
        print("No operational events were triggered with the current parameters.\n")


def _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits: dict | None = None) -> list[dict]:
    limits = get_mechanical_limits(mechanical_limits)
    key = (_data_signature(Data), _mesh_signature(Mesh), "constrained_pure_time", _make_hashable(limits))
    if key in _OPERATIONAL_CACHE:
        return _OPERATIONAL_CACHE[key]

    candidates = []
    for candidate in _scan_candidates(Data):
        evaluation = evaluate_mechanical_limits(candidate["up_force_1"], candidate["torque"], limits)
        if not evaluation["is_valid"]:
            continue
        try:
            timing = drilling_time_breakdown(Data, Mesh, candidate["l1"], candidate["R"])
            merged = dict(candidate)
            merged.update(
                {
                    "drilling_time_h": float(timing["total_time_h"]),
                    "average_rop_mph": float(timing["average_rop_mph"]),
                    "timing": timing,
                    "mechanical_limits": limits,
                }
            )
            candidates.append(merged)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    _OPERATIONAL_CACHE[key] = candidates
    return candidates


def minimal_constrained_drilling_time(Data, Mesh, mechanical_limits: dict | None = None) -> list:
    candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    if not candidates:
        raise ValueError("No valid configuration was found for the constrained drilling-time optimization.")
    best = min(candidates, key=lambda item: item["drilling_time_h"])
    return [best["l1"], best["R"]]


def constrained_drilling_time_informations(Data, Mesh, mechanical_limits: dict | None = None) -> dict:
    candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    if not candidates:
        raise ValueError("No valid configuration was found for the constrained drilling-time optimization.")
    return min(candidates, key=lambda item: item["drilling_time_h"])


def constrained_drilling_time_information_table(Data, Mesh, mechanical_limits: dict | None = None) -> None:
    best = constrained_drilling_time_informations(Data, Mesh, mechanical_limits=mechanical_limits)
    timing = best["timing"]
    summary = pd.DataFrame(
        {
            "Value": np.round(
                [
                    best["l1"],
                    best["l2"],
                    best["l3"],
                    best["R"],
                    best["angle_deg"],
                    best["up_force_1"],
                    best["torque"],
                    timing["total_length_m"],
                    timing["total_time_h"],
                    timing["average_rop_mph"],
                    timing["average_wob_N"],
                    timing["average_dls_deg_per_30m"],
                    timing["max_cumulative_torque_Nm"],
                ],
                3,
            )
        },
        index=[
            "L1 (m)",
            "L2 (m)",
            "L3 (m)",
            "Radius (m)",
            "Angle (deg)",
            "Top axial force (N)",
            "Torque (N*m)",
            "Total trajectory length (m)",
            "Constrained drilling time (h)",
            "Average effective ROP (m/h)",
            "Average effective WOB (N)",
            "Average DLS (deg/30m)",
            "Max cumulative drilling torque (N*m)",
        ],
    )
    print("\n--- Result table for minimal constrained drilling time ---")
    print(summary)
    print("")


def _scan_total_time_candidates(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> list[dict]:
    params = get_operational_parameters(Data, operational_parameters)
    limits = get_mechanical_limits(mechanical_limits)
    key = (
        _data_signature(Data),
        _mesh_signature(Mesh),
        "total_time",
        _make_hashable(params),
        _make_hashable(limits),
    )
    if key in _OPERATIONAL_CACHE:
        return _OPERATIONAL_CACHE[key]

    candidates = []
    for candidate in _scan_candidates(Data):
        evaluation = evaluate_mechanical_limits(candidate["up_force_1"], candidate["torque"], limits)
        if not evaluation["is_valid"]:
            continue
        try:
            base_timing = drilling_time_breakdown(Data, Mesh, candidate["l1"], candidate["R"])
            operational = operational_time_breakdown(
                Data,
                Mesh,
                candidate["l1"],
                candidate["R"],
                drilling_timing=base_timing,
                operational_parameters=params,
            )
            merged = dict(candidate)
            merged.update(
                {
                    "drilling_time_h": float(base_timing["total_time_h"]),
                    "operational_time_h": float(operational["total_operational_time_h"]),
                    "total_time_h": float(operational["total_time_h"]),
                    "operational": operational,
                    "mechanical_limits": limits,
                }
            )
            candidates.append(merged)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    _OPERATIONAL_CACHE[key] = candidates
    return candidates


def minimal_total_time(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> list:
    candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    if not candidates:
        raise ValueError("No valid configuration was found for the total-time optimization.")
    best = min(candidates, key=lambda item: item["total_time_h"])
    return [best["l1"], best["R"]]


def total_time_informations(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> dict:
    candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    if not candidates:
        raise ValueError("No valid configuration was found for the total-time optimization.")
    return min(candidates, key=lambda item: item["total_time_h"])


def total_time_information_table(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    best = total_time_informations(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    operational = best["operational"]
    summary = pd.DataFrame(
        {
            "Value": np.round(
                [
                    best["l1"],
                    best["l2"],
                    best["l3"],
                    best["R"],
                    best["angle_deg"],
                    best["up_force_1"],
                    best["torque"],
                    best["drilling_time_h"],
                    best["operational_time_h"],
                    best["total_time_h"],
                    operational["by_category"]["bit_trip"]["count"],
                    operational["by_category"]["routine_stop"]["count"],
                    operational["by_category"]["casing_cement"]["count"],
                ],
                3,
            )
        },
        index=[
            "L1 (m)",
            "L2 (m)",
            "L3 (m)",
            "Radius (m)",
            "Angle (deg)",
            "Top axial force (N)",
            "Torque (N*m)",
            "Constrained drilling time (h)",
            "Operational time (h)",
            "Constrained total time (h)",
            "Number of bit trips",
            "Number of routine stops",
            "Number of casing/cement events",
        ],
    )
    print("\n--- Result table for minimal constrained total time (drilling + operations) ---")
    print(summary)
    print("")


def total_time_for_best_existing_trajectories_table(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    force_l1, force_R = minimal_tension(Data)
    torque_l1, torque_R = minimal_torque(Data)
    drill_l1, drill_R = minimal_constrained_drilling_time(Data, Mesh, mechanical_limits=mechanical_limits)
    total_l1, total_R = minimal_total_time(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    rows = []
    for label, l1, R in [
        ("Minimal axial force", force_l1, force_R),
        ("Minimal torque", torque_l1, torque_R),
        ("Minimal constrained drilling time", drill_l1, drill_R),
        ("Minimal constrained total time", total_l1, total_R),
    ]:
        operational = operational_time_breakdown(Data, Mesh, l1, R, operational_parameters=operational_parameters)
        up_t1, *_ = ax.up_tension(Data, l1, R)
        *_, torque = ax.down_tension(Data, l1, R)
        eval_limits = evaluate_mechanical_limits(up_t1, torque, mechanical_limits)
        rows.append(
            {
                "Objective": label,
                "L1 (m)": round(l1, 3),
                "R (m)": round(R, 3),
                "Top axial force (N)": round(float(up_t1), 3),
                "Torque (N*m)": round(float(torque), 3),
                "Mechanically valid": eval_limits["is_valid"],
                "Pure drilling time (h)": round(operational["drilling_time_h"], 3),
                "Operational time (h)": round(operational["total_operational_time_h"], 3),
                "Total time (h)": round(operational["total_time_h"], 3),
            }
        )
    table = pd.DataFrame(rows)
    print("\n--- Total time for the four optimization conditions ---")
    print(table.to_string(index=False))
    print("")


# ==========================
# Plot helpers for 4 conditions
# ==========================

def _best_mechanical_candidates(Data) -> tuple[dict, dict]:
    mech_candidates = _scan_candidates(Data)
    if not mech_candidates:
        raise ValueError("No valid mechanical candidates were found.")
    best_force = min(mech_candidates, key=lambda item: item["up_force_1"])
    best_torque = min(mech_candidates, key=lambda item: item["torque"])
    return best_force, best_torque


def _best_constrained_time_candidate(Data, Mesh, mechanical_limits: dict | None = None) -> dict:
    candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    if not candidates:
        raise ValueError("No valid constrained drilling-time candidates were found.")
    return min(candidates, key=lambda item: item["drilling_time_h"])


def _best_total_time_candidate(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> dict:
    candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    if not candidates:
        raise ValueError("No valid constrained total-time candidates were found.")
    return min(candidates, key=lambda item: item["total_time_h"])


def _series_varying_radius_for_fixed_l1(
    Data,
    Mesh,
    l1_force: float,
    l1_torque: float,
    l1_time: float,
    l1_total: float,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    total_candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    force_rows = sorted([c for c in mech_candidates if np.isclose(c["l1"], l1_force)], key=lambda item: item["R"])
    torque_rows = sorted([c for c in mech_candidates if np.isclose(c["l1"], l1_torque)], key=lambda item: item["R"])
    time_rows = sorted([c for c in time_candidates if np.isclose(c["l1"], l1_time)], key=lambda item: item["R"])
    total_rows = sorted([c for c in total_candidates if np.isclose(c["l1"], l1_total)], key=lambda item: item["R"])

    if not force_rows or not torque_rows or not time_rows or not total_rows:
        raise ValueError("Could not build all radius-varying series for the four conditions.")

    return {
        "force": {
            "x": [row["R"] for row in force_rows],
            "y": [row["up_force_1"] for row in force_rows],
            "best_x": float(min(force_rows, key=lambda item: item["up_force_1"])["R"]),
            "ylabel": "Axial force ($N$)",
            "title": f"Axial force varying $R$ for best $L_1$ = {l1_force:.1f} m",
        },
        "torque": {
            "x": [row["R"] for row in torque_rows],
            "y": [row["torque"] for row in torque_rows],
            "best_x": float(min(torque_rows, key=lambda item: item["torque"])["R"]),
            "ylabel": "Torque ($N*m$)",
            "title": f"Torque varying $R$ for best $L_1$ = {l1_torque:.1f} m",
        },
        "time": {
            "x": [row["R"] for row in time_rows],
            "y": [row["drilling_time_h"] for row in time_rows],
            "best_x": float(min(time_rows, key=lambda item: item["drilling_time_h"])["R"]),
            "ylabel": "Constrained drilling time ($h$)",
            "title": f"Constrained drilling time varying $R$ for best $L_1$ = {l1_time:.1f} m",
        },
        "total": {
            "x": [row["R"] for row in total_rows],
            "y": [row["total_time_h"] for row in total_rows],
            "best_x": float(min(total_rows, key=lambda item: item["total_time_h"])["R"]),
            "ylabel": "Constrained total time ($h$)",
            "title": f"Constrained total time varying $R$ for best $L_1$ = {l1_total:.1f} m",
        },
    }


def _series_varying_l1_for_fixed_radius(
    Data,
    Mesh,
    r_force: float,
    r_torque: float,
    r_time: float,
    r_total: float,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    total_candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    force_rows = sorted([c for c in mech_candidates if np.isclose(c["R"], r_force)], key=lambda item: item["l1"])
    torque_rows = sorted([c for c in mech_candidates if np.isclose(c["R"], r_torque)], key=lambda item: item["l1"])
    time_rows = sorted([c for c in time_candidates if np.isclose(c["R"], r_time)], key=lambda item: item["l1"])
    total_rows = sorted([c for c in total_candidates if np.isclose(c["R"], r_total)], key=lambda item: item["l1"])

    if not force_rows or not torque_rows or not time_rows or not total_rows:
        raise ValueError("Could not build all L1-varying series for the four conditions.")

    return {
        "force": {
            "x": [row["l1"] for row in force_rows],
            "y": [row["up_force_1"] for row in force_rows],
            "best_x": float(min(force_rows, key=lambda item: item["up_force_1"])["l1"]),
            "ylabel": "Axial force ($N$)",
            "title": f"Axial force varying $L_1$ for best $R$ = {r_force:.1f} m",
        },
        "torque": {
            "x": [row["l1"] for row in torque_rows],
            "y": [row["torque"] for row in torque_rows],
            "best_x": float(min(torque_rows, key=lambda item: item["torque"])["l1"]),
            "ylabel": "Torque ($N*m$)",
            "title": f"Torque varying $L_1$ for best $R$ = {r_torque:.1f} m",
        },
        "time": {
            "x": [row["l1"] for row in time_rows],
            "y": [row["drilling_time_h"] for row in time_rows],
            "best_x": float(min(time_rows, key=lambda item: item["drilling_time_h"])["l1"]),
            "ylabel": "Constrained drilling time ($h$)",
            "title": f"Constrained drilling time varying $L_1$ for best $R$ = {r_time:.1f} m",
        },
        "total": {
            "x": [row["l1"] for row in total_rows],
            "y": [row["total_time_h"] for row in total_rows],
            "best_x": float(min(total_rows, key=lambda item: item["total_time_h"])["l1"]),
            "ylabel": "Constrained total time ($h$)",
            "title": f"Constrained total time varying $L_1$ for best $R$ = {r_total:.1f} m",
        },
    }


def _series_best_metric_for_each_l1(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_constrained_drilling_time_candidates(Data, Mesh, mechanical_limits=mechanical_limits)
    total_candidates = _scan_total_time_candidates(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    if not mech_candidates or not time_candidates or not total_candidates:
        raise ValueError("Could not build the four-condition best-per-L1 series.")

    best_force_per_l1 = {}
    best_torque_per_l1 = {}
    best_time_per_l1 = {}
    best_total_per_l1 = {}

    for candidate in mech_candidates:
        l1 = candidate["l1"]
        if l1 not in best_force_per_l1 or candidate["up_force_1"] < best_force_per_l1[l1]["up_force_1"]:
            best_force_per_l1[l1] = candidate
        if l1 not in best_torque_per_l1 or candidate["torque"] < best_torque_per_l1[l1]["torque"]:
            best_torque_per_l1[l1] = candidate

    for candidate in time_candidates:
        l1 = candidate["l1"]
        if l1 not in best_time_per_l1 or candidate["drilling_time_h"] < best_time_per_l1[l1]["drilling_time_h"]:
            best_time_per_l1[l1] = candidate

    for candidate in total_candidates:
        l1 = candidate["l1"]
        if l1 not in best_total_per_l1 or candidate["total_time_h"] < best_total_per_l1[l1]["total_time_h"]:
            best_total_per_l1[l1] = candidate

    force_x = sorted(best_force_per_l1.keys())
    torque_x = sorted(best_torque_per_l1.keys())
    time_x = sorted(best_time_per_l1.keys())
    total_x = sorted(best_total_per_l1.keys())

    return {
        "force": {
            "x": force_x,
            "y": [best_force_per_l1[l1]["up_force_1"] for l1 in force_x],
            "ylabel": "Axial force ($N$)",
            "title": "Best axial force for each $L_1$ using the best $R$",
        },
        "torque": {
            "x": torque_x,
            "y": [best_torque_per_l1[l1]["torque"] for l1 in torque_x],
            "ylabel": "Torque ($N*m$)",
            "title": "Best torque for each $L_1$ using the best $R$",
        },
        "time": {
            "x": time_x,
            "y": [best_time_per_l1[l1]["drilling_time_h"] for l1 in time_x],
            "ylabel": "Constrained drilling time ($h$)",
            "title": "Best constrained drilling time for each $L_1$ using the best $R$",
        },
        "total": {
            "x": total_x,
            "y": [best_total_per_l1[l1]["total_time_h"] for l1 in total_x],
            "ylabel": "Constrained total time ($h$)",
            "title": "Best constrained total time for each $L_1$ using the best $R$",
        },
    }


def _plot_metric_family(series: dict, x_label: str) -> None:
    plt.rcParams.update(DEFAULT_STYLE)
    order = ["force", "torque", "time", "total"]
    colors = {
        "force": "tab:blue",
        "torque": "tab:orange",
        "time": "tab:green",
        "total": "tab:red",
    }
    legend_labels = {
        "force": "Minimum axial force",
        "torque": "Minimum torque",
        "time": "Minimum constrained drilling time",
        "total": "Minimum constrained total time",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    axes = axes.ravel()

    for ax_plot, key in zip(axes, order):
        x_vals = np.asarray(series[key]["x"], dtype=float)
        y_vals = np.asarray(series[key]["y"], dtype=float)

        ax_plot.plot(
            x_vals,
            y_vals,
            linestyle="-",
            linewidth=2.2,
            color=colors[key],
            label=legend_labels[key],
        )
        ax_plot.set_title(series[key]["title"], pad=10)
        ax_plot.set_xlabel(x_label)
        ax_plot.set_ylabel(series[key]["ylabel"])
        ax_plot.grid(alpha=0.35, linewidth=0.8)
        ax_plot.legend(loc="best", frameon=True)
        ax_plot.margins(x=0.02, y=0.08)

    plt.show()


def plot_metrics_vs_radius_for_best_l1_4_conditions(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    best_force, best_torque = _best_mechanical_candidates(Data)
    best_time = _best_constrained_time_candidate(Data, Mesh, mechanical_limits=mechanical_limits)
    best_total = _best_total_time_candidate(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    series = _series_varying_radius_for_fixed_l1(
        Data,
        Mesh,
        l1_force=best_force["l1"],
        l1_torque=best_torque["l1"],
        l1_time=best_time["l1"],
        l1_total=best_total["l1"],
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    _plot_metric_family(series, "Radius ($m$)")


def plot_metrics_vs_l1_for_best_r_4_conditions(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    best_force, best_torque = _best_mechanical_candidates(Data)
    best_time = _best_constrained_time_candidate(Data, Mesh, mechanical_limits=mechanical_limits)
    best_total = _best_total_time_candidate(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    series = _series_varying_l1_for_fixed_radius(
        Data,
        Mesh,
        r_force=best_force["R"],
        r_torque=best_torque["R"],
        r_time=best_time["R"],
        r_total=best_total["R"],
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    _plot_metric_family(series, "Length $L_1$ ($m$)")


def plot_best_metric_per_l1_using_best_r_4_conditions(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    series = _series_best_metric_for_each_l1(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
    _plot_metric_family(series, "Length $L_1$ ($m$)")
