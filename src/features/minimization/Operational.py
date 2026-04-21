from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

import Auxiliaries as ax
from Minimal import DEFAULT_STYLE, drilling_time_breakdown


DEFAULT_OPERATIONAL_PARAMETERS = {
    "trip_fixed_time_h": 2.0,
    "trip_time_per_meter_h": 0.0025,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": 60.0,
    "min_spacing_between_bit_trips_m": 150.0,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "abrupt_transition_threshold": 0.18,
    "abrupt_transition_extra_wear": 0.30,
    "transition_short_stop_time_h": 0.75,
    "transition_short_stop_instead_of_trip": False,
    "reset_bit_run_on_casing": True,
    "casing_events": [],
    "lithology_wear_factors": {
        "Sandstone": 1.00,
        "Limestone": 1.12,
        "Dolomite": 1.25,
        "Evaporite": 0.75,
        "Undefined": 1.00,
    },
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

    if Data is not None and hasattr(Data, "operational_parameters"):
        user_data_params = getattr(Data, "operational_parameters")
        if user_data_params is not None:
            params.update(deepcopy(user_data_params))

    if operational_parameters is not None:
        params.update(deepcopy(operational_parameters))

    wear_factors = deepcopy(DEFAULT_OPERATIONAL_PARAMETERS["lithology_wear_factors"])
    wear_factors.update(params.get("lithology_wear_factors", {}))
    params["lithology_wear_factors"] = wear_factors
    params["casing_events"] = _normalize_casing_events(params.get("casing_events", []))

    positive_keys = [
        "trip_fixed_time_h",
        "trip_time_per_meter_h",
        "bit_run_length_limit_m",
        "bit_run_time_limit_h",
        "min_spacing_between_bit_trips_m",
        "routine_stop_every_m",
        "routine_stop_time_h",
        "fatigue_dls_threshold_deg_per_30m",
        "fatigue_dls_multiplier",
        "fatigue_torque_ratio_threshold",
        "fatigue_torque_multiplier",
        "abrupt_transition_threshold",
        "abrupt_transition_extra_wear",
        "transition_short_stop_time_h",
    ]
    for key in positive_keys:
        if params[key] < 0.0:
            raise ValueError(f"'{key}' must be non-negative.")

    if params["bit_run_length_limit_m"] <= 0.0:
        raise ValueError("'bit_run_length_limit_m' must be positive.")
    if params["bit_run_time_limit_h"] <= 0.0:
        raise ValueError("'bit_run_time_limit_h' must be positive.")
    if params["routine_stop_every_m"] <= 0.0:
        raise ValueError("'routine_stop_every_m' must be positive.")
    if params["fatigue_dls_threshold_deg_per_30m"] <= 0.0:
        raise ValueError("'fatigue_dls_threshold_deg_per_30m' must be positive.")
    if not (0.0 <= params["fatigue_torque_ratio_threshold"] < 1.0):
        raise ValueError("'fatigue_torque_ratio_threshold' must satisfy 0 <= value < 1.")

    return params


def trip_time_from_measured_depth(measured_depth_m: float, params: dict) -> float:
    depth = max(float(measured_depth_m), 0.0)
    return float(params["trip_fixed_time_h"] + params["trip_time_per_meter_h"] * (depth + depth))


def _bit_wear_increment(row: dict, params: dict, torque_reference: float, previous_lithology: str | None) -> tuple[float, bool]:
    ds = float(row["element_length_m"])
    lithology = row["lithology"]
    wear_factor = float(params["lithology_wear_factors"].get(lithology, 1.0))

    dls = float(row["dls_deg_per_30m"])
    dls_threshold = float(params["fatigue_dls_threshold_deg_per_30m"])
    dls_excess = max(0.0, (dls - dls_threshold) / dls_threshold)
    dls_multiplier = 1.0 + float(params["fatigue_dls_multiplier"]) * dls_excess

    torque_ratio = 0.0 if torque_reference <= 0.0 else float(row["cumulative_torque_Nm"]) / torque_reference
    torque_threshold = float(params["fatigue_torque_ratio_threshold"])
    if torque_ratio > torque_threshold:
        denominator = max(1.0 - torque_threshold, 1.0e-8)
        torque_excess = (torque_ratio - torque_threshold) / denominator
    else:
        torque_excess = 0.0
    torque_multiplier = 1.0 + float(params["fatigue_torque_multiplier"]) * torque_excess

    abrupt_transition = False
    if previous_lithology is not None and lithology != previous_lithology:
        previous_factor = float(params["lithology_wear_factors"].get(previous_lithology, 1.0))
        current_factor = float(params["lithology_wear_factors"].get(lithology, 1.0))
        if (current_factor - previous_factor) >= float(params["abrupt_transition_threshold"]):
            abrupt_transition = True

    wear = ds * wear_factor * dls_multiplier * torque_multiplier
    if abrupt_transition:
        wear *= 1.0 + float(params["abrupt_transition_extra_wear"])

    return float(wear), abrupt_transition


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
    drilled_since_bit_trip_m = 0.0
    equivalent_bit_run_m = 0.0
    base_time_since_bit_trip_h = 0.0
    drilled_since_routine_m = 0.0
    drilled_since_last_bit_trip_m = 0.0
    previous_lithology = None
    previous_tvd_m = float(Data.P0[1])

    pending_casing_events = deepcopy(params["casing_events"])

    for element, row in zip(elements, rows):
        ds = float(row["element_length_m"])
        depth_end_m = max(float(element["y0"]), float(element["y1"]))
        measured_depth_m += ds
        drilled_since_bit_trip_m += ds
        drilled_since_last_bit_trip_m += ds
        drilled_since_routine_m += ds
        base_time_since_bit_trip_h += float(row["time_h"])

        wear_increment, abrupt_transition = _bit_wear_increment(row, params, torque_reference, previous_lithology)
        equivalent_bit_run_m += wear_increment

        while pending_casing_events and previous_tvd_m < pending_casing_events[0]["depth_m"] <= depth_end_m:
            casing_event = pending_casing_events.pop(0)
            added_time_h = float(casing_event["fixed_time_h"])
            if casing_event["include_trip"]:
                added_time_h += trip_time_from_measured_depth(measured_depth_m, params)

            events.append(
                {
                    "category": "casing_cement",
                    "cause": casing_event["name"],
                    "depth_tvd_m": float(casing_event["depth_m"]),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": float(added_time_h),
                }
            )
            total_casing_time_h += float(added_time_h)

            if bool(params["reset_bit_run_on_casing"]):
                drilled_since_bit_trip_m = 0.0
                equivalent_bit_run_m = 0.0
                base_time_since_bit_trip_h = 0.0
                drilled_since_last_bit_trip_m = 0.0

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

        can_trigger_bit_trip = drilled_since_last_bit_trip_m >= float(params["min_spacing_between_bit_trips_m"])
        reached_run_limit = equivalent_bit_run_m >= float(params["bit_run_length_limit_m"])
        reached_time_limit = base_time_since_bit_trip_h >= float(params["bit_run_time_limit_h"])

        if abrupt_transition and bool(params["transition_short_stop_instead_of_trip"]):
            short_stop_h = float(params["transition_short_stop_time_h"])
            events.append(
                {
                    "category": "transition_stop",
                    "cause": f"Abrupt lithology transition to {row['lithology']}",
                    "depth_tvd_m": float(depth_end_m),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": short_stop_h,
                }
            )
            total_transition_time_h += short_stop_h
            abrupt_transition = False

        if can_trigger_bit_trip and (abrupt_transition or reached_run_limit or reached_time_limit):
            if abrupt_transition:
                cause = f"Bit change due to abrupt transition to {row['lithology']}"
            elif reached_run_limit and reached_time_limit:
                cause = "Bit change due to equivalent run-length and run-time limits"
            elif reached_run_limit:
                cause = "Bit change due to equivalent run-length limit"
            else:
                cause = "Bit change due to run-time limit"

            added_trip_h = trip_time_from_measured_depth(measured_depth_m, params)
            events.append(
                {
                    "category": "bit_trip",
                    "cause": cause,
                    "depth_tvd_m": float(depth_end_m),
                    "measured_depth_m": float(measured_depth_m),
                    "added_time_h": float(added_trip_h),
                }
            )
            total_trip_time_h += float(added_trip_h)

            drilled_since_bit_trip_m = 0.0
            equivalent_bit_run_m = 0.0
            base_time_since_bit_trip_h = 0.0
            drilled_since_last_bit_trip_m = 0.0

        previous_lithology = row["lithology"]
        previous_tvd_m = depth_end_m

    total_operational_time_h = float(
        total_trip_time_h + total_routine_time_h + total_casing_time_h + total_transition_time_h
    )
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


def _scan_total_time_candidates(Data, Mesh, operational_parameters: dict | None = None) -> list[dict]:
    params = get_operational_parameters(Data, operational_parameters)
    key = (_data_signature(Data), _mesh_signature(Mesh), _make_hashable(params))
    if key in _OPERATIONAL_CACHE:
        return _OPERATIONAL_CACHE[key]

    l1_values = np.arange(100.0, Data.max + Data.l1_step, Data.l1_step)
    r_values = np.arange(Data.min_radius, Data.max_radius + Data.radius_step, Data.radius_step)

    candidates = []
    for l1 in l1_values:
        for R in r_values:
            try:
                config = ax.validate_configuration(Data, l1, R)
                up_t1, *_ = ax.up_tension(Data, l1, R)
                *_, torque = ax.down_tension(Data, l1, R)
                base_timing = drilling_time_breakdown(Data, Mesh, l1, R)
                operational = operational_time_breakdown(
                    Data,
                    Mesh,
                    l1,
                    R,
                    drilling_timing=base_timing,
                    operational_parameters=params,
                )
                merged = dict(config)
                merged.update(
                    {
                        "up_force_1": float(up_t1),
                        "torque": float(torque),
                        "drilling_time_h": float(base_timing["total_time_h"]),
                        "operational_time_h": float(operational["total_operational_time_h"]),
                        "total_time_h": float(operational["total_time_h"]),
                        "operational": operational,
                    }
                )
                candidates.append(merged)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue

    _OPERATIONAL_CACHE[key] = candidates
    return candidates


def minimal_total_time(Data, Mesh, operational_parameters: dict | None = None) -> list:
    candidates = _scan_total_time_candidates(Data, Mesh, operational_parameters=operational_parameters)
    if not candidates:
        raise ValueError("No valid configuration was found for the total-time optimization.")
    best = min(candidates, key=lambda item: item["total_time_h"])
    return [best["l1"], best["R"]]


def total_time_informations(Data, Mesh, operational_parameters: dict | None = None) -> dict:
    candidates = _scan_total_time_candidates(Data, Mesh, operational_parameters=operational_parameters)
    if not candidates:
        raise ValueError("No valid configuration was found for the total-time optimization.")
    return min(candidates, key=lambda item: item["total_time_h"])


def total_time_information_table(Data, Mesh, operational_parameters: dict | None = None) -> None:
    best = total_time_informations(Data, Mesh, operational_parameters=operational_parameters)
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
            "Pure drilling time (h)",
            "Operational time (h)",
            "Total time (h)",
            "Number of bit trips",
            "Number of routine stops",
            "Number of casing/cement events",
        ],
    )
    print("\n--- Result table for minimal total time (drilling + operations) ---")
    print(summary)
    print("")


def total_time_for_best_existing_trajectories_table(Data, Mesh, operational_parameters: dict | None = None) -> None:
    from Minimal import minimal_drilling_time, minimal_tension, minimal_torque

    force_l1, force_R = minimal_tension(Data)
    torque_l1, torque_R = minimal_torque(Data)
    drill_l1, drill_R = minimal_drilling_time(Data, Mesh)
    total_l1, total_R = minimal_total_time(Data, Mesh, operational_parameters=operational_parameters)

    rows = []
    for label, l1, R in [
        ("Minimal axial force", force_l1, force_R),
        ("Minimal torque", torque_l1, torque_R),
        ("Minimal drilling time", drill_l1, drill_R),
        ("Minimal total time", total_l1, total_R),
    ]:
        operational = operational_time_breakdown(Data, Mesh, l1, R, operational_parameters=operational_parameters)
        rows.append(
            {
                "Objective": label,
                "L1 (m)": round(l1, 3),
                "R (m)": round(R, 3),
                "Pure drilling time (h)": round(operational["drilling_time_h"], 3),
                "Operational time (h)": round(operational["total_operational_time_h"], 3),
                "Total time (h)": round(operational["total_time_h"], 3),
            }
        )

    table = pd.DataFrame(rows)
    print("\n--- Total time for the best trajectories ---")
    print(table.to_string(index=False))
    print("")
