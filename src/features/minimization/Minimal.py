import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import Auxiliaries as ax


DEFAULT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "figure.figsize": (7.5, 5.0),
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}


_MECH_CACHE: dict[tuple, list[dict]] = {}
_TIME_CACHE: dict[tuple, list[dict]] = {}


def _data_signature(Data) -> tuple:
    if hasattr(Data, "cache_signature"):
        return Data.cache_signature()
    return (id(Data),)


def _mesh_signature(Mesh) -> tuple:
    if hasattr(Mesh, "cache_signature"):
        return Mesh.cache_signature()
    return (id(Mesh),)


def _apply_plot_style() -> None:
    plt.rcParams.update(DEFAULT_STYLE)


def _require_candidates(candidates):
    if not candidates:
        raise ValueError(
            "No valid configuration was found in the searched domain. "
            "Check the geometric limits, radius interval, and pipe lengths."
        )


def _scan_candidates(Data):
    key = _data_signature(Data)
    if key in _MECH_CACHE:
        return _MECH_CACHE[key]

    l1_values = np.arange(100.0, Data.max + Data.l1_step, Data.l1_step)
    r_values = np.arange(Data.min_radius, Data.max_radius + Data.radius_step, Data.radius_step)

    candidates = []
    for l1 in l1_values:
        for R in r_values:
            try:
                config = ax.validate_configuration(Data, l1, R)
                up_t1, up_t2, up_t3 = ax.up_tension(Data, l1, R)
                down_t1, down_t2, down_t3, torque = ax.down_tension(Data, l1, R)
                config.update(
                    {
                        "up_force_1": float(up_t1),
                        "up_force_2": float(up_t2),
                        "up_force_3": float(up_t3),
                        "down_force_1": float(down_t1),
                        "down_force_2": float(down_t2),
                        "down_force_3": float(down_t3),
                        "torque": float(torque),
                        "neutral_line": float(ax.Nl(Data, l1, R)),
                    }
                )
                candidates.append(config)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue

    _MECH_CACHE[key] = candidates
    return candidates


def minimal_tension(Data) -> list:
    candidates = _scan_candidates(Data)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["up_force_1"])
    return [best["l1"], best["R"]]


def minimal_torque(Data) -> list:
    candidates = _scan_candidates(Data)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["torque"])
    return [best["l1"], best["R"]]


def drilling_informations(Data) -> list:
    candidates = _scan_candidates(Data)
    _require_candidates(candidates)

    best_tension = min(candidates, key=lambda item: item["up_force_1"])
    best_torque = min(candidates, key=lambda item: item["torque"])

    tension_results = [
        (best_tension["up_force_1"], best_tension["up_force_2"], best_tension["up_force_3"]),
        (
            best_tension["down_force_1"],
            best_tension["down_force_2"],
            best_tension["down_force_3"],
            best_tension["torque"],
        ),
        best_tension["angle_deg"],
        best_tension["neutral_line"],
        (best_tension["l1"], best_tension["l2"], best_tension["l3"]),
        best_tension["lc"],
        best_tension["l1"],
        best_tension["R"],
    ]

    torque_results = [
        (best_torque["up_force_1"], best_torque["up_force_2"], best_torque["up_force_3"]),
        (
            best_torque["down_force_1"],
            best_torque["down_force_2"],
            best_torque["down_force_3"],
            best_torque["torque"],
        ),
        best_torque["angle_deg"],
        best_torque["neutral_line"],
        (best_torque["l1"], best_torque["l2"], best_torque["l3"]),
        best_torque["lc"],
        best_torque["l1"],
        best_torque["R"],
    ]

    return [tension_results, torque_results]


def drilling_informations_table(data):
    results = drilling_informations(data)
    for i, result in enumerate(results):
        up_forces, down_forces, angle, neutral_line, lengths, length_command, l1, R = result
        l1, l2, l3 = lengths
        f1_up, f2_up, f3_up = up_forces
        f1_down, f2_down, f3_down, torque = down_forces
        values = np.round(
            [
                l1,
                l2,
                l3,
                R,
                f1_up,
                f2_up,
                f3_up,
                f1_down,
                f2_down,
                f3_down,
                torque,
                angle,
                neutral_line,
                length_command,
            ],
            2,
        )
        index_labels = [
            "L1 (m):",
            "L2 (m):",
            "L3 (m):",
            "Radius (m):",
            "Up axial force L1 (N):",
            "Up axial force L2 (N):",
            "Up axial force L3 (N):",
            "Down axial force L1 (N):",
            "Down axial force L2 (N):",
            "Down axial force L3 (N):",
            "Torque (N*m):",
            "Angle (°):",
            "Neutral line (m):",
            "Length command (m):",
        ]

        if i == 0:
            print("\n--- Result table for minimal axial force ---")
        else:
            print("\n--- Result table for minimal torque ---")
        table = pd.DataFrame(values, columns=[""], index=index_labels)
        print(table)
        print("")


def _finish_plot(ax_plot, title: str, xlabel: str, ylabel: str, equal: bool = False) -> None:
    if equal:
        ax_plot.set_aspect("equal")
    ax_plot.set_title(title)
    ax_plot.set_xlabel(xlabel)
    ax_plot.set_ylabel(ylabel)
    ax_plot.grid(alpha=0.35, linewidth=0.8)
    plt.tight_layout()


def drilling_draw(Data) -> None:
    force_l1, force_R = minimal_tension(Data)
    x_force, y_force = ax.points_coordinates(Data, force_l1, force_R)
    torque_l1, torque_R = minimal_torque(Data)
    x_torque, y_torque = ax.points_coordinates(Data, torque_l1, torque_R)

    _apply_plot_style()

    fig, ax_plot = plt.subplots(figsize=(8.5, 6.2))
    ax_plot.plot(x_force, y_force, label="Minimal axial force")
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=35, zorder=4)
    ax_plot.invert_yaxis()
    _finish_plot(
        ax_plot,
        "Type-1 well trajectory for the minimal axial force",
        "Distance ($m$)",
        "Depth ($m$)",
        equal=True,
    )
    ax_plot.legend()
    plt.show()

    fig, ax_plot = plt.subplots(figsize=(8.5, 6.2))
    ax_plot.plot(x_torque, y_torque, linestyle="--", label="Minimal torque")
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=35, zorder=4)
    ax_plot.invert_yaxis()
    _finish_plot(
        ax_plot,
        "Type-1 well trajectory for the minimal torque",
        "Distance ($m$)",
        "Depth ($m$)",
        equal=True,
    )
    ax_plot.legend()
    plt.show()


def tension_in_radius(Data, l1) -> list:
    torque_data = []
    axial_force_data = []
    radius_values = []

    for R in np.arange(Data.min_radius, Data.max_radius + Data.radius_step, Data.radius_step):
        try:
            ax.validate_configuration(Data, l1, R)
            up_f1, *_ = ax.up_tension(Data, l1, R)
            *_, torque = ax.down_tension(Data, l1, R)
            axial_force_data.append(float(up_f1))
            torque_data.append(float(torque))
            radius_values.append(float(R))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    if not radius_values:
        raise ValueError("No valid configurations were found for the requested l1.")

    _apply_plot_style()

    fig, ax_plot = plt.subplots(figsize=(8.0, 5.4))
    ax_plot.plot(radius_values, axial_force_data, marker="o", label=f"Axial force for section {l1} m")
    best_idx = int(np.argmin(axial_force_data))
    ax_plot.scatter([radius_values[best_idx]], [axial_force_data[best_idx]], s=60, zorder=5)
    _finish_plot(ax_plot, "Axial-force behavior across the radius", "Radius ($m$)", "Axial force ($N$)")
    ax_plot.legend()
    plt.show()

    fig, ax_plot = plt.subplots(figsize=(8.0, 5.4))
    ax_plot.plot(radius_values, torque_data, marker="s", linestyle="--", label=f"Torque for section {l1} m")
    best_idx = int(np.argmin(torque_data))
    ax_plot.scatter([radius_values[best_idx]], [torque_data[best_idx]], s=60, zorder=5)
    _finish_plot(ax_plot, "Torque behavior across the radius", "Radius ($m$)", "Torque ($N*m$)")
    ax_plot.legend()
    plt.show()


def tension_in_section1(Data, R) -> list:
    axial_force_data = []
    l1_data = []
    torque_data = []

    for l1 in np.arange(100.0, Data.max + Data.l1_step, Data.l1_step):
        try:
            ax.validate_configuration(Data, l1, R)
            up_f1, *_ = ax.up_tension(Data, l1, R)
            *_, torque = ax.down_tension(Data, l1, R)
            axial_force_data.append(float(up_f1))
            l1_data.append(float(l1))
            torque_data.append(float(torque))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    if not l1_data:
        raise ValueError("No valid configurations were found for the requested radius.")

    _apply_plot_style()

    fig, ax_plot = plt.subplots(figsize=(8.0, 5.4))
    ax_plot.plot(l1_data, axial_force_data, marker="o", label=f"Axial force for radius {R} m")
    best_idx = int(np.argmin(axial_force_data))
    ax_plot.scatter([l1_data[best_idx]], [axial_force_data[best_idx]], s=60, zorder=5)
    _finish_plot(
        ax_plot,
        "Axial-force behavior over the length of section 1",
        "Length ($m$)",
        "Axial force ($N$)",
    )
    ax_plot.legend()
    plt.show()

    fig, ax_plot = plt.subplots(figsize=(8.0, 5.4))
    ax_plot.plot(l1_data, torque_data, marker="s", linestyle="--", label=f"Torque for radius {R} m")
    best_idx = int(np.argmin(torque_data))
    ax_plot.scatter([l1_data[best_idx]], [torque_data[best_idx]], s=60, zorder=5)
    _finish_plot(
        ax_plot,
        "Torque behavior over the length of section 1",
        "Length ($m$)",
        "Torque ($N*m$)",
    )
    ax_plot.legend()
    plt.show()


def tension_graphic(Data) -> list:
    candidates = _scan_candidates(Data)
    _require_candidates(candidates)

    best_force_per_l1 = {}
    best_torque_per_l1 = {}
    for candidate in candidates:
        l1 = candidate["l1"]
        if l1 not in best_force_per_l1 or candidate["up_force_1"] < best_force_per_l1[l1]["up_force_1"]:
            best_force_per_l1[l1] = candidate
        if l1 not in best_torque_per_l1 or candidate["torque"] < best_torque_per_l1[l1]["torque"]:
            best_torque_per_l1[l1] = candidate

    l1_force = sorted(best_force_per_l1.keys())
    force_values = [best_force_per_l1[l1]["up_force_1"] for l1 in l1_force]
    l1_torque = sorted(best_torque_per_l1.keys())
    torque_values = [best_torque_per_l1[l1]["torque"] for l1 in l1_torque]

    _apply_plot_style()

    fig, ax_plot = plt.subplots(figsize=(9.0, 5.4))
    ax_plot.plot(l1_force, force_values, marker="o", label="Axial force")
    _finish_plot(
        ax_plot,
        "Best axial-force response for each section-1 length",
        "Length ($m$)",
        "Axial force ($N$)",
    )
    ax_plot.legend()
    plt.show()

    fig, ax_plot = plt.subplots(figsize=(9.0, 5.4))
    ax_plot.plot(l1_torque, torque_values, marker="s", linestyle="--", label="Torque")
    _finish_plot(
        ax_plot,
        "Best torque response for each section-1 length",
        "Length ($m$)",
        "Torque ($N*m$)",
    )
    ax_plot.legend()
    plt.show()


def drilling_time_breakdown(Data, Mesh, l1: float, R: float, ds_target: float | None = None) -> dict:
    """Compute drilling time with lithology, DLS, WOB, and torque penalties."""
    config = ax.validate_configuration(Data, l1, R)
    elements = ax.trajectory_elements(Data, l1, R, ds_target=ds_target)
    params = Data.drilling_time_parameters

    rows = []
    total_time_h = 0.0
    total_length = 0.0
    cumulative_torque = 0.0

    for index, element in enumerate(elements, start=1):
        depth_mid = float(element["y_mid"])
        segment = Mesh.segment_at_depth(depth_mid)
        rop_base = float(segment["rop"])

        angle_deg = float(element["inclination_deg"])
        dls = float(element["dls_deg_per_30m"])
        curvature = float(element["curvature"])
        ds = float(element["length"])

        f_inc = ax.inclination_factor(angle_deg, params)
        f_dls = ax.dls_factor(dls, params)

        wob_transfer = ax.wob_transfer_factor(angle_deg, dls, params)
        wob_effective = float(params["surface_wob"]) * wob_transfer
        f_wob = ax.wob_factor(wob_effective, params)

        contact_force_per_length = ax.local_contact_force_per_length(Data, angle_deg, curvature, wob_effective)
        torque_increment = float(Data.µ * contact_force_per_length * ds * float(params["bit_radius"]))
        cumulative_torque += torque_increment
        f_torque = ax.torque_factor(cumulative_torque, params)

        rop_effective = rop_base * f_inc * f_dls * f_wob * f_torque
        if rop_effective <= 0:
            raise ValueError("The effective ROP became non-positive.")

        time_h = float(ds / rop_effective)
        total_time_h += time_h
        total_length += ds

        rows.append(
            {
                "id": index,
                "section": element["section"],
                "depth_mid_m": depth_mid,
                "element_length_m": ds,
                "inclination_deg": angle_deg,
                "curvature_1pm": curvature,
                "dls_deg_per_30m": dls,
                "lithology": segment["lithology"],
                "rop_base_mph": rop_base,
                "wob_transfer": wob_transfer,
                "wob_effective_N": wob_effective,
                "contact_force_per_length_Npm": contact_force_per_length,
                "torque_increment_Nm": torque_increment,
                "cumulative_torque_Nm": cumulative_torque,
                "f_inclination": f_inc,
                "f_dls": f_dls,
                "f_wob": f_wob,
                "f_torque": f_torque,
                "rop_effective_mph": rop_effective,
                "time_h": time_h,
            }
        )

    by_lithology = {}
    by_section = {}
    for row in rows:
        lith = row["lithology"]
        sec = row["section"]
        by_lithology.setdefault(lith, {"length_m": 0.0, "time_h": 0.0})
        by_section.setdefault(sec, {"length_m": 0.0, "time_h": 0.0})
        by_lithology[lith]["length_m"] += row["element_length_m"]
        by_lithology[lith]["time_h"] += row["time_h"]
        by_section[sec]["length_m"] += row["element_length_m"]
        by_section[sec]["time_h"] += row["time_h"]

    return {
        "l1": float(config["l1"]),
        "l2": float(config["l2"]),
        "l3": float(config["l3"]),
        "R": float(config["R"]),
        "angle_deg": float(config["angle_deg"]),
        "lc": float(config["lc"]),
        "ld": float(config["ld"]),
        "elements": rows,
        "by_lithology": by_lithology,
        "by_section": by_section,
        "total_length_m": float(total_length),
        "total_time_h": float(total_time_h),
        "average_rop_mph": float(total_length / total_time_h if total_time_h > 0 else np.nan),
        "max_cumulative_torque_Nm": float(max([row["cumulative_torque_Nm"] for row in rows], default=0.0)),
        "average_wob_N": float(np.mean([row["wob_effective_N"] for row in rows]) if rows else np.nan),
        "average_dls_deg_per_30m": float(np.mean([row["dls_deg_per_30m"] for row in rows]) if rows else np.nan),
    }


def _scan_candidates_with_time(Data, Mesh):
    key = (_data_signature(Data), _mesh_signature(Mesh))
    if key in _TIME_CACHE:
        return _TIME_CACHE[key]

    base_candidates = _scan_candidates(Data)
    candidates = []
    for candidate in base_candidates:
        try:
            timing = drilling_time_breakdown(Data, Mesh, candidate["l1"], candidate["R"])
            merged = dict(candidate)
            merged.update(
                {
                    "drilling_time_h": float(timing["total_time_h"]),
                    "average_rop_mph": float(timing["average_rop_mph"]),
                    "timing": timing,
                }
            )
            candidates.append(merged)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    _TIME_CACHE[key] = candidates
    return candidates


def minimal_drilling_time(Data, Mesh) -> list:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["drilling_time_h"])
    return [best["l1"], best["R"]]


def drilling_time_informations(Data, Mesh) -> dict:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)
    return min(candidates, key=lambda item: item["drilling_time_h"])


def drilling_time_information_table(Data, Mesh) -> None:
    best = drilling_time_informations(Data, Mesh)
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
            "Up axial force at top (N)",
            "Mechanical torque (N*m)",
            "Total trajectory length (m)",
            "Total drilling time (h)",
            "Average effective ROP (m/h)",
            "Average effective WOB (N)",
            "Average DLS (deg/30m)",
            "Max cumulative drilling torque (N*m)",
        ],
    )
    print("\n--- Result table for minimal drilling time ---")
    print(summary)
    print("")

    lith_df = pd.DataFrame(timing["by_lithology"]).T
    lith_df = lith_df[["length_m", "time_h"]].sort_index()
    lith_df["average_rop_mph"] = lith_df["length_m"] / lith_df["time_h"]
    lith_df = np.round(lith_df, 3)
    print("--- Time breakdown by lithology ---")
    print(lith_df)
    print("")

    section_df = pd.DataFrame(timing["by_section"]).T
    section_df = section_df[["length_m", "time_h"]].sort_index()
    section_df["average_rop_mph"] = section_df["length_m"] / section_df["time_h"]
    section_df = np.round(section_df, 3)
    print("--- Time breakdown by trajectory section ---")
    print(section_df)
    print("")


def drilling_time_for_best_mechanical_trajectories_table(Data, Mesh) -> None:
    force_l1, force_R = minimal_tension(Data)
    torque_l1, torque_R = minimal_torque(Data)

    force_cfg = ax.validate_configuration(Data, force_l1, force_R)
    torque_cfg = ax.validate_configuration(Data, torque_l1, torque_R)
    force_timing = drilling_time_breakdown(Data, Mesh, force_l1, force_R)
    torque_timing = drilling_time_breakdown(Data, Mesh, torque_l1, torque_R)

    table = pd.DataFrame(
        [
            {
                "Objective": "Minimal axial force",
                "L1 (m)": round(force_l1, 3),
                "R (m)": round(force_R, 3),
                "Angle (deg)": round(force_cfg["angle_deg"], 3),
                "Total drilling time (h)": round(force_timing["total_time_h"], 3),
                "Average effective ROP (m/h)": round(force_timing["average_rop_mph"], 3),
            },
            {
                "Objective": "Minimal torque",
                "L1 (m)": round(torque_l1, 3),
                "R (m)": round(torque_R, 3),
                "Angle (deg)": round(torque_cfg["angle_deg"], 3),
                "Total drilling time (h)": round(torque_timing["total_time_h"], 3),
                "Average effective ROP (m/h)": round(torque_timing["average_rop_mph"], 3),
            },
        ]
    )
    print("\n--- Drilling time for the best mechanical trajectories ---")
    print(table.to_string(index=False))
    print("")


def drilling_time_curve(Data, Mesh) -> None:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)

    best_time_per_l1 = {}
    for candidate in candidates:
        l1 = candidate["l1"]
        if l1 not in best_time_per_l1 or candidate["drilling_time_h"] < best_time_per_l1[l1]["drilling_time_h"]:
            best_time_per_l1[l1] = candidate

    xs = sorted(best_time_per_l1.keys())
    ys = [best_time_per_l1[x]["drilling_time_h"] for x in xs]

    _apply_plot_style()
    fig, ax_plot = plt.subplots(figsize=(9.0, 5.4))
    ax_plot.plot(xs, ys, marker="o")
    _finish_plot(ax_plot, "Best drilling-time response for each section-1 length", "Length L1 (m)", "Drilling time (h)")
    plt.show()


def drilling_time_vs_radius(Data, Mesh, l1: float) -> None:
    radius_values = []
    time_values = []

    for R in np.arange(Data.min_radius, Data.max_radius + Data.radius_step, Data.radius_step):
        try:
            timing = drilling_time_breakdown(Data, Mesh, l1, R)
            radius_values.append(float(R))
            time_values.append(float(timing["total_time_h"]))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

    if not radius_values:
        raise ValueError("No valid configurations were found for the requested l1.")

    _apply_plot_style()
    fig, ax_plot = plt.subplots(figsize=(9.0, 5.4))
    ax_plot.plot(radius_values, time_values, marker="o")
    _finish_plot(ax_plot, "Drilling-time behavior across the radius", "Radius (m)", "Drilling time (h)")
    plt.show()


def _mesh_plot_limits(Data, Mesh, x_values, timing) -> tuple[float, float, float]:
    margin_x = float(Data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    x_min = min(0.0, min(x_values) - 0.05 * max(Data.P3[0], 1.0))
    x_max = max(max(x_values), Data.P3[0]) + margin_x
    y_max = max(
        [segment["end"] for segment in Mesh.segments]
        + [Data.P3[1], max([row["depth_mid_m"] for row in timing["elements"]], default=0.0)]
    )
    return float(x_min), float(x_max), float(y_max)


def _prepare_mesh_axes(ax_plot, Data, Mesh, x_values, timing):
    x_min, x_max, y_max = _mesh_plot_limits(Data, Mesh, x_values, timing)
    alpha = float(Data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))

    used_labels = set()
    for segment in Mesh.segments:
        color = ax.LITHOLOGY_COLORS.get(segment["lithology"], "#dddddd")
        label = segment["lithology"] if segment["lithology"] not in used_labels else None
        if label is not None:
            used_labels.add(label)
        rect = patches.Rectangle(
            (x_min, segment["start"]),
            x_max - x_min,
            segment["end"] - segment["start"],
            facecolor=color,
            edgecolor="white",
            alpha=alpha,
            linewidth=0.8,
            label=label,
            zorder=0,
        )
        ax_plot.add_patch(rect)
        ax_plot.text(
            x_max - 0.02 * (x_max - x_min),
            0.5 * (segment["start"] + segment["end"]),
            segment["lithology"],
            ha="right",
            va="center",
            fontsize=9,
            alpha=0.9,
        )

    ax_plot.set_aspect("equal")
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_ylim(0.0, y_max)
    ax_plot.invert_yaxis()
    ax_plot.set_xlabel("Horizontal distance (m)")
    ax_plot.set_ylabel("Depth (m)")
    ax_plot.grid(alpha=0.25, linewidth=0.8)


def plot_trajectory_with_geological_mesh(Data, Mesh, l1: float | None = None, R: float | None = None, optimize: str = "time") -> None:
    if l1 is None or R is None:
        if optimize == "time":
            l1, R = minimal_drilling_time(Data, Mesh)
        elif optimize == "force":
            l1, R = minimal_tension(Data)
        elif optimize == "torque":
            l1, R = minimal_torque(Data)
        else:
            raise ValueError("'optimize' must be 'time', 'force', or 'torque'.")

    timing = drilling_time_breakdown(Data, Mesh, l1, R)
    x_values, y_values = ax.points_coordinates(Data, l1, R)

    _apply_plot_style()
    fig, ax_plot = plt.subplots(figsize=(11.5, 7.5))
    _prepare_mesh_axes(ax_plot, Data, Mesh, x_values, timing)

    ax_plot.plot(x_values, y_values, label="Well trajectory", zorder=3)
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=55, zorder=4)
    ax_plot.annotate("P0", (Data.P0[0], Data.P0[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.annotate("P3", (Data.P3[0], Data.P3[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.set_title("Type-1 trajectory over the geological mesh")
    ax_plot.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()


def save_trajectory_with_geological_mesh(Data, Mesh, filename: str, l1: float | None = None, R: float | None = None, optimize: str = "time") -> str:
    if l1 is None or R is None:
        if optimize == "time":
            l1, R = minimal_drilling_time(Data, Mesh)
        elif optimize == "force":
            l1, R = minimal_tension(Data)
        elif optimize == "torque":
            l1, R = minimal_torque(Data)
        else:
            raise ValueError("'optimize' must be 'time', 'force', or 'torque'.")

    timing = drilling_time_breakdown(Data, Mesh, l1, R)
    x_values, y_values = ax.points_coordinates(Data, l1, R)

    _apply_plot_style()
    fig, ax_plot = plt.subplots(figsize=(11.5, 7.5))
    _prepare_mesh_axes(ax_plot, Data, Mesh, x_values, timing)

    ax_plot.plot(x_values, y_values, label="Well trajectory", zorder=3)
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=55, zorder=4)
    ax_plot.annotate("P0", (Data.P0[0], Data.P0[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.annotate("P3", (Data.P3[0], Data.P3[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.set_title("Type-1 trajectory over the geological mesh")
    ax_plot.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename




def _best_mechanical_candidates(Data) -> tuple[dict, dict]:
    candidates = _scan_candidates(Data)
    _require_candidates(candidates)
    best_force = min(candidates, key=lambda item: item["up_force_1"])
    best_torque = min(candidates, key=lambda item: item["torque"])
    return best_force, best_torque


def _best_time_candidate(Data, Mesh) -> dict:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)
    return min(candidates, key=lambda item: item["drilling_time_h"])


def _series_varying_radius_for_fixed_l1(Data, Mesh, l1_force: float, l1_torque: float, l1_time: float) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_candidates_with_time(Data, Mesh)

    force_rows = sorted(
        [c for c in mech_candidates if np.isclose(c["l1"], l1_force)],
        key=lambda item: item["R"],
    )
    torque_rows = sorted(
        [c for c in mech_candidates if np.isclose(c["l1"], l1_torque)],
        key=lambda item: item["R"],
    )
    time_rows = sorted(
        [c for c in time_candidates if np.isclose(c["l1"], l1_time)],
        key=lambda item: item["R"],
    )

    if not force_rows or not torque_rows or not time_rows:
        raise ValueError("Could not build all radius-varying series.")

    return {
        "force": {
            "fixed_l1": float(l1_force),
            "x": [row["R"] for row in force_rows],
            "y": [row["up_force_1"] for row in force_rows],
            "best_x": float(min(force_rows, key=lambda item: item["up_force_1"])["R"]),
            "ylabel": "Axial force ($N$)",
            "title": f"Axial force varying $R$ for best $L_1$ = {l1_force:.1f} m",
        },
        "torque": {
            "fixed_l1": float(l1_torque),
            "x": [row["R"] for row in torque_rows],
            "y": [row["torque"] for row in torque_rows],
            "best_x": float(min(torque_rows, key=lambda item: item["torque"])["R"]),
            "ylabel": "Torque ($N*m$)",
            "title": f"Torque varying $R$ for best $L_1$ = {l1_torque:.1f} m",
        },
        "time": {
            "fixed_l1": float(l1_time),
            "x": [row["R"] for row in time_rows],
            "y": [row["drilling_time_h"] for row in time_rows],
            "best_x": float(min(time_rows, key=lambda item: item["drilling_time_h"])["R"]),
            "ylabel": "Drilling time ($h$)",
            "title": f"Drilling time varying $R$ for best $L_1$ = {l1_time:.1f} m",
        },
    }


def _series_varying_l1_for_fixed_radius(Data, Mesh, r_force: float, r_torque: float, r_time: float) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_candidates_with_time(Data, Mesh)

    force_rows = sorted(
        [c for c in mech_candidates if np.isclose(c["R"], r_force)],
        key=lambda item: item["l1"],
    )
    torque_rows = sorted(
        [c for c in mech_candidates if np.isclose(c["R"], r_torque)],
        key=lambda item: item["l1"],
    )
    time_rows = sorted(
        [c for c in time_candidates if np.isclose(c["R"], r_time)],
        key=lambda item: item["l1"],
    )

    if not force_rows or not torque_rows or not time_rows:
        raise ValueError("Could not build all L1-varying series.")

    return {
        "force": {
            "fixed_r": float(r_force),
            "x": [row["l1"] for row in force_rows],
            "y": [row["up_force_1"] for row in force_rows],
            "best_x": float(min(force_rows, key=lambda item: item["up_force_1"])["l1"]),
            "ylabel": "Axial force ($N$)",
            "title": f"Axial force varying $L_1$ for best $R$ = {r_force:.1f} m",
        },
        "torque": {
            "fixed_r": float(r_torque),
            "x": [row["l1"] for row in torque_rows],
            "y": [row["torque"] for row in torque_rows],
            "best_x": float(min(torque_rows, key=lambda item: item["torque"])["l1"]),
            "ylabel": "Torque ($N*m$)",
            "title": f"Torque varying $L_1$ for best $R$ = {r_torque:.1f} m",
        },
        "time": {
            "fixed_r": float(r_time),
            "x": [row["l1"] for row in time_rows],
            "y": [row["drilling_time_h"] for row in time_rows],
            "best_x": float(min(time_rows, key=lambda item: item["drilling_time_h"])["l1"]),
            "ylabel": "Drilling time ($h$)",
            "title": f"Drilling time varying $L_1$ for best $R$ = {r_time:.1f} m",
        },
    }


def _series_best_metric_for_each_l1(Data, Mesh) -> dict:
    mech_candidates = _scan_candidates(Data)
    time_candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(mech_candidates)
    _require_candidates(time_candidates)

    best_force_per_l1 = {}
    best_torque_per_l1 = {}
    best_time_per_l1 = {}

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

    force_x = sorted(best_force_per_l1.keys())
    torque_x = sorted(best_torque_per_l1.keys())
    time_x = sorted(best_time_per_l1.keys())

    return {
        "force": {
            "x": force_x,
            "y": [best_force_per_l1[l1]["up_force_1"] for l1 in force_x],
            "best_r": [best_force_per_l1[l1]["R"] for l1 in force_x],
            "ylabel": "Axial force ($N$)",
            "title": "Best axial force for each $L_1$ using the best $R$",
        },
        "torque": {
            "x": torque_x,
            "y": [best_torque_per_l1[l1]["torque"] for l1 in torque_x],
            "best_r": [best_torque_per_l1[l1]["R"] for l1 in torque_x],
            "ylabel": "Torque ($N*m$)",
            "title": "Best torque for each $L_1$ using the best $R$",
        },
        "time": {
            "x": time_x,
            "y": [best_time_per_l1[l1]["drilling_time_h"] for l1 in time_x],
            "best_r": [best_time_per_l1[l1]["R"] for l1 in time_x],
            "ylabel": "Drilling time ($h$)",
            "title": "Best drilling time for each $L_1$ using the best $R$",
        },
    }


def _plot_metric_family(series: dict, x_label: str) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 12.2), sharex=False)
    metrics = ["force", "torque", "time"]
    colors = {
        "force": "tab:blue",
        "torque": "tab:red",
        "time": "tab:green",
    }
    labels = {
        "force": "Axial force",
        "torque": "Torque",
        "time": "Drilling time",
    }

    for ax_plot, key in zip(axes, metrics):
        x_vals = np.asarray(series[key]["x"], dtype=float)
        y_vals = np.asarray(series[key]["y"], dtype=float)
        ax_plot.plot(
            x_vals,
            y_vals,
            linestyle="-",
            color=colors[key],
            label=labels[key],
        )
        ax_plot.set_title(series[key]["title"])
        ax_plot.set_xlabel(x_label)
        ax_plot.set_ylabel(series[key]["ylabel"])
        ax_plot.grid(alpha=0.35, linewidth=0.8)
        ax_plot.legend(loc="best")

    plt.tight_layout(pad=1.1, h_pad=1.0)
    plt.show()


def plot_metrics_vs_radius_for_best_l1(Data, Mesh) -> None:
    """Plot axial force, torque, and drilling time varying R for each objective's best L1."""
    best_force, best_torque = _best_mechanical_candidates(Data)
    best_time = _best_time_candidate(Data, Mesh)
    series = _series_varying_radius_for_fixed_l1(
        Data,
        Mesh,
        l1_force=best_force["l1"],
        l1_torque=best_torque["l1"],
        l1_time=best_time["l1"],
    )
    _plot_metric_family(series, "Radius ($m$)")


def plot_metrics_vs_l1_for_best_r(Data, Mesh) -> None:
    """Plot axial force, torque, and drilling time varying L1 for each objective's best R."""
    best_force, best_torque = _best_mechanical_candidates(Data)
    best_time = _best_time_candidate(Data, Mesh)
    series = _series_varying_l1_for_fixed_radius(
        Data,
        Mesh,
        r_force=best_force["R"],
        r_torque=best_torque["R"],
        r_time=best_time["R"],
    )
    _plot_metric_family(series, "Length $L_1$ ($m$)")


def plot_best_metric_per_l1_using_best_r(Data, Mesh) -> None:
    """Plot the best metric value for each L1 using the corresponding best R."""
    series = _series_best_metric_for_each_l1(Data, Mesh)
    _plot_metric_family(series, "Length $L_1$ ($m$)")

def optimization_summary(Data, Mesh=None) -> dict:
    force_l1, force_R = minimal_tension(Data)
    torque_l1, torque_R = minimal_torque(Data)

    force_time = drilling_time_breakdown(Data, Mesh, force_l1, force_R) if Mesh is not None else None
    torque_time = drilling_time_breakdown(Data, Mesh, torque_l1, torque_R) if Mesh is not None else None

    summary = {
        "minimal_axial_force": {
            "l1": force_l1,
            "R": force_R,
            "configuration": ax.validate_configuration(Data, force_l1, force_R),
            "up_forces": ax.up_tension(Data, force_l1, force_R),
            "down_results": ax.down_tension(Data, force_l1, force_R),
            "time_information": force_time,
        },
        "minimal_torque": {
            "l1": torque_l1,
            "R": torque_R,
            "configuration": ax.validate_configuration(Data, torque_l1, torque_R),
            "up_forces": ax.up_tension(Data, torque_l1, torque_R),
            "down_results": ax.down_tension(Data, torque_l1, torque_R),
            "time_information": torque_time,
        },
    }
    if Mesh is not None:
        time_best = drilling_time_informations(Data, Mesh)
        summary["minimal_drilling_time"] = {
            "l1": time_best["l1"],
            "R": time_best["R"],
            "configuration": ax.validate_configuration(Data, time_best["l1"], time_best["R"]),
            "time_information": time_best,
        }
    return summary


def optimization_summary_table(Data, Mesh=None) -> None:
    rows = []

    force_l1, force_R = minimal_tension(Data)
    force_cfg = ax.validate_configuration(Data, force_l1, force_R)
    force_up = ax.up_tension(Data, force_l1, force_R)
    force_down = ax.down_tension(Data, force_l1, force_R)
    force_time = drilling_time_breakdown(Data, Mesh, force_l1, force_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append(
        {
            "Objective": "Minimal axial force",
            "L1 (m)": round(force_l1, 3),
            "R (m)": round(force_R, 3),
            "Angle (deg)": round(force_cfg["angle_deg"], 3),
            "Top axial force (N)": round(force_up[0], 3),
            "Torque (N*m)": round(force_down[3], 3),
            "Total time (h)": round(force_time, 3) if Mesh is not None else np.nan,
        }
    )

    torque_l1, torque_R = minimal_torque(Data)
    torque_cfg = ax.validate_configuration(Data, torque_l1, torque_R)
    torque_up = ax.up_tension(Data, torque_l1, torque_R)
    torque_down = ax.down_tension(Data, torque_l1, torque_R)
    torque_time = drilling_time_breakdown(Data, Mesh, torque_l1, torque_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append(
        {
            "Objective": "Minimal torque",
            "L1 (m)": round(torque_l1, 3),
            "R (m)": round(torque_R, 3),
            "Angle (deg)": round(torque_cfg["angle_deg"], 3),
            "Top axial force (N)": round(torque_up[0], 3),
            "Torque (N*m)": round(torque_down[3], 3),
            "Total time (h)": round(torque_time, 3) if Mesh is not None else np.nan,
        }
    )

    if Mesh is not None:
        time_best = drilling_time_informations(Data, Mesh)
        rows.append(
            {
                "Objective": "Minimal drilling time",
                "L1 (m)": round(time_best["l1"], 3),
                "R (m)": round(time_best["R"], 3),
                "Angle (deg)": round(time_best["angle_deg"], 3),
                "Top axial force (N)": round(time_best["up_force_1"], 3),
                "Torque (N*m)": round(time_best["torque"], 3),
                "Total time (h)": round(time_best["drilling_time_h"], 3),
            }
        )

    table = pd.DataFrame(rows)
    print("\n--- Unified optimization summary ---")
    print(table.to_string(index=False))
    print("")


def plot_optimized_trajectories_with_mesh(Data, Mesh) -> None:
    force_l1, force_R = minimal_tension(Data)
    torque_l1, torque_R = minimal_torque(Data)
    time_l1, time_R = minimal_drilling_time(Data, Mesh)

    time_timing = drilling_time_breakdown(Data, Mesh, time_l1, time_R)
    x_time, y_time = ax.points_coordinates(Data, time_l1, time_R)
    x_force, y_force = ax.points_coordinates(Data, force_l1, force_R)
    x_torque, y_torque = ax.points_coordinates(Data, torque_l1, torque_R)

    _apply_plot_style()
    fig, ax_plot = plt.subplots(figsize=(11.5, 7.5))
    _prepare_mesh_axes(ax_plot, Data, Mesh, x_time, time_timing)

    ax_plot.plot(x_force, y_force, linestyle="-", label="Minimal axial force", zorder=3)
    ax_plot.plot(x_torque, y_torque, linestyle="--", label="Minimal torque", zorder=3)
    ax_plot.plot(x_time, y_time, linestyle="-.", label="Minimal drilling time", zorder=3)
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=55, zorder=4)
    ax_plot.annotate("P0", (Data.P0[0], Data.P0[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.annotate("P3", (Data.P3[0], Data.P3[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.set_title("Optimal Type-1 trajectories over the geological mesh")
    ax_plot.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()