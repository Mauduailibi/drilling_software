
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import features.minimization.Auxiliaries as ax


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
        raise ValueError("No valid configuration was found in the searched domain.")


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

    def _pack(best):
        return [
            (best["up_force_1"], best["up_force_2"], best["up_force_3"]),
            (best["down_force_1"], best["down_force_2"], best["down_force_3"], best["torque"]),
            best["angle_deg"],
            best["neutral_line"],
            (best["l1"], best["l2"], best["l3"]),
            best["lc"],
            best["l1"],
            best["R"],
        ]

    return [_pack(best_tension), _pack(best_torque)]


def drilling_informations_table(data):
    results = drilling_informations(data)
    for i, result in enumerate(results):
        up_forces, down_forces, angle, neutral_line, lengths, length_command, l1, R = result
        l1, l2, l3 = lengths
        f1_up, f2_up, f3_up = up_forces
        f1_down, f2_down, f3_down, torque = down_forces
        values = np.round([l1, l2, l3, R, f1_up, f2_up, f3_up, f1_down, f2_down, f3_down, torque, angle, neutral_line, length_command], 2)
        index_labels = [
            "L1 (m):", "L2 (m):", "L3 (m):", "Radius (m):",
            "Up axial force L1 (N):", "Up axial force L2 (N):", "Up axial force L3 (N):",
            "Down axial force L1 (N):", "Down axial force L2 (N):", "Down axial force L3 (N):",
            "Torque (N*m):", "Angle (°):", "Neutral line (m):", "Length command (m):",
        ]
        print("\n--- Result table for minimal axial force ---" if i == 0 else "\n--- Result table for minimal torque ---")
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


def drilling_time_breakdown(Data, Mesh, l1: float, R: float, ds_target: float | None = None) -> dict:
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
        "l1": float(config["l1"]), "l2": float(config["l2"]), "l3": float(config["l3"]),
        "R": float(config["R"]), "angle_deg": float(config["angle_deg"]), "lc": float(config["lc"]),
        "ld": float(config["ld"]), "elements": rows,
        "by_lithology": by_lithology, "by_section": by_section,
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
            merged.update({"drilling_time_h": float(timing["total_time_h"]), "average_rop_mph": float(timing["average_rop_mph"]), "timing": timing})
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
                [best["l1"], best["l2"], best["l3"], best["R"], best["angle_deg"], best["up_force_1"], best["torque"],
                 timing["total_length_m"], timing["total_time_h"], timing["average_rop_mph"], timing["average_wob_N"],
                 timing["average_dls_deg_per_30m"], timing["max_cumulative_torque_Nm"]], 3
            )
        },
        index=[
            "L1 (m)", "L2 (m)", "L3 (m)", "Radius (m)", "Angle (deg)", "Up axial force at top (N)",
            "Mechanical torque (N*m)", "Total trajectory length (m)", "Total drilling time (h)",
            "Average effective ROP (m/h)", "Average effective WOB (N)", "Average DLS (deg/30m)",
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


def optimization_summary_table(Data, Mesh=None) -> None:
    rows = []
    force_l1, force_R = minimal_tension(Data)
    force_cfg = ax.validate_configuration(Data, force_l1, force_R)
    force_up = ax.up_tension(Data, force_l1, force_R)
    force_down = ax.down_tension(Data, force_l1, force_R)
    force_time = drilling_time_breakdown(Data, Mesh, force_l1, force_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append({"Objective": "Minimal axial force", "L1 (m)": round(force_l1, 3), "R (m)": round(force_R, 3), "Angle (deg)": round(force_cfg["angle_deg"], 3), "Top axial force (N)": round(force_up[0], 3), "Torque (N*m)": round(force_down[3], 3), "Total time (h)": round(force_time, 3) if Mesh is not None else np.nan})
    torque_l1, torque_R = minimal_torque(Data)
    torque_cfg = ax.validate_configuration(Data, torque_l1, torque_R)
    torque_up = ax.up_tension(Data, torque_l1, torque_R)
    torque_down = ax.down_tension(Data, torque_l1, torque_R)
    torque_time = drilling_time_breakdown(Data, Mesh, torque_l1, torque_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append({"Objective": "Minimal torque", "L1 (m)": round(torque_l1, 3), "R (m)": round(torque_R, 3), "Angle (deg)": round(torque_cfg["angle_deg"], 3), "Top axial force (N)": round(torque_up[0], 3), "Torque (N*m)": round(torque_down[3], 3), "Total time (h)": round(torque_time, 3) if Mesh is not None else np.nan})
    if Mesh is not None:
        time_best = drilling_time_informations(Data, Mesh)
        rows.append({"Objective": "Minimal drilling time", "L1 (m)": round(time_best["l1"], 3), "R (m)": round(time_best["R"], 3), "Angle (deg)": round(time_best["angle_deg"], 3), "Top axial force (N)": round(time_best["up_force_1"], 3), "Torque (N*m)": round(time_best["torque"], 3), "Total time (h)": round(time_best["drilling_time_h"], 3)})
    table = pd.DataFrame(rows)
    print("\n--- Unified optimization summary ---")
    print(table.to_string(index=False))
    print("")


def _prepare_mesh_axes(ax_plot, Data, Mesh, x_values, y_values):
    margin_x = float(Data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    x_min = min(0.0, min(x_values) - 0.05 * max(Data.P3[0], 1.0))
    x_max = max(max(x_values), Data.P3[0]) + margin_x
    y_max = max([segment["end"] for segment in Mesh.segments] + [Data.P3[1], max(y_values)])
    alpha = float(Data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))
    used_labels = set()
    for segment in Mesh.segments:
        color = ax.LITHOLOGY_COLORS.get(segment["lithology"], "#dddddd")
        label = segment["lithology"] if segment["lithology"] not in used_labels else None
        if label is not None:
            used_labels.add(label)
        rect = patches.Rectangle((x_min, segment["start"]), x_max - x_min, segment["end"] - segment["start"], facecolor=color, edgecolor="white", alpha=alpha, linewidth=0.8, label=label, zorder=0)
        ax_plot.add_patch(rect)
    ax_plot.set_aspect("equal")
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_ylim(0.0, y_max)
    ax_plot.invert_yaxis()
    ax_plot.set_xlabel("Horizontal distance (m)")
    ax_plot.set_ylabel("Depth (m)")
    ax_plot.grid(alpha=0.25, linewidth=0.8)


def plot_metrics_vs_radius_for_best_l1(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from features.minimization.Operational import plot_metrics_vs_radius_for_best_l1_4_conditions

    plot_metrics_vs_radius_for_best_l1_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )


def plot_metrics_vs_l1_for_best_r(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from features.minimization.Operational import plot_metrics_vs_l1_for_best_r_4_conditions

    plot_metrics_vs_l1_for_best_r_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )


def plot_best_metric_per_l1_using_best_r(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from features.minimization.Operational import plot_best_metric_per_l1_using_best_r_4_conditions

    plot_best_metric_per_l1_using_best_r_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
