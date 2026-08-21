from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import features.minimization.Auxiliaries as ax
from features.minimization.Data_base import DataSet, mesh
from features.minimization.Minimal import (
    DEFAULT_STYLE,
    drilling_time_breakdown,
    plot_best_metric_per_l1_using_best_r,
    plot_metrics_vs_l1_for_best_r,
    plot_metrics_vs_radius_for_best_l1,
)
from features.minimization.Operational import operational_time_table


# ============================================================
# USER INPUT
# ============================================================
L1_SELECTED = 1200.0
R_SELECTED = 500.0
PLOT_GLOBAL_OPTIMIZATION_CURVES = True
SAVE_SELECTED_TRAJECTORY = False
SELECTED_TRAJECTORY_FILENAME = "selected_trajectory_geological_mesh.png"

OPERATIONAL_PARAMETERS = {
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
    "casing_events": [
        {
            "depth_m": 2000.0,
            "name": "Casing shoe / cementing",
            "fixed_time_h": 10.0,
            "include_trip": True,
        }
    ],
}


Data = DataSet(
    P0=(0, 0),
    P3=(1000, 3000),
    ro_fluid=1737.5,
    ro_command=8000,
    ro_drillpipe=8000,
    ro_heavypipe=8000,
    diameters_command=(0.2032, 0.1143),
    diameters_drillpipe=(0.127, 0.1086104),
    diameters_heavypipe=(0.1524, 0.1143),
    µ=0.23,
    z=(5000 * 8) * 4.44822,
    lp=36,
    max=2300,
    radius=(100, 600),
    drilling_time_parameters={
        "trajectory_step": 1.0,
        "reference_dls_deg_per_30m": 3.0,
        "surface_wob": 1.60e5,
        "optimal_wob": 1.80e5,
        "torque_limit": 1.20e4,
        "mesh_plot_alpha": 0.45,
    },
)

Mesh = mesh(
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



SECTION_COLORS = {
    "L1": "#0b3c5d",
    "L2": "#8c510a",
    "L3": "#1b5e20",
    "Command": "#7f0000",
    "Radius": "#4d4d4d",
}


def _trajectory_plot_data(Data, l1: float, R: float) -> dict:
    config = ax.validate_configuration(Data, l1, R)
    x0, y0 = Data.P0
    p1 = (x0, y0 + config["l1"])
    curve_x, curve_y = ax.curve_points(Data, l1, R)
    p2 = (float(curve_x[-1]), float(curve_y[-1]))
    p3 = (float(Data.P3[0]), float(Data.P3[1]))
    center = (float(Data.P0[0] + R), float(Data.P0[1] + config["l1"]))

    l3 = float(config["l3"])
    lc = float(config["lc"])
    command_fraction = 0.0 if l3 <= 0.0 else max(0.0, min(1.0, (l3 - lc) / l3))
    command_start = (
        p2[0] + command_fraction * (p3[0] - p2[0]),
        p2[1] + command_fraction * (p3[1] - p2[1]),
    )

    return {
        "config": config,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "center": center,
        "curve_x": curve_x,
        "curve_y": curve_y,
        "command_start": command_start,
    }

plt.rcParams.update(DEFAULT_STYLE)


def selected_trajectory_information(Data, Mesh, l1: float, R: float) -> dict:
    config = ax.validate_configuration(Data, l1, R)
    up_forces = ax.up_tension(Data, l1, R)
    down_forces = ax.down_tension(Data, l1, R)
    neutral_line = ax.Nl(Data, l1, R)
    timing = drilling_time_breakdown(Data, Mesh, l1, R)

    return {
        "configuration": config,
        "up_forces": up_forces,
        "down_forces": down_forces,
        "neutral_line": float(neutral_line),
        "timing": timing,
    }


def selected_trajectory_information_table(Data, Mesh, l1: float, R: float) -> None:
    info = selected_trajectory_information(Data, Mesh, l1, R)
    config = info["configuration"]
    up_f1, up_f2, up_f3 = info["up_forces"]
    down_f1, down_f2, down_f3, torque = info["down_forces"]
    timing = info["timing"]

    summary = pd.DataFrame(
        {
            "Value": np.round(
                [
                    config["l1"],
                    config["l2"],
                    config["l3"],
                    config["R"],
                    config["angle_deg"],
                    config["lc"],
                    config["ld"],
                    info["neutral_line"],
                    up_f1,
                    up_f2,
                    up_f3,
                    down_f1,
                    down_f2,
                    down_f3,
                    torque,
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
            "Command length lc (m)",
            "Remaining tangent length ld (m)",
            "Neutral line (m)",
            "Up axial force L1 (N)",
            "Up axial force L2 (N)",
            "Up axial force L3 (N)",
            "Down axial force L1 (N)",
            "Down axial force L2 (N)",
            "Down axial force L3 (N)",
            "Torque (N*m)",
            "Total trajectory length (m)",
            "Total drilling time (h)",
            "Average effective ROP (m/h)",
            "Average effective WOB (N)",
            "Average DLS (deg/30m)",
            "Max cumulative drilling torque (N*m)",
        ],
    )

    print("\n--- Result table for the selected trajectory ---")
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


def plot_selected_trajectory_with_geological_mesh(
    Data,
    Mesh,
    l1: float,
    R: float,
    title: str = "Type-1 trajectory for the selected configuration",
    filename: str | None = None,
    y_margin: float = 120.0,
) -> str | None:
    plot_data = _trajectory_plot_data(Data, l1, R)

    x_values = [Data.P0[0], plot_data["p1"][0], *plot_data["curve_x"], Data.P3[0]]
    y_values = [Data.P0[1], plot_data["p1"][1], *plot_data["curve_y"], Data.P3[1]]

    margin_x = float(Data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    alpha = float(Data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))

    x_min = min(0.0, min(x_values) - 0.05 * max(Data.P3[0], 1.0))
    x_max = max(max(x_values), Data.P3[0]) + margin_x
    mesh_y_max = max(segment["end"] for segment in Mesh.segments) if Mesh.segments else Data.P3[1]
    y_min = float(min(Data.P0[1], min(y_values)) - y_margin)
    y_max = float(max(mesh_y_max, Data.P3[1], max(y_values)) + y_margin)

    fig, ax_plot = plt.subplots(figsize=(12.0, 8.0))

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

    x0, y0 = Data.P0
    p1 = plot_data["p1"]
    p2 = plot_data["p2"]
    p3 = plot_data["p3"]
    center = plot_data["center"]
    command_start = plot_data["command_start"]

    ax_plot.plot([x0, p1[0]], [y0, p1[1]], color=SECTION_COLORS["L1"], linewidth=3.0, label="L1 - Vertical section", zorder=3)
    ax_plot.plot(plot_data["curve_x"], plot_data["curve_y"], color=SECTION_COLORS["L2"], linewidth=3.0, label="L2 - Curved section", zorder=3)
    ax_plot.plot([p2[0], p3[0]], [p2[1], p3[1]], color=SECTION_COLORS["L3"], linewidth=3.0, label="L3 - Inclined section", zorder=3)
    ax_plot.plot(
        [command_start[0], p3[0]],
        [command_start[1], p3[1]],
        color=SECTION_COLORS["Command"],
        linewidth=4.0,
        label="Command section",
        zorder=4,
    )
    ax_plot.plot(
        [center[0], p1[0]],
        [center[1], p1[1]],
        linestyle="--",
        color=SECTION_COLORS["Radius"],
        linewidth=1.8,
        label="Radius to curvature center",
        zorder=2,
    )
    ax_plot.plot(
        [center[0], p2[0]],
        [center[1], p2[1]],
        linestyle="--",
        color=SECTION_COLORS["Radius"],
        linewidth=1.8,
        label="_nolegend_",
        zorder=2,
    )
    ax_plot.scatter([center[0]], [center[1]], s=45, color=SECTION_COLORS["Radius"], zorder=5)
    ax_plot.annotate("C", center, xytext=(8, -6), textcoords="offset points", color=SECTION_COLORS["Radius"])

    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=60, color="black", zorder=5)
    ax_plot.annotate("P0", (Data.P0[0], Data.P0[1]), xytext=(8, -12), textcoords="offset points")
    ax_plot.annotate("P3", (Data.P3[0], Data.P3[1]), xytext=(8, -12), textcoords="offset points")

    ax_plot.set_aspect("equal")
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_ylim(y_min, y_max)
    ax_plot.invert_yaxis()
    ax_plot.set_title(title)
    ax_plot.set_xlabel("Horizontal distance (m)")
    ax_plot.set_ylabel("Depth (m)")
    ax_plot.grid(alpha=0.25, linewidth=0.8)
    ax_plot.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))

    if filename is not None:
        output_path = Path(filename)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return str(output_path)

    plt.show()
    return None


print(f"Selected configuration: l1 = {L1_SELECTED:.1f} m, R = {R_SELECTED:.1f} m")
selected_trajectory_information_table(Data, Mesh, L1_SELECTED, R_SELECTED)
operational_time_table(Data, Mesh, L1_SELECTED, R_SELECTED, operational_parameters=OPERATIONAL_PARAMETERS)

if SAVE_SELECTED_TRAJECTORY:
    saved_path = plot_selected_trajectory_with_geological_mesh(
        Data,
        Mesh,
        L1_SELECTED,
        R_SELECTED,
        title="Type-1 trajectory for the selected configuration",
        filename=SELECTED_TRAJECTORY_FILENAME,
    )
    print(f"Saved selected trajectory plot to: {saved_path}")
else:
    plot_selected_trajectory_with_geological_mesh(
        Data,
        Mesh,
        L1_SELECTED,
        R_SELECTED,
        title="Type-1 trajectory for the selected configuration",
    )

if PLOT_GLOBAL_OPTIMIZATION_CURVES:
    plot_metrics_vs_radius_for_best_l1(Data, Mesh)
    plot_metrics_vs_l1_for_best_r(Data, Mesh)
    plot_best_metric_per_l1_using_best_r(Data, Mesh)
