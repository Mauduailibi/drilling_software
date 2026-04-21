from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import Auxiliaries as ax
from Data_base import DataSet, mesh
from Minimal import (
    DEFAULT_STYLE,
    drilling_time_breakdown,
    plot_best_metric_per_l1_using_best_r,
    plot_metrics_vs_l1_for_best_r,
    plot_metrics_vs_radius_for_best_l1,
)
from Operational import operational_time_table


# ============================================================
# USER INPUT
# ============================================================
L1_SELECTED = 1500.0
R_SELECTED = 600.0
PLOT_GLOBAL_OPTIMIZATION_CURVES = True
SAVE_SELECTED_TRAJECTORY = False
SELECTED_TRAJECTORY_FILENAME = "selected_trajectory_geological_mesh.png"

OPERATIONAL_PARAMETERS = {
    "trip_fixed_time_h": 2.0,
    "trip_time_per_meter_h": 0.0025,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": 60.0,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "min_spacing_between_bit_trips_m": 150.0,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "abrupt_transition_threshold": 0.18,
    "abrupt_transition_extra_wear": 0.30,
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
    x_values, y_values = ax.points_coordinates(Data, l1, R)

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

    ax_plot.plot(x_values, y_values, color="black", linewidth=2.6, label="Well trajectory", zorder=3)
    ax_plot.scatter([Data.P0[0], Data.P3[0]], [Data.P0[1], Data.P3[1]], s=60, color="black", zorder=4)
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
    plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

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
