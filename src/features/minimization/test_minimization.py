from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import Auxiliaries as ax
from Data_base import DataSet, mesh
from Minimal import (
    DEFAULT_STYLE,
    drilling_informations_table,
    drilling_time_information_table,
    minimal_drilling_time,
    minimal_tension,
    minimal_torque,
    optimization_summary_table,
    plot_best_metric_per_l1_using_best_r,
    plot_metrics_vs_l1_for_best_r,
    plot_metrics_vs_radius_for_best_l1,
)
from Operational import (
    minimal_total_time,
    operational_time_table,
    total_time_for_best_existing_trajectories_table,
    total_time_information_table,
)


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


plt.rcParams.update(DEFAULT_STYLE)


def plot_single_trajectory_with_geological_mesh(
    Data,
    Mesh,
    l1: float,
    R: float,
    title: str,
    filename: str | None = None,
    y_margin: float = 120.0,
) -> str | None:
    """Plot one optimized trajectory over the geological mesh with side legend and Y padding."""
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


force_l1, force_r = minimal_tension(Data)
print(f"Minimal axial-force configuration: l1 = {force_l1:.1f} m, R = {force_r:.1f} m")

torque_l1, torque_r = minimal_torque(Data)
print(f"Minimal torque configuration: l1 = {torque_l1:.1f} m, R = {torque_r:.1f} m")

time_l1, time_r = minimal_drilling_time(Data, Mesh)
print(f"Minimal drilling-time configuration: l1 = {time_l1:.1f} m, R = {time_r:.1f} m")

total_l1, total_r = minimal_total_time(Data, Mesh, operational_parameters=OPERATIONAL_PARAMETERS)
print(f"Minimal total-time configuration: l1 = {total_l1:.1f} m, R = {total_r:.1f} m")

optimization_summary_table(Data, Mesh)
drilling_informations_table(Data)
drilling_time_information_table(Data, Mesh)
total_time_information_table(Data, Mesh, operational_parameters=OPERATIONAL_PARAMETERS)
total_time_for_best_existing_trajectories_table(Data, Mesh, operational_parameters=OPERATIONAL_PARAMETERS)

print("\n--- Operational-time breakdown for the minimal total-time trajectory ---")
operational_time_table(Data, Mesh, total_l1, total_r, operational_parameters=OPERATIONAL_PARAMETERS)

plot_single_trajectory_with_geological_mesh(
    Data,
    Mesh,
    force_l1,
    force_r,
    title="Type-1 trajectory for the minimal axial force",
)
plot_single_trajectory_with_geological_mesh(
    Data,
    Mesh,
    torque_l1,
    torque_r,
    title="Type-1 trajectory for the minimal torque",
)
plot_single_trajectory_with_geological_mesh(
    Data,
    Mesh,
    time_l1,
    time_r,
    title="Type-1 trajectory for the minimal drilling time",
)
plot_single_trajectory_with_geological_mesh(
    Data,
    Mesh,
    total_l1,
    total_r,
    title="Type-1 trajectory for the minimal total time",
)

plot_metrics_vs_radius_for_best_l1(Data, Mesh)
plot_metrics_vs_l1_for_best_r(Data, Mesh)
plot_best_metric_per_l1_using_best_r(Data, Mesh)
