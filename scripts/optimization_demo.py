"""Demo de pesquisa dos quatro objetivos de otimização Tipo 1 sobre geologia 3D.

Isto não é um módulo pytest. Execute após instalar o pacote::

    python scripts/optimization_demo.py

Troque ``GEOLOGIA`` para ``"flat"``, ``"dipping"`` ou ``"facies"`` (ver
``scripts/example_meshes.py``). As figuras abrem uma de cada vez.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

import drilling.features.minimization.auxiliaries as ax
import drilling.features.minimization.plot as mplot
from drilling.features.minimization.data_base import DataSet
from drilling.features.minimization.minimal import (
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
from example_meshes import build_model
from drilling.features.minimization.operational import (
    minimal_total_time,
    operational_time_table,
    total_time_for_best_existing_trajectories_table,
    total_time_information_table,
)


Data = DataSet(
    P0=(0, 0, 0),
    P3=(1000, 300, 3000),
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

# ------------------------------------------------------------------
# GEOLOGIA
#   "facies"  -> horizontes mergulhados, dobrados e falhados + mudanca
#                lateral de facies (arenito <-> dolomita). E o modelo 3D
#                completo: no mesmo Z a rocha muda com (x, y).
#   "dipping" -> a mesma coluna, so com estrutura (sem mudanca lateral).
#   "flat"    -> camadas horizontais. E um caso PARTICULAR do modelo 3D
#                (todas as superficies constantes), util como referencia.
# ------------------------------------------------------------------
GEOLOGIA = "facies"

Mesh = build_model(GEOLOGIA)


OPERATIONAL_PARAMETERS = {
    "trip_fixed_time_h": 2.0,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": 60.0,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "min_spacing_between_bit_trips_m": 150.0,
    "lithology_min_run_m": 30.0,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "casing_events": [
        {
            "depth_m": 2000.0,
            "name": "Casing shoe / cementing",
            "fixed_time_h": 10.0,
            "include_trip": True,
        }
    ],
}


SECTION_COLORS = {
    "L1": "#0b3c5d",
    "L2": "#8c510a",
    "L3": "#1b5e20",
    "Command": "#7f0000",
    "Radius": "#4d4d4d",
}

POINT_COLORS = {
    "P0": "#111111",
    "P1": "#3f007d",
    "P2": "#005a32",
    "P3": "#7f2704",
    "C": "#4d4d4d",
}


def _trajectory_plot_data(Data, l1: float, R: float) -> dict:
    return ax.trajectory_plot_data(Data, l1, R)

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
    plot_data = _trajectory_plot_data(Data, l1, R)

    s_values = [plot_data["p0_s"], plot_data["p1_s"], *plot_data["curve_s"], plot_data["p3_s"]]
    z_values = [plot_data["p0_z"], plot_data["p1_z"], *plot_data["curve_z"], plot_data["p3_z"]]

    margin_x = float(Data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    alpha = float(Data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))

    s_min = min(0.0, min(s_values) - 0.05 * max(Data.departure, 1.0))
    s_max = max(max(s_values), Data.departure) + margin_x
    mesh_z_max = Mesh.z_max
    z_min = float(min(Data.P0[2], min(z_values)) - y_margin)
    z_max = float(max(mesh_z_max, Data.P3[2], max(z_values)) + y_margin)

    fig, ax_plot = plt.subplots(figsize=(12.0, 8.0))

    # One raster sampled on the well plane, so dipping contacts and lateral facies
    # changes show up as they really are instead of as full-width depth bands.
    mesh_handles = mplot.plot_mesh_cross_section(
        ax_plot, Data, Mesh, (s_min, s_max), (z_min, z_max), alpha=alpha
    )

    p1 = plot_data["p1_plane"]
    p2 = plot_data["p2_plane"]
    p3 = plot_data["p3_plane"]
    center = plot_data["center_plane"]
    command_start = plot_data["command_start_plane"]

    ax_plot.plot([plot_data["p0_s"], p1[0]], [plot_data["p0_z"], p1[1]], color=SECTION_COLORS["L1"], linewidth=3.0, label="L1 - Vertical section", zorder=3)
    ax_plot.plot(plot_data["curve_s"], plot_data["curve_z"], color=SECTION_COLORS["L2"], linewidth=3.0, label="L2 - Curved section", zorder=3)
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
    ax_plot.scatter([plot_data["p0_s"]], [plot_data["p0_z"]], s=65, color=POINT_COLORS["P0"], label="P0 - Initial point", zorder=5)
    ax_plot.scatter([p1[0]], [p1[1]], s=65, color=POINT_COLORS["P1"], label="P1 - Start of curved section", zorder=5)
    ax_plot.scatter([p2[0]], [p2[1]], s=65, color=POINT_COLORS["P2"], label="P2 - Start of inclined section", zorder=5)
    ax_plot.scatter([p3[0]], [p3[1]], s=65, color=POINT_COLORS["P3"], label="P3 - Target point", zorder=5)
    ax_plot.scatter([center[0]], [center[1]], s=55, color=POINT_COLORS["C"], label="C - Curvature center", zorder=5)

    ax_plot.set_aspect("equal")
    ax_plot.set_xlim(s_min, s_max)
    ax_plot.set_ylim(z_min, z_max)
    ax_plot.invert_yaxis()
    ax_plot.set_title(title)
    ax_plot.set_xlabel("Horizontal distance in well plane (m)")
    ax_plot.set_ylabel("Depth (m)")
    ax_plot.grid(alpha=0.25, linewidth=0.8)
    handles = ax_plot.get_legend_handles_labels()[0] + mesh_handles
    ax_plot.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
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


def main() -> None:
    """Executa os quatro objetivos e mostra os gráficos de pesquisa."""
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

    print(f"\nGeologia: '{GEOLOGIA}' -> {Mesh}")
    print("Abrindo 8 figuras, uma de cada vez. FECHE cada janela para ver a proxima:")
    print("  1) malha 3D + trajetoria     2-5) secao geologica de cada objetivo")
    print("  6-8) curvas globais de otimizacao\n")

    mplot.plot_trajectory_with_mesh_3d(
        Data,
        Mesh,
        total_l1,
        total_r,
        title=f"Geological horizons and minimal-total-time trajectory ({GEOLOGIA})",
    )
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


if __name__ == "__main__":
    main()
