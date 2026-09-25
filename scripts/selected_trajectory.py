"""Inspeciona uma configuração Tipo 1 (L1, R) sobre o modelo geológico 3D.

Isto não é um módulo pytest. Execute após instalar o pacote::

    python scripts/selected_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTPUT_DIR = ROOT / "outputs"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drilling.features.minimization.auxiliaries as ax
import drilling.features.minimization.plot as mplot
from drilling.features.minimization.data_base import DataSet
from drilling.features.minimization.minimal import (
    DEFAULT_STYLE,
    drilling_time_breakdown,
    plot_best_metric_per_l1_using_best_r,
    plot_metrics_vs_l1_for_best_r,
    plot_metrics_vs_radius_for_best_l1,
)
from example_meshes import build_model
from drilling.features.minimization.operational import operational_time_table


# ============================================================
# USER INPUT
# ============================================================
L1_SELECTED = 1200.0
R_SELECTED = 500.0
PLOT_GLOBAL_OPTIMIZATION_CURVES = True
SAVE_SELECTED_TRAJECTORY = False
SELECTED_TRAJECTORY_FILENAME = OUTPUT_DIR / "selected_trajectory_geological_mesh.png"

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

Mesh = build_model(GEOLOGIA)


SECTION_COLORS = {
    "L1": "#0b3c5d",
    "L2": "#8c510a",
    "L3": "#1b5e20",
    "Command": "#7f0000",
    "Radius": "#4d4d4d",
}


def _trajectory_plot_data(Data, l1: float, R: float) -> dict:
    return ax.trajectory_plot_data(Data, l1, R)

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
    ax_plot.scatter([center[0]], [center[1]], s=45, color=SECTION_COLORS["Radius"], zorder=5)
    ax_plot.annotate("C", center, xytext=(8, -6), textcoords="offset points", color=SECTION_COLORS["Radius"])

    ax_plot.scatter([plot_data["p0_s"], p3[0]], [plot_data["p0_z"], p3[1]], s=60, color="black", zorder=5)
    ax_plot.annotate("P0", (plot_data["p0_s"], plot_data["p0_z"]), xytext=(8, -12), textcoords="offset points")
    ax_plot.annotate("P3", p3, xytext=(8, -12), textcoords="offset points")

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
    """Mostra as tabelas e os gráficos da configuração selecionada."""
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
        n_figures = 5 if PLOT_GLOBAL_OPTIMIZATION_CURVES else 2
        print(f"\nGeologia: '{GEOLOGIA}' -> {Mesh}")
        print(f"Abrindo {n_figures} figuras, uma de cada vez. FECHE cada janela para ver a proxima:")
        print("  1) malha 3D + trajetoria     2) secao geologica no plano do poco")
        if PLOT_GLOBAL_OPTIMIZATION_CURVES:
            print("  3-5) curvas globais de otimizacao")
        print("")

        mplot.plot_trajectory_with_mesh_3d(
            Data,
            Mesh,
            L1_SELECTED,
            R_SELECTED,
            title=f"Geological horizons and selected trajectory ({GEOLOGIA})",
        )
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


if __name__ == "__main__":
    main()
