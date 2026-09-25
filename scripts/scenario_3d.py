"""How a genuinely three-dimensional geology moves the optimal trajectory.

Three geological models share the same stratigraphic column, the same rock
properties and the same well target; they differ only in geometry:

1. ``flat``    - horizontal layers, the layer-cake the earlier implementation assumed
2. ``dipping`` - the same column tilted, folded over an anticline and cut by a fault
3. ``facies``  - the dipping column, but the hard dolomite unit passes laterally
                 into sandstone near the well head

The column contains a slow, abrasive dolomite between about 1200 m and 1800 m. In
models 1 and 2 it extends over the whole field, so every candidate has to drill
through it. In model 3 it only exists beyond roughly 300 m of horizontal offset,
which a deep kick-off reaches below its base and a shallow kick-off does not - so
the trajectories are no longer equivalent and the optimal kick-off point moves.

That is the effect a stack of parallelepipeds could not represent: at one depth the
rock is sandstone at one ``(x, y)`` and dolomite at another.

Run with::

    python scripts/scenario_3d.py

Figures and the CSV table are written to ``outputs/scenario_3d/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTPUT_DIR = ROOT / "outputs" / "scenario_3d"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import drilling.features.minimization.auxiliaries as ax
import drilling.features.minimization.plot as mplot
from drilling.features.minimization.data_base import DataSet
from drilling.features.minimization.minimal import DEFAULT_STYLE, minimal_drilling_time, minimal_tension, minimal_torque
from drilling.features.minimization.operational import minimal_total_time, operational_time_breakdown
from example_meshes import build_model


plt.rcParams.update(DEFAULT_STYLE)

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
        {"depth_m": 2000.0, "name": "Casing shoe / cementing", "fixed_time_h": 10.0, "include_trip": True}
    ],
}


def build_data() -> DataSet:
    return DataSet(
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
        drilling_time_parameters={"trajectory_step": 1.0, "mesh_plot_alpha": 0.55},
        friction_model="lithology",
    )


def evaluate(Data, Mesh, label: str) -> list[dict]:
    """Run the four objectives on one geological model."""
    objectives = {
        "Minimal axial force": minimal_tension(Data, Mesh),
        "Minimal torque": minimal_torque(Data, Mesh),
        "Minimal drilling time": minimal_drilling_time(Data, Mesh),
        "Minimal total time": minimal_total_time(Data, Mesh, operational_parameters=OPERATIONAL_PARAMETERS),
    }

    rows = []
    for objective, (l1, R) in objectives.items():
        config = ax.validate_configuration(Data, l1, R)
        mechanics = ax.mechanical_summary(Data, Mesh, l1, R)
        operational = operational_time_breakdown(
            Data, Mesh, l1, R, operational_parameters=OPERATIONAL_PARAMETERS
        )
        rows.append(
            {
                "Scenario": label,
                "Objective": objective,
                "KOP l1 (m)": round(float(l1), 1),
                "R (m)": round(float(R), 1),
                "Angle (deg)": round(config["angle_deg"], 2),
                "Top axial force (kN)": round(mechanics["up_force_1"] / 1000.0, 1),
                "Torque (kN.m)": round(mechanics["torque"] / 1000.0, 3),
                "Drilling time (h)": round(operational["drilling_time_h"], 1),
                "Total time (h)": round(operational["total_time_h"], 1),
                "Dolomite drilled (m)": round(
                    operational["base_timing"]["by_lithology"].get("Dolomite", {"length_m": 0.0})["length_m"], 1
                ),
                "Bit trips": operational["by_category"]["bit_trip"]["count"],
            }
        )
    return rows


def plot_section(Data, Mesh, label: str, l1: float, R: float, filename: str) -> None:
    plot_data = ax.trajectory_plot_data(Data, l1, R)
    s_min, s_max = -60.0, float(Data.departure) + 120.0
    z_min, z_max = 0.0, 3300.0

    figure, axes = plt.subplots(figsize=(9.0, 9.5))
    handles = mplot.plot_mesh_cross_section(axes, Data, Mesh, (s_min, s_max), (z_min, z_max))
    axes.plot([plot_data["p0_s"], plot_data["p1_s"]], [plot_data["p0_z"], plot_data["p1_z"]],
              color="#0b3c5d", linewidth=3.0, label="L1 - Vertical section", zorder=3)
    axes.plot(plot_data["curve_s"], plot_data["curve_z"],
              color="#8c510a", linewidth=3.0, label="L2 - Curved section", zorder=3)
    axes.plot([plot_data["p2_s"], plot_data["p3_s"]], [plot_data["p2_z"], plot_data["p3_z"]],
              color="#1b5e20", linewidth=3.0, label="L3 - Inclined section", zorder=3)
    axes.scatter([plot_data["p1_s"]], [plot_data["p1_z"]], s=70, color="#3f007d", zorder=5, label="KOP")
    axes.set_aspect("equal")
    axes.set_xlim(s_min, s_max)
    axes.set_ylim(z_min, z_max)
    axes.invert_yaxis()
    axes.set_xlabel("Horizontal distance in well plane (m)")
    axes.set_ylabel("Depth (m)")
    axes.set_title(f"{label}: KOP = {l1:.0f} m, R = {R:.0f} m")
    axes.grid(alpha=0.2, linewidth=0.7)
    axes.legend(handles=axes.get_legend_handles_labels()[0] + handles,
                loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
    figure.savefig(filename, dpi=140)
    plt.close(figure)


def main() -> None:
    Data = build_data()
    scenarios = [
        ("Flat layers", "flat"),
        ("Dipping + fold + fault", "dipping"),
        ("Lateral facies change", "facies"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label, kind in scenarios:
        Mesh = build_model(kind)
        print(f"\nEvaluating '{label}' -> {Mesh}")
        rows = evaluate(Data, Mesh, label)
        all_rows.extend(rows)
        best_time = next(row for row in rows if row["Objective"] == "Minimal total time")
        plot_section(Data, Mesh, label, best_time["KOP l1 (m)"], best_time["R (m)"],
                     OUTPUT_DIR / f"scenario_{kind}_section.png")
        mplot.plot_trajectory_with_mesh_3d(
            Data, Mesh, best_time["KOP l1 (m)"], best_time["R (m)"],
            title=f"{label} - minimal total time", filename=OUTPUT_DIR / f"scenario_{kind}_3d.png",
        )

    table = pd.DataFrame(all_rows)
    print("\n=== Optimal trajectory per objective and geological scenario ===")
    print(table.to_string(index=False))

    print("\n=== Kick-off point shift relative to the flat-layer model ===")
    pivot = table.pivot(index="Objective", columns="Scenario", values="KOP l1 (m)")
    pivot = pivot[[label for label, _ in scenarios]]
    for label, _ in scenarios[1:]:
        pivot[f"shift vs flat ({label})"] = pivot[label] - pivot["Flat layers"]
    print(pivot.to_string())

    table.to_csv(OUTPUT_DIR / "scenario_3d_results.csv", index=False)
    print(f"\nSaved scenario_3d_results.csv and the section/3D figures to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
