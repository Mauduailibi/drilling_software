"""
post.py

Sensitivity-analysis post-processing script for the Type-I directional-well
optimization framework.

This script evaluates the influence of three numerical discretizations:

1) L1 search increment sensitivity, using a fixed R increment.
2) R search increment sensitivity, using a fixed L1 increment.
3) Trajectory-element length sensitivity in the drilling-time module, using
   fixed L1 and R search increments.

For each analysis, the script compares consecutive meshes:
100 -> 50, 50 -> 25, 25 -> 10, and 10 -> 1.
The relative error is computed only at common x-points between two consecutive
meshes. Two families of curves are considered:

- L1 using best R: for each objective, R is fixed at the optimum value found
  in that mesh and the response is plotted as a function of L1.
- R using best L1: for each objective, L1 is fixed at the optimum value found
  in that mesh and the response is plotted as a function of R.

Outputs are written to outputs/post_outputs/ by default:
- Overlay plots for each sensitivity analysis.
- CSV tables with relative errors.
- LaTeX tables with relative errors.
- CSV files with the plotted series.

Run examples:
    python scripts/sensitivity/post_corrigido_min_l1.py --run all
    python scripts/sensitivity/post_corrigido_min_l1.py --run l1 --fixed-radius-step-for-l1 50
    python scripts/sensitivity/post_corrigido_min_l1.py --run r --fixed-l1-step-for-r 10
    python scripts/sensitivity/post_corrigido_min_l1.py --run trajectory --fixed-l1-step-for-trajectory 50 --fixed-radius-step-for-trajectory 50
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import copy
import gc
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drilling.features.minimization.auxiliaries as ax
import drilling.features.minimization.plot as mplot
import drilling.features.minimization.minimal as Minimal
import drilling.features.minimization.operational as Operational
from drilling.features.minimization.data_base import DataSet, mesh
from drilling.features.minimization.minimal import DEFAULT_STYLE, _scan_candidates, drilling_time_breakdown
from drilling.features.minimization.operational import operational_time_breakdown


# ============================================================
# Sensitivity-analysis inputs
# ============================================================
L1_GRID_STEPS = [100.0, 50.0, 25.0, 10.0, 1.0]
R_GRID_STEPS = [100.0, 50.0, 25.0, 10.0, 1.0]
TRAJECTORY_STEPS = [100.0, 50.0, 25.0, 10.0, 1.0]

METRICS_ALL = {
    "force": "up_force_1",
    "torque": "torque",
    "drilling_time": "drilling_time_h",
    "total_time": "total_time_h",
}

METRICS_TIME = {
    "drilling_time": "drilling_time_h",
    "total_time": "total_time_h",
}

METRIC_LABELS = {
    "force": "Top axial force (N)",
    "torque": "Torque (N*m)",
    "drilling_time": "Drilling time (h)",
    "total_time": "Total time (h)",
}


# ============================================================
# Lightweight LaTeX table writer
# ============================================================
def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_latex_value(value: object, float_format: str = "%.4f") -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return float_format % float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _latex_escape(value)


def write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.4f") -> None:
    """Write a simple LaTeX tabular without requiring pandas/Jinja2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(df.columns)
    colspec = "l" + "c" * max(ncols - 1, 0)
    lines = [r"\begin{tabular}{" + colspec + "}", r"\hline"]
    latex_newline = " " + chr(92) * 2
    header = " & ".join(_latex_escape(column) for column in df.columns) + latex_newline
    lines.append(header)
    lines.append(r"\hline")
    for _, row in df.iterrows():
        formatted = [_format_latex_value(row[column], float_format) for column in df.columns]
        lines.append(" & ".join(formatted) + latex_newline)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Base case definition used in the TCC final study case
# ============================================================
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
    "operation_merge_distance_m": 30.0,
    "casing_connection_length_m": 9.0,
    "casing_connection_time_h": 0.10,
    "casing_trip_speed_mph": 300.0,
    "casing_logging_time_h": 5.0,
    "cement_pumping_time_h": 2.5,
    "cement_curing_time_h": 12.0,
    "casing_events": [
        {"depth_m": 300.0, "name": "Surface casing / cementing", "fixed_time_h": 0.0, "include_trip": True},
        {"depth_m": 1500.0, "name": "Intermediate casing near end of curve / cementing", "fixed_time_h": 0.0, "include_trip": True},
        {"depth_m": 2400.0, "name": "Production casing / liner cementing", "fixed_time_h": 0.0, "include_trip": True},
    ],
}


def make_data(
    l1_step: float = 10.0,
    radius_step: float = 50.0,
    trajectory_step: float = 1.0,
) -> DataSet:
    """Create the final study-case DataSet with user-defined numerical steps."""
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
        radius=(300, 600),
        drilling_time_parameters={
            "trajectory_step": float(trajectory_step),
            "reference_dls_deg_per_30m": 3.0,
            "surface_wob": 1.60e5,
            "optimal_wob": 1.80e5,
            "torque_limit": 1.20e4,
            "min_inclination_factor": 0.85,
            "inclination_reduction": 0.15,
            "inclination_exponent": 1.00,
            "min_dls_factor": 0.50,
            "dls_reduction": 0.50,
            "dls_exponent": 1.00,
            "min_wob_factor": 0.90,
            "min_torque_factor": 0.90,
            "torque_reduction": 0.10,
            "torque_exponent": 1.00,
            "mesh_plot_alpha": 0.45,
        },
    )
    Data.min_l1 = 300.0
    Data.l1_step = float(l1_step)
    Data.radius_step = float(radius_step)
    return Data


def make_mesh():
    """Create the final study-case lithological mesh."""
    return mesh(
        shale=[[2100, 2600]],
        siltstone=[[0, 300]],
        sandstone=[[700, 1900], [2600, 3000]],
        limestone=[[300, 700]],
        dolomite=[[1900, 2100]],
        evaporite=[],
        rop_values={
            "Dolomite": 3.2,
            "Limestone": 4.5,
            "Sandstone": 7.0,
            "Siltstone": 7.4,
            "Shale": 7.8,
            "Evaporite": 9.2,
        },
    )


# ============================================================
# Internal helpers
# ============================================================
def _clear_framework_caches() -> None:
    Minimal._MECH_CACHE.clear()
    Minimal._TIME_CACHE.clear()
    Operational._OPERATIONAL_CACHE.clear()
    gc.collect()


def _safe_key(l1: float, R: float) -> tuple[float, float]:
    return (round(float(l1), 8), round(float(R), 8))


def evaluate_candidate_grid(
    Data: DataSet,
    Mesh: mesh,
    operational_parameters: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate mechanics, drilling time, and total time for all valid candidates.

    A minimum vertical-section length is enforced here to guarantee that all
    post-processing results are restricted to feasible trajectories with

        L1 >= Data.min_l1.

    This additional filter is intentionally applied after the mechanical scan,
    so the post-processing stage remains protected even if the candidate
    generator returns points below the adopted lower bound.
    """
    if verbose:
        print(
            "Evaluating grid: "
            f"l1_step={Data.l1_step}, radius_step={Data.radius_step}, "
            f"trajectory_step={Data.drilling_time_parameters['trajectory_step']}"
        )

    min_l1 = float(getattr(Data, "min_l1", 300.0))

    mechanical_candidates = [
        candidate
        for candidate in _scan_candidates(Data)
        if float(candidate["l1"]) >= min_l1
    ]

    if not mechanical_candidates:
        raise RuntimeError(
            f"No mechanically feasible candidate satisfies the minimum "
            f"vertical-section constraint L1 >= {min_l1:g} m."
        )

    records: list[dict] = []

    total = len(mechanical_candidates)
    for idx, candidate in enumerate(mechanical_candidates, start=1):
        if verbose and (idx == 1 or idx == total or idx % max(1, total // 10) == 0):
            print(f"  candidate {idx}/{total}")

        l1 = float(candidate["l1"])
        R = float(candidate["R"])
        try:
            timing = drilling_time_breakdown(Data, Mesh, l1, R)
            operational = operational_time_breakdown(
                Data,
                Mesh,
                l1,
                R,
                drilling_timing=timing,
                operational_parameters=operational_parameters,
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

        records.append(
            {
                "l1": l1,
                "R": R,
                "up_force_1": float(candidate["up_force_1"]),
                "torque": float(candidate["torque"]),
                "drilling_time_h": float(timing["total_time_h"]),
                "total_time_h": float(operational["total_time_h"]),
                "operational_time_h": float(operational["total_operational_time_h"]),
                "average_rop_mph": float(timing["average_rop_mph"]),
                "angle_deg": float(candidate["angle_deg"]),
                "l2": float(candidate["l2"]),
                "l3": float(candidate["l3"]),
            }
        )

    if not records:
        raise RuntimeError(
            f"No candidate with L1 >= {min_l1:g} m could be evaluated "
            "for the selected grid."
        )

    df = pd.DataFrame(records).sort_values(["l1", "R"]).reset_index(drop=True)

    # Final safety filter before selecting best candidates and generating tables.
    df = df[df["l1"] >= min_l1].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            f"No post-processed candidate satisfies L1 >= {min_l1:g} m."
        )

    if verbose:
        print(f"  evaluated candidates with L1 >= {min_l1:g} m: {len(df)}")
    return df


def identify_best_candidates(df: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    """Return one optimum candidate for each requested metric."""
    rows = []
    for metric_name, column in metric_map.items():
        idx = df[column].idxmin()
        row = df.loc[idx]
        rows.append(
            {
                "metric": metric_name,
                "column": column,
                "best_l1": float(row["l1"]),
                "best_R": float(row["R"]),
                "best_value": float(row[column]),
            }
        )
    return pd.DataFrame(rows)


def build_fixed_best_series(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    metric_map: dict[str, str],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build L1 using best R and R using best L1 series for each metric."""
    series = {"l1_using_best_r": {}, "r_using_best_l1": {}}
    for _, best in best_df.iterrows():
        metric_name = str(best["metric"])
        column = metric_map[metric_name]
        best_r = float(best["best_R"])
        best_l1 = float(best["best_l1"])

        l1_df = df[np.isclose(df["R"], best_r)][["l1", column]].copy()
        l1_df = l1_df.rename(columns={"l1": "x", column: "y"}).sort_values("x")
        l1_df["metric"] = metric_name
        l1_df["fixed_R"] = best_r
        series["l1_using_best_r"][metric_name] = l1_df.reset_index(drop=True)

        r_df = df[np.isclose(df["l1"], best_l1)][["R", column]].copy()
        r_df = r_df.rename(columns={"R": "x", column: "y"}).sort_values("x")
        r_df["metric"] = metric_name
        r_df["fixed_l1"] = best_l1
        series["r_using_best_l1"][metric_name] = r_df.reset_index(drop=True)
    return series


def flatten_series_to_dataframe(
    all_runs: dict[float, dict],
    curve_type: str,
    metrics: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for step, run in all_runs.items():
        for metric in metrics:
            series_df = run["series"][curve_type][metric]
            for _, row in series_df.iterrows():
                rows.append(
                    {
                        "mesh_step": step,
                        "curve_type": curve_type,
                        "metric": metric,
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                    }
                )
    return pd.DataFrame(rows)


def relative_error_on_common_points(coarse: pd.DataFrame, fine: pd.DataFrame) -> tuple[float, int]:
    """Return mean relative error (%) and number of common x-points."""
    coarse_map = {round(float(row["x"]), 8): float(row["y"]) for _, row in coarse.iterrows()}
    fine_map = {round(float(row["x"]), 8): float(row["y"]) for _, row in fine.iterrows()}
    common_x = sorted(set(coarse_map).intersection(fine_map))
    if not common_x:
        return np.nan, 0

    errors = []
    for x in common_x:
        y_coarse = coarse_map[x]
        y_fine = fine_map[x]
        denominator = max(abs(y_fine), 1.0e-12)
        errors.append(abs(y_coarse - y_fine) / denominator * 100.0)
    return float(np.mean(errors)), len(common_x)


def compute_error_tables(
    all_runs: dict[float, dict],
    ordered_steps: list[float],
    metric_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute curve-wise and combined error tables."""
    curve_rows = []
    combined_rows = []
    metrics = list(metric_map.keys())
    curve_types = ["l1_using_best_r", "r_using_best_l1"]

    for coarse_step, fine_step in zip(ordered_steps[:-1], ordered_steps[1:]):
        comparison = f"{coarse_step:g} -> {fine_step:g}"
        combined_by_metric = {metric: [] for metric in metrics}
        combined_count_by_metric = {metric: 0 for metric in metrics}

        for curve_type in curve_types:
            for metric in metrics:
                coarse_series = all_runs[coarse_step]["series"][curve_type][metric]
                fine_series = all_runs[fine_step]["series"][curve_type][metric]
                error_pct, n_common = relative_error_on_common_points(coarse_series, fine_series)
                curve_rows.append(
                    {
                        "comparison": comparison,
                        "curve_type": curve_type,
                        "metric": metric,
                        "mean_relative_error_pct": error_pct,
                        "common_points": n_common,
                    }
                )
                if np.isfinite(error_pct) and n_common > 0:
                    combined_by_metric[metric].append(error_pct)
                    combined_count_by_metric[metric] += n_common

        for metric in metrics:
            values = combined_by_metric[metric]
            combined_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "global_mean_relative_error_pct": float(np.mean(values)) if values else np.nan,
                    "total_common_points": combined_count_by_metric[metric],
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(combined_rows)


def save_error_tables(
    curve_errors: pd.DataFrame,
    combined_errors: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> None:
    curve_csv = output_dir / f"{prefix}_curve_errors.csv"
    combined_csv = output_dir / f"{prefix}_global_errors.csv"
    curve_tex = output_dir / f"{prefix}_curve_errors.tex"
    combined_tex = output_dir / f"{prefix}_global_errors.tex"

    curve_errors.to_csv(curve_csv, index=False)
    combined_errors.to_csv(combined_csv, index=False)

    write_latex_table(curve_errors, curve_tex, float_format="%.4f")
    write_latex_table(combined_errors, combined_tex, float_format="%.4f")

    print(f"Saved error tables: {curve_csv}")
    print(f"Saved error tables: {combined_csv}")


def save_best_candidate_table(all_runs: dict[float, dict], output_dir: Path, prefix: str) -> None:
    rows = []
    for step, run in all_runs.items():
        best_df = run["best"].copy()
        best_df.insert(0, "mesh_step", step)
        rows.append(best_df)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output_dir / f"{prefix}_best_candidates.csv", index=False)
    write_latex_table(table, output_dir / f"{prefix}_best_candidates.tex", float_format="%.4f")


def save_series_tables(all_runs: dict[float, dict], output_dir: Path, prefix: str, metric_map: dict[str, str]) -> None:
    metrics = list(metric_map.keys())
    for curve_type in ["l1_using_best_r", "r_using_best_l1"]:
        table = flatten_series_to_dataframe(all_runs, curve_type, metrics)
        table.to_csv(output_dir / f"{prefix}_{curve_type}_series.csv", index=False)


def plot_overlay(
    all_runs: dict[float, dict],
    ordered_steps: list[float],
    metric_map: dict[str, str],
    curve_type: str,
    x_label: str,
    title_prefix: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(DEFAULT_STYLE)
    metrics = list(metric_map.keys())
    n_metrics = len(metrics)
    if n_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=(12.0, 4.8), constrained_layout=True)
        axes = np.ravel(axes)

    for ax_plot, metric in zip(axes, metrics):
        for step in ordered_steps:
            series_df = all_runs[step]["series"][curve_type][metric]
            ax_plot.plot(
                series_df["x"].to_numpy(dtype=float),
                series_df["y"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=f"step = {step:g}",
            )
        ax_plot.set_title(METRIC_LABELS.get(metric, metric))
        ax_plot.set_xlabel(x_label)
        ax_plot.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax_plot.grid(alpha=0.35, linewidth=0.8)
        ax_plot.legend(loc="best", frameon=True)
        ax_plot.margins(x=0.02, y=0.08)

    fig.suptitle(title_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def run_mesh_family(
    family_name: str,
    varied_steps: list[float],
    l1_step_factory,
    radius_step_factory,
    trajectory_step_factory,
    metric_map: dict[str, str],
    output_dir: Path,
    verbose: bool = True,
) -> dict[float, dict]:
    all_runs = {}
    Mesh = make_mesh()

    for step in varied_steps:
        _clear_framework_caches()
        Data = make_data(
            l1_step=l1_step_factory(step),
            radius_step=radius_step_factory(step),
            trajectory_step=trajectory_step_factory(step),
        )
        df = evaluate_candidate_grid(Data, Mesh, OPERATIONAL_PARAMETERS, verbose=verbose)
        best_df = identify_best_candidates(df, metric_map)
        series = build_fixed_best_series(df, best_df, metric_map)
        all_runs[float(step)] = {"data": df, "best": best_df, "series": series}

        step_label = str(step).replace(".", "p")
        df.to_csv(output_dir / f"{family_name}_full_candidates_step_{step_label}.csv", index=False)
        _clear_framework_caches()

    return all_runs


def run_l1_sensitivity(fixed_radius_step: float, trajectory_step: float, output_dir: Path, verbose: bool = True) -> None:
    prefix = f"l1_sensitivity_fixed_Rstep_{fixed_radius_step:g}_Ltraj_{trajectory_step:g}"
    print(f"\n=== L1 grid sensitivity | fixed R step = {fixed_radius_step:g} m ===")
    all_runs = run_mesh_family(
        family_name=prefix,
        varied_steps=L1_GRID_STEPS,
        l1_step_factory=lambda step: step,
        radius_step_factory=lambda step: fixed_radius_step,
        trajectory_step_factory=lambda step: trajectory_step,
        metric_map=METRICS_ALL,
        output_dir=output_dir,
        verbose=verbose,
    )
    curve_errors, combined_errors = compute_error_tables(all_runs, L1_GRID_STEPS, METRICS_ALL)
    save_error_tables(curve_errors, combined_errors, output_dir, prefix)
    save_best_candidate_table(all_runs, output_dir, prefix)
    save_series_tables(all_runs, output_dir, prefix, METRICS_ALL)

    plot_overlay(
        all_runs,
        L1_GRID_STEPS,
        METRICS_ALL,
        curve_type="l1_using_best_r",
        x_label="Length $L_1$ (m)",
        title_prefix="L1-step sensitivity: L1 using best R",
        output_path=output_dir / f"{prefix}_l1_using_best_r.png",
    )
    plot_overlay(
        all_runs,
        L1_GRID_STEPS,
        METRICS_ALL,
        curve_type="r_using_best_l1",
        x_label="Radius R (m)",
        title_prefix="L1-step sensitivity: R using best L1",
        output_path=output_dir / f"{prefix}_r_using_best_l1.png",
    )


def run_r_sensitivity(fixed_l1_step: float, trajectory_step: float, output_dir: Path, verbose: bool = True) -> None:
    prefix = f"r_sensitivity_fixed_L1step_{fixed_l1_step:g}_Ltraj_{trajectory_step:g}"
    print(f"\n=== R grid sensitivity | fixed L1 step = {fixed_l1_step:g} m ===")
    all_runs = run_mesh_family(
        family_name=prefix,
        varied_steps=R_GRID_STEPS,
        l1_step_factory=lambda step: fixed_l1_step,
        radius_step_factory=lambda step: step,
        trajectory_step_factory=lambda step: trajectory_step,
        metric_map=METRICS_ALL,
        output_dir=output_dir,
        verbose=verbose,
    )
    curve_errors, combined_errors = compute_error_tables(all_runs, R_GRID_STEPS, METRICS_ALL)
    save_error_tables(curve_errors, combined_errors, output_dir, prefix)
    save_best_candidate_table(all_runs, output_dir, prefix)
    save_series_tables(all_runs, output_dir, prefix, METRICS_ALL)

    plot_overlay(
        all_runs,
        R_GRID_STEPS,
        METRICS_ALL,
        curve_type="l1_using_best_r",
        x_label="Length $L_1$ (m)",
        title_prefix="R-step sensitivity: L1 using best R",
        output_path=output_dir / f"{prefix}_l1_using_best_r.png",
    )
    plot_overlay(
        all_runs,
        R_GRID_STEPS,
        METRICS_ALL,
        curve_type="r_using_best_l1",
        x_label="Radius R (m)",
        title_prefix="R-step sensitivity: R using best L1",
        output_path=output_dir / f"{prefix}_r_using_best_l1.png",
    )


def run_trajectory_step_sensitivity(
    fixed_l1_step: float,
    fixed_radius_step: float,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    prefix = f"trajectory_step_sensitivity_fixed_L1step_{fixed_l1_step:g}_Rstep_{fixed_radius_step:g}"
    print(
        "\n=== Trajectory-step sensitivity | "
        f"fixed L1 step = {fixed_l1_step:g} m, fixed R step = {fixed_radius_step:g} m ==="
    )
    all_runs = run_mesh_family(
        family_name=prefix,
        varied_steps=TRAJECTORY_STEPS,
        l1_step_factory=lambda step: fixed_l1_step,
        radius_step_factory=lambda step: fixed_radius_step,
        trajectory_step_factory=lambda step: step,
        metric_map=METRICS_TIME,
        output_dir=output_dir,
        verbose=verbose,
    )
    curve_errors, combined_errors = compute_error_tables(all_runs, TRAJECTORY_STEPS, METRICS_TIME)
    save_error_tables(curve_errors, combined_errors, output_dir, prefix)
    save_best_candidate_table(all_runs, output_dir, prefix)
    save_series_tables(all_runs, output_dir, prefix, METRICS_TIME)

    plot_overlay(
        all_runs,
        TRAJECTORY_STEPS,
        METRICS_TIME,
        curve_type="l1_using_best_r",
        x_label="Length $L_1$ (m)",
        title_prefix="Trajectory-step sensitivity: L1 using best R",
        output_path=output_dir / f"{prefix}_l1_using_best_r.png",
    )
    plot_overlay(
        all_runs,
        TRAJECTORY_STEPS,
        METRICS_TIME,
        curve_type="r_using_best_l1",
        x_label="Radius R (m)",
        title_prefix="Trajectory-step sensitivity: R using best L1",
        output_path=output_dir / f"{prefix}_r_using_best_l1.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-processing sensitivity analysis for the TCC drilling optimization code.")
    parser.add_argument("--run", choices=["all", "l1", "r", "trajectory", "dry-run"], default="all")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "post_outputs"))
    parser.add_argument("--fixed-radius-step-for-l1", type=float, default=50.0)
    parser.add_argument("--fixed-l1-step-for-r", type=float, default=10.0)
    parser.add_argument("--fixed-trajectory-step-for-l1-r", type=float, default=10.0)
    parser.add_argument("--fixed-l1-step-for-trajectory", type=float, default=50.0)
    parser.add_argument("--fixed-radius-step-for-trajectory", type=float, default=50.0)
    parser.add_argument("--quiet", action="store_true", help="Reduce progress printing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run == "dry-run":
        Data = make_data()
        Mesh = make_mesh()
        print("Dry run successful.")
        print(f"Data: l1_step={Data.l1_step}, radius_step={Data.radius_step}, trajectory_step={Data.drilling_time_parameters['trajectory_step']}")
        print(f"Mesh layers: {Mesh.n_layers} ({', '.join(Mesh.lithologies)})")
        return

    verbose = not args.quiet

    if args.run in {"all", "l1"}:
        run_l1_sensitivity(
            fixed_radius_step=args.fixed_radius_step_for_l1,
            trajectory_step=args.fixed_trajectory_step_for_l1_r,
            output_dir=output_dir,
            verbose=verbose,
        )

    if args.run in {"all", "r"}:
        run_r_sensitivity(
            fixed_l1_step=args.fixed_l1_step_for_r,
            trajectory_step=args.fixed_trajectory_step_for_l1_r,
            output_dir=output_dir,
            verbose=verbose,
        )

    if args.run in {"all", "trajectory"}:
        run_trajectory_step_sensitivity(
            fixed_l1_step=args.fixed_l1_step_for_trajectory,
            fixed_radius_step=args.fixed_radius_step_for_trajectory,
            output_dir=output_dir,
            verbose=verbose,
        )

    print("\nSensitivity analysis completed.")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
