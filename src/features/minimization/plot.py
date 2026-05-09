import matplotlib.patches as patches
import numpy as np

import features.minimization.Auxiliaries as ax
from features.minimization.Minimal import DEFAULT_STYLE


OBJECTIVE_STYLES = {
    "force": {"label": "Minimum axial force", "color": "#2563eb"},
    "torque": {"label": "Minimum torque", "color": "#d97706"},
    "time": {"label": "Minimum drilling time", "color": "#059669"},
    "total": {"label": "Minimum total time", "color": "#dc2626"},
}


def apply_chart_style(figure):
    figure.patch.set_facecolor("#ffffff")
    for axis in figure.axes:
        axis.set_facecolor("#ffffff")
        axis.grid(alpha=0.25, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")


def clear_figure(figure):
    figure.clear()
    figure.set_facecolor("#ffffff")


def trajectory_plot_data(data, l1: float, radius: float) -> dict:
    config = ax.validate_configuration(data, l1, radius)
    x0, y0 = data.P0
    p1 = (x0, y0 + config["l1"])
    curve_x, curve_y = ax.curve_points(data, l1, radius)
    p2 = (float(curve_x[-1]), float(curve_y[-1]))
    p3 = (float(data.P3[0]), float(data.P3[1]))
    center = (float(data.P0[0] + radius), float(data.P0[1] + config["l1"]))
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
        "command_start": command_start,
        "curve_x": curve_x,
        "curve_y": curve_y,
    }


def prepare_mesh_axes(axis, data, geological_mesh, x_values, y_values):
    margin_x = float(data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    alpha = float(data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))

    x_min = min(0.0, min(x_values) - 0.05 * max(data.P3[0], 1.0))
    x_max = max(max(x_values), data.P3[0]) + margin_x
    y_max = max([segment["end"] for segment in geological_mesh.segments] + [data.P3[1], max(y_values)])

    used_labels = set()
    for segment in geological_mesh.segments:
        color = ax.LITHOLOGY_COLORS.get(segment["lithology"], "#d1d5db")
        label = segment["lithology"] if segment["lithology"] not in used_labels else None
        if label:
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
        axis.add_patch(rect)

    axis.set_aspect("equal")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(0.0, y_max)
    axis.invert_yaxis()
    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("Depth (m)")
    axis.grid(alpha=0.25, linewidth=0.8)


def plot_trajectories(
    figure,
    data,
    geological_mesh,
    results,
    visible_objectives=None,
    show_command_sections=False,
    show_radius_lines=False,
):
    clear_figure(figure)
    axis = figure.add_subplot(111)
    visible_objectives = list(results.keys()) if visible_objectives is None else list(visible_objectives)

    all_x = [data.P0[0], data.P3[0]]
    all_y = [data.P0[1], data.P3[1]]
    trajectory_data = {}

    for key, result in results.items():
        plot_data = trajectory_plot_data(data, result["l1"], result["R"])
        trajectory_data[key] = plot_data
        if key in visible_objectives:
            all_x.extend([plot_data["p1"][0], plot_data["p2"][0], *plot_data["curve_x"], data.P3[0]])
            all_y.extend([plot_data["p1"][1], plot_data["p2"][1], *plot_data["curve_y"], data.P3[1]])

    prepare_mesh_axes(axis, data, geological_mesh, all_x, all_y)

    for key, plot_data in trajectory_data.items():
        if key not in visible_objectives:
            continue
        style = OBJECTIVE_STYLES[key]
        axis.plot(
            [data.P0[0], plot_data["p1"][0]],
            [data.P0[1], plot_data["p1"][1]],
            color=style["color"],
            linewidth=2.4,
            alpha=0.95,
            label=style["label"],
            zorder=3,
        )
        axis.plot(plot_data["curve_x"], plot_data["curve_y"], color=style["color"], linewidth=2.4, alpha=0.95, zorder=3)
        axis.plot(
            [plot_data["p2"][0], plot_data["p3"][0]],
            [plot_data["p2"][1], plot_data["p3"][1]],
            color=style["color"],
            linewidth=2.4,
            alpha=0.95,
            zorder=3,
        )
        if show_command_sections:
            axis.plot(
                [plot_data["command_start"][0], plot_data["p3"][0]],
                [plot_data["command_start"][1], plot_data["p3"][1]],
                color=style["color"],
                linewidth=5.0,
                alpha=0.45,
                solid_capstyle="round",
                zorder=4,
            )
            axis.scatter([plot_data["command_start"][0]], [plot_data["command_start"][1]], color=style["color"], s=28, marker="s", zorder=5)
        if show_radius_lines:
            axis.plot(
                [plot_data["center"][0], plot_data["p1"][0]],
                [plot_data["center"][1], plot_data["p1"][1]],
                color=style["color"],
                linestyle="--",
                linewidth=1.3,
                alpha=0.55,
                zorder=2,
            )
            axis.plot(
                [plot_data["center"][0], plot_data["p2"][0]],
                [plot_data["center"][1], plot_data["p2"][1]],
                color=style["color"],
                linestyle="--",
                linewidth=1.3,
                alpha=0.55,
                zorder=2,
            )
            axis.scatter([plot_data["center"][0]], [plot_data["center"][1]], color=style["color"], s=26, marker="x", zorder=5)
        axis.scatter([plot_data["p1"][0], plot_data["p2"][0]], [plot_data["p1"][1], plot_data["p2"][1]], color=style["color"], s=34, zorder=4)

    axis.scatter([data.P0[0]], [data.P0[1]], s=52, color="#111827", label="P0", zorder=5)
    axis.scatter([data.P3[0]], [data.P3[1]], s=58, color="#7c2d12", label="P3", zorder=5)
    axis.set_title("Optimized trajectories over geological mesh", pad=12)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    figure.tight_layout()
    apply_chart_style(figure)


def _plot_metric_family(figure, series: dict, x_label: str):
    clear_figure(figure)
    axes = figure.subplots(2, 2)
    axes = np.asarray(axes).ravel()

    for axis, key in zip(axes, ["force", "torque", "time", "total"]):
        item = series[key]
        style = OBJECTIVE_STYLES[key]
        x_vals = np.asarray(item["x"], dtype=float)
        y_vals = np.asarray(item["y"], dtype=float)
        axis.plot(x_vals, y_vals, color=style["color"], linewidth=2.2)
        if "best_x" in item:
            axis.axvline(float(item["best_x"]), color=style["color"], linestyle="--", alpha=0.45, linewidth=1.4)
        axis.set_title(item["title"], pad=8)
        axis.set_xlabel(x_label)
        axis.set_ylabel(item["ylabel"])
        axis.margins(x=0.02, y=0.08)
        axis.grid(alpha=0.30, linewidth=0.8)

    figure.tight_layout()
    apply_chart_style(figure)


def plot_global_curves(figure, series_by_scope: dict, selected_scope: str):
    if selected_scope == "radius":
        _plot_metric_family(figure, series_by_scope["radius"], "Radius (m)")
    elif selected_scope == "l1":
        _plot_metric_family(figure, series_by_scope["l1"], "Length L1 (m)")
    else:
        _plot_metric_family(figure, series_by_scope["best_per_l1"], "Length L1 (m)")


def plot_time_breakdown(figure, results):
    clear_figure(figure)
    axes = figure.subplots(1, 2)
    labels = [OBJECTIVE_STYLES[key]["label"] for key in results]
    colors = [OBJECTIVE_STYLES[key]["color"] for key in results]
    drilling = [results[key]["drilling_time_h"] for key in results]
    operational = [results[key]["operational_time_h"] for key in results]
    total = [results[key]["total_time_h"] for key in results]

    x = np.arange(len(labels))
    axes[0].bar(x, drilling, color=colors, alpha=0.80, label="Pure drilling")
    axes[0].bar(x, operational, bottom=drilling, color="#64748b", alpha=0.62, label="Operations")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Time (h)")
    axes[0].set_title("Drilling and operational time")
    axes[0].legend(frameon=True)

    width = 0.24
    axes[1].bar(x - width, drilling, width=width, color="#2563eb", alpha=0.78, label="Drilling")
    axes[1].bar(x, operational, width=width, color="#f59e0b", alpha=0.78, label="Operations")
    axes[1].bar(x + width, total, width=width, color="#16a34a", alpha=0.78, label="Total")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Time (h)")
    axes[1].set_title("Objective comparison")
    axes[1].legend(frameon=True)

    figure.tight_layout()
    apply_chart_style(figure)


def use_default_matplotlib_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(DEFAULT_STYLE)
