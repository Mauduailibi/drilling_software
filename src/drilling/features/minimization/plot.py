"""Gráficos Matplotlib da aba Minimization e dos scripts de pesquisa.

Esses plotters recebem resultados já calculados. Não devem reexecutar a
malha de otimização.

As trajetórias são desenhadas no plano vertical do poço: eixo horizontal
``s`` (afastamento a partir de ``P0`` no azimute do alvo) e eixo vertical
``z`` (profundidade). O fundo é a seção do modelo geológico nesse plano, de
modo que contatos inclinados e mudanças laterais de fácies aparecem como o
poço de fato os atravessa.
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import drilling.features.minimization.auxiliaries as ax
from drilling.features.minimization.geology import LITHOLOGY_COLORS, LITHOLOGY_NAMES
from drilling.features.minimization.minimal import DEFAULT_STYLE


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
    """Pontos de controle e curva de uma trajetória no plano do poço ``(s, z)``.

    Envolve ``auxiliaries.trajectory_plot_data`` e expõe as chaves no plano
    usadas pelos gráficos 2D: ``p0``, ``p1``, ``p2``, ``p3``, ``center``,
    ``command_start``, ``curve_x`` (afastamento ``s``) e ``curve_y``
    (profundidade).
    """
    base = ax.trajectory_plot_data(data, l1, radius)
    return {
        "config": base["config"],
        "p0": (base["p0_s"], base["p0_z"]),
        "p1": base["p1_plane"],
        "p2": base["p2_plane"],
        "p3": base["p3_plane"],
        "center": base["center_plane"],
        "command_start": base["command_start_plane"],
        "curve_x": base["curve_s"],
        "curve_y": base["curve_z"],
    }


def lithology_colormap():
    """Colormap discreto e norma indexados pelo código de litologia."""
    colors = [LITHOLOGY_COLORS.get(name, "#dddddd") for name in LITHOLOGY_NAMES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(LITHOLOGY_NAMES) + 0.5), cmap.N)
    return cmap, norm


def lithology_legend_handles(codes) -> list:
    """Patches de legenda para as litologias presentes em ``codes``."""
    present = sorted(int(code) for code in np.unique(codes))
    return [
        Patch(facecolor=LITHOLOGY_COLORS.get(LITHOLOGY_NAMES[code], "#dddddd"),
              edgecolor="none", label=LITHOLOGY_NAMES[code])
        for code in present
    ]


def plot_mesh_cross_section(
    axis,
    data,
    geological_mesh,
    s_range,
    z_range,
    ns: int = 500,
    nz: int = 500,
    alpha: float | None = None,
) -> list:
    """Desenha a seção geológica atrás de um gráfico de trajetória.

    Returns
    -------
    list
        Alças de legenda das litologias visíveis na seção.
    """
    if alpha is None:
        alpha = float(data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))
    s_values, z_values, codes = ax.mesh_cross_section(data, geological_mesh, s_range, z_range, ns, nz)
    cmap, norm = lithology_colormap()
    axis.pcolormesh(
        s_values, z_values, codes, cmap=cmap, norm=norm,
        alpha=alpha, shading="auto", zorder=0, rasterized=True,
    )
    return lithology_legend_handles(codes)


def prepare_mesh_axes(axis, data, geological_mesh, x_values, y_values) -> list:
    """Enquadra o gráfico no plano do poço e desenha a seção geológica ao fundo.

    ``x_values`` são afastamentos ``s`` e ``y_values`` profundidades.

    Returns
    -------
    list
        Alças de legenda das litologias visíveis.
    """
    margin_x = float(data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
    departure = float(data.departure)

    x_min = min(0.0, min(x_values) - 0.05 * max(departure, 1.0))
    x_max = max(max(x_values), departure) + margin_x
    y_min = min(0.0, float(geological_mesh.z_min))
    y_max = max(float(geological_mesh.z_max), float(data.P3[2]), max(y_values))

    handles = plot_mesh_cross_section(axis, data, geological_mesh, (x_min, x_max), (y_min, y_max))

    axis.set_aspect("equal")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.invert_yaxis()
    axis.set_xlabel("Horizontal distance in the well plane (m)")
    axis.set_ylabel("Depth (m)")
    axis.grid(alpha=0.25, linewidth=0.8)
    return handles


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

    p0 = (0.0, float(data.P0[2]))
    p3 = (float(data.departure), float(data.P3[2]))
    all_x = [p0[0], p3[0]]
    all_y = [p0[1], p3[1]]
    trajectory_data = {}

    for key, result in results.items():
        plot_data = trajectory_plot_data(data, result["l1"], result["R"])
        trajectory_data[key] = plot_data
        if key in visible_objectives:
            all_x.extend([plot_data["p1"][0], plot_data["p2"][0], *plot_data["curve_x"], p3[0]])
            all_y.extend([plot_data["p1"][1], plot_data["p2"][1], *plot_data["curve_y"], p3[1]])

    lithology_handles = prepare_mesh_axes(axis, data, geological_mesh, all_x, all_y)

    for key, plot_data in trajectory_data.items():
        if key not in visible_objectives:
            continue
        style = OBJECTIVE_STYLES[key]
        axis.plot(
            [p0[0], plot_data["p1"][0]],
            [p0[1], plot_data["p1"][1]],
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

    axis.scatter([p0[0]], [p0[1]], s=52, color="#111827", label="P0", zorder=5)
    axis.scatter([p3[0]], [p3[1]], s=58, color="#7c2d12", label="P3", zorder=5)
    axis.set_title("Optimized trajectories over the geological section", pad=12)
    handles = axis.get_legend_handles_labels()[0] + lithology_handles
    if handles:
        axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    figure.tight_layout()
    apply_chart_style(figure)


def plot_trajectory_with_mesh_3d(
    data,
    geological_mesh,
    l1: float,
    radius: float,
    title: str = "Type-1 trajectory in 3D",
    filename: str | None = None,
    max_nodes: int = 60,
):
    """Desenha o poço junto com os horizontes geológicos em coordenadas do mundo.

    Cada horizonte é uma superfície colorida pela fácies da camada logo abaixo,
    o que torna visíveis contatos inclinados ou dobrados e mudanças laterais de
    fácies; uma pilha de blocos opacos esconderia os dois.

    Parameters
    ----------
    filename : str or None, optional
        Caminho do arquivo a salvar. Sem ele a figura é exibida com ``plt.show``.

    Returns
    -------
    str or None
        Caminho absoluto do arquivo salvo, se houver.
    """
    import matplotlib.pyplot as plt

    plot_data = ax.trajectory_plot_data(data, l1, radius)
    alpha_2d = float(data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))
    alpha = float(data.drilling_time_parameters.get("mesh_plot_alpha_3d", min(0.45, 1.5 * alpha_2d)))

    fig = plt.figure(figsize=(11.0, 8.0))
    axis = fig.add_subplot(111, projection="3d")
    axis.computed_zorder = False

    grid = geological_mesh.grid
    step_x = max(1, int(np.ceil(grid.nx / max_nodes)))
    step_y = max(1, int(np.ceil(grid.ny / max_nodes)))
    X, Y = grid.meshgrid()
    X, Y = X[::step_y, ::step_x], Y[::step_y, ::step_x]

    cmap, norm = lithology_colormap()
    drawn_codes = []
    # Each horizon is drawn once, coloured by the layer immediately below it; the
    # base of the stack takes the colour of the last layer.
    for index in range(geological_mesh.n_layers + 1):
        codes = geological_mesh.layer_codes[min(index, geological_mesh.n_layers - 1)][::step_y, ::step_x]
        drawn_codes.append(codes)
        axis.plot_surface(
            X, Y, geological_mesh.horizons[index][::step_y, ::step_x],
            facecolors=cmap(norm(codes)), alpha=alpha, shade=False,
            linewidth=0.0, antialiased=True, zorder=1,
        )

    p0, p1, p2, p3 = plot_data["p0"], plot_data["p1"], plot_data["p2"], plot_data["p3"]
    well_zorder = 20
    halo = {"color": "white", "linewidth": 6.5, "solid_capstyle": "round", "zorder": well_zorder}
    axis.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], **halo)
    axis.plot(plot_data["curve_x"], plot_data["curve_y"], plot_data["curve_zw"], **halo)
    axis.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], **halo)
    axis.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
              color="#0b3c5d", linewidth=3.2, label="L1 - Vertical section", zorder=well_zorder + 1)
    axis.plot(plot_data["curve_x"], plot_data["curve_y"], plot_data["curve_zw"],
              color="#8c510a", linewidth=3.2, label="L2 - Curved section", zorder=well_zorder + 1)
    axis.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]],
              color="#1b5e20", linewidth=3.2, label="L3 - Inclined section", zorder=well_zorder + 1)
    axis.scatter([p0[0]], [p0[1]], [p0[2]], color="black", s=55, zorder=well_zorder + 2)
    axis.scatter([p3[0]], [p3[1]], [p3[2]], color="black", s=55, zorder=well_zorder + 2)
    axis.text(p0[0], p0[1], p0[2], "  P0", zorder=well_zorder + 2)
    axis.text(p3[0], p3[1], p3[2], "  P3", zorder=well_zorder + 2)

    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("Depth z (m)")
    axis.set_title(title)
    z_max = max(geological_mesh.z_max, float(data.P3[2]), float(p1[2]), float(p2[2]))
    axis.set_zlim(z_max, min(geological_mesh.z_min, float(data.P0[2])))
    handles = axis.get_legend_handles_labels()[0]
    handles += lithology_legend_handles(np.concatenate([c.ravel() for c in drawn_codes]))
    axis.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()

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
