"""Gráficos Matplotlib da aba Minimization.

Esses plotters recebem resultados já calculados. Não devem reexecutar a
malha de otimização. A geologia pode ser a ``mesh`` de intervalos ou a
``GridGeology`` lida de um GRDECL; só a segunda tem vista 3D.
"""
import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import drilling.features.minimization.auxiliaries as ax
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


GRID_LITHOLOGY_COLORS = ["#d8b365", "#5ab4ac", "#8c8c8c", "#c2a5cf", "#bdb76b", "#a6611a", "#80cdc1", "#f6e8c3"]
"""Cores de fundo das litologias de uma malha GRDECL, na ordem em que aparecem no arquivo."""


def lithology_color(geological_mesh, name: str) -> str:
    """Cor de uma litologia: nomes fixos da ``mesh`` ou posição na lista da malha GRDECL."""
    if name in ax.LITHOLOGY_COLORS:
        return ax.LITHOLOGY_COLORS[name]
    names = list(getattr(geological_mesh, "lithology_names", []))
    if name in names:
        return GRID_LITHOLOGY_COLORS[names.index(name) % len(GRID_LITHOLOGY_COLORS)]
    return "#d1d5db"


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

    if hasattr(geological_mesh, "section_polygons"):
        # Desenha na ordem das litologias da malha, para a legenda seguir a da vista 3D.
        order = {name: index for index, name in enumerate(geological_mesh.lithology_names)}
        polygons = sorted(geological_mesh.section_polygons(), key=lambda polygon: order[polygon["lithology"]])
        y_min = min([0.0] + [min(polygon["depth"]) for polygon in polygons])
        y_max = max([data.P3[1], max(y_values)] + [max(polygon["depth"]) for polygon in polygons])
        used_labels = set()
        for polygon in polygons:
            name = polygon["lithology"]
            label = name if name not in used_labels else None
            used_labels.add(name)
            axis.fill(
                polygon["horizontal"],
                polygon["depth"],
                facecolor=lithology_color(geological_mesh, name),
                edgecolor="white",
                linewidth=0.6,
                alpha=alpha,
                label=label,
                zorder=0,
            )
    else:
        y_min = 0.0
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
    axis.set_ylim(y_min, y_max)
    axis.invert_yaxis()
    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("Depth below wellhead (m)")
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

    axis.scatter([data.P0[0]], [data.P0[1]], s=52, color="#111827", label="Wellhead", zorder=5)
    axis.scatter([data.P3[0]], [data.P3[1]], s=58, color="#7c2d12", label="Target", zorder=5)
    axis.set_title("Optimized trajectories in the well plane", pad=12)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    figure.tight_layout()
    apply_chart_style(figure)


def trajectory_polyline(data, l1: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Pontos ``(horizontal, depth)`` do poço completo: vertical, curva e tangente."""
    plot_data = trajectory_plot_data(data, l1, radius)
    horizontal = np.concatenate([[data.P0[0]], plot_data["curve_x"], [plot_data["p3"][0]]])
    depth = np.concatenate([[data.P0[1]], plot_data["curve_y"], [plot_data["p3"][1]]])
    return horizontal, depth


_FACE_NEIGHBORS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
"""Deslocamento ``(dk, dj, di)`` da célula vizinha de cada face de ``_cell_faces``."""

CELL_ALPHA_3D = 0.2
"""Opacidade das células na vista 3D: baixa, para as trajetórias aparecerem através delas."""


def _cell_faces(corners) -> list:
    """As 6 faces de uma célula corner-point ``(2, 2, 2, 3)``: topo, base, J-, J+, I-, I+."""
    c = corners
    return [
        [c[0, 0, 0], c[0, 0, 1], c[0, 1, 1], c[0, 1, 0]],
        [c[1, 0, 0], c[1, 0, 1], c[1, 1, 1], c[1, 1, 0]],
        [c[0, 0, 0], c[0, 0, 1], c[1, 0, 1], c[1, 0, 0]],
        [c[0, 1, 0], c[0, 1, 1], c[1, 1, 1], c[1, 1, 0]],
        [c[0, 0, 0], c[0, 1, 0], c[1, 1, 0], c[1, 0, 0]],
        [c[0, 0, 1], c[0, 1, 1], c[1, 1, 1], c[1, 0, 1]],
    ]


def _visible_faces(grid, cells, z_range) -> tuple[list, list, list]:
    """Faces na borda do bloco ou entre litologias diferentes, cortadas em ``z_range``.

    Faces entre duas células iguais ficam de fora: empilhar faces translúcidas
    deixaria o bloco opaco e esconderia as trajetórias.
    """
    selected = set(cells)
    faces, codes, points = [], [], []
    for cell in cells:
        k, j, i = cell
        code = int(grid.lithology[k, j, i])
        corners = grid.cell_corners(k, j, i)
        corners[..., 2] = np.clip(corners[..., 2], *z_range)
        points.append(corners.reshape(-1, 3))
        for face, (dk, dj, di) in zip(_cell_faces(corners), _FACE_NEIGHBORS):
            neighbor = (k + dk, j + dj, i + di)
            if neighbor in selected:
                if int(grid.lithology[neighbor]) == code or neighbor < cell:
                    continue
            faces.append(face)
            codes.append(code)
    return faces, codes, points


def plot_trajectories_3d(figure, data, geology, results, visible_objectives=None, zoom_to_grid=False):
    """Vista 3D do trecho da malha em volta do poço e das trajetórias.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figura a redesenhar.
    data : DataSet
        Geometria no plano do poço (``P0`` na cabeça, ``P3`` no alvo).
    geology : GridGeology
        Malha posicionada; fornece ``to_world`` e ``cells_around_well``.
    results : dict
        Resultados de ``calculate_minimization`` por objetivo.
    visible_objectives : list of str, optional
        Objetivos a desenhar; o padrão é todos.
    zoom_to_grid : bool, optional
        Enquadra só as células da malha e corta as trajetórias nesse recorte;
        útil quando o reservatório é fino perto do comprimento do poço.
    """
    clear_figure(figure)
    # Ordem de desenho fixa (células atrás, trajetórias na frente): a ordenação
    # automática do mplot3d esconde linhas atrás de faces translúcidas.
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    visible_objectives = list(results.keys()) if visible_objectives is None else list(visible_objectives)

    cells, z_range = geology.cells_around_well()
    grid = getattr(geology, "grid", None)
    cell_points = []
    if grid is not None and cells:
        faces, codes, cell_points = _visible_faces(grid, cells, z_range)
        colors = [lithology_color(geology, grid.lithology_names[code]) for code in codes]
        collection = Poly3DCollection(faces, facecolors=colors, edgecolors=(1, 1, 1, 0.35), linewidths=0.3, alpha=CELL_ALPHA_3D)
        collection.set_zorder(1)
        axis.add_collection3d(collection)
        for index in sorted(set(codes)):
            name = grid.lithology_names[index]
            axis.plot([], [], [], color=lithology_color(geology, name), linewidth=8, alpha=0.6, label=name)
        cell_points = np.concatenate(cell_points)

    zoom = zoom_to_grid and len(cell_points) > 0
    if zoom:
        box = (cell_points.min(axis=0), cell_points.max(axis=0))
        points = [*box]
    else:
        points = [geology.wellhead, geology.target, *cell_points]

    for key, result in results.items():
        if key not in visible_objectives:
            continue
        horizontal, depth = trajectory_polyline(data, result["l1"], result["R"])
        if zoom:
            # Reamostra a cada metro para o corte no recorte não pular o trecho dentro dele.
            length = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(horizontal), np.diff(depth)))])
            fine = np.linspace(0.0, length[-1], max(2, int(length[-1]) + 1))
            horizontal, depth = np.interp(fine, length, horizontal), np.interp(fine, length, depth)
        xyz = geology.to_world(horizontal, depth)
        if zoom:
            outside = np.any((xyz < box[0]) | (xyz > box[1]), axis=1)
            xyz[outside] = np.nan
        style = OBJECTIVE_STYLES[key]
        axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=style["color"], linewidth=2.6, label=style["label"], zorder=10)

    for point, color, label in ((geology.wellhead, "#111827", "Wellhead"), (geology.target, "#7c2d12", "Target")):
        if not zoom or np.all((point >= box[0]) & (point <= box[1])):
            axis.scatter(*point, s=52, color=color, label=label, depthshade=False, zorder=11)

    points = np.asarray(points, dtype=float)
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(upper[2], lower[2])
    # Proporção real, mas sem deixar um eixo fino demais para ser lido.
    aspect = np.maximum(span / span.max(), 0.2)
    axis.set_box_aspect(aspect)
    for axis_, size in zip((axis.xaxis, axis.yaxis, axis.zaxis), aspect):
        axis_.set_major_locator(MaxNLocator(int(3 + 4 * size)))
    axis.set_xlabel("X (m)", labelpad=10)
    axis.set_ylabel("Y (m)", labelpad=10)
    axis.set_zlabel("Depth (m)", labelpad=12)
    axis.set_title("Grid section crossed by the well" if zoom else "Optimized trajectories in the reservoir grid", pad=12)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.16, top=0.94)
    figure.legend(*axis.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=True)
    figure.patch.set_facecolor("#ffffff")


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
