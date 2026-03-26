import numpy as np
import pyvista as pv
from .logic import normalize

def get_plot_xylim(p1, pt, center, arc_points, margem=100):
    pontos_fixos = np.array([p1, pt, center])
    xs_fixos = pontos_fixos[:, 0]
    xs_arco = arc_points[:, 0]
    todos_xs = np.concatenate([xs_fixos, xs_arco])
    xmin = np.min(todos_xs)
    xmax = np.max(todos_xs)
    xlim = (np.round(xmin, 0) - margem, np.round(xmax, 0) + margem)
    
    ys_fixos = pontos_fixos[:, 1]
    ys_arco = arc_points[:, 1]
    todos_ys = np.concatenate([ys_fixos, ys_arco])
    ymin = np.min(todos_ys)
    ymax = np.max(todos_ys)
    ylim = (np.round(ymin, 0) - margem, np.round(ymax, 0) + margem)
    
    return xlim, ylim

def plot_case_1(plotter, result, show_project_trajectory, show_points_coordinates):
    plotter.clear()
    
    p1  = result["p1"]
    pt  = result["pt"]
    Pin = result["Pin"]
    Pbd = result["Pbd"]
    v   = result["v"]
    center  = result["center"]
    arc_pts = result["arc"]
    tp      = result["tangent_point"]

    xlim, ylim = get_plot_xylim(p1, pt, center, arc_pts)
    xlim = (-300, 1300)
    ylim = (-300, 1300)
    zlim = (np.floor(np.round(pt[2], 0) - 100), 100)
    grid_step = 100

    plotter.show_bounds(
        bounds=(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]),
        grid=False,
        location="outer",
        all_edges=True,
        ticks="outside",
        font_size=15,
        fmt="%.0f"
    )

    num_x = int(np.ceil((xlim[1] - xlim[0]) / grid_step)) + 1
    num_y = int(np.ceil((ylim[1] - ylim[0]) / grid_step)) + 1
    num_z = int(np.ceil((zlim[1] - zlim[0]) / grid_step)) + 1

    xs = np.linspace(xlim[0], xlim[1], num_x)
    ys = np.linspace(ylim[0], ylim[1], num_y)
    zs = np.linspace(zlim[0], zlim[1], num_z)

    y_back = ylim[1]
    x_side = xlim[0]

    for x in xs:
        plotter.add_mesh(pv.Line((x, ylim[0], zlim[0]), (x, ylim[1], zlim[0])), color="lightgray", line_width=1)
    for y in ys:
        plotter.add_mesh(pv.Line((xlim[0], y, zlim[0]), (xlim[1], y, zlim[0])), color="lightgray", line_width=1)

    for x in xs:
        plotter.add_mesh(pv.Line((x, y_back, zlim[0]), (x, y_back, zlim[1])), color="lightgray", line_width=1)
    for z in zs:
        plotter.add_mesh(pv.Line((xlim[0], y_back, z), (xlim[1], y_back, z)), color="lightgray", line_width=1)

    for y in ys:
        plotter.add_mesh(pv.Line((x_side, y, zlim[0]), (x_side, y, zlim[1])), color="lightgray", line_width=1)
    for z in zs:
        plotter.add_mesh(pv.Line((x_side, ylim[0], z), (x_side, ylim[1], z)), color="lightgray", line_width=1)

    legend_items = []

    plotter.add_mesh(pv.Spline(arc_pts, len(arc_pts)), color="blue", line_width=4)
    legend_items.append(("Arc segment", "blue"))

    if show_project_trajectory:
        vec_in = Pbd - Pin
        vec_out = pt - Pbd
        len_in = np.linalg.norm(vec_in)
        len_out = np.linalg.norm(vec_out)
        u_in = vec_in / len_in
        u_out = vec_out / len_out

        tangent_dist = min(200, len_in * 0.45, len_out * 0.45)
        p_curve_start = Pbd - u_in * tangent_dist
        p_curve_end = Pbd + u_out * tangent_dist

        t_vals = np.linspace(0, 1, 30)
        curve_pts = []
        for t in t_vals:
            pt_interp = (1 - t) ** 2 * p_curve_start + 2 * (1 - t) * t * Pbd + t ** 2 * p_curve_end
            curve_pts.append(pt_interp)

        full_path = [Pin] + curve_pts + [pt]
        plotter.add_mesh(pv.lines_from_points(np.array(full_path)), color="brown", line_width=4)
        legend_items.append(("Project trajectory", "brown"))

        R_traj = np.linalg.norm(pt - p1) * 0.015
        plotter.add_mesh(pv.Sphere(R_traj, p_curve_start), color="black")
        plotter.add_mesh(pv.Sphere(R_traj, p_curve_end), color="black")

    for P in arc_pts:
        plotter.add_mesh(pv.Line(center, P), color="gray", line_width=1, opacity=0.6)

    R = np.linalg.norm(pt - p1) * 0.015

    plotter.add_mesh(pv.Sphere(R, p1), color="orange")
    plotter.add_mesh(pv.Sphere(R, pt), color="red")
    plotter.add_mesh(pv.Sphere(R, tp), color="cyan")
    plotter.add_mesh(pv.Sphere(R, center), color="gray", opacity=0.6)
    plotter.add_mesh(pv.Sphere(R, Pin), color="black")

    v_dir = normalize(v)
    v_scale = np.linalg.norm(pt - p1) * 0.25

    arrow = pv.Arrow(start=p1, direction=v_dir, scale=v_scale, tip_length=0.25, tip_radius=0.06, shaft_radius=0.02)
    plotter.add_mesh(arrow, color="purple")
    legend_items.append(("P1 direction vector", "purple"))

    def label_point(P, name, offset=(75, 0, 0)):
        txt = f"{name}\n({P[0]:.0f}, {P[1]:.0f}, {P[2]:.0f})"
        plotter.add_point_labels([P + np.array(offset)], [txt], font_size=11, shape_opacity=0.0)

    if show_points_coordinates:
        for P, n in [(Pin, "Pin"), (Pbd, "Pbd"), (p1, "p1"), (tp, "TP"), (pt, "pt"), (center, "Centro")]:
            label_point(P, n)

    plotter.camera.position = (0, -6000, -3000)
    plotter.camera.focal_point = (0, 0, -1500)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.parallel_projection = True

    plotter.add_text("Case A 3D trajectory", font_size=15)

    legend_items.append(("Intermediate Control Point", "orange", "circle"))
    legend_items.append(("Target Point", "red", "circle"))
    legend_items.append(("Project points", "black", "circle"))

    plotter.add_legend(legend_items, bcolor="white", face="rectangle", size=(0.3, 0.3), loc="upper left")

def plot_case_2(plotter, result, show_project_trajectory, show_points_coordinates):
    plotter.clear()

    Pin = result["Pin"]
    Pbd = result["Pbd"]
    p1  = result["p1"]
    pt  = result["pt"]
    v   = result["v"]
    center = result["center"]
    arc_pts = result["arc"]
    tp = result["tangent_point"]

    xlim, ylim = get_plot_xylim(p1, pt, center, arc_pts)
    xlim = (-200, 1100)
    ylim = (-400, 700)
    zlim = (np.floor(np.round(pt[2], 0) - 100), 100)
    grid_step = 100

    plotter.show_bounds(
        bounds=(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]),
        grid=False,
        location="outer",
        all_edges=True,
        ticks="outside",
        font_size=15,
        fmt="%.0f"
    )

    num_x = int(np.ceil((xlim[1] - xlim[0]) / grid_step)) + 1
    num_y = int(np.ceil((ylim[1] - ylim[0]) / grid_step)) + 1
    num_z = int(np.ceil((zlim[1] - zlim[0]) / grid_step)) + 1

    xs = np.linspace(xlim[0], xlim[1], num_x)
    ys = np.linspace(ylim[0], ylim[1], num_y)
    zs = np.linspace(zlim[0], zlim[1], num_z)

    y_back = ylim[0]
    x_side = xlim[0]

    for x in xs:
        plotter.add_mesh(pv.Line((x, ylim[0], zlim[0]), (x, ylim[1], zlim[0])), color="lightgray", line_width=1)
    for y in ys:
        plotter.add_mesh(pv.Line((xlim[0], y, zlim[0]), (xlim[1], y, zlim[0])), color="lightgray", line_width=1)

    for x in xs:
        plotter.add_mesh(pv.Line((x, y_back, zlim[0]), (x, y_back, zlim[1])), color="lightgray", line_width=1)
    for z in zs:
        plotter.add_mesh(pv.Line((xlim[0], y_back, z), (xlim[1], y_back, z)), color="lightgray", line_width=1)

    for y in ys:
        plotter.add_mesh(pv.Line((x_side, y, zlim[0]), (x_side, y, zlim[1])), color="lightgray", line_width=1)
    for z in zs:
        plotter.add_mesh(pv.Line((x_side, ylim[0], z), (x_side, ylim[1], z)), color="lightgray", line_width=1)

    legend_items = []

    plotter.add_mesh(pv.Spline(arc_pts, len(arc_pts)), color="blue", line_width=4)
    legend_items.append(("Arc segment", "blue"))

    plotter.add_mesh(pv.Line(tp, pt), color="green", line_width=4)
    legend_items.append(("Straight segment", "green"))

    if show_project_trajectory:
        vec_in = Pbd - Pin
        vec_out = pt - Pbd
        len_in = np.linalg.norm(vec_in)
        len_out = np.linalg.norm(vec_out)
        u_in = vec_in / len_in
        u_out = vec_out / len_out

        tangent_dist = min(200, len_in * 0.45, len_out * 0.45)
        p_curve_start = Pbd - u_in * tangent_dist
        p_curve_end = Pbd + u_out * tangent_dist

        t_vals = np.linspace(0, 1, 30)
        curve_pts = []
        for t in t_vals:
            pt_interp = (1 - t) ** 2 * p_curve_start + 2 * (1 - t) * t * Pbd + t ** 2 * p_curve_end
            curve_pts.append(pt_interp)

        full_path = [Pin] + curve_pts + [pt]
        plotter.add_mesh(pv.lines_from_points(np.array(full_path)), color="brown", line_width=4)
        legend_items.append(("Project trajectory", "brown"))

        R_traj = np.linalg.norm(pt - p1) * 0.015
        plotter.add_mesh(pv.Sphere(R_traj, p_curve_start), color="black")
        plotter.add_mesh(pv.Sphere(R_traj, p_curve_end), color="black")

    for P in arc_pts:
        plotter.add_mesh(pv.Line(center, P), color="gray", line_width=1, opacity=0.6)

    R = np.linalg.norm(pt - p1) * 0.015

    plotter.add_mesh(pv.Sphere(R, p1), color="orange")
    plotter.add_mesh(pv.Sphere(R, pt), color="red")
    plotter.add_mesh(pv.Sphere(R, tp), color="cyan")
    plotter.add_mesh(pv.Sphere(R, center), color="gray", opacity=0.6)
    plotter.add_mesh(pv.Sphere(R, Pin), color="black")

    v_dir = normalize(v)
    v_scale = np.linalg.norm(pt - p1) * 0.25

    arrow = pv.Arrow(start=p1, direction=v_dir, scale=v_scale, tip_length=0.25, tip_radius=0.06, shaft_radius=0.02)
    plotter.add_mesh(arrow, color="purple")
    legend_items.append(("P1 directional vector", "purple"))

    def label_point(P, name, offset=(75, 0, 0)):
        txt = f"{name})"
        plotter.add_point_labels([P + np.array(offset)], [txt], font_size=11, shape_opacity=0.0)

    if show_points_coordinates:
        for P, n in [(Pin, "Pin"), (p1, "p1"), (tp, "TP"), (pt, "pt"), (center, "Centro")]:
            label_point(P, n)

    plotter.camera.position = (0, -6000, -3000)
    plotter.camera.focal_point = (0, 0, -1500)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.parallel_projection = True

    plotter.add_text("Case B 3D trajectory", font_size=15)

    legend_items.append(("Intermediate Control Point", "orange", "circle"))
    legend_items.append(("Tangency Point", "cyan", "circle"))
    legend_items.append(("Target Point", "red", "circle"))
    legend_items.append(("Project points", "black", "circle"))

    plotter.add_legend(legend_items, bcolor="white", face="rectangle", size=(0.2, 0.3), loc="upper left")