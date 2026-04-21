import numpy as np
from scipy.optimize import fsolve


def _normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def dls_to_radius(dls_val, ref_length=30.0):
    if dls_val <= 0:
        return np.inf
    return ref_length / np.radians(dls_val)


def rodrigues_rotation(v, k, theta):
    v = np.asarray(v)
    k = normalize(np.asarray(k))
    return (
            v * np.cos(theta)
            + np.cross(k, v) * np.sin(theta)
            + k * np.dot(k, v) * (1 - np.cos(theta))
    )


def generate_initial_guesses(p1, pt, v, n_random=3, scale=1.0):
    mid = 0.5 * (p1 + pt)
    guesses = [mid]
    vn = normalize(v)
    for mag in (0.5 * scale, scale):
        guesses.append(mid + mag * vn)
        guesses.append(mid - mag * vn)
    rng = np.random.default_rng(42)
    for _ in range(n_random):
        guesses.append(mid + rng.normal(scale=0.2 * scale, size=3))
    return guesses


def find_arc_center_radius(p1, pt, v):
    def F(c):
        r1 = p1 - c
        r2 = pt - c
        return [
            np.dot(r1, r1) - np.dot(r2, r2),
            np.dot(v, r1),
            np.dot(c - p1, np.cross(pt - p1, v))
        ]

    scale = np.linalg.norm(pt - p1) + 1.0
    for g in generate_initial_guesses(p1, pt, v, 6, scale):
        sol, _, ier, _ = fsolve(F, g, full_output=True)
        if ier == 1:
            return sol, np.linalg.norm(p1 - sol)
    raise RuntimeError("Solver did not converge")


def generate_arc_points(p1, pt, center, radius, n=300):
    u = normalize(p1 - center)
    ptv = pt - center
    w = ptv - np.dot(ptv, u) * u
    if np.linalg.norm(w) < 1e-9:
        w = np.cross(u, [0, 0, 1])
    w = normalize(w)
    ang = _normalize_angle(np.arctan2(
        np.dot(w, ptv / radius),
        np.dot(u, ptv / radius)
    ))
    t = np.linspace(0, ang, n)
    arc = center + radius * (
            np.outer(np.cos(t), u) + np.outer(np.sin(t), w)
    )
    return arc, ang


def validate_trajectory_case1(p1, pt, v, arc, turn_angle, max_ang_deg=70):
    status = []
    status.append(("Deeper Target", pt[2] <= p1[2], f"{pt[2]:.0f}", f"≤ {p1[2]:.0f}"))
    dot = np.dot(v, pt - p1)
    status.append(("General Direction", dot > 0, f"{dot:.2f}", "> 0"))
    ang_deg = abs(np.degrees(turn_angle))
    status.append(("Total Angle", ang_deg <= max_ang_deg, f"{ang_deg:.1f}°", f"≤ {max_ang_deg}°"))
    arc_z_max = arc[:, 2].max()
    status.append(("No Climb", arc_z_max <= p1[2] + 1.0, f"{arc_z_max:.1f}", f"≤ {p1[2]:.0f}"))
    full_path = np.vstack((arc, pt))
    diffs = full_path[1:] - full_path[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    valid_idx = norms > 1e-6
    tangents = diffs[valid_idx] / norms[valid_idx, np.newaxis]
    max_inc = np.degrees(np.arccos(np.clip(np.abs(tangents[:, 2]), -1.0, 1.0))).max()
    status.append(("Max Inclination", max_inc <= 60.0, f"{max_inc:.1f}°", "≤ 60.0°"))
    return status


def solve_case1(Pin, Pbd, p1, pt, v):
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v = normalize(np.asarray(v, float))
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    center, radius = find_arc_center_radius(p1, pt, v)
    arc, turn_angle = generate_arc_points(p1, pt, center, radius)
    arc_length = radius * abs(turn_angle)
    status = validate_trajectory_case1(p1, pt, v, arc, turn_angle)

    return {
        "Pin": Pin,
        "Pbd": Pbd,
        "p1": p1,
        "pt": pt,
        "v": v,
        "center": center,
        "radius": radius,
        "arc": arc,
        "tangent_point": arc[-1],
        "turn_angle": turn_angle,
        "arc_length": arc_length,
        "status": status
    }


def compute_case2_trajectory(p1, pt, v_init, dls_deg=3.0):
    radius = dls_to_radius(dls_deg)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v_init = normalize(np.asarray(v_init, float))
    vec_p1_pt = pt - p1
    plane_normal = np.cross(v_init, vec_p1_pt)

    if np.linalg.norm(plane_normal) < 1e-6:
        return {
            "arc": np.array([p1]),
            "line": np.vstack((p1, pt)),
            "center": p1,
            "tangent_point": p1,
            "radius": np.inf,
            "turn_angle": 0.0
        }

    plane_normal = normalize(plane_normal)
    radius_dir = np.cross(plane_normal, v_init)
    center = p1 + radius * radius_dir
    vec_c_pt = pt - center
    dist_c_pt = np.linalg.norm(vec_c_pt)

    if dist_c_pt < radius:
        raise ValueError("Impossible geometry: target inside radius.")

    angle_offset = np.arccos(radius / dist_c_pt)
    vec_c_tan_dir = rodrigues_rotation(
        normalize(vec_c_pt), plane_normal, -angle_offset
    )
    tp = center + radius * vec_c_tan_dir
    u = normalize(p1 - center)
    w = normalize(np.cross(plane_normal, u))
    vec_c_tan = normalize(tp - center)
    cos_ang = np.dot(u, vec_c_tan)
    sin_ang = np.dot(w, vec_c_tan)
    angle_total = np.arctan2(sin_ang, cos_ang)

    if angle_total < 0:
        angle_total += 2 * np.pi

    thetas = np.linspace(0, angle_total, 120)
    arc = center + radius * (
            np.outer(np.cos(thetas), u)
            + np.outer(np.sin(thetas), w)
    )

    return {
        "arc": arc,
        "line": np.vstack((tp, pt)),
        "center": center,
        "tangent_point": tp,
        "radius": radius,
        "turn_angle": angle_total
    }


def validate_trajectory_case2(p1, pt, v_init, result, max_ang_deg=70):
    status = []
    status.append(("Deeper Target", pt[2] <= p1[2], f"Zt: {pt[2]:.0f}", f"≤ {p1[2]:.0f}"))
    dot = np.dot(v_init, pt - p1)
    status.append(("General Direction", dot > 0, f"{dot:.2f}", "> 0"))
    ang_deg = np.degrees(result["turn_angle"])
    status.append(("Total Angle", ang_deg <= max_ang_deg, f"{ang_deg:.1f}°", f"≤ {max_ang_deg}°"))
    arc_z_max = result["arc"][:, 2].max()
    status.append(("No Climb", arc_z_max <= p1[2] + 1.0, f"Zmax: {arc_z_max:.1f}", f"≤ {p1[2]:.0f}"))
    full_path = np.vstack((result["arc"], pt))
    diffs = full_path[1:] - full_path[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    valid_idx = norms > 1e-6
    tangents = diffs[valid_idx] / norms[valid_idx, np.newaxis]
    max_inc = np.degrees(np.arccos(np.clip(np.abs(tangents[:, 2]), -1.0, 1.0))).max()
    status.append(("Max Inclination", max_inc <= 60.0, f"{max_inc:.1f}°", "≤ 60.0°"))
    return status


def solve_case2(Pin, Pbd, p1, pt, v):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v = normalize(np.asarray(v, float))
    dls_deg = 3.0
    result = compute_case2_trajectory(p1, pt, v, dls_deg)

    if np.isinf(result["radius"]):
        total_length = np.linalg.norm(pt - p1)
    else:
        len_arc = result["radius"] * result["turn_angle"]
        len_line = np.linalg.norm(pt - result["tangent_point"])
        total_length = len_arc + len_line

    status = validate_trajectory_case2(p1, pt, v, result)

    return {
        "Pin": Pin,
        "Pbd": Pbd,
        "p1": p1,
        "pt": pt,
        "v": v,
        "center": result["center"],
        "radius": result["radius"],
        "arc": result["arc"],
        "tangent_point": result["tangent_point"],
        "turn_angle": result["turn_angle"],
        "total_length": total_length,
        "status": status
    }

def project_direction(Pin, Pbd):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    return normalize(Pbd - Pin)


def angle_between(u, v):
    u = normalize(np.asarray(u, float))
    v = normalize(np.asarray(v, float))
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arccos(c)


def signed_angle_on_plane(u, v, plane_normal):
    u = normalize(np.asarray(u, float))
    v = normalize(np.asarray(v, float))
    plane_normal = normalize(np.asarray(plane_normal, float))
    cross_uv = np.cross(u, v)
    sin_val = np.dot(plane_normal, cross_uv)
    cos_val = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arctan2(sin_val, cos_val)


def build_alignment_arc(p_start, v_start, v_end, radius, n=120):
    p_start = np.asarray(p_start, float)
    v_start = normalize(np.asarray(v_start, float))
    v_end = normalize(np.asarray(v_end, float))

    if np.isinf(radius):
        return {
            "arc": np.array([p_start]),
            "center": p_start,
            "radius": np.inf,
            "turn_angle": 0.0,
            "arc_length": 0.0,
            "end_point": p_start,
            "end_tangent": v_end,
            "plane_normal": np.zeros(3)
        }

    plane_normal = np.cross(v_start, v_end)
    norm_plane = np.linalg.norm(plane_normal)

    if norm_plane < 1e-9:
        if np.dot(v_start, v_end) > 0:
            return {
                "arc": np.array([p_start]),
                "center": p_start,
                "radius": radius,
                "turn_angle": 0.0,
                "arc_length": 0.0,
                "end_point": p_start,
                "end_tangent": v_end,
                "plane_normal": np.zeros(3)
            }
        aux = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(aux, v_start)) > 0.9:
            aux = np.array([0.0, 1.0, 0.0])
        plane_normal = normalize(np.cross(v_start, aux))
    else:
        plane_normal = plane_normal / norm_plane

    radius_dir = normalize(np.cross(plane_normal, v_start))
    center = p_start + radius * radius_dir

    turn_angle = signed_angle_on_plane(v_start, v_end, plane_normal)

    if turn_angle < 0:
        plane_normal = -plane_normal
        radius_dir = normalize(np.cross(plane_normal, v_start))
        center = p_start + radius * radius_dir
        turn_angle = signed_angle_on_plane(v_start, v_end, plane_normal)

    r0 = p_start - center
    thetas = np.linspace(0.0, turn_angle, n)
    arc = np.array([center + rodrigues_rotation(r0, plane_normal, th) for th in thetas])
    end_point = arc[-1]
    arc_length = radius * abs(turn_angle)

    return {
        "arc": arc,
        "center": center,
        "radius": radius,
        "turn_angle": turn_angle,
        "arc_length": arc_length,
        "end_point": end_point,
        "end_tangent": v_end,
        "plane_normal": plane_normal
    }


def compute_case2_from_start(p_start, pt, v_start, dls_deg=3.0):
    p_start = np.asarray(p_start, float)
    pt = np.asarray(pt, float)
    v_start = normalize(np.asarray(v_start, float))
    radius = dls_to_radius(dls_deg)

    vec_start_target = pt - p_start
    plane_normal = np.cross(v_start, vec_start_target)

    if np.linalg.norm(plane_normal) < 1e-6:
        return {
            "arc": np.array([p_start]),
            "line": np.vstack((p_start, pt)),
            "center": p_start,
            "tangent_point": p_start,
            "radius": np.inf,
            "turn_angle": 0.0,
            "arc_length": 0.0,
            "line_length": np.linalg.norm(pt - p_start),
            "total_length": np.linalg.norm(pt - p_start)
        }

    plane_normal = normalize(plane_normal)
    radius_dir = np.cross(plane_normal, v_start)
    center = p_start + radius * radius_dir
    vec_c_pt = pt - center
    dist_c_pt = np.linalg.norm(vec_c_pt)

    if dist_c_pt < radius:
        raise ValueError("Impossible geometry: target inside radius.")

    angle_offset = np.arccos(radius / dist_c_pt)
    vec_c_tan_dir = rodrigues_rotation(
        normalize(vec_c_pt), plane_normal, -angle_offset
    )
    tp = center + radius * vec_c_tan_dir

    u = normalize(p_start - center)
    w = normalize(np.cross(plane_normal, u))
    vec_c_tan = normalize(tp - center)

    cos_ang = np.dot(u, vec_c_tan)
    sin_ang = np.dot(w, vec_c_tan)
    angle_total = np.arctan2(sin_ang, cos_ang)

    if angle_total < 0:
        angle_total += 2 * np.pi

    thetas = np.linspace(0, angle_total, 120)
    arc = center + radius * (
        np.outer(np.cos(thetas), u) + np.outer(np.sin(thetas), w)
    )

    arc_length = radius * angle_total
    line_length = np.linalg.norm(pt - tp)
    total_length = arc_length + line_length

    return {
        "arc": arc,
        "line": np.vstack((tp, pt)),
        "center": center,
        "tangent_point": tp,
        "radius": radius,
        "turn_angle": angle_total,
        "arc_length": arc_length,
        "line_length": line_length,
        "total_length": total_length
    }

def compute_initial_alignment(Pin, Pbd, p1, v, dls_deg=3.0, alpha_max_deg=None):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    v = normalize(np.asarray(v, float))

    u_proj = project_direction(Pin, Pbd)
    radius = dls_to_radius(dls_deg)

    alpha = angle_between(v, u_proj)
    alpha_deg = np.degrees(alpha)

    if alpha_max_deg is not None and alpha_deg > alpha_max_deg:
        raise ValueError(
            f"Initial alignment angle exceeds limit: {alpha_deg:.2f}° > {alpha_max_deg:.2f}°"
        )

    alignment = build_alignment_arc(
        p_start=p1,
        v_start=v,
        v_end=u_proj,
        radius=radius
    )

    p_align = np.asarray(alignment["end_point"], float)

    return {
        "project_direction": u_proj,
        "alignment_arc": alignment["arc"],
        "alignment_center": alignment["center"],
        "alignment_radius": alignment["radius"],
        "alignment_turn_angle": alignment["turn_angle"],
        "alignment_turn_angle_deg": alpha_deg,
        "alignment_arc_length": alignment["arc_length"],
        "alignment_end_point": p_align,
        "alignment_end_tangent": u_proj
    }

def point_along_direction(p0, direction, distance):
    p0 = np.asarray(p0, float)
    direction = normalize(np.asarray(direction, float))
    return p0 + distance * direction

def distance_to_reach_z(p0, direction, z_target):
    p0 = np.asarray(p0, float)
    direction = normalize(np.asarray(direction, float))

    if abs(direction[2]) < 1e-9:
        return None

    s = (z_target - p0[2]) / direction[2]

    if s < 0:
        return None

    return float(s)

def compute_preferred_start_distance(p_align, u_proj, Pbd):
    p_align = np.asarray(p_align, float)
    u_proj = normalize(np.asarray(u_proj, float))
    Pbd = np.asarray(Pbd, float)

    return distance_to_reach_z(p_align, u_proj, Pbd[2])

def try_case2_candidate(p_align, u_proj, pt, s, dls_deg=3.0):
    p_align = np.asarray(p_align, float)
    u_proj = normalize(np.asarray(u_proj, float))
    pt = np.asarray(pt, float)

    p_start_main = point_along_direction(p_align, u_proj, s)
    result = compute_case2_from_start(
        p_start=p_start_main,
        pt=pt,
        v_start=u_proj,
        dls_deg=dls_deg
    )

    hold_line = np.vstack((p_align, p_start_main))
    hold_length = np.linalg.norm(p_start_main - p_align)

    total_length = hold_length + result["total_length"]

    return {
        "start_point": p_start_main,
        "hold_line": hold_line,
        "hold_length": hold_length,
        "main_curve": result,
        "total_length": total_length
    }

def search_best_case3_start(
    p_align,
    u_proj,
    pt,
    Pbd,
    dls_deg=3.0,
    step=30.0,
    min_s=0.0
):
    p_align = np.asarray(p_align, float)
    u_proj = normalize(np.asarray(u_proj, float))
    pt = np.asarray(pt, float)
    Pbd = np.asarray(Pbd, float)

    s_pref = compute_preferred_start_distance(p_align, u_proj, Pbd)

    if s_pref is None:
        s_pref = 0.0

    s_pref = max(float(s_pref), float(min_s))

    candidates = []
    s = s_pref

    while s >= min_s - 1e-9:
        try:
            candidate = try_case2_candidate(
                p_align=p_align,
                u_proj=u_proj,
                pt=pt,
                s=s,
                dls_deg=dls_deg
            )
            candidate["s"] = float(s)
            candidates.append(candidate)
        except Exception:
            pass

        s -= step

    if not candidates:
        raise ValueError("No feasible start point found for case 3 main curve.")

    candidates.sort(key=lambda c: (-c["s"], c["total_length"]))
    best = candidates[0]

    return {
        "preferred_s": s_pref,
        "tested_candidates": candidates,
        "best_candidate": best
    }

def compute_case3_trajectory(
    Pin,
    Pbd,
    p1,
    pt,
    v,
    dls_deg=3.0,
    alpha_max_deg=None,
    step=30.0
):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v = normalize(np.asarray(v, float))

    alignment = compute_initial_alignment(
        Pin=Pin,
        Pbd=Pbd,
        p1=p1,
        v=v,
        dls_deg=dls_deg,
        alpha_max_deg=alpha_max_deg
    )

    p_align = alignment["alignment_end_point"]
    u_proj = alignment["alignment_end_tangent"]

    search = search_best_case3_start(
        p_align=p_align,
        u_proj=u_proj,
        pt=pt,
        Pbd=Pbd,
        dls_deg=dls_deg,
        step=step,
        min_s=0.0
    )

    best = search["best_candidate"]
    main = best["main_curve"]

    total_length = (
        alignment["alignment_arc_length"]
        + best["hold_length"]
        + main["total_length"]
    )

    return {
        "Pin": Pin,
        "Pbd": Pbd,
        "p1": p1,
        "pt": pt,
        "v": v,
        "project_direction": alignment["project_direction"],
        "alignment_arc": alignment["alignment_arc"],
        "alignment_center": alignment["alignment_center"],
        "alignment_radius": alignment["alignment_radius"],
        "alignment_turn_angle": alignment["alignment_turn_angle"],
        "alignment_turn_angle_deg": alignment["alignment_turn_angle_deg"],
        "alignment_arc_length": alignment["alignment_arc_length"],
        "alignment_end_point": alignment["alignment_end_point"],
        "hold_line": best["hold_line"],
        "hold_length": best["hold_length"],
        "main_start_point": best["start_point"],
        "main_arc": main["arc"],
        "main_line": main["line"],
        "main_center": main["center"],
        "main_radius": main["radius"],
        "main_tangent_point": main["tangent_point"],
        "main_turn_angle": main["turn_angle"],
        "main_arc_length": main["arc_length"],
        "main_line_length": main["line_length"],
        "preferred_s": search["preferred_s"],
        "selected_s": best["s"],
        "total_length": total_length,
        "tested_candidates": search["tested_candidates"]
    }

def validate_trajectory_case3(
    Pin,
    Pbd,
    p1,
    pt,
    v,
    result,
    alpha1_max_deg=20.0,
    alpha_main_max_deg=70.0,
    total_angle_max_deg=90.0,
    max_inclination_deg=60.0
):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v = normalize(np.asarray(v, float))

    status = []

    status.append(("Deeper Target", pt[2] <= p1[2], f"{pt[2]:.0f}", f"≤ {p1[2]:.0f}"))

    dot = np.dot(v, pt - p1)
    status.append(("General Direction", dot > 0, f"{dot:.2f}", "> 0"))

    alpha1_deg = abs(np.degrees(result["alignment_turn_angle"]))
    status.append(("Initial Turn Angle", alpha1_deg <= alpha1_max_deg, f"{alpha1_deg:.1f}°", f"≤ {alpha1_max_deg:.1f}°"))

    alpha_main_deg = abs(np.degrees(result["main_turn_angle"]))
    status.append(("Main Turn Angle", alpha_main_deg <= alpha_main_max_deg, f"{alpha_main_deg:.1f}°", f"≤ {alpha_main_max_deg:.1f}°"))

    total_turn_deg = alpha1_deg + alpha_main_deg
    status.append(("Total Turn Angle", total_turn_deg <= total_angle_max_deg, f"{total_turn_deg:.1f}°", f"≤ {total_angle_max_deg:.1f}°"))

    segments = []

    if len(result["alignment_arc"]) > 1:
        segments.append(result["alignment_arc"])

    if len(result["hold_line"]) > 1:
        segments.append(result["hold_line"])

    if len(result["main_arc"]) > 1:
        segments.append(result["main_arc"])

    if len(result["main_line"]) > 1:
        segments.append(result["main_line"])

    if segments:
        full_path = np.vstack(segments)
    else:
        full_path = np.array([p1, pt])

    arc_z_max = full_path[:, 2].max()
    status.append(("No Climb", arc_z_max <= p1[2] + 1.0, f"{arc_z_max:.1f}", f"≤ {p1[2]:.0f}"))

    diffs = full_path[1:] - full_path[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    valid_idx = norms > 1e-6

    if np.any(valid_idx):
        tangents = diffs[valid_idx] / norms[valid_idx, np.newaxis]
        max_inc = np.degrees(np.arccos(np.clip(np.abs(tangents[:, 2]), -1.0, 1.0))).max()
    else:
        max_inc = 0.0

    status.append(("Max Inclination", max_inc <= max_inclination_deg, f"{max_inc:.1f}°", f"≤ {max_inclination_deg:.1f}°"))

    return status

def solve_case3(
    Pin,
    Pbd,
    p1,
    pt,
    v,
    dls_deg=3.0,
    alpha1_max_deg=20.0,
    alpha_main_max_deg=70.0,
    total_angle_max_deg=90.0,
    max_inclination_deg=60.0,
    step=30.0
):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v = normalize(np.asarray(v, float))

    result = compute_case3_trajectory(
        Pin=Pin,
        Pbd=Pbd,
        p1=p1,
        pt=pt,
        v=v,
        dls_deg=dls_deg,
        alpha_max_deg=alpha1_max_deg,
        step=step
    )

    status = validate_trajectory_case3(
        Pin=Pin,
        Pbd=Pbd,
        p1=p1,
        pt=pt,
        v=v,
        result=result,
        alpha1_max_deg=alpha1_max_deg,
        alpha_main_max_deg=alpha_main_max_deg,
        total_angle_max_deg=total_angle_max_deg,
        max_inclination_deg=max_inclination_deg
    )

    return {
        "Pin": Pin,
        "Pbd": Pbd,
        "p1": p1,
        "pt": pt,
        "v": v,
        "project_direction": result["project_direction"],
        "alignment_arc": result["alignment_arc"],
        "alignment_center": result["alignment_center"],
        "alignment_radius": result["alignment_radius"],
        "alignment_turn_angle": result["alignment_turn_angle"],
        "alignment_turn_angle_deg": result["alignment_turn_angle_deg"],
        "alignment_arc_length": result["alignment_arc_length"],
        "alignment_end_point": result["alignment_end_point"],
        "hold_line": result["hold_line"],
        "hold_length": result["hold_length"],
        "main_start_point": result["main_start_point"],
        "main_arc": result["main_arc"],
        "main_line": result["main_line"],
        "main_center": result["main_center"],
        "main_radius": result["main_radius"],
        "main_tangent_point": result["main_tangent_point"],
        "main_turn_angle": result["main_turn_angle"],
        "main_arc_length": result["main_arc_length"],
        "main_line_length": result["main_line_length"],
        "preferred_s": result["preferred_s"],
        "selected_s": result["selected_s"],
        "total_length": result["total_length"],
        "tested_candidates": result["tested_candidates"],
        "status": status
    }