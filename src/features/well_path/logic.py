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
    raise RuntimeError("Solver não convergiu")

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
    status.append(("Alvo mais profundo", pt[2] <= p1[2], f"{pt[2]:.0f}", f"≤ {p1[2]:.0f}"))
    dot = np.dot(v, pt - p1)
    status.append(("Direção geral", dot > 0, f"{dot:.2f}", "> 0"))
    ang_deg = abs(np.degrees(turn_angle))
    status.append(("Ângulo total", ang_deg <= max_ang_deg, f"{ang_deg:.1f}°", f"≤ {max_ang_deg}°"))
    arc_z_max = arc[:, 2].max()
    status.append(("Sem subida", arc_z_max <= p1[2] + 1.0, f"{arc_z_max:.1f}", f"≤ {p1[2]:.0f}"))
    full_path = np.vstack((arc, pt))
    diffs = full_path[1:] - full_path[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    valid_idx = norms > 1e-6
    tangents = diffs[valid_idx] / norms[valid_idx, np.newaxis]
    max_inc = np.degrees(np.arccos(np.clip(np.abs(tangents[:, 2]), -1.0, 1.0))).max()
    status.append(("Inclinação máx", max_inc <= 60.0, f"{max_inc:.1f}°", "≤ 60.0°"))
    return status

def solve_case1(Pin, Pbd, p1, pt, v):
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v  = normalize(np.asarray(v, float))
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
        raise ValueError("Geometria impossível: alvo dentro do raio.")
        
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
    status.append(("Alvo mais profundo", pt[2] <= p1[2], f"Zt: {pt[2]:.0f}", f"≤ {p1[2]:.0f}"))
    dot = np.dot(v_init, pt - p1)
    status.append(("Direção geral", dot > 0, f"{dot:.2f}", "> 0"))
    ang_deg = np.degrees(result["turn_angle"])
    status.append(("Ângulo total", ang_deg <= max_ang_deg, f"{ang_deg:.1f}°", f"≤ {max_ang_deg}°"))
    arc_z_max = result["arc"][:, 2].max()
    status.append(("Sem subida", arc_z_max <= p1[2] + 1.0, f"Zmax: {arc_z_max:.1f}", f"≤ {p1[2]:.0f}"))
    full_path = np.vstack((result["arc"], pt))
    diffs = full_path[1:] - full_path[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    valid_idx = norms > 1e-6
    tangents = diffs[valid_idx] / norms[valid_idx, np.newaxis]
    max_inc = np.degrees(np.arccos(np.clip(np.abs(tangents[:, 2]), -1.0, 1.0))).max()
    status.append(("Inclinação máx", max_inc <= 60.0, f"{max_inc:.1f}°", "≤ 60.0°"))
    return status

def solve_case2(Pin, Pbd, p1, pt, v):
    Pin = np.asarray(Pin, float)
    Pbd = np.asarray(Pbd, float)
    p1 = np.asarray(p1, float)
    pt = np.asarray(pt, float)
    v  = normalize(np.asarray(v, float))
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