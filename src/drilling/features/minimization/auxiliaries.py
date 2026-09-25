
"""Geometria, forças e núcleos de fator de ROP da trajetória Tipo 1.

Auxiliares geométricos públicos
-------------------------------
``theta``, ``lenght``, ``curve_points``, ``validate_configuration``,
``up_tension``, ``down_tension``, ``buckling``, ``Nl``, ``trajectory_elements``.

Núcleo vetorizado
-----------------
``trajectory_arrays`` discretiza a trajetória em arrays; ``evaluate_trajectory``
acrescenta as propriedades da rocha amostradas no modelo geológico;
``soft_string_profile`` integra torque e arraste com atrito por elemento
(``friction_model="lithology"``); ``mechanical_summary`` agrupa as saídas
mecânicas de um candidato.

Coordenadas: ``P0``/``P3`` são ``(x, y, z)`` com ``z`` profundidade positiva.
O poço fica no plano vertical de azimute ``Data.azimuth``; ``s`` é a distância
horizontal nesse plano (``plane_to_world`` converte ``(s, z)`` em ``(x, y, z)``).

A grafia ``lenght`` faz parte da API pública histórica e não deve ser
renomeada. As fórmulas deste arquivo estão congeladas pelos testes golden.
"""

import numpy as np

from drilling.features.minimization.geology import LITHOLOGY_COLORS  # noqa: F401 (reexportado)


EPS = 1.0e-10


def signal_change(s) -> tuple:
    """Indica se o sinal muda ao longo do array e devolve o primeiro índice."""
    arr = np.asarray(s, dtype=float).flatten()
    if arr.size == 0:
        return ("No", None)

    signs = np.sign(arr)

    first_nonzero_idx = None
    for idx, value in enumerate(signs):
        if value != 0:
            first_nonzero_idx = idx
            break

    if first_nonzero_idx is None:
        return ("No", None)

    current = signs[first_nonzero_idx]
    for idx in range(first_nonzero_idx + 1, len(signs)):
        if signs[idx] == 0:
            continue
        if signs[idx] != current:
            return ("Yes", idx)
    return ("No", None)


def plane_to_world(Data, s, depth):
    """Converte a distância horizontal ``s`` no plano do poço e a profundidade em ``(x, y, z)``."""
    az = float(Data.azimuth)
    x = Data.P0[0] + np.asarray(s, dtype=float) * np.cos(az)
    y = Data.P0[1] + np.asarray(s, dtype=float) * np.sin(az)
    z = np.asarray(depth, dtype=float)
    if np.ndim(x) == 0:
        return float(x), float(y), float(z)
    return x, y, z


def well_plane_s(Data, x, y) -> float:
    """Distância horizontal de ``(x, y)`` até ``P0`` no plano do poço Tipo 1."""
    return float(np.hypot(np.asarray(x, dtype=float) - Data.P0[0], np.asarray(y, dtype=float) - Data.P0[1]))


def _reconstruct_target_residual(Data, l1: float, R: float, angle: float, l3: float) -> float:
    s3, z3 = Data.departure, Data.P3[2]
    rx = -R * np.cos(angle) + l3 * np.sin(angle)
    rz = l3 * np.cos(angle) + R * np.sin(angle)
    return abs(rx - (s3 - R)) + abs(rz - (z3 - l1))


def theta(Data, l1, R) -> float:
    """Inclinação final estimada da curva de build-up, em radianos.

    Parameters
    ----------
    Data : DataSet
        Geometria do poço (usa ``departure`` e ``P3[2]``).
    l1, R : float
        Comprimento do trecho vertical e raio de curvatura, em metros.

    Returns
    -------
    float
        Ângulo positivo em ``(0, π/2]``.

    Raises
    ------
    ValueError
        Se a geometria não for um poço Tipo 1 válido.
    """
    s3, z3 = Data.departure, Data.P3[2]
    radicand_l3 = ((s3 - R) ** 2) + (z3 - l1) ** 2 - (R ** 2)
    if radicand_l3 <= EPS:
        raise ValueError("Invalid geometry: l3 is not real or non-positive.")
    l3 = np.sqrt(radicand_l3)

    disc = R ** 2 - l1 ** 2 + 2 * l1 * z3 + l3 ** 2 - z3 ** 2
    if disc < -EPS:
        raise ValueError("Invalid geometry: discriminant is negative.")
    disc = max(disc, 0.0)

    den = -l1 + l3 + z3
    if abs(den) <= EPS:
        raise ValueError("Invalid geometry: zero denominator in angle calculation.")

    sqrt_disc = np.sqrt(disc)
    candidates = [
        2 * np.arctan((R - sqrt_disc) / den),
        2 * np.arctan((R + sqrt_disc) / den),
    ]

    valid_candidates = []
    for candidate in candidates:
        if np.isfinite(candidate) and candidate > 0:
            valid_candidates.append(candidate)

    if not valid_candidates:
        raise ValueError("No valid angle candidate was found.")

    angle = min(
        valid_candidates,
        key=lambda candidate: _reconstruct_target_residual(Data, l1, R, candidate, l3),
    )

    if not (0.0 < angle < np.pi / 2 + EPS):
        raise ValueError("The computed angle is outside the admissible interval.")

    return float(angle)


def lenght(Data, l1, R) -> list:
    """Devolve os comprimentos dos três trechos ``(l1, l2, l3)``.

    O nome da função preserva a grafia histórica ``lenght``.

    Parameters
    ----------
    Data : DataSet
        Geometria do poço.
    l1, R : float
        Comprimento do trecho vertical e raio de curvatura, em metros.

    Returns
    -------
    tuple of float
        ``l1``, comprimento de arco ``l2 = R * theta`` e comprimento de tangente ``l3``.
    """
    angle = theta(Data, l1, R)
    l3_sq = ((Data.departure - R) ** 2) + (Data.P3[2] - l1) ** 2 - (R ** 2)
    if l3_sq <= EPS:
        raise ValueError("Invalid geometry: the tangent section length is not positive.")
    l3 = float(np.sqrt(l3_sq))
    l2 = float(R * angle)
    return float(l1), l2, l3


def curve_points_plane(Data, l1, R):
    """Curva de build-up no plano do poço: distância horizontal ``s`` e profundidade ``z``."""
    angle = np.linspace(np.pi, np.pi + theta(Data, l1, R), 1000)
    l1, *_ = lenght(Data, l1, R)
    s = R * np.cos(angle) + R
    z = Data.P0[2] + l1 - R * np.sin(angle)
    return s, z


def curve_points(Data, l1, R):
    """Coordenadas ``(x, y, z)`` da curva de build-up; ``z`` é profundidade."""
    s, z = curve_points_plane(Data, l1, R)
    return plane_to_world(Data, s, z)


def points_coordinates(Data, l1, R):
    s_vals = [0.0, 0.0]
    z_vals = [Data.P0[2], Data.P0[2] + l1]

    curve_s, curve_z = curve_points_plane(Data, l1, R)
    s_vals.extend(np.asarray(curve_s, dtype=float).tolist())
    z_vals.extend(np.asarray(curve_z, dtype=float).tolist())

    s_vals.append(Data.departure)
    z_vals.append(Data.P3[2])
    return plane_to_world(Data, np.asarray(s_vals, dtype=float), np.asarray(z_vals, dtype=float))


def buckling(Data, l1, R) -> float:
    psb = Data.z
    angle = theta(Data, l1, R)
    alpha = 1 - (Data.ro_fluid / Data.ro_command)
    ws = Data.lambd_command

    denominator = ws * alpha * np.cos(angle) * 9.81
    if denominator <= EPS:
        raise ValueError("Invalid buckling calculation: non-positive denominator.")

    lc = round((psb * 1.2) / denominator)
    if lc <= 0:
        raise ValueError("Invalid buckling result: lc must be positive.")

    while True:
        if (lc % 9) >= 5 and (lc % 9) != 0:
            lc += 1
        elif (lc % 9) < 5 and (lc % 9) != 0:
            lc -= 1
        else:
            break

    return float(lc)


def _denominator_check(value: float, message: str) -> None:
    if abs(value) <= EPS:
        raise ValueError(message)


def Nl(Data, l1, R, Mesh=None) -> float:
    if use_numeric_friction(Data, Mesh):
        return float(soft_string_profile(Data, Mesh, l1, R, "down")["neutral_point_m"])
    lc = buckling(Data, l1, R)
    angle = theta(Data, l1, R)
    z_f = Data.P3[2]
    ro_fluid = Data.ro_fluid
    ro_command = Data.ro_command
    ro_heavypipe = Data.ro_heavypipe
    ro_drillpipe = Data.ro_drillpipe
    g = Data.g
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    lambd_drill = Data.lambd_drill
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    z = Data.z
    µ = Data.µ
    lp = Data.lp

    ff = ro_fluid * g * z_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)

    denom = lambd_command * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_command)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in command section.")
    neutral_line = (z + ff) / denom

    if neutral_line < lc:
        return float(neutral_line)

    z_ic = z_f - (lc * np.cos(angle))
    fic = (
        ro_fluid * g * z_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
    )

    denom = lambd_heavy * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in heavy-pipe section.")
    a = (ff + z - fic) / denom
    b = lc * (
        (lambd_command - lambd_heavy) * np.cos(angle)
        - (µ * ((lambd_command * (1 - (ro_fluid / ro_command))) - (lambd_heavy * (1 - (ro_fluid / ro_heavypipe))))) * np.sin(angle)
    )
    c = lambd_heavy * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle))
    _denominator_check(c, "Invalid neutral-line correction in heavy-pipe section.")
    neutral_line = a - (b / c)

    if neutral_line <= (lc + lp):
        return float(neutral_line)

    z_ip = z_f - ((lc + lp) * np.cos(angle))
    fip = (
        ro_fluid * g * z_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4
    )

    denom = lambd_drill * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in drill-pipe section.")
    a = (ff + z - fic - fip) / denom
    b = lc * (
        (lambd_command - lambd_drill) * np.cos(angle)
        - (µ * ((lambd_command * (1 - (ro_fluid / ro_command))) - (lambd_drill * (1 - (ro_fluid / ro_drillpipe))))) * np.sin(angle)
    )
    c = lambd_drill * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle))
    _denominator_check(c, "Invalid neutral-line correction in drill-pipe section.")
    d = lp * (
        (lambd_heavy - lambd_drill) * np.cos(angle)
        - (µ * ((lambd_heavy * (1 - (ro_fluid / ro_heavypipe))) - (lambd_drill * (1 - (ro_fluid / ro_drillpipe))))) * np.sin(angle)
    )

    neutral_line = a - (b / c) - (d / c)
    return float(neutral_line)


def _vertical_effective_weight(Data, l1: float) -> float:
    return (Data.lambd_drill - Data.ro_fluid * Data.area_drill) * Data.g * l1


def validate_configuration(Data, l1, R, angle_limit_deg: float | None = None) -> dict:
    """Aceita um par (L1, R) ou levanta erro se a geometria Tipo 1 for inadmissível.

    Parameters
    ----------
    Data : DataSet
        Geometria do poço e o limite angular padrão.
    l1, R : float
        Comprimentos candidatos.
    angle_limit_deg : float or None, optional
        Substitui ``Data.angle_limit_deg`` (padrão 52°).

    Returns
    -------
    dict
        ``l1``, ``l2``, ``l3``, ``R``, ``angle``, ``angle_deg``, ``lc``, ``ld``.
    """
    if l1 <= 0 or R <= 0:
        raise ValueError("l1 and R must be positive.")

    l1, l2, l3 = lenght(Data, l1, R)
    angle = theta(Data, l1, R)
    angle_deg = float(np.degrees(angle))
    if angle_limit_deg is None:
        angle_limit_deg = getattr(Data, "angle_limit_deg", 52.0)

    if angle_deg > angle_limit_deg + 1e-9:
        raise ValueError("Configuration rejected: angle above the admissible limit.")

    lc = buckling(Data, l1, R)
    ld = l3 - Data.lp - lc

    if lc <= 0:
        raise ValueError("Configuration rejected: non-positive command length.")
    if ld < -EPS:
        raise ValueError("Configuration rejected: the tangent section is shorter than command + heavy pipe.")

    return {
        "l1": float(l1),
        "l2": float(l2),
        "l3": float(l3),
        "R": float(R),
        "angle": float(angle),
        "angle_deg": angle_deg,
        "lc": float(lc),
        "ld": float(max(ld, 0.0)),
    }


def up_tension(Data, l1, R, Mesh=None) -> list:
    """Força axial no topo de L1, no início da curva e em P3 (içamento).

    Parameters
    ----------
    Data : DataSet
        Propriedades mecânicas.
    l1, R : float
        Configuração em avaliação.
    Mesh : HorizonModel or None, optional
        Necessário apenas com ``friction_model="lithology"``, quando o
        resultado vem de ``soft_string_profile``.

    Returns
    -------
    tuple of float
        ``(tension_1, tension_2, tension_3)`` em newtons.
    """
    if use_numeric_friction(Data, Mesh):
        profile = soft_string_profile(Data, Mesh, l1, R, "up")
        return profile["tension_1"], profile["tension_2"], profile["tension_3"]
    config = validate_configuration(Data, l1, R)
    lc = config["lc"]
    l1 = config["l1"]
    l3 = config["l3"]
    angle = config["angle"]
    g = Data.g
    ro_fluid = Data.ro_fluid
    ro_drillpipe = Data.ro_drillpipe
    lambd_drill = Data.lambd_drill
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    µ = Data.µ
    lp = Data.lp
    p3 = Data.P3
    area_drill = Data.area_drill
    area_command = Data.area_command
    area_heavy = Data.area_heavy

    weight_3 = ((lambd_command * lc) + (lambd_heavy * lp) + (lambd_drill * (l3 - lc - lp))) * g
    buoyancy_3 = (
        (area_drill * (l3 - lp - lc) * ro_fluid)
        + (area_command * lc * ro_fluid)
        + (area_heavy * lp * ro_fluid)
    ) * g

    z_f = p3[2]
    z_ic = z_f - (lc * np.cos(angle))
    z_ip = z_f - ((lc + lp) * np.cos(angle))
    fic = ro_fluid * g * z_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
    fip = ro_fluid * g * z_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4
    ff = ro_fluid * g * z_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)

    tension_3 = weight_3 * np.cos(angle) + µ * (weight_3 - buoyancy_3) * np.sin(angle) + fic + fip - ff

    angle_variation = np.arange(0.001, angle, 0.001)[::-1]
    if angle_variation.size == 0:
        tension_2 = tension_3
        tension_1 = tension_2 + _vertical_effective_weight(Data, l1)
        return float(tension_1), float(tension_2), float(tension_3)

    dns = []
    for value in angle_variation:
        dn = tension_3 * value - (1 - (ro_fluid / ro_drillpipe)) * g * lambd_drill * R * np.sin(value) * value
        dns.append(dn)

    conditions = signal_change(np.sign(dns))
    condition_signal_change = conditions[0]
    condition_where_change = conditions[1]

    if condition_signal_change == "No":
        if dns[0] > 0:
            a = lambd_drill * g * R
            c1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            c2 = -((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))
            tension_2 = c1 * np.sin(0) + c2 * np.cos(0) + k * (np.exp(-µ * 0))
        else:
            a = lambd_drill * g * R
            b1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            b2 = ((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))
            tension_2 = b1 * np.sin(0) + b2 * np.cos(0) + k * (np.exp(µ * 0))
    else:
        a = lambd_drill * g * R
        b1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        b2 = ((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))

        tension_change = (
            b1 * np.sin(angle_variation[condition_where_change])
            + b2 * np.cos(angle_variation[condition_where_change])
            + k * (np.exp(µ * angle_variation[condition_where_change]))
        )

        c1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        c2 = -((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_change - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))
        tension_2 = c1 * np.sin(0) + c2 * np.cos(0) + k * (np.exp(-µ * 0))

    tension_1 = tension_2 + _vertical_effective_weight(Data, l1)
    return float(tension_1), float(tension_2), float(tension_3)


def down_tension(Data, l1, R, Mesh=None) -> list:
    """Força axial ao descer a coluna, mais o torque de atrito.

    Parameters
    ----------
    Data : DataSet
        Propriedades mecânicas.
    l1, R : float
        Configuração em avaliação.
    Mesh : HorizonModel or None, optional
        Necessário apenas com ``friction_model="lithology"``.

    Returns
    -------
    tuple of float
        ``(tension_1, tension_2, tension_3, torque)`` em N e N·m.
    """
    if use_numeric_friction(Data, Mesh):
        profile = soft_string_profile(Data, Mesh, l1, R, "down")
        return profile["tension_1"], profile["tension_2"], profile["tension_3"], profile["torque"]
    config = validate_configuration(Data, l1, R)
    lc = config["lc"]
    l1 = config["l1"]
    l3 = config["l3"]
    angle = config["angle"]
    ld = config["ld"]
    g = Data.g
    ro_fluid = Data.ro_fluid
    ro_drillpipe = Data.ro_drillpipe
    lambd_drill = Data.lambd_drill
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    µ = Data.µ
    lp = Data.lp
    p3 = Data.P3
    area_drill = Data.area_drill
    area_command = Data.area_command
    area_heavy = Data.area_heavy
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    z = Data.z

    weight_3 = ((lambd_command * lc) + (lambd_heavy * lp) + (lambd_drill * (l3 - lc - lp))) * g
    buoyancy_3 = (
        (area_drill * (l3 - lp - lc) * ro_fluid)
        + (area_command * lc * ro_fluid)
        + (area_heavy * lp * ro_fluid)
    ) * g

    z_f = p3[2]
    z_ic = z_f - (lc * np.cos(angle))
    z_ip = z_f - ((lc + lp) * np.cos(angle))
    ff = ro_fluid * g * z_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)
    fic = ro_fluid * g * z_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
    fip = ro_fluid * g * z_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4

    tension_3 = weight_3 * (np.cos(angle) - µ * np.sin(angle)) + (µ * buoyancy_3 * np.sin(angle)) + fic + fip - z - ff

    angle_variation = np.arange(0.01, angle, 0.01)[::-1]
    if angle_variation.size == 0:
        tension_2 = tension_3
        fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
        fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
        fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)
        torque = (d_ext_command / 2) * fat3_command + (d_ext_heavy / 2) * fat3_heavy + (d_ext_drill / 2) * fat3_drill
        tension_1 = tension_2 + _vertical_effective_weight(Data, l1)
        return float(tension_1), float(tension_2), float(tension_3), float(torque)

    dns = []
    for value in angle_variation:
        dn = tension_3 * value - (1 - (ro_fluid / ro_drillpipe)) * g * lambd_drill * R * np.sin(value) * value
        dns.append(dn)

    conditions = signal_change(np.sign(dns))
    condition_signal_change = conditions[0]
    condition_where_change = conditions[1]

    if condition_signal_change == "No":
        if dns[0] > 0:
            a = lambd_drill * g * R
            b1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            b2 = ((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))
            tension_2 = b1 * np.sin(0) + b2 * np.cos(0) + k * (np.exp(µ * 0))
            b3 = b1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
            medial_n_force = b3 * (1 - np.cos(angle)) + b2 * np.sin(angle) + (k / µ) * ((np.exp(µ * angle)) - 1)
            frictional_torque_2 = (µ * medial_n_force * d_ext_drill) / 2
        else:
            a = lambd_drill * g * R
            c1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            c2 = -((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))
            tension_2 = c1 * np.sin(0) + c2 * np.cos(0) + k * (np.exp(-µ * 0))
            c3 = c1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
            medial_n_force = c3 * (1 - np.cos(angle)) + c2 * np.sin(angle) - (k / µ) * ((np.exp(-µ * angle)) - 1)
            frictional_torque_2 = -((µ * medial_n_force * d_ext_drill) / 2)

        fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
        fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
        fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)
        torque3_command = (d_ext_command / 2) * fat3_command
        torque3_heavy = (d_ext_heavy / 2) * fat3_heavy
        torque3_drill = (d_ext_drill / 2) * fat3_drill
        frictional_torque_3 = torque3_command + torque3_heavy + torque3_drill
        torque = frictional_torque_2 + frictional_torque_3
    else:
        a = lambd_drill * g * R
        c1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        c2 = -((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_3 - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))

        c3 = c1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
        medial_n_force = -c3 * (np.cos(angle_variation[condition_where_change]) - np.cos(angle)) + c2 * (
            np.sin(angle_variation[condition_where_change]) - np.sin(angle)
        ) - (k / µ) * (np.exp(-µ * angle_variation[condition_where_change]) - np.exp(-µ * angle))
        frictional_torque_2_change = -((µ * medial_n_force * d_ext_drill) / 2)

        tension_change = (
            c1 * np.sin(angle_variation[condition_where_change])
            + c2 * np.cos(angle_variation[condition_where_change])
            + k * (np.exp(-µ * angle_variation[condition_where_change]))
        )

        b1 = (a / (1 + µ ** 2)) * (((µ ** 2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        b2 = ((a * µ) / (1 + µ ** 2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_change - b1 * np.sin(angle_variation[condition_where_change]) - b2 * np.cos(angle_variation[condition_where_change])) / (
            np.exp(µ * angle_variation[condition_where_change])
        )

        b3 = b1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
        medial_n_force = (
            b3 * (1 - np.cos(angle_variation[condition_where_change]))
            + b2 * np.sin(angle_variation[condition_where_change])
            + (k / µ) * ((np.exp(µ * angle_variation[condition_where_change])) - 1)
        )
        frictional_torque_2 = ((µ * medial_n_force * d_ext_drill) / 2) + frictional_torque_2_change

        fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
        fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
        fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)
        torque3_command = (d_ext_command / 2) * fat3_command
        torque3_heavy = (d_ext_heavy / 2) * fat3_heavy
        torque3_drill = (d_ext_drill / 2) * fat3_drill
        frictional_torque_3 = torque3_command + torque3_heavy + torque3_drill
        torque = frictional_torque_2 + frictional_torque_3
        tension_2 = b1 * np.sin(0) + b2 * np.cos(0) + k * (np.exp(µ * 0))

    tension_1 = tension_2 + _vertical_effective_weight(Data, l1)
    return float(tension_1), float(tension_2), float(tension_3), float(torque)





def inclination_angle_deg(d_horizontal, d_depth):
    return np.degrees(np.arctan2(np.abs(d_horizontal), np.abs(d_depth) + EPS))


def dogleg_severity_deg_per_30m(curvature):
    return np.where(np.asarray(curvature, dtype=float) > 0.0, np.degrees(np.asarray(curvature, dtype=float) * 30.0), 0.0)


def inclination_factor(angle_deg, params: dict):
    """Penalidade de ROP pela inclinação do poço (1,0 = sem penalidade)."""
    reduction = float(params["inclination_reduction"])
    exponent = float(params["inclination_exponent"])
    lower = max(0.85, float(params["min_inclination_factor"]))
    normalized = np.clip(np.asarray(angle_deg, dtype=float) / 90.0, 0.0, 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.clip(factor, lower, 1.0)


def dls_factor(dls_deg_per_30m, params: dict):
    """Penalidade de ROP pela severidade de dogleg."""
    reduction = float(params["dls_reduction"])
    exponent = float(params["dls_exponent"])
    lower = max(0.50, float(params["min_dls_factor"]))
    reference = float(params["reference_dls_deg_per_30m"])
    normalized = np.clip(np.asarray(dls_deg_per_30m, dtype=float) / reference, 0.0, 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.clip(factor, lower, 1.0)


def wob_transfer_factor(angle_deg, dls_deg_per_30m, params: dict):
    """Fração do peso sobre broca aplicado na superfície que de fato chega à broca."""
    reference_dls = float(params["reference_dls_deg_per_30m"])
    a = float(params["drag_inclination_coeff"])
    b = float(params["drag_dls_coeff"])
    exponent = float(params["wob_transfer_exponent"])
    inc_term = np.sin(np.radians(np.maximum(np.asarray(angle_deg, dtype=float), 0.0)))
    dls_term = np.maximum(np.asarray(dls_deg_per_30m, dtype=float), 0.0) / reference_dls
    transfer = np.exp(-(a * (inc_term ** exponent) + b * (dls_term ** exponent)))
    return np.clip(transfer, 0.0, 1.0)


def wob_factor(wob_effective, params: dict):
    """Penalidade de ROP por operar abaixo do peso sobre broca ótimo."""
    optimal_wob = float(params["optimal_wob"])
    lower = max(0.90, float(params["min_wob_factor"]))
    exponent = float(params["wob_factor_exponent"])
    ratio = np.maximum(np.asarray(wob_effective, dtype=float) / optimal_wob, 0.0)
    return np.clip(ratio ** exponent, lower, 1.0)


def local_contact_force_per_length(Data, angle_deg, curvature, wob_effective):
    """Força de contato com a parede por unidade de comprimento (termos de gravidade e curvatura)."""
    angle_rad = np.radians(np.maximum(np.asarray(angle_deg, dtype=float), 0.0))
    gravity_contact = abs(Data.buoyed_linear_weight_avg) * np.sin(angle_rad)
    curvature_contact = np.abs(np.asarray(wob_effective, dtype=float)) * np.maximum(
        np.asarray(curvature, dtype=float), 0.0
    )
    return gravity_contact + curvature_contact


def torque_factor(cumulative_torque, params: dict):
    """Penalidade de ROP quando o torque acumulado se aproxima do limite da coluna."""
    reduction = float(params["torque_reduction"])
    exponent = float(params["torque_exponent"])
    lower = max(0.90, float(params["min_torque_factor"]))
    limit = float(params["torque_limit"])
    normalized = np.clip(np.asarray(cumulative_torque, dtype=float) / limit, 0.0, 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.clip(factor, lower, 1.0)


_SECTION_NAMES = ("vertical", "curve", "tangent")

_ELEMENT_KEYS = (
    "x0", "y0", "z0", "x1", "y1", "z1",
    "x_mid", "y_mid", "z_mid",
    "s0", "s1", "s_mid",
    "dx", "dy", "dz", "length",
    "inclination_deg", "curvature", "dls_deg_per_30m",
    "measured_depth_m", "measured_depth_mid_m",
)


def trajectory_arrays(Data, l1: float, R: float, ds_target: float | None = None) -> dict:
    """Discretiza o poço Tipo 1 em elementos de trecho vertical, curva e tangente, como arrays.

    É o núcleo vetorizado de :func:`trajectory_elements`: o modelo geológico é
    amostrado no ponto médio de cada elemento e o otimizador avalia milhares de
    trajetórias, então a geometria sai em arrays e não em lista de dicionários.

    Parameters
    ----------
    Data : DataSet
        Geometria e o ``trajectory_step`` padrão.
    l1, R : float
        Configuração a discretizar.
    ds_target : float or None, optional
        Comprimento do elemento em metros. O padrão é
        ``Data.drilling_time_parameters['trajectory_step']``.

    Returns
    -------
    dict
        ``config``, ``n_elements``, ``section``/``section_code`` e arrays por
        elemento (extremidades e ponto médio em ``x, y, z`` e ``s``, comprimento,
        inclinação, curvatura, DLS e profundidade medida).
    """
    config = validate_configuration(Data, l1, R)
    l1 = config["l1"]
    l2 = config["l2"]
    l3 = config["l3"]
    angle = config["angle"]

    if ds_target is None:
        ds_target = float(Data.drilling_time_parameters["trajectory_step"])
    if ds_target <= 0:
        raise ValueError("'ds_target' must be positive.")

    p0 = Data.P0

    n1 = max(1, int(np.ceil(l1 / ds_target)))
    x_v = np.full(n1 + 1, float(p0[0]))
    y_v = np.full(n1 + 1, float(p0[1]))
    z_v = np.linspace(float(p0[2]), float(p0[2]) + l1, n1 + 1)

    n2 = max(20, int(np.ceil(l2 / ds_target)))
    phi = np.linspace(0.0, angle, n2 + 1)
    s_arc = R * (1.0 - np.cos(phi))
    z_arc = float(p0[2]) + l1 + R * np.sin(phi)
    x_c, y_c, z_c = plane_to_world(Data, s_arc, z_arc)

    n3 = max(1, int(np.ceil(l3 / ds_target)))
    x_t = np.linspace(float(x_c[-1]), float(Data.P3[0]), n3 + 1)
    y_t = np.linspace(float(y_c[-1]), float(Data.P3[1]), n3 + 1)
    z_t = np.linspace(float(z_c[-1]), float(Data.P3[2]), n3 + 1)

    xa = np.concatenate([x_v[:-1], x_c[:-1], x_t[:-1]])
    ya = np.concatenate([y_v[:-1], y_c[:-1], y_t[:-1]])
    za = np.concatenate([z_v[:-1], z_c[:-1], z_t[:-1]])
    xb = np.concatenate([x_v[1:], x_c[1:], x_t[1:]])
    yb = np.concatenate([y_v[1:], y_c[1:], y_t[1:]])
    zb = np.concatenate([z_v[1:], z_c[1:], z_t[1:]])
    section_code = np.concatenate(
        [np.zeros(n1, dtype=np.int8), np.ones(n2, dtype=np.int8), np.full(n3, 2, dtype=np.int8)]
    )

    dx, dy, dz = xb - xa, yb - ya, zb - za
    ds = np.sqrt(dx * dx + dy * dy + dz * dz)

    inclination = inclination_angle_deg(np.hypot(dx, dy), dz)
    # Inside the build section the inclination is the arc angle itself, which is
    # exact, whereas the chord-based value above is only a secant approximation.
    inclination[n1 : n1 + n2] = np.degrees(0.5 * (phi[:-1] + phi[1:]))

    curvature = np.zeros_like(ds)
    curvature[n1 : n1 + n2] = 1.0 / R

    keep = ds > EPS
    if not np.any(keep):
        raise ValueError("The discretised trajectory has no element of positive length.")
    if not np.all(keep):
        xa, ya, za = xa[keep], ya[keep], za[keep]
        xb, yb, zb = xb[keep], yb[keep], zb[keep]
        dx, dy, dz, ds = dx[keep], dy[keep], dz[keep], ds[keep]
        inclination, curvature = inclination[keep], curvature[keep]
        section_code = section_code[keep]

    measured_depth = np.cumsum(ds)
    s0 = np.hypot(xa, ya)
    s1 = np.hypot(xb, yb)

    return {
        "config": config,
        "n_elements": int(ds.size),
        "section_code": section_code,
        "section": np.array(_SECTION_NAMES, dtype=object)[section_code],
        "x0": xa, "y0": ya, "z0": za,
        "x1": xb, "y1": yb, "z1": zb,
        "x_mid": 0.5 * (xa + xb), "y_mid": 0.5 * (ya + yb), "z_mid": 0.5 * (za + zb),
        "s0": s0, "s1": s1, "s_mid": 0.5 * (s0 + s1),
        "dx": dx, "dy": dy, "dz": dz,
        "length": ds,
        "inclination_deg": inclination,
        "curvature": curvature,
        "dls_deg_per_30m": dogleg_severity_deg_per_30m(curvature),
        "measured_depth_m": measured_depth,
        "measured_depth_mid_m": measured_depth - 0.5 * ds,
        "total_length_m": float(measured_depth[-1]),
    }


def trajectory_elements(Data, l1: float, R: float, ds_target: float | None = None) -> list[dict]:
    """Lista de dicionários por elemento, para quem itera elemento a elemento.

    Mesmos campos de :func:`trajectory_arrays`, com ``section`` em cada item.
    """
    return elements_from_arrays(trajectory_arrays(Data, l1, R, ds_target=ds_target))


def elements_from_arrays(geometry: dict) -> list[dict]:
    """Converte a saída de :func:`trajectory_arrays` em lista de dicionários."""
    sections = geometry["section"]
    columns = {key: geometry[key] for key in _ELEMENT_KEYS}
    return [
        {"section": sections[i], **{key: float(values[i]) for key, values in columns.items()}}
        for i in range(geometry["n_elements"])
    ]


def evaluate_trajectory(Data, Mesh, l1: float, R: float, ds_target: float | None = None) -> dict:
    """Geometria de um candidato mais as propriedades da rocha ao longo dele.

    Todos os cálculos seguintes (tempo de broca, torque e arraste, eventos
    operacionais) partem daqui, de modo que a trajetória é discretizada e o
    modelo geológico consultado uma vez por candidato, e não uma vez por
    consumidor. Acrescenta ``lithology``, ``lithology_code``, ``rop_base``,
    ``mu`` e ``mu_from_mesh`` ao dicionário de :func:`trajectory_arrays`.
    """
    geometry = trajectory_arrays(Data, l1, R, ds_target=ds_target)
    if Mesh is None:
        geometry["lithology"] = np.full(geometry["n_elements"], "Undefined", dtype=object)
        geometry["lithology_code"] = np.zeros(geometry["n_elements"], dtype=np.int8)
        geometry["rop_base"] = np.full(geometry["n_elements"], np.nan)
        geometry["mu"] = np.full(geometry["n_elements"], float(Data.µ))
        geometry["mu_from_mesh"] = False
        return geometry

    sampled = Mesh.sample(geometry["x_mid"], geometry["y_mid"], geometry["z_mid"])
    geometry["lithology"] = sampled["lithology"]
    geometry["lithology_code"] = sampled["code"]
    geometry["rop_base"] = sampled["rop"]
    use_mesh_mu = sampled["mu"] is not None and getattr(Data, "friction_model", "constant") == "lithology"
    geometry["mu"] = sampled["mu"] if use_mesh_mu else np.full(geometry["n_elements"], float(Data.µ))
    geometry["mu_from_mesh"] = bool(use_mesh_mu)
    return geometry


def use_numeric_friction(Data, Mesh) -> bool:
    """Indica se torque e arraste devem ser integrados numericamente em vez da solução fechada."""
    if Mesh is None or getattr(Data, "friction_model", "constant") != "lithology":
        return False
    if not bool(getattr(Mesh, "has_mu", False)):
        raise ValueError(
            "friction_model='lithology' needs a geological model carrying friction "
            "coefficients. Build the mesh with 'mu_values' (or 'default_mu'), or set "
            "friction_model='constant'."
        )
    return True


def _string_properties(Data, md_from_bit: np.ndarray, lc: float) -> tuple:
    """Peso linear, área da seção e diâmetro externo do tubo em cada posição.

    ``md_from_bit`` é a profundidade medida contada a partir da broca: os
    comandos ocupam os primeiros ``lc`` metros e o heavy-weight os ``Data.lp``
    seguintes.
    """
    in_command = md_from_bit < lc
    in_heavy = (~in_command) & (md_from_bit < lc + Data.lp)
    lambd = np.where(in_command, Data.lambd_command, np.where(in_heavy, Data.lambd_heavy, Data.lambd_drill))
    area = np.where(in_command, Data.area_command, np.where(in_heavy, Data.area_heavy, Data.area_drill))
    d_ext = np.where(in_command, Data.d_ext_command, np.where(in_heavy, Data.d_ext_heavy, Data.d_ext_drill))
    return lambd, area, d_ext


def _pressure_end_effects(Data, angle: float, lc: float) -> tuple[float, float, float]:
    """Pressão do fluido nas mudanças de seção da coluna.

    ``ff`` atua na broca, ``fic`` no ombro comando/heavy-weight e ``fip`` no
    ombro heavy-weight/drill pipe. São os mesmos termos da solução fechada,
    reproduzidos aqui para que os dois caminhos compartilhem as mesmas
    condições de contorno.
    """
    z_f = float(Data.P3[2])
    z_ic = z_f - lc * np.cos(angle)
    z_ip = z_f - (lc + Data.lp) * np.cos(angle)
    quarter_pi = np.pi / 4.0
    ff = Data.ro_fluid * Data.g * z_f * quarter_pi * (Data.d_ext_command ** 2 - Data.d_int_command ** 2)
    fic = Data.ro_fluid * Data.g * z_ic * quarter_pi * (
        (Data.d_ext_command ** 2 - Data.d_ext_heavy ** 2) - (Data.d_int_command ** 2 - Data.d_int_heavy ** 2)
    )
    fip = Data.ro_fluid * Data.g * z_ip * quarter_pi * (
        (Data.d_ext_heavy ** 2 - Data.d_ext_drill ** 2) - (Data.d_int_heavy ** 2 - Data.d_int_drill ** 2)
    )
    return float(ff), float(fic), float(fip)


def soft_string_profile(Data, Mesh, l1: float, R: float, direction: str = "up", geometry: dict | None = None) -> dict:
    """Integra torque e arraste numericamente com coeficiente de atrito por elemento.

    As soluções fechadas :func:`up_tension` / :func:`down_tension` resolvem a
    equação do cabo com um único coeficiente de atrito para o poço inteiro e
    não representam um atrito que muda com a rocha atravessada. Esta função
    marcha o modelo soft-string (Johancsik) da broca até a superfície, tomando
    ``mu`` do modelo geológico em cada elemento::

        N_i   = |T_i * dalpha_i + w_buoyed_i * ds_i * sin(alpha_i)|
        T_i+1 = T_i + w_air_i * ds_i * cos(alpha_i) +/- mu_i * N_i
        M_i+1 = M_i + mu_i * N_i * r_i

    O trecho vertical não tem atrito (sin(alpha) = 0) e é somado em forma
    fechada, como na solução analítica.

    Com ``mu`` uniforme reproduz os resultados da solução fechada, o que é
    verificado em ``tests/test_geology.py``.

    Parameters
    ----------
    Data : DataSet
        Propriedades mecânicas.
    Mesh : HorizonModel
        Modelo geológico com ``mu_values``.
    l1, R : float
        Configuração em avaliação.
    direction : {"up", "down"}, optional
        Içamento ou descida da coluna.
    geometry : dict or None, optional
        Saída pré-calculada de :func:`evaluate_trajectory`.

    Returns
    -------
    dict
        ``tension_1/2/3``, ``torque``, ``neutral_point_m`` e os perfis ao longo
        da coluna.
    """
    if direction not in ("up", "down"):
        raise ValueError("'direction' must be either 'up' or 'down'.")

    geom = geometry if geometry is not None else evaluate_trajectory(Data, Mesh, l1, R)
    config = geom["config"]
    angle = float(config["angle"])
    lc = float(config["lc"])

    # March from the bit upwards: elements are stored surface-first, and only the
    # build and tangent sections generate contact force.
    curved_and_tangent = geom["section_code"] >= 1
    order = np.flatnonzero(curved_and_tangent)[::-1]

    ds = geom["length"][order]
    alpha = np.radians(geom["inclination_deg"][order])
    curvature = geom["curvature"][order]
    mu = geom["mu"][order]
    # Going up, the hole angle decreases through the build section.
    d_alpha = -curvature * ds

    total_length = float(geom["total_length_m"])
    md_from_bit_mid = total_length - geom["measured_depth_mid_m"][order]
    md_from_bit_end = total_length - (geom["measured_depth_m"][order] - ds)

    lambd, area, d_ext = _string_properties(Data, md_from_bit_mid, lc)
    w_air = lambd * Data.g
    w_buoyed = (lambd - Data.ro_fluid * area) * Data.g
    radius = 0.5 * d_ext

    ff, fic, fip = _pressure_end_effects(Data, angle, lc)
    friction_sign = 1.0 if direction == "up" else -1.0

    tension = -ff
    if direction == "down":
        tension -= float(Data.z)

    # Shoulder forces enter once, when the march passes the section change.
    shoulders = [(lc, fic), (lc + Data.lp, fip)]

    n = ds.size
    tensions = np.empty(n + 1, dtype=float)
    torques = np.empty(n + 1, dtype=float)
    tensions[0] = tension
    torques[0] = 0.0
    torque = 0.0
    contact_total = 0.0
    shoulder_index = 0

    for i in range(n):
        normal = abs(tension * d_alpha[i] + w_buoyed[i] * ds[i] * np.sin(alpha[i]))
        friction = mu[i] * normal
        tension = tension + w_air[i] * ds[i] * np.cos(alpha[i]) + friction_sign * friction
        while shoulder_index < len(shoulders) and md_from_bit_end[i] >= shoulders[shoulder_index][0]:
            tension += shoulders[shoulder_index][1]
            shoulder_index += 1
        torque += friction * radius[i]
        contact_total += normal
        tensions[i + 1] = tension
        torques[i + 1] = torque

    for remaining in range(shoulder_index, len(shoulders)):
        tension += shoulders[remaining][1]
    tensions[-1] = tension

    n_tangent = int(np.count_nonzero(geom["section_code"] == 2))
    tension_3 = float(tensions[min(n_tangent, n)])
    tension_2 = float(tension)
    tension_1 = tension_2 + _vertical_effective_weight(Data, l1)

    # The neutral point is where the running-in axial profile changes sign, measured
    # from the bit; it falls out of the profile instead of needing its own formula.
    neutral_point = _zero_crossing(md_from_bit_end, tensions[1:])

    return {
        "config": config,
        "direction": direction,
        "tension_1": float(tension_1),
        "tension_2": float(tension_2),
        "tension_3": tension_3,
        "torque": float(torque),
        "neutral_point_m": neutral_point,
        "total_contact_force_N": float(contact_total),
        "measured_depth_from_bit_m": md_from_bit_end,
        "tension_profile_N": tensions[1:],
        "torque_profile_Nm": torques[1:],
        "mu_profile": mu,
        "mu_from_mesh": bool(geom.get("mu_from_mesh", False)),
    }


def _zero_crossing(positions: np.ndarray, values: np.ndarray) -> float:
    """Primeira troca de sinal de ``values``, interpolada linearmente em ``positions``."""
    if values.size < 2:
        return float("nan")
    sign_change = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    if sign_change.size == 0:
        return float("nan")
    index = int(sign_change[0])
    v0, v1 = float(values[index]), float(values[index + 1])
    p0, p1 = float(positions[index]), float(positions[index + 1])
    if v1 == v0:
        return p0
    return p0 + (p1 - p0) * (-v0) / (v1 - v0)


def mechanical_summary(Data, Mesh, l1: float, R: float, geometry: dict | None = None) -> dict:
    """Forças axiais, torque e ponto neutro de um candidato.

    Agrupa as três chamadas mecânicas para que o caminho numérico discretize a
    trajetória e consulte o modelo geológico uma vez, e não três, por candidato.
    """
    if not use_numeric_friction(Data, Mesh):
        up = up_tension(Data, l1, R)
        down = down_tension(Data, l1, R)
        return {
            "up_force_1": float(up[0]), "up_force_2": float(up[1]), "up_force_3": float(up[2]),
            "down_force_1": float(down[0]), "down_force_2": float(down[1]),
            "down_force_3": float(down[2]), "torque": float(down[3]),
            "neutral_line": float(Nl(Data, l1, R)),
            "mu_from_mesh": False,
        }

    geom = geometry if geometry is not None else evaluate_trajectory(Data, Mesh, l1, R)
    up = soft_string_profile(Data, Mesh, l1, R, "up", geometry=geom)
    down = soft_string_profile(Data, Mesh, l1, R, "down", geometry=geom)
    return {
        "up_force_1": float(up["tension_1"]), "up_force_2": float(up["tension_2"]),
        "up_force_3": float(up["tension_3"]),
        "down_force_1": float(down["tension_1"]), "down_force_2": float(down["tension_2"]),
        "down_force_3": float(down["tension_3"]), "torque": float(down["torque"]),
        "neutral_line": float(down["neutral_point_m"]),
        "mu_from_mesh": True,
    }


def trajectory_plot_data(Data, l1: float, R: float) -> dict:
    """Coordenadas no plano do poço ``(s, z)`` e no mundo ``(x, y, z)`` para os gráficos Tipo 1."""
    config = validate_configuration(Data, l1, R)
    l1 = config["l1"]
    R = config["R"]
    curve_s, curve_z = curve_points_plane(Data, l1, R)
    curve_x, curve_y, curve_zw = curve_points(Data, l1, R)

    p0_s, p0_z = 0.0, float(Data.P0[2])
    p1_s, p1_z = 0.0, float(Data.P0[2] + l1)
    p2_s, p2_z = float(curve_s[-1]), float(curve_z[-1])
    p3_s, p3_z = float(Data.departure), float(Data.P3[2])
    center_s, center_z = float(R), float(Data.P0[2] + l1)

    l3 = float(config["l3"])
    lc = float(config["lc"])
    command_fraction = 0.0 if l3 <= 0.0 else max(0.0, min(1.0, (l3 - lc) / l3))
    command_s = p2_s + command_fraction * (p3_s - p2_s)
    command_z = p2_z + command_fraction * (p3_z - p2_z)

    p0 = Data.P0
    p1 = plane_to_world(Data, p1_s, p1_z)
    p2 = (float(curve_x[-1]), float(curve_y[-1]), float(curve_zw[-1]))
    p3 = Data.P3
    center = plane_to_world(Data, center_s, center_z)
    command_start = plane_to_world(Data, command_s, command_z)

    return {
        "config": config,
        "p0_s": p0_s, "p0_z": p0_z,
        "p1_s": p1_s, "p1_z": p1_z,
        "p2_s": p2_s, "p2_z": p2_z,
        "p3_s": p3_s, "p3_z": p3_z,
        "center_s": center_s, "center_z": center_z,
        "curve_s": curve_s, "curve_z": curve_z,
        "curve_x": curve_x, "curve_y": curve_y, "curve_zw": curve_zw,
        "command_s": command_s, "command_z": command_z,
        "p0": p0, "p1": p1, "p2": p2, "p3": p3, "center": center,
        "command_start": command_start,
        "p1_plane": (p1_s, p1_z),
        "p2_plane": (p2_s, p2_z),
        "p3_plane": (p3_s, p3_z),
        "center_plane": (center_s, center_z),
        "command_start_plane": (command_s, command_z),
    }


def mesh_cross_section(Data, Mesh, s_range, z_range, ns: int = 500, nz: int = 500):
    """Amostra o modelo geológico no plano vertical que contém o poço.

    Um poço Tipo 1 fica no plano vertical que passa por ``P0`` com o azimute do
    alvo; cada nó ``(s, z)`` desse plano é levado a coordenadas do mundo e o
    modelo é consultado ali. Ao contrário dos retângulos por intervalo de
    profundidade que substitui, mostra contatos inclinados e mudanças laterais
    de fácies como são de fato ao longo do poço.

    Returns
    -------
    tuple
        ``(s_values, z_values, codes)`` com ``codes`` de forma ``(nz, ns)``.
    """
    s_values = np.linspace(float(s_range[0]), float(s_range[1]), int(ns))
    z_values = np.linspace(float(z_range[0]), float(z_range[1]), int(nz))
    S, Z = np.meshgrid(s_values, z_values)
    X, Y, _ = plane_to_world(Data, S, Z)
    codes = Mesh.lithology_code_at_points(X, Y, Z)
    return s_values, z_values, codes
