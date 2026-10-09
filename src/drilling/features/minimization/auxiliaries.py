
"""Geometria, forças e núcleos de fator de ROP da trajetória Tipo 1.

Auxiliares geométricos públicos
-------------------------------
``theta``, ``lenght``, ``curve_points``, ``validate_configuration``,
``up_tension``, ``down_tension``, ``buckling``, ``Nl``, ``trajectory_elements``.

A grafia ``lenght`` faz parte da API pública histórica e não deve ser
renomeada nesta fase. As fórmulas deste arquivo estão congeladas pelos testes golden.
"""

import numpy as np


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


def _reconstruct_target_residual(Data, l1: float, R: float, angle: float, l3: float) -> float:
    x3, y3 = Data.P3
    rx = -R * np.cos(angle) + l3 * np.sin(angle)
    ry = l3 * np.cos(angle) + R * np.sin(angle)
    return abs(rx - (x3 - R)) + abs(ry - (y3 - l1))


def theta(Data, l1, R) -> float:
    """Inclinação final estimada da curva de build-up, em radianos.

    Parameters
    ----------
    Data : DataSet
        Geometria do poço (usa ``P3``).
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
    x3, y3 = Data.P3
    radicand_l3 = ((x3 - R) ** 2) + (y3 - l1) ** 2 - (R ** 2)
    if radicand_l3 <= EPS:
        raise ValueError("Invalid geometry: l3 is not real or non-positive.")
    l3 = np.sqrt(radicand_l3)

    disc = R ** 2 - l1 ** 2 + 2 * l1 * y3 + l3 ** 2 - y3 ** 2
    if disc < -EPS:
        raise ValueError("Invalid geometry: discriminant is negative.")
    disc = max(disc, 0.0)

    den = -l1 + l3 + y3
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
    l3_sq = ((Data.P3[0] - R) ** 2) + (Data.P3[1] - l1) ** 2 - (R ** 2)
    if l3_sq <= EPS:
        raise ValueError("Invalid geometry: the tangent section length is not positive.")
    l3 = float(np.sqrt(l3_sq))
    l2 = float(R * angle)
    return float(l1), l2, l3


def curve_points(Data, l1, R) -> list:
    angle = np.linspace(np.pi, np.pi + theta(Data, l1, R), 1000)
    l1, *_ = lenght(Data, l1, R)
    y = Data.P0[1] + l1 - R * np.sin(angle)
    x = Data.P0[0] + R * np.cos(angle) + R
    return x, y


def points_coordinates(Data, l1, R) -> list:
    x = [Data.P0[0], Data.P0[0]]
    y = [Data.P0[1], Data.P0[1] + l1]

    points = curve_points(Data, l1, R)
    for i in range(len(points[0])):
        x.append(points[0][i])
        y.append(points[1][i])

    x.append(Data.P3[0])
    y.append(Data.P3[1])
    return x, y


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


def Nl(Data, l1, R) -> float:
    lc = buckling(Data, l1, R)
    angle = theta(Data, l1, R)
    y_f = Data.P3[1]
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

    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)

    denom = lambd_command * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_command)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in command section.")
    neutral_line = (z + ff) / denom

    if neutral_line < lc:
        return float(neutral_line)

    y_ic = y_f - (lc * np.cos(angle))
    fic = (
        ro_fluid * g * y_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
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

    y_ip = y_f - ((lc + lp) * np.cos(angle))
    fip = (
        ro_fluid * g * y_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4
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


def up_tension(Data, l1, R) -> list:
    """Força axial no topo de L1, no início da curva e em P3 (içamento).

    Parameters
    ----------
    Data : DataSet
        Propriedades mecânicas.
    l1, R : float
        Configuração em avaliação.

    Returns
    -------
    tuple of float
        ``(tension_1, tension_2, tension_3)`` em newtons.
    """
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

    y_f = p3[1]
    y_ic = y_f - (lc * np.cos(angle))
    y_ip = y_f - ((lc + lp) * np.cos(angle))
    fic = ro_fluid * g * y_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
    fip = ro_fluid * g * y_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4
    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)

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


def down_tension(Data, l1, R) -> list:
    """Força axial ao descer a coluna, mais o torque de atrito.

    Parameters
    ----------
    Data : DataSet
        Propriedades mecânicas.
    l1, R : float
        Configuração em avaliação.

    Returns
    -------
    tuple of float
        ``(tension_1, tension_2, tension_3, torque)`` em N e N·m.
    """
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

    y_f = p3[1]
    y_ic = y_f - (lc * np.cos(angle))
    y_ip = y_f - ((lc + lp) * np.cos(angle))
    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command ** 2 - d_int_command ** 2) / 4)
    fic = ro_fluid * g * y_ic * np.pi * ((d_ext_command ** 2 - d_ext_heavy ** 2) - (d_int_command ** 2 - d_int_heavy ** 2)) / 4
    fip = ro_fluid * g * y_ip * np.pi * ((d_ext_heavy ** 2 - d_ext_drill ** 2) - (d_int_heavy ** 2 - d_int_drill ** 2)) / 4

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


LITHOLOGY_COLORS = {
    "Shale": "#8c8c8c",
    "Siltstone": "#bdb76b",
    "Sandstone": "#d8b365",
    "Limestone": "#f6e8c3",
    "Dolomite": "#5ab4ac",
    "Evaporite": "#c2a5cf",
    "Marl": "#9ecae1",
    "Undefined": "#dddddd",
}


def inclination_angle_deg(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + EPS)))


def dogleg_severity_deg_per_30m(curvature: float) -> float:
    if curvature <= 0.0:
        return 0.0
    return float(np.degrees(curvature * 30.0))


# Os fatores abaixo aceitam escalares ou arrays NumPy (um valor por elemento do poço).


def inclination_factor(angle_deg, params: dict):
    reduction = float(params["inclination_reduction"])
    exponent = float(params["inclination_exponent"])
    lower = max(0.85, float(params["min_inclination_factor"]))
    normalized = np.minimum(np.maximum(angle_deg / 90.0, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.maximum(lower, np.minimum(1.0, factor))


def dls_factor(dls_deg_per_30m, params: dict):
    reduction = float(params["dls_reduction"])
    exponent = float(params["dls_exponent"])
    lower = max(0.50, float(params["min_dls_factor"]))
    reference = float(params["reference_dls_deg_per_30m"])
    normalized = np.minimum(np.maximum(dls_deg_per_30m / reference, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.maximum(lower, np.minimum(1.0, factor))


def wob_transfer_factor(angle_deg, dls_deg_per_30m, params: dict):
    reference_dls = float(params["reference_dls_deg_per_30m"])
    a = float(params["drag_inclination_coeff"])
    b = float(params["drag_dls_coeff"])
    exponent = float(params["wob_transfer_exponent"])
    inc_term = np.sin(np.radians(np.maximum(angle_deg, 0.0)))
    dls_term = np.maximum(dls_deg_per_30m, 0.0) / reference_dls
    transfer = np.exp(-(a * (inc_term ** exponent) + b * (dls_term ** exponent)))
    return np.minimum(1.0, np.maximum(0.0, transfer))


def wob_factor(wob_effective, params: dict):
    optimal_wob = float(params["optimal_wob"])
    lower = max(0.90, float(params["min_wob_factor"]))
    exponent = float(params["wob_factor_exponent"])
    ratio = np.maximum(wob_effective / optimal_wob, 0.0)
    factor = np.minimum(1.0, ratio ** exponent)
    return np.maximum(lower, factor)


def local_contact_force_per_length(Data, angle_deg, curvature, wob_effective):
    angle_rad = np.radians(np.maximum(angle_deg, 0.0))
    gravity_contact = abs(Data.buoyed_linear_weight_avg) * np.sin(angle_rad)
    curvature_contact = np.abs(wob_effective) * np.maximum(curvature, 0.0)
    return gravity_contact + curvature_contact


def torque_factor(cumulative_torque, params: dict):
    reduction = float(params["torque_reduction"])
    exponent = float(params["torque_exponent"])
    lower = max(0.90, float(params["min_torque_factor"]))
    limit = float(params["torque_limit"])
    normalized = np.minimum(np.maximum(cumulative_torque / limit, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized ** exponent)
    return np.maximum(lower, np.minimum(1.0, factor))


ELEMENT_KEYS = ("x0", "y0", "x1", "y1", "x_mid", "y_mid", "dx", "dy", "length", "inclination_deg", "curvature", "dls_deg_per_30m")
"""Campos numéricos de cada elemento do poço; ``section`` é o único campo de texto."""


def _element_arrays(x, y, section: str, curvature: float, inclination_deg=None) -> dict:
    """Elementos entre pontos consecutivos de ``x``, ``y``; descarta os de comprimento nulo."""
    xa, xb = x[:-1], x[1:]
    ya, yb = y[:-1], y[1:]
    dx = xb - xa
    dy = yb - ya
    ds = np.hypot(dx, dy)
    if inclination_deg is None:
        inclination_deg = np.degrees(np.arctan2(np.abs(dx), np.abs(dy) + EPS))
    keep = ds > EPS
    n = int(keep.sum())
    return {
        "section": [section] * n,
        "x0": xa[keep], "y0": ya[keep], "x1": xb[keep], "y1": yb[keep],
        "x_mid": 0.5 * (xa + xb)[keep], "y_mid": 0.5 * (ya + yb)[keep],
        "dx": dx[keep], "dy": dy[keep], "length": ds[keep],
        "inclination_deg": np.broadcast_to(inclination_deg, ds.shape)[keep],
        "curvature": np.full(n, float(curvature)),
        "dls_deg_per_30m": np.full(n, dogleg_severity_deg_per_30m(curvature)),
    }


def trajectory_element_arrays(Data, l1: float, R: float, ds_target: float | None = None) -> dict:
    """Discretiza o poço Tipo 1 em elementos, com um array NumPy por campo.

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
        ``section`` (lista) e os arrays de ``ELEMENT_KEYS``, na ordem do poço:
        trecho vertical, curva e tangente.
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

    x0, y0 = float(Data.P0[0]), float(Data.P0[1])
    n1 = max(1, int(np.ceil(l1 / ds_target)))
    vertical = _element_arrays(np.linspace(x0, x0, n1 + 1), np.linspace(y0, y0 + l1, n1 + 1), "vertical", 0.0)

    n2 = max(20, int(np.ceil(l2 / ds_target)))
    phi = np.linspace(0.0, angle, n2 + 1)
    x_arc = Data.P0[0] + R * (1.0 - np.cos(phi))
    y_arc = Data.P0[1] + l1 + R * np.sin(phi)
    phi_mid = 0.5 * (phi[:-1] + phi[1:])
    curve = _element_arrays(x_arc, y_arc, "curve", float(1.0 / R), inclination_deg=np.degrees(phi_mid))

    x2 = float(x_arc[-1])
    y2 = float(y_arc[-1])
    x3 = float(Data.P3[0])
    y3 = float(Data.P3[1])
    n3 = max(1, int(np.ceil(l3 / ds_target)))
    tangent = _element_arrays(np.linspace(x2, x3, n3 + 1), np.linspace(y2, y3, n3 + 1), "tangent", 0.0)

    parts = (vertical, curve, tangent)
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in ELEMENT_KEYS}
    arrays["section"] = [name for part in parts for name in part["section"]]
    return arrays


def trajectory_elements(Data, l1: float, R: float, ds_target: float | None = None):
    """Discretiza o poço Tipo 1 em elementos de trecho vertical, curva e tangente.

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
    list of dict
        Cada elemento tem extremidades, comprimento, inclinação, curvatura e DLS.
        Mesmo conteúdo de ``trajectory_element_arrays``, um dicionário por elemento.
    """
    arrays = trajectory_element_arrays(Data, l1, R, ds_target=ds_target)
    columns = [arrays[key].tolist() for key in ELEMENT_KEYS]
    return [
        {"section": section, **dict(zip(ELEMENT_KEYS, values))}
        for section, *values in zip(arrays["section"], *columns)
    ]
