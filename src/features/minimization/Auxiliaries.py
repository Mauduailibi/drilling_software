import numpy as np


EPS = 1.0e-10


def signal_change(s) -> tuple:
    """Return whether the sign changes along the array and the first index."""
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
    """Estimated final angle in the curve (radians)."""
    x3, y3 = Data.P3
    radicand_l3 = ((x3 - R) ** 2) + (y3 - l1) ** 2 - (R**2)
    if radicand_l3 <= EPS:
        raise ValueError("Invalid geometry: l3 is not real or non-positive.")
    l3 = np.sqrt(radicand_l3)

    disc = R**2 - l1**2 + 2 * l1 * y3 + l3**2 - y3**2
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
    """Returns the lengths l1, l2, and l3 of the three sections."""
    angle = theta(Data, l1, R)
    l3_sq = ((Data.P3[0] - R) ** 2) + (Data.P3[1] - l1) ** 2 - (R**2)
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

    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command**2 - d_int_command**2) / 4)

    denom = lambd_command * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_command)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in command section.")
    neutral_line = (z + ff) / denom

    if neutral_line < lc:
        return float(neutral_line)

    y_ic = y_f - (lc * np.cos(angle))
    fic = (
        ro_fluid
        * g
        * y_ic
        * np.pi
        * ((d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2))
        / 4
    )

    denom = lambd_heavy * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in heavy-pipe section.")
    a = (ff + z - fic) / denom
    b = lc * (
        (lambd_command - lambd_heavy) * np.cos(angle)
        - (
            µ
            * (
                (lambd_command * (1 - (ro_fluid / ro_command)))
                - (lambd_heavy * (1 - (ro_fluid / ro_heavypipe)))
            )
        )
        * np.sin(angle)
    )
    c = lambd_heavy * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle))
    _denominator_check(c, "Invalid neutral-line correction in heavy-pipe section.")
    neutral_line = a - (b / c)

    if neutral_line <= (lc + lp):
        return float(neutral_line)

    y_ip = y_f - ((lc + lp) * np.cos(angle))
    fip = (
        ro_fluid
        * g
        * y_ip
        * np.pi
        * ((d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2))
        / 4
    )

    denom = lambd_drill * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle))
    _denominator_check(denom, "Invalid neutral-line calculation in drill-pipe section.")
    a = (ff + z - fic - fip) / denom
    b = lc * (
        (lambd_command - lambd_drill) * np.cos(angle)
        - (
            µ
            * (
                (lambd_command * (1 - (ro_fluid / ro_command)))
                - (lambd_drill * (1 - (ro_fluid / ro_drillpipe)))
            )
        )
        * np.sin(angle)
    )
    c = lambd_drill * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle))
    _denominator_check(c, "Invalid neutral-line correction in drill-pipe section.")
    d = lp * (
        (lambd_heavy - lambd_drill) * np.cos(angle)
        - (
            µ
            * (
                (lambd_heavy * (1 - (ro_fluid / ro_heavypipe)))
                - (lambd_drill * (1 - (ro_fluid / ro_drillpipe)))
            )
        )
        * np.sin(angle)
    )

    neutral_line = a - (b / c) - (d / c)
    return float(neutral_line)


def _vertical_effective_weight(Data, l1: float) -> float:
    return (Data.lambd_drill - Data.ro_fluid * Data.area_drill) * Data.g * l1


def validate_configuration(Data, l1, R, angle_limit_deg: float | None = None) -> dict:
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
        raise ValueError(
            "Configuration rejected: the tangent section is shorter than command + heavy pipe."
        )

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
    fic = (
        ro_fluid
        * g
        * y_ic
        * np.pi
        * ((d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2))
        / 4
    )
    fip = (
        ro_fluid
        * g
        * y_ip
        * np.pi
        * ((d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2))
        / 4
    )
    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command**2 - d_int_command**2) / 4)

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
            c1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            c2 = -((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))
            tension_2 = c1 * np.sin(0) + c2 * np.cos(0) + k * (np.exp(-µ * 0))
        else:
            a = lambd_drill * g * R
            b1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            b2 = ((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))
            tension_2 = b1 * np.sin(0) + b2 * np.cos(0) + k * (np.exp(µ * 0))
    else:
        a = lambd_drill * g * R
        b1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        b2 = ((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))

        tension_change = (
            b1 * np.sin(angle_variation[condition_where_change])
            + b2 * np.cos(angle_variation[condition_where_change])
            + k * (np.exp(µ * angle_variation[condition_where_change]))
        )

        c1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        c2 = -((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_change - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))
        tension_2 = c1 * np.sin(0) + c2 * np.cos(0) + k * (np.exp(-µ * 0))

    tension_1 = tension_2 + _vertical_effective_weight(Data, l1)
    return float(tension_1), float(tension_2), float(tension_3)


def down_tension(Data, l1, R) -> list:
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
    ff = ro_fluid * g * y_f * (np.pi * (d_ext_command**2 - d_int_command**2) / 4)
    fic = (
        ro_fluid
        * g
        * y_ic
        * np.pi
        * ((d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2))
        / 4
    )
    fip = (
        ro_fluid
        * g
        * y_ip
        * np.pi
        * ((d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2))
        / 4
    )

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
            b1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            b2 = ((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            k = (tension_3 - b1 * np.sin(angle) - b2 * np.cos(angle)) / (np.exp(µ * angle))

            tension_2 = b1 * np.sin(0) + b2 * np.cos(0) + k * (np.exp(µ * 0))

            b3 = b1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
            medial_n_force = b3 * (1 - np.cos(angle)) + b2 * np.sin(angle) + (k / µ) * ((np.exp(µ * angle)) - 1)
            frictional_torque_2 = (µ * medial_n_force * d_ext_drill) / 2
        else:
            a = lambd_drill * g * R
            c1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            c2 = -((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
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
        c1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        c2 = -((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (tension_3 - c1 * np.sin(angle) - c2 * np.cos(angle)) / (np.exp(-µ * angle))

        c3 = c1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)
        medial_n_force = -c3 * (np.cos(angle_variation[condition_where_change]) - np.cos(angle)) + c2 * (
            np.sin(angle_variation[condition_where_change]) - np.sin(angle)
        ) - (k / µ) * (
            np.exp(-µ * angle_variation[condition_where_change]) - np.exp(-µ * angle)
        )
        frictional_torque_2_change = -((µ * medial_n_force * d_ext_drill) / 2)

        tension_change = (
            c1 * np.sin(angle_variation[condition_where_change])
            + c2 * np.cos(angle_variation[condition_where_change])
            + k * (np.exp(-µ * angle_variation[condition_where_change]))
        )

        b1 = (a / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        b2 = ((a * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        k = (
            tension_change
            - b1 * np.sin(angle_variation[condition_where_change])
            - b2 * np.cos(angle_variation[condition_where_change])
        ) / (np.exp(µ * angle_variation[condition_where_change]))

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


import numpy as np

LITHOLOGY_COLORS = {
    "Sandstone": "#d8b365",
    "Limestone": "#f6e8c3",
    "Dolomite": "#5ab4ac",
    "Evaporite": "#c2a5cf",
    "Undefined": "#dddddd",
}


def inclination_angle_deg(dx: float, dy: float) -> float:
    """Angle relative to the vertical direction in degrees."""
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + EPS)))


def dogleg_severity_deg_per_30m(curvature: float) -> float:
    """Convert curvature (1/m) to dogleg severity in deg/30 m."""
    if curvature <= 0.0:
        return 0.0
    return float(np.degrees(curvature * 30.0))


def inclination_factor(angle_deg: float, params: dict) -> float:
    reduction = float(params["inclination_reduction"])
    exponent = float(params["inclination_exponent"])
    lower = float(params["min_inclination_factor"])
    normalized = min(max(angle_deg / 90.0, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized**exponent)
    return float(max(lower, min(1.0, factor)))


def dls_factor(dls_deg_per_30m: float, params: dict) -> float:
    reduction = float(params["dls_reduction"])
    exponent = float(params["dls_exponent"])
    lower = float(params["min_dls_factor"])
    reference = float(params["reference_dls_deg_per_30m"])
    normalized = min(max(dls_deg_per_30m / reference, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized**exponent)
    return float(max(lower, min(1.0, factor)))


def wob_transfer_factor(angle_deg: float, dls_deg_per_30m: float, params: dict) -> float:
    """Fraction of surface WOB effectively transferred to the bit."""
    reference_dls = float(params["reference_dls_deg_per_30m"])
    a = float(params["drag_inclination_coeff"])
    b = float(params["drag_dls_coeff"])
    exponent = float(params["wob_transfer_exponent"])
    inc_term = np.sin(np.radians(max(angle_deg, 0.0)))
    dls_term = max(dls_deg_per_30m, 0.0) / reference_dls
    transfer = np.exp(-(a * (inc_term**exponent) + b * (dls_term**exponent)))
    return float(min(1.0, max(0.0, transfer)))


def wob_factor(wob_effective: float, params: dict) -> float:
    optimal_wob = float(params["optimal_wob"])
    lower = float(params["min_wob_factor"])
    exponent = float(params["wob_factor_exponent"])
    ratio = max(wob_effective / optimal_wob, 0.0)
    factor = min(1.0, ratio**exponent)
    return float(max(lower, factor))


def local_contact_force_per_length(Data, angle_deg: float, curvature: float, wob_effective: float) -> float:
    """Simple contact-force estimate per length.

    The first term represents lateral contact generated by buoyed weight in an
    inclined interval, while the second term introduces additional contact due
    to curvature under the applied WOB.
    """
    angle_rad = np.radians(max(angle_deg, 0.0))
    gravity_contact = abs(Data.buoyed_linear_weight_avg) * np.sin(angle_rad)
    curvature_contact = abs(wob_effective) * max(curvature, 0.0)
    return float(gravity_contact + curvature_contact)


def torque_factor(cumulative_torque: float, params: dict) -> float:
    reduction = float(params["torque_reduction"])
    exponent = float(params["torque_exponent"])
    lower = float(params["min_torque_factor"])
    limit = float(params["torque_limit"])
    normalized = min(max(cumulative_torque / limit, 0.0), 1.0)
    factor = 1.0 - reduction * (normalized**exponent)
    return float(max(lower, min(1.0, factor)))


def _line_elements(x0: float, y0: float, x1: float, y1: float, n_steps: int, section: str, curvature: float = 0.0):
    elements = []
    xs = np.linspace(x0, x1, n_steps + 1)
    ys = np.linspace(y0, y1, n_steps + 1)
    for i in range(n_steps):
        xa, xb = float(xs[i]), float(xs[i + 1])
        ya, yb = float(ys[i]), float(ys[i + 1])
        dx = xb - xa
        dy = yb - ya
        ds = float(np.hypot(dx, dy))
        if ds <= EPS:
            continue
        angle = inclination_angle_deg(dx, dy)
        elements.append(
            {
                "section": section,
                "x0": xa,
                "y0": ya,
                "x1": xb,
                "y1": yb,
                "x_mid": 0.5 * (xa + xb),
                "y_mid": 0.5 * (ya + yb),
                "dx": dx,
                "dy": dy,
                "length": ds,
                "inclination_deg": angle,
                "curvature": float(curvature),
                "dls_deg_per_30m": dogleg_severity_deg_per_30m(curvature),
            }
        )
    return elements


def trajectory_elements(Data, l1: float, R: float, ds_target: float | None = None):
    """Discretize the Type-1 trajectory into local elements."""
    config = validate_configuration(Data, l1, R)
    l1 = config["l1"]
    l2 = config["l2"]
    l3 = config["l3"]
    angle = config["angle"]

    if ds_target is None:
        ds_target = float(Data.drilling_time_parameters["trajectory_step"])
    if ds_target <= 0:
        raise ValueError("'ds_target' must be positive.")

    elements = []

    n1 = max(1, int(np.ceil(l1 / ds_target)))
    elements.extend(_line_elements(Data.P0[0], Data.P0[1], Data.P0[0], Data.P0[1] + l1, n1, "vertical", 0.0))

    n2 = max(20, int(np.ceil(l2 / ds_target)))
    phi = np.linspace(0.0, angle, n2 + 1)
    x_arc = Data.P0[0] + R * (1.0 - np.cos(phi))
    y_arc = Data.P0[1] + l1 + R * np.sin(phi)
    for i in range(n2):
        xa, xb = float(x_arc[i]), float(x_arc[i + 1])
        ya, yb = float(y_arc[i]), float(y_arc[i + 1])
        dx = xb - xa
        dy = yb - ya
        ds = float(np.hypot(dx, dy))
        if ds <= EPS:
            continue
        phi_mid = 0.5 * (phi[i] + phi[i + 1])
        curvature = float(1.0 / R)
        elements.append(
            {
                "section": "curve",
                "x0": xa,
                "y0": ya,
                "x1": xb,
                "y1": yb,
                "x_mid": 0.5 * (xa + xb),
                "y_mid": 0.5 * (ya + yb),
                "dx": dx,
                "dy": dy,
                "length": ds,
                "inclination_deg": float(np.degrees(phi_mid)),
                "curvature": curvature,
                "dls_deg_per_30m": dogleg_severity_deg_per_30m(curvature),
            }
        )

    x2 = float(x_arc[-1])
    y2 = float(y_arc[-1])
    x3 = float(Data.P3[0])
    y3 = float(Data.P3[1])
    n3 = max(1, int(np.ceil(l3 / ds_target)))
    elements.extend(_line_elements(x2, y2, x3, y3, n3, "tangent", 0.0))

    return elements