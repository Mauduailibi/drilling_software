
"""Busca em malha de candidatos Tipo 1 e decomposição do tempo de broca.

``_scan_candidates`` enumera o domínio padrão L1 × R. ``minimal_tension``
e ``minimal_torque`` escolhem os ótimos mecânicos. ``drilling_time_breakdown``
transforma um par (L1, R) em uma estimativa de tempo sobre o modelo geológico.

``Mesh`` (um ``HorizonModel``) só entra nos objetivos mecânicos quando
``Data.friction_model == "lithology"``; no modelo de atrito constante o
resultado mecânico independe da geologia e o cache ignora a malha.

Os auxiliares de ``print`` em CLI deste arquivo são saída de pesquisa legado;
a GUI usa ``drilling.features.minimization.plot``. Não altere passos de
varredura nem fórmulas aqui sem atualizar os snapshots golden.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drilling.features.minimization.auxiliaries as ax


DEFAULT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "figure.figsize": (7.5, 5.0),
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}

_MECH_CACHE: dict[tuple, list[dict]] = {}
_TIME_CACHE: dict[tuple, list[dict]] = {}


def _data_signature(Data) -> tuple:
    if hasattr(Data, "cache_signature"):
        return Data.cache_signature()
    return (id(Data),)


def _mesh_signature(Mesh) -> tuple:
    if hasattr(Mesh, "cache_signature"):
        return Mesh.cache_signature()
    return (id(Mesh),)


def _apply_plot_style() -> None:
    plt.rcParams.update(DEFAULT_STYLE)


def _require_candidates(candidates):
    if not candidates:
        raise ValueError("No valid configuration was found in the searched domain.")


def _mech_cache_key(Data, Mesh) -> tuple:
    # The mechanical result only depends on the geology when friction comes from it,
    # so the mesh is kept out of the key otherwise to preserve cache hits.
    mesh_part = _mesh_signature(Mesh) if ax.use_numeric_friction(Data, Mesh) else None
    return (_data_signature(Data), mesh_part)


def _scan_candidates(Data, Mesh=None):
    key = _mech_cache_key(Data, Mesh)
    if key in _MECH_CACHE:
        return _MECH_CACHE[key]

    min_l1 = float(getattr(Data, "min_l1", 100.0))
    l1_values = np.arange(min_l1, Data.max + Data.l1_step, Data.l1_step)
    r_values = np.arange(Data.min_radius, Data.max_radius + Data.radius_step, Data.radius_step)

    candidates = []
    for l1 in l1_values:
        for R in r_values:
            try:
                config = ax.validate_configuration(Data, l1, R)
                config.update(ax.mechanical_summary(Data, Mesh, l1, R))
                candidates.append(config)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue

    _MECH_CACHE[key] = candidates
    return candidates


def minimal_tension(Data, Mesh=None) -> list:
    candidates = _scan_candidates(Data, Mesh)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["up_force_1"])
    return [best["l1"], best["R"]]


def minimal_torque(Data, Mesh=None) -> list:
    candidates = _scan_candidates(Data, Mesh)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["torque"])
    return [best["l1"], best["R"]]


def drilling_informations(Data, Mesh=None) -> list:
    candidates = _scan_candidates(Data, Mesh)
    _require_candidates(candidates)
    best_tension = min(candidates, key=lambda item: item["up_force_1"])
    best_torque = min(candidates, key=lambda item: item["torque"])

    def _pack(best):
        return [
            (best["up_force_1"], best["up_force_2"], best["up_force_3"]),
            (best["down_force_1"], best["down_force_2"], best["down_force_3"], best["torque"]),
            best["angle_deg"],
            best["neutral_line"],
            (best["l1"], best["l2"], best["l3"]),
            best["lc"],
            best["l1"],
            best["R"],
        ]

    return [_pack(best_tension), _pack(best_torque)]


def drilling_informations_table(data, Mesh=None):
    results = drilling_informations(data, Mesh)
    for i, result in enumerate(results):
        up_forces, down_forces, angle, neutral_line, lengths, length_command, l1, R = result
        l1, l2, l3 = lengths
        f1_up, f2_up, f3_up = up_forces
        f1_down, f2_down, f3_down, torque = down_forces
        values = np.round([l1, l2, l3, R, f1_up, f2_up, f3_up, f1_down, f2_down, f3_down, torque, angle, neutral_line, length_command], 2)
        index_labels = [
            "L1 (m):", "L2 (m):", "L3 (m):", "Radius (m):",
            "Up axial force L1 (N):", "Up axial force L2 (N):", "Up axial force L3 (N):",
            "Down axial force L1 (N):", "Down axial force L2 (N):", "Down axial force L3 (N):",
            "Torque (N*m):", "Angle (°):", "Neutral line (m):", "Length command (m):",
        ]
        print("\n--- Result table for minimal axial force ---" if i == 0 else "\n--- Result table for minimal torque ---")
        table = pd.DataFrame(values, columns=[""], index=index_labels)
        print(table)
        print("")


def _group_totals(keys: np.ndarray, lengths: np.ndarray, times: np.ndarray) -> dict:
    """Comprimento perfurado e tempo totais por chave distinta, na ordem de primeira ocorrência."""
    totals = {}
    for key in dict.fromkeys(keys.tolist()):
        mask = keys == key
        totals[str(key)] = {
            "length_m": float(lengths[mask].sum()),
            "time_h": float(times[mask].sum()),
        }
    return totals


def drilling_time_breakdown(
    Data,
    Mesh,
    l1: float,
    R: float,
    ds_target: float | None = None,
    detail: bool = True,
    geometry: dict | None = None,
) -> dict:
    """Integra o tempo de broca ao longo da trajetória discretizada.

    Parameters
    ----------
    Data : DataSet
        Dados mecânicos e parâmetros de fator de ROP.
    Mesh : HorizonModel
        Modelo geológico de onde vêm a ROP base (e ``mu``, se aplicável).
    l1, R : float
        Configuração em cronometragem.
    ds_target : float or None, optional
        Comprimento de elemento encaminhado a ``trajectory_arrays``.
    detail : bool, optional
        ``False`` pula a montagem das linhas por elemento: a varredura de
        candidatos avalia milhares de trajetórias e só precisa dos totais.
    geometry : dict or None, optional
        Saída pré-calculada de ``auxiliaries.evaluate_trajectory``.

    Returns
    -------
    dict
        Linhas por elemento (vazias com ``detail=False``), totais,
        decomposições por litologia e trecho, ``geometry`` e ``arrays`` (as
        colunas consumidas pelo modelo operacional).
    """
    if Mesh is None and geometry is None:
        raise ValueError("drilling_time_breakdown needs a geological model to read the ROP from.")
    geom = geometry if geometry is not None else ax.evaluate_trajectory(
        Data, Mesh, l1, R, ds_target=ds_target
    )
    config = geom["config"]
    params = Data.drilling_time_parameters

    ds = geom["length"]
    angle_deg = geom["inclination_deg"]
    dls = geom["dls_deg_per_30m"]
    curvature = geom["curvature"]
    rop_base = geom["rop_base"]
    mu = geom["mu"]

    f_inc_raw = ax.inclination_factor(angle_deg, params)
    f_dls = ax.dls_factor(dls, params)
    wob_transfer = ax.wob_transfer_factor(angle_deg, dls, params)
    wob_effective = float(params["surface_wob"]) * wob_transfer
    f_wob = ax.wob_factor(wob_effective, params)

    contact_force = ax.local_contact_force_per_length(Data, angle_deg, curvature, wob_effective)
    torque_increment = mu * contact_force * ds * float(params["bit_radius"])
    cumulative_torque = np.cumsum(torque_increment)
    f_torque = ax.torque_factor(cumulative_torque, params)

    # Inside a build section the dogleg penalty already stands in for the
    # inclination penalty, so the two are not compounded.
    f_dls_active = f_dls < (1.0 - ax.EPS)
    f_inc = np.where(f_dls_active, 1.0, f_inc_raw)
    f_rop_total = f_inc * f_dls * f_wob * f_torque

    rop_effective = rop_base * f_rop_total
    if np.any(~np.isfinite(rop_effective)) or np.any(rop_effective <= 0):
        raise ValueError("The effective ROP became non-positive.")
    time_h = ds / rop_effective

    lithology = geom["lithology"]
    section = geom["section"]
    total_time_h = float(time_h.sum())
    total_length = float(ds.sum())

    rows = []
    if detail:
        rows = [
            {
                "id": index + 1,
                "section": section[index],
                "depth_mid_m": float(geom["z_mid"][index]),
                "x_mid_m": float(geom["x_mid"][index]),
                "y_mid_m": float(geom["y_mid"][index]),
                "measured_depth_m": float(geom["measured_depth_m"][index]),
                "element_length_m": float(ds[index]),
                "inclination_deg": float(angle_deg[index]),
                "curvature_1pm": float(curvature[index]),
                "dls_deg_per_30m": float(dls[index]),
                "lithology": lithology[index],
                "mu": float(mu[index]),
                "rop_base_mph": float(rop_base[index]),
                "wob_transfer": float(wob_transfer[index]),
                "wob_effective_N": float(wob_effective[index]),
                "contact_force_per_length_Npm": float(contact_force[index]),
                "torque_increment_Nm": float(torque_increment[index]),
                "cumulative_torque_Nm": float(cumulative_torque[index]),
                "f_inclination": float(f_inc[index]),
                "f_inclination_raw": float(f_inc_raw[index]),
                "f_dls": float(f_dls[index]),
                "f_dls_active": bool(f_dls_active[index]),
                "f_wob": float(f_wob[index]),
                "f_torque": float(f_torque[index]),
                "f_rop_total": float(f_rop_total[index]),
                "rop_effective_mph": float(rop_effective[index]),
                "time_h": float(time_h[index]),
            }
            for index in range(geom["n_elements"])
        ]

    return {
        "l1": float(config["l1"]), "l2": float(config["l2"]), "l3": float(config["l3"]),
        "R": float(config["R"]), "angle_deg": float(config["angle_deg"]), "lc": float(config["lc"]),
        "ld": float(config["ld"]), "elements": rows,
        "by_lithology": _group_totals(lithology, ds, time_h),
        "by_section": _group_totals(section, ds, time_h),
        "total_length_m": total_length,
        "total_time_h": total_time_h,
        "average_rop_mph": float(total_length / total_time_h if total_time_h > 0 else np.nan),
        "max_cumulative_torque_Nm": float(cumulative_torque[-1]) if cumulative_torque.size else 0.0,
        "average_wob_N": float(np.mean(wob_effective)),
        "average_dls_deg_per_30m": float(np.mean(dls)),
        "mu_from_mesh": bool(geom.get("mu_from_mesh", False)),
        "geometry": geom,
        # The columns the operational model consumes, so it never has to be handed
        # thousands of per-element dictionaries just to iterate over them.
        "arrays": {
            "element_length_m": ds,
            "lithology": lithology,
            "dls_deg_per_30m": dls,
            "cumulative_torque_Nm": cumulative_torque,
            "time_h": time_h,
            "depth_end_m": np.maximum(geom["z0"], geom["z1"]),
        },
    }


def _scan_candidates_with_time(Data, Mesh):
    key = (_data_signature(Data), _mesh_signature(Mesh))
    if key in _TIME_CACHE:
        return _TIME_CACHE[key]

    base_candidates = _scan_candidates(Data, Mesh)
    candidates = []
    for candidate in base_candidates:
        try:
            # Summary only: keeping the per-element rows of every candidate would
            # hold millions of dictionaries in the cache for no benefit.
            timing = drilling_time_breakdown(
                Data, Mesh, candidate["l1"], candidate["R"], detail=False
            )
            merged = dict(candidate)
            merged.update(
                {
                    "drilling_time_h": float(timing["total_time_h"]),
                    "average_rop_mph": float(timing["average_rop_mph"]),
                }
            )
            candidates.append(merged)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue
    _TIME_CACHE[key] = candidates
    return candidates


def minimal_drilling_time(Data, Mesh) -> list:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)
    best = min(candidates, key=lambda item: item["drilling_time_h"])
    return [best["l1"], best["R"]]


def drilling_time_informations(Data, Mesh) -> dict:
    candidates = _scan_candidates_with_time(Data, Mesh)
    _require_candidates(candidates)
    best = dict(min(candidates, key=lambda item: item["drilling_time_h"]))
    best["timing"] = drilling_time_breakdown(Data, Mesh, best["l1"], best["R"])
    return best


def drilling_time_information_table(Data, Mesh) -> None:
    best = drilling_time_informations(Data, Mesh)
    timing = best["timing"]
    summary = pd.DataFrame(
        {
            "Value": np.round(
                [best["l1"], best["l2"], best["l3"], best["R"], best["angle_deg"], best["up_force_1"], best["torque"],
                 timing["total_length_m"], timing["total_time_h"], timing["average_rop_mph"], timing["average_wob_N"],
                 timing["average_dls_deg_per_30m"], timing["max_cumulative_torque_Nm"]], 3
            )
        },
        index=[
            "L1 (m)", "L2 (m)", "L3 (m)", "Radius (m)", "Angle (deg)", "Up axial force at top (N)",
            "Mechanical torque (N*m)", "Total trajectory length (m)", "Total drilling time (h)",
            "Average effective ROP (m/h)", "Average effective WOB (N)", "Average DLS (deg/30m)",
            "Max cumulative drilling torque (N*m)",
        ],
    )
    print("\n--- Result table for minimal drilling time ---")
    print(summary)
    print("")
    lith_df = pd.DataFrame(timing["by_lithology"]).T
    lith_df = lith_df[["length_m", "time_h"]].sort_index()
    lith_df["average_rop_mph"] = lith_df["length_m"] / lith_df["time_h"]
    lith_df = np.round(lith_df, 3)
    print("--- Time breakdown by lithology ---")
    print(lith_df)
    print("")
    section_df = pd.DataFrame(timing["by_section"]).T
    section_df = section_df[["length_m", "time_h"]].sort_index()
    section_df["average_rop_mph"] = section_df["length_m"] / section_df["time_h"]
    section_df = np.round(section_df, 3)
    print("--- Time breakdown by trajectory section ---")
    print(section_df)
    print("")


def optimization_summary_table(Data, Mesh=None) -> None:
    rows = []
    force_l1, force_R = minimal_tension(Data, Mesh)
    force_cfg = ax.validate_configuration(Data, force_l1, force_R)
    force_up = ax.up_tension(Data, force_l1, force_R, Mesh)
    force_down = ax.down_tension(Data, force_l1, force_R, Mesh)
    force_time = drilling_time_breakdown(Data, Mesh, force_l1, force_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append({"Objective": "Minimal axial force", "L1 (m)": round(force_l1, 3), "R (m)": round(force_R, 3), "Angle (deg)": round(force_cfg["angle_deg"], 3), "Top axial force (N)": round(force_up[0], 3), "Torque (N*m)": round(force_down[3], 3), "Total time (h)": round(force_time, 3) if Mesh is not None else np.nan})
    torque_l1, torque_R = minimal_torque(Data, Mesh)
    torque_cfg = ax.validate_configuration(Data, torque_l1, torque_R)
    torque_up = ax.up_tension(Data, torque_l1, torque_R, Mesh)
    torque_down = ax.down_tension(Data, torque_l1, torque_R, Mesh)
    torque_time = drilling_time_breakdown(Data, Mesh, torque_l1, torque_R)["total_time_h"] if Mesh is not None else np.nan
    rows.append({"Objective": "Minimal torque", "L1 (m)": round(torque_l1, 3), "R (m)": round(torque_R, 3), "Angle (deg)": round(torque_cfg["angle_deg"], 3), "Top axial force (N)": round(torque_up[0], 3), "Torque (N*m)": round(torque_down[3], 3), "Total time (h)": round(torque_time, 3) if Mesh is not None else np.nan})
    if Mesh is not None:
        time_best = drilling_time_informations(Data, Mesh)
        rows.append({"Objective": "Minimal drilling time", "L1 (m)": round(time_best["l1"], 3), "R (m)": round(time_best["R"], 3), "Angle (deg)": round(time_best["angle_deg"], 3), "Top axial force (N)": round(time_best["up_force_1"], 3), "Torque (N*m)": round(time_best["torque"], 3), "Total time (h)": round(time_best["drilling_time_h"], 3)})
    table = pd.DataFrame(rows)
    print("\n--- Unified optimization summary ---")
    print(table.to_string(index=False))
    print("")


def plot_metrics_vs_radius_for_best_l1(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from drilling.features.minimization.operational import plot_metrics_vs_radius_for_best_l1_4_conditions

    plot_metrics_vs_radius_for_best_l1_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )


def plot_metrics_vs_l1_for_best_r(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from drilling.features.minimization.operational import plot_metrics_vs_l1_for_best_r_4_conditions

    plot_metrics_vs_l1_for_best_r_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )


def plot_best_metric_per_l1_using_best_r(
    Data,
    Mesh,
    operational_parameters: dict | None = None,
    mechanical_limits: dict | None = None,
) -> None:
    from drilling.features.minimization.operational import plot_best_metric_per_l1_using_best_r_4_conditions

    plot_best_metric_per_l1_using_best_r_4_conditions(
        Data,
        Mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )
