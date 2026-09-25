"""Orquestração pública dos quatro objetivos de otimização Tipo 1.

Este módulo não introduz matemática nova. Ele chama os varredores mecânicos,
de tempo de broca e operacionais já existentes e empacota o payload que a
aba Minimization já exibia.
"""

from __future__ import annotations

from drilling.features.minimization.minimal import drilling_time_breakdown
from drilling.features.minimization.operational import (
    _best_constrained_time_candidate,
    _best_mechanical_candidates,
    _best_total_time_candidate,
    _series_best_metric_for_each_l1,
    _series_varying_l1_for_fixed_radius,
    _series_varying_radius_for_fixed_l1,
    evaluate_mechanical_limits,
    operational_time_breakdown,
)


def calculate_minimization(data, geological_mesh, operational_parameters, mechanical_limits):
    """Executa os quatro objetivos na malha padrão e monta as séries dos gráficos.

    Parameters
    ----------
    data : DataSet
        Entradas mecânicas e de tempo de broca.
    geological_mesh : HorizonModel
        Modelo geológico com ROP (e ``mu``, para ``friction_model="lithology"``).
    operational_parameters : dict
        Sobreposição de manobras, revestimento e limites de corrida da broca.
    mechanical_limits : dict
        Tetos opcionais de força axial no topo e de torque.

    Returns
    -------
    dict
        Mapeamento com ``results`` (um registro por objetivo) e ``series``
        (famílias de curvas em raio, L1 e melhor-por-L1).
    """
    best_force, best_torque = _best_mechanical_candidates(data, geological_mesh)
    best_time = _best_constrained_time_candidate(data, geological_mesh, mechanical_limits=mechanical_limits)
    best_total = _best_total_time_candidate(
        data,
        geological_mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    selected = {
        "force": best_force,
        "torque": best_torque,
        "time": best_time,
        "total": best_total,
    }

    results = {}
    for key, candidate in selected.items():
        l1 = float(candidate["l1"])
        radius = float(candidate["R"])
        timing = candidate.get("timing") or drilling_time_breakdown(data, geological_mesh, l1, radius)
        operational = candidate.get("operational") or operational_time_breakdown(
            data,
            geological_mesh,
            l1,
            radius,
            drilling_timing=timing,
            operational_parameters=operational_parameters,
        )
        mechanical = evaluate_mechanical_limits(candidate["up_force_1"], candidate["torque"], mechanical_limits)
        results[key] = {
            **candidate,
            "l1": l1,
            "R": radius,
            "timing": timing,
            "operational": operational,
            "mechanical": mechanical,
            "drilling_time_h": float(timing["total_time_h"]),
            "operational_time_h": float(operational["total_operational_time_h"]),
            "total_time_h": float(operational["total_time_h"]),
        }

    series = {
        "radius": _series_varying_radius_for_fixed_l1(
            data,
            geological_mesh,
            l1_force=best_force["l1"],
            l1_torque=best_torque["l1"],
            l1_time=best_time["l1"],
            l1_total=best_total["l1"],
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
        "l1": _series_varying_l1_for_fixed_radius(
            data,
            geological_mesh,
            r_force=best_force["R"],
            r_torque=best_torque["R"],
            r_time=best_time["R"],
            r_total=best_total["R"],
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
        "best_per_l1": _series_best_metric_for_each_l1(
            data,
            geological_mesh,
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
    }
    return {
        "data": data,
        "mesh": geological_mesh,
        "operational_parameters": operational_parameters,
        "mechanical_limits": mechanical_limits,
        "results": results,
        "series": series,
    }
