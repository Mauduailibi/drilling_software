"""Geometria do poço planejado desenhada na aba Well Path (``logic.project_trajectory``)."""

from __future__ import annotations

import numpy as np
import pytest

from drilling.features.well_path.defaults import DEFAULT_WELL_PATH_INPUT
from drilling.features.well_path.logic import dls_to_radius, normalize, project_trajectory


def _defaults():
    return (
        DEFAULT_WELL_PATH_INPUT.pin.as_array(),
        DEFAULT_WELL_PATH_INPUT.pbd.as_array(),
        DEFAULT_WELL_PATH_INPUT.pt.as_array(),
    )


def test_project_goes_straight_to_kickoff_and_curves_only_after_it() -> None:
    """O desenho passa exatamente por ``Pbd`` e só começa a curvar nele, tangente ao trecho reto."""
    pin, pbd, pt = _defaults()
    project = project_trajectory(pin, pbd, pt)
    points = project["points"]

    assert np.allclose(points[0], pin)
    assert np.allclose(points[1], pbd)
    assert np.allclose(points[-1], pt)
    assert np.allclose(project["kickoff_point"], pbd)

    # Antes de Pbd é só a reta Pin→Pbd; logo depois de Pbd a direção ainda é a do projeto.
    straight = normalize(pbd - pin)
    first_build_step = normalize(points[2] - points[1])
    assert np.degrees(np.arccos(np.clip(first_build_step @ straight, -1, 1))) < 0.5


def test_build_section_has_constant_dls_and_ends_tangent_to_target() -> None:
    """O ganho de ângulo tem raio do DLS de projeto e termina na tangente que chega ao alvo."""
    pin, pbd, pt = _defaults()
    project = project_trajectory(pin, pbd, pt, dls_deg=3.0)
    radius = dls_to_radius(3.0)
    assert project["radius"] == pytest.approx(radius)

    arc = project["points"][1:-1]
    straight = normalize(pbd - pin)
    to_target = normalize(pt - pbd)
    normal = normalize(np.cross(straight, to_target))
    center = pbd + radius * np.cross(normal, straight)
    assert np.allclose(np.linalg.norm(arc - center, axis=1), radius)

    end_of_build = project["end_of_build"]
    assert np.allclose(arc[-1], end_of_build)
    tangent = normalize(pt - end_of_build)
    assert abs(tangent @ normalize(end_of_build - center)) < 1e-6


def test_target_inside_build_radius_falls_back_to_straight_segments() -> None:
    """Sem arco possível com o DLS de projeto, o desenho liga Pin→Pbd→pt por retas."""
    pin = np.array([0.0, 0.0, 0.0])
    pbd = np.array([0.0, 0.0, -1000.0])
    pt = np.array([50.0, 0.0, -1100.0])
    project = project_trajectory(pin, pbd, pt, dls_deg=3.0)
    assert np.allclose(project["points"], [pin, pbd, pt])
    assert project["end_of_build"] is None


def test_target_straight_ahead_needs_no_build() -> None:
    """Alvo na direção do projeto: reta até ``Pbd`` e reta até o alvo, sem arco."""
    pin = np.array([0.0, 0.0, 0.0])
    pbd = np.array([0.0, 0.0, -1000.0])
    pt = np.array([0.0, 0.0, -2500.0])
    project = project_trajectory(pin, pbd, pt)
    assert np.allclose(project["points"], [pin, pbd, pt])
    assert project["end_of_build"] is None
