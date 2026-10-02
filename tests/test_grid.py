"""Leitor GRDECL e consulta de litologia ao longo do poço (``minimization.grid``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from drilling.features.minimization.data_base import mesh
from drilling.features.minimization.defaults import DEFAULT_TARGET, DEFAULT_WELLHEAD, build_default_data
from drilling.features.minimization.grid import OUTSIDE_GRID, GridGeology, read_grdecl
from drilling.features.minimization.minimal import drilling_time_breakdown

SAMPLE_GRID = Path(__file__).parent / "data" / "kvl_quarter_five_spot.grdecl"


def _write_layered_grid(path: Path, nx: int, ny: int, size_xy: float, layer_tops: list[float], bottom: float, facies: list[int]) -> Path:
    """GRDECL de camadas horizontais com pilares verticais e um keyword FACIES por camada."""
    nz = len(layer_tops)
    xs = np.linspace(0.0, size_xy, nx + 1)
    ys = np.linspace(0.0, size_xy, ny + 1)
    coord = [f"{x} {y} 0 {x} {y} {bottom}" for y in ys for x in xs]
    bounds = list(layer_tops) + [bottom]
    zcorn = []
    for k in range(nz):
        for depth in (bounds[k], bounds[k + 1]):
            zcorn.append(f"{4 * nx * ny}*{depth}")
    facies_values = [f"{nx * ny}*{code}" for code in facies]
    path.write_text(
        "-- grade sintética de teste\n"
        "NOECHO\n"
        f"SPECGRID\n {nx} {ny} {nz} 1 F /\n"
        "MAPAXES\n 0 1 0 0 1 0 /\n"
        f"COORD\n {' '.join(coord)} /\n"
        f"ZCORN\n {' '.join(zcorn)} /\n"
        f"ACTNUM\n {nx * ny * nz}*1 /\n"
        f"FACIES\n {' '.join(facies_values)} /\n"
        f"PORO\n {nx * ny * nz}*0.2 /\n",
        encoding="utf-8",
    )
    return path


def test_sample_grid_dimensions_and_lithologies() -> None:
    """O GRDECL do KVL é lido e cada camada recebe o sedimento de maior fração."""
    grid = read_grdecl(SAMPLE_GRID)
    assert (grid.nx, grid.ny, grid.nz) == (10, 10, 3)
    assert grid.lithology_names == ["SED1", "SED2", "SED3", "SED4"]
    assert grid.depth_range() == (0.0, 30.0)
    assert grid.actnum.all()
    # Camada 1: SED3 = 0.50; camada 2: SED2 = 0.55; camada 3: SED1 = 0.65.
    dominant = [grid.lithology_names[code] for code in grid.lithology[:, 0, 0]]
    assert dominant == ["SED3", "SED2", "SED1"]
    assert grid.lithology_counts() == {"SED1": 100, "SED2": 100, "SED3": 100, "SED4": 0}


def test_sample_grid_lookup_uses_rop_coefficients() -> None:
    """``segment_at`` devolve a litologia da célula e ROP = base × coeficiente."""
    grid = read_grdecl(SAMPLE_GRID)
    geology = GridGeology(
        grid,
        wellhead=(50.0, 50.0, -500.0),
        target=(950.0, 950.0, 25.0),
        base_rop=12.0,
        rop_coefficients={"SED1": 2.0, "SED3": 0.5},
        outside_rop_coefficient=1.5,
    )
    assert geology.P3 == pytest.approx((900.0 * np.sqrt(2.0), 525.0))
    assert geology.segment_at(0.0, 400.0) == {"lithology": OUTSIDE_GRID, "rop": 18.0}
    assert geology.segment_at(0.0, 505.0) == {"lithology": "SED3", "rop": 6.0}
    assert geology.segment_at(600.0, 515.0) == {"lithology": "SED2", "rop": 12.0}
    assert geology.segment_at(geology.P3[0], 525.0) == {"lithology": "SED1", "rop": 24.0}
    assert geology.segment_at(0.0, 531.0)["lithology"] == OUTSIDE_GRID


def test_lookup_outside_grid_footprint_is_outside() -> None:
    """Fora da projeção horizontal da malha não há litologia, mesmo na profundidade certa."""
    geology = GridGeology(read_grdecl(SAMPLE_GRID), wellhead=(-1000.0, 500.0, 0.0), target=(500.0, 500.0, 15.0), base_rop=10.0)
    assert geology.segment_at(500.0, 15.0)["lithology"] == OUTSIDE_GRID
    assert geology.segment_at(1500.0, 15.0)["lithology"] == "SED2"


@pytest.mark.parametrize(
    "target",
    [(3000.0, 500.0, 20.0), (500.0, 500.0, 40.0), (500.0, 500.0, -5.0)],
    ids=["beside the grid", "below the grid", "above the grid"],
)
def test_target_outside_grid_is_rejected(target) -> None:
    """Um alvo fora das células ativas não deixa a otimização começar."""
    with pytest.raises(ValueError, match="not inside an active cell"):
        GridGeology(read_grdecl(SAMPLE_GRID), wellhead=(0.0, 0.0, -3000.0), target=target, base_rop=10.0)


def test_reader_ignores_unused_keywords_and_reads_facies(tmp_path: Path) -> None:
    """Keywords sem uso (MAPAXES, PORO, NOECHO) e a sintaxe ``n*valor`` não atrapalham a leitura."""
    path = _write_layered_grid(tmp_path / "facies.grdecl", 2, 3, 100.0, [0.0, 40.0], 100.0, facies=[7, 3])
    grid = read_grdecl(path)
    assert (grid.nx, grid.ny, grid.nz) == (2, 3, 2)
    assert grid.lithology_names == ["Facies 3", "Facies 7"]
    assert [grid.lithology_names[code] for code in grid.lithology[:, 0, 0]] == ["Facies 7", "Facies 3"]


def test_layered_grid_matches_interval_mesh(tmp_path: Path) -> None:
    """Camadas horizontais na GRDECL dão o mesmo tempo de broca que a ``mesh`` de intervalos."""
    tops = [0.0, 400.0, 1500.0, 2200.0]
    path = _write_layered_grid(tmp_path / "layers.grdecl", 6, 6, 3000.0, tops, 3200.0, facies=[1, 2, 3, 1])
    grid = read_grdecl(path)
    base_rop = 10.0
    coefficients = {"Facies 1": 1.8, "Facies 2": 0.95, "Facies 3": 2.4}
    geology = GridGeology(grid, wellhead=(100.0, 100.0, 0.0), target=(700.0, 900.0, 3000.0), base_rop=base_rop, rop_coefficients=coefficients)

    reference = mesh(
        sandstone=[[0, 400], [2200, 3200]],
        dolomite=[[400, 1500]],
        evaporite=[[1500, 2200]],
        rop_values={"Sandstone": 18.0, "Dolomite": 9.5, "Evaporite": 24.0},
    )

    data, _ = build_default_data()
    assert geology.P3 == pytest.approx(data.P3)
    from_grid = drilling_time_breakdown(data, geology, 1200.0, 500.0)
    from_mesh = drilling_time_breakdown(data, reference, 1200.0, 500.0)
    assert from_grid["total_time_h"] == pytest.approx(from_mesh["total_time_h"], rel=1e-12)
    assert [row["rop_base_mph"] for row in from_grid["elements"]] == pytest.approx(
        [row["rop_base_mph"] for row in from_mesh["elements"]]
    )


def test_no_grid_means_uniform_base_rop() -> None:
    """Sem malha carregada todo o poço usa o coeficiente de fora da malha."""
    geology = GridGeology(None, wellhead=(0.0, 0.0, 0.0), target=(1000.0, 0.0, 3000.0), base_rop=15.0, outside_rop_coefficient=0.8)
    assert geology.segment_at(500.0, 1500.0) == {"lithology": OUTSIDE_GRID, "rop": 12.0}
    assert geology.section_polygons() == []
    assert geology.cells_around_well() == ([], (0.0, 3150.0))


@pytest.mark.parametrize(
    "wellhead, target",
    [((0.0, 0.0, 0.0), (0.0, 0.0, 100.0)), ((0.0, 0.0, 100.0), (50.0, 0.0, 10.0))],
)
def test_invalid_well_positions(wellhead, target) -> None:
    """O alvo precisa estar mais fundo e afastado horizontalmente da cabeça do poço."""
    with pytest.raises(ValueError):
        GridGeology(None, wellhead, target, base_rop=10.0)


def test_gui_builds_geology_from_loaded_grid() -> None:
    """A aba Minimization monta ``GridGeology`` a partir da malha e da tabela de coeficientes."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from drilling.features.minimization.view import MinimizationView

    _app = QApplication.instance() or QApplication([])
    view = MinimizationView()
    with pytest.raises(ValueError, match="Load a GRDECL grid"):
        view.build_data_from_inputs()

    view.set_grid(read_grdecl(SAMPLE_GRID))
    assert view.lithology_table.rowCount() == 5
    view.lithology_table.cellWidget(0, 2).setValue(2.0)
    view.wellhead_input.setText("-50, 50, -2975")
    view.target_input.setText("950, 50, 25")

    data, geology, operational, _ = view.build_data_from_inputs()
    assert data.P3 == pytest.approx((1000.0, 3000.0))
    assert geology.rop_coefficients["SED1"] == 2.0
    assert geology.segment_at(1000.0, 2999.0)["lithology"] == "SED1"
    assert set(operational["lithology_wear_factors"]) == {"SED1", "SED2", "SED3", "SED4", OUTSIDE_GRID}


def test_synthetic_basin_generator_covers_gui_defaults(tmp_path: Path) -> None:
    """A malha de ``scripts/make_test_grid.py`` é lida e contém o poço padrão da GUI."""
    script = Path(__file__).parents[1] / "scripts" / "make_test_grid.py"
    spec = importlib.util.spec_from_file_location("make_test_grid", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    path = tmp_path / "basin.grdecl"
    generator.write_grdecl(path, generator.build(nx=12, ny=12, size=3000.0))
    grid = read_grdecl(path)
    assert (grid.nx, grid.ny, grid.nz) == (12, 12, 12)
    assert all(count > 0 for count in grid.lithology_counts().values())

    geology = GridGeology(grid, DEFAULT_WELLHEAD, DEFAULT_TARGET, base_rop=15.0)
    data, _ = build_default_data()
    assert geology.P3 == pytest.approx(data.P3)
    timing = drilling_time_breakdown(data, geology, 1200.0, 500.0)
    assert OUTSIDE_GRID not in timing["by_lithology"]
    assert len(timing["by_lithology"]) >= 3
