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
BASIN_GRID = Path(__file__).parent / "data" / "synthetic_basin.grdecl"


def _write_layered_grid(
    path: Path,
    nx: int,
    ny: int,
    size_xy: float,
    layer_tops: list[float],
    bottom: float,
    facies: list,
    table: str = "",
) -> Path:
    """GRDECL de camadas horizontais com pilares verticais, um código FACIES por camada e ``table`` (LITHTAB) opcional."""
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
        f"{table}"
        f"PORO\n {nx * ny * nz}*0.2 /\n",
        encoding="utf-8",
    )
    return path


def test_sample_grid_dimensions_and_lithologies() -> None:
    """O GRDECL de exemplo traz o código de litologia de cada célula e a tabela LITHTAB."""
    grid = read_grdecl(SAMPLE_GRID)
    assert (grid.nx, grid.ny, grid.nz) == (10, 10, 3)
    assert grid.lithology_names == ["Sandstone", "Siltstone", "Shale"]
    assert grid.rop_coefficients == {"Sandstone": 1.30, "Siltstone": 1.00, "Shale": 0.80}
    assert grid.depth_range() == (0.0, 30.0)
    assert grid.actnum.all()
    by_layer = [grid.lithology_names[code] for code in grid.lithology[:, 0, 0]]
    assert by_layer == ["Shale", "Siltstone", "Sandstone"]
    assert grid.lithology_counts() == {"Sandstone": 100, "Siltstone": 100, "Shale": 100}


def test_sample_grid_lookup_uses_file_coefficients() -> None:
    """ROP = base × coeficiente de LITHTAB; ``rop_coefficients`` sobrepõe o do arquivo."""
    grid = read_grdecl(SAMPLE_GRID)
    geology = GridGeology(
        grid,
        wellhead=(50.0, 50.0, -500.0),
        target=(950.0, 950.0, 25.0),
        base_rop=12.0,
        rop_coefficients={"Sandstone": 2.0},
        outside_rop_coefficient=1.5,
    )
    assert geology.P3 == pytest.approx((900.0 * np.sqrt(2.0), 525.0))
    assert geology.segment_at(0.0, 400.0) == {"lithology": OUTSIDE_GRID, "rop": 18.0}
    assert geology.segment_at(0.0, 505.0) == {"lithology": "Shale", "rop": pytest.approx(9.6)}
    assert geology.segment_at(600.0, 515.0) == {"lithology": "Siltstone", "rop": 12.0}
    assert geology.segment_at(geology.P3[0], 525.0) == {"lithology": "Sandstone", "rop": 24.0}
    assert geology.segment_at(0.0, 531.0)["lithology"] == OUTSIDE_GRID


def test_lookup_outside_grid_footprint_is_outside() -> None:
    """Fora da projeção horizontal da malha não há litologia, mesmo na profundidade certa."""
    geology = GridGeology(read_grdecl(SAMPLE_GRID), wellhead=(-1000.0, 500.0, 0.0), target=(500.0, 500.0, 15.0), base_rop=10.0)
    assert geology.segment_at(500.0, 15.0)["lithology"] == OUTSIDE_GRID
    assert geology.segment_at(1500.0, 15.0)["lithology"] == "Siltstone"


@pytest.mark.parametrize(
    "target",
    [(3000.0, 500.0, 20.0), (500.0, 500.0, 40.0), (500.0, 500.0, -5.0)],
    ids=["beside the grid", "below the grid", "above the grid"],
)
def test_target_outside_grid_is_rejected(target) -> None:
    """Um alvo fora das células ativas não deixa a otimização começar."""
    with pytest.raises(ValueError, match="not inside an active cell"):
        GridGeology(read_grdecl(SAMPLE_GRID), wellhead=(0.0, 0.0, -3000.0), target=target, base_rop=10.0)


def test_reader_ignores_unused_keywords_and_reads_codes_without_table(tmp_path: Path) -> None:
    """Keywords sem uso (MAPAXES, PORO, NOECHO) e ``n*valor`` não atrapalham; sem LITHTAB, os códigos viram nomes."""
    path = _write_layered_grid(tmp_path / "facies.grdecl", 2, 3, 100.0, [0.0, 40.0], 100.0, facies=[7, 3])
    grid = read_grdecl(path)
    assert (grid.nx, grid.ny, grid.nz) == (2, 3, 2)
    assert grid.lithology_names == ["Lithology 3", "Lithology 7"]
    assert grid.rop_coefficients == {}
    assert [grid.lithology_names[code] for code in grid.lithology[:, 0, 0]] == ["Lithology 7", "Lithology 3"]


def test_lithology_table_names_and_defaults(tmp_path: Path) -> None:
    """Nomes entre aspas podem ter espaços, o coeficiente omitido vale 1 e códigos fora da tabela ganham nome padrão."""
    table = "LITHTAB\n  1 'Fine sandstone' 1.25 /\n  2 'Marl' /\n  9 'Unused' 0.5 /\n/\n"
    path = _write_layered_grid(tmp_path / "table.grdecl", 2, 2, 100.0, [0.0, 30.0, 60.0], 100.0, facies=[1, 2, 4], table=table)
    grid = read_grdecl(path)
    assert grid.lithology_names == ["Fine sandstone", "Marl", "Lithology 4", "Unused"]
    assert grid.rop_coefficients == {"Fine sandstone": 1.25, "Marl": 1.0, "Unused": 0.5}
    assert grid.lithology_counts() == {"Fine sandstone": 4, "Marl": 4, "Lithology 4": 4, "Unused": 0}


@pytest.mark.parametrize(
    "facies, table, message",
    [
        ([1, 2], "LITHTAB\n 1 'A' 1.0 /\n 1 'B' 1.0 /\n/\n", "code 1 twice"),
        ([1, 2], "LITHTAB\n 1 'A' 1.0 /\n 2 'A' 1.0 /\n/\n", "name 'A' twice"),
        ([1, 2], "LITHTAB\n 1 'A' -1 /\n/\n", "must be positive"),
        ([1.5, 2], "", "integer lithology codes"),
    ],
    ids=["duplicate code", "duplicate name", "negative coefficient", "non-integer code"],
)
def test_invalid_lithology_data_is_rejected(tmp_path: Path, facies, table, message) -> None:
    """Dados de litologia inválidos geram erro com o motivo."""
    path = _write_layered_grid(tmp_path / "bad.grdecl", 2, 2, 100.0, [0.0, 50.0], 100.0, facies=facies, table=table)
    with pytest.raises(ValueError, match=message):
        read_grdecl(path)


def test_grid_without_lithology_is_rejected(tmp_path: Path) -> None:
    """Sem LITHOLOGY/FACIES o leitor não adivinha a litologia."""
    path = _write_layered_grid(tmp_path / "nolith.grdecl", 2, 2, 100.0, [0.0], 100.0, facies=[1])
    path.write_text(path.read_text().replace("FACIES", "SED1"), encoding="utf-8")
    with pytest.raises(ValueError, match="no LITHOLOGY"):
        read_grdecl(path)


def test_layered_grid_matches_interval_mesh(tmp_path: Path) -> None:
    """Camadas horizontais na GRDECL dão o mesmo tempo de broca que a ``mesh`` de intervalos."""
    tops = [0.0, 400.0, 1500.0, 2200.0]
    table = "LITHTAB\n  1 'Sandstone' 1.8 /\n  2 'Dolomite' 0.95 /\n  3 'Evaporite' 2.4 /\n/\n"
    path = _write_layered_grid(tmp_path / "layers.grdecl", 6, 6, 3000.0, tops, 3200.0, facies=[1, 2, 3, 1], table=table)
    grid = read_grdecl(path)
    geology = GridGeology(grid, wellhead=(100.0, 100.0, 0.0), target=(700.0, 900.0, 3000.0), base_rop=10.0)

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
    assert view.lithology_table.rowCount() == 4
    file_values = [view.lithology_table.cellWidget(row, 2).value() for row in range(4)]
    assert file_values == [1.30, 1.00, 0.80, 1.00]
    view.lithology_table.cellWidget(0, 2).setValue(2.0)
    view.wellhead_input.setText("-50, 50, -2975")
    view.target_input.setText("950, 50, 25")

    data, geology, operational, _ = view.build_data_from_inputs()
    assert data.P3 == pytest.approx((1000.0, 3000.0))
    assert geology.rop_coefficients["Sandstone"] == 2.0
    assert geology.rop_coefficients["Shale"] == 0.80
    assert geology.segment_at(1000.0, 2999.0)["lithology"] == "Sandstone"
    assert set(operational["lithology_wear_factors"]) == {"Sandstone", "Siltstone", "Shale", OUTSIDE_GRID}


def _load_generator():
    script = Path(__file__).parents[1] / "scripts" / "make_test_grid.py"
    spec = importlib.util.spec_from_file_location("make_test_grid", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator


def test_basin_grid_is_what_the_generator_writes(tmp_path: Path) -> None:
    """``tests/data/synthetic_basin.grdecl`` está em dia com ``scripts/make_test_grid.py``."""
    generator = _load_generator()
    path = tmp_path / "basin.grdecl"
    generator.write_grdecl(path, generator.build(nx=20, ny=20, size=3000.0))
    assert path.read_text(encoding="utf-8") == BASIN_GRID.read_text(encoding="utf-8")


def test_basin_grid_is_realistic() -> None:
    """A malha de teste tem refinamento, falha, acunhamento e as cinco litologias com seus coeficientes."""
    grid = read_grdecl(BASIN_GRID)
    assert (grid.nx, grid.ny, grid.nz) == (20, 20, 60)
    assert grid.rop_coefficients == {"Sandstone": 1.30, "Siltstone": 1.00, "Marl": 0.90, "Shale": 0.80, "Limestone": 0.60}
    assert all(count > 0 for count in grid.lithology_counts().values())
    assert grid.depth_range()[1] > 3500.0

    thickness = grid.zcorn[:, 1] - grid.zcorn[:, 0]
    assert thickness.min() >= 0.0
    assert np.allclose(grid.zcorn[1:, 0], grid.zcorn[:-1, 1])
    # Acunhamento: células de espessura zero ficam inativas.
    assert (~grid.actnum).sum() > 0
    assert np.all(thickness.max(axis=(2, 4))[~grid.actnum] <= 0.01)
    # Falha normal: os cantos de colunas vizinhas sobre um mesmo pilar diferem em uma única linha.
    jump = np.abs(grid.zcorn[:, :, :, :, 1:, 0] - grid.zcorn[:, :, :, :, :-1, 1]).max(axis=(0, 1, 2, 3))
    assert np.count_nonzero(jump > 1.0) == 1
    assert jump.max() == pytest.approx(90.0)


def test_basin_grid_contains_gui_default_well() -> None:
    """O poço padrão da GUI cai no reservatório arenoso e atravessa as cinco litologias."""
    geology = GridGeology(read_grdecl(BASIN_GRID), DEFAULT_WELLHEAD, DEFAULT_TARGET, base_rop=15.0)
    data, _ = build_default_data()
    assert geology.P3 == pytest.approx(data.P3)
    assert geology.segment_at(*geology.P3)["lithology"] == "Sandstone"
    timing = drilling_time_breakdown(data, geology, 1200.0, 500.0)
    assert OUTSIDE_GRID not in timing["by_lithology"]
    assert set(timing["by_lithology"]) == {"Sandstone", "Siltstone", "Marl", "Shale", "Limestone"}
