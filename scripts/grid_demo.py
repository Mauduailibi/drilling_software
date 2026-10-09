"""Otimização Tipo 1 sobre uma malha GRDECL, com a vista no plano do poço e em 3D.

Uso::

    python scripts/grid_demo.py tests/data/synthetic_basin.grdecl

Os coeficientes de ROP vêm da tabela ``LITHTAB`` do arquivo; ``--coef Shale=0.7``
sobrepõe o de uma litologia.

As figuras vão para ``outputs/grid_demo/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from drilling.core import parse_vector3
from drilling.features.minimization import calculate_minimization, read_grdecl
from drilling.features.minimization.defaults import DEFAULT_BASE_ROP, build_default_data
from drilling.features.minimization.grid import GridGeology
from drilling.features.minimization.operational import DEFAULT_MECHANICAL_LIMITS
from drilling.features.minimization.plot import plot_trajectories, plot_trajectories_3d

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "grid_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grdecl", type=Path)
    parser.add_argument("--wellhead", default="1000, 1500, 0", help="x, y, z da cabeça do poço")
    parser.add_argument("--target", default="2000, 1500, 3000", help="x, y, z do alvo")
    parser.add_argument("--base-rop", type=float, default=DEFAULT_BASE_ROP)
    parser.add_argument("--coef", action="append", default=[], help="LITOLOGIA=coeficiente, sobrepõe o do arquivo (repetível)")
    parser.add_argument("--l1-step", type=float, default=10.0, help="passo de L1 da varredura (m)")
    args = parser.parse_args()

    grid = read_grdecl(args.grdecl)
    coefficients = {name: float(value) for name, value in (item.split("=") for item in args.coef)}
    geology = GridGeology(
        grid,
        parse_vector3(args.wellhead),
        parse_vector3(args.target),
        base_rop=args.base_rop,
        rop_coefficients=coefficients,
    )
    print(f"Grid {grid.nx} x {grid.ny} x {grid.nz}; lithologies (active cells): {grid.lithology_counts()}")
    print(f"Well plane target P3 = ({geology.P3[0]:.1f}, {geology.P3[1]:.1f}) m")

    data, operational = build_default_data()
    data.P3 = geology.P3
    data.l1_step = args.l1_step

    start = time.perf_counter()
    payload = calculate_minimization(data, geology, operational, dict(DEFAULT_MECHANICAL_LIMITS))
    print(f"Optimization finished in {time.perf_counter() - start:.1f} s")
    for key, result in payload["results"].items():
        by_lithology = {name: round(values["length_m"], 1) for name, values in result["timing"]["by_lithology"].items()}
        print(f"  {key:7s} L1={result['l1']:7.1f} R={result['R']:6.1f} total={result['total_time_h']:8.2f} h  length by lithology: {by_lithology}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10, 7))
    plot_trajectories(figure, data, geology, payload["results"])
    figure.savefig(OUTPUT_DIR / "well_plane.png", dpi=130)
    figure = Figure(figsize=(10, 8))
    plot_trajectories_3d(figure, data, geology, payload["results"])
    figure.savefig(OUTPUT_DIR / "grid_3d.png", dpi=130)
    figure = Figure(figsize=(10, 8))
    plot_trajectories_3d(figure, data, geology, payload["results"], zoom_to_grid=True)
    figure.savefig(OUTPUT_DIR / "grid_3d_zoom.png", dpi=130)
    print(f"Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
