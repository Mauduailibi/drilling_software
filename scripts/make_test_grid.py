"""Gera uma malha GRDECL sintética para testar a otimização sobre uma malha de verdade.

A malha é só um arquivo de teste: o sistema não constrói malhas, ele as lê.
Ela segue o formato que o sistema espera de um usuário que já definiu as
litologias (código ``LITHOLOGY`` por célula e tabela ``LITHTAB`` com nome e
coeficiente de ROP) e cobre o poço inteiro, do fundo do mar até abaixo do
alvo padrão da GUI:

* 3 km × 3 km em planta, 30 × 30 colunas de 100 m;
* 12 camadas de 150 a 400 m, do fundo do mar (0 m) até ~3,4 km;
* mergulho para leste e um anticlinal, mais fortes nas camadas profundas;
* um canal arenoso sinuoso em três camadas, ou seja, litologias diferentes
  na mesma profundidade.

Os coeficientes seguem a facilidade de perfuração: arenito 1,30, siltito 1,00,
folhelho 0,80 e calcário 0,60.

Uso::

    python scripts/make_test_grid.py                       # outputs/synthetic_basin.grdecl
    python scripts/make_test_grid.py --nx 60 --ny 60 -o minha_malha.grdecl

Com a malha padrão, a GUI roda com Wellhead = 1000, 1500, 0 e
Target = 2000, 1500, 3000 (o mesmo P3 = (1000, 3000) dos goldens).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "synthetic_basin.grdecl"

# Código: (nome, coeficiente de ROP). Rochas mais friáveis têm coeficiente maior.
LITHOLOGIES = {
    1: ("Sandstone", 1.30),
    2: ("Siltstone", 1.00),
    3: ("Shale", 0.80),
    4: ("Limestone", 0.60),
}
SANDSTONE, SILTSTONE, SHALE, LIMESTONE = 1, 2, 3, 4

# Espessura (m) e litologia de cada camada, do topo para a base.
LAYERS = [
    (150.0, SHALE),
    (250.0, SANDSTONE),
    (300.0, SHALE),
    (250.0, SILTSTONE),
    (350.0, LIMESTONE),
    (300.0, SHALE),
    (250.0, SANDSTONE),
    (400.0, SHALE),
    (200.0, LIMESTONE),
    (350.0, SILTSTONE),
    (300.0, SANDSTONE),
    (300.0, SHALE),
]
CHANNEL_LAYERS = (2, 5, 7)
"""Camadas de folhelho cortadas pelo canal arenoso."""


def surfaces(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    """Profundidade ``(nz + 1, ...)`` de cada horizonte nos pontos ``x``, ``y``."""
    thickness = np.array([layer[0] for layer in LAYERS])
    flat = np.concatenate([[0.0], np.cumsum(thickness)])
    dip = 150.0 * (x / size)
    anticline = -250.0 * np.exp(-((x - 0.55 * size) ** 2 + (y - 0.5 * size) ** 2) / (2 * (0.25 * size) ** 2))
    structure = dip + anticline
    # O fundo do mar é plano; a estrutura cresce com a profundidade.
    weight = flat / flat[-1]
    return flat[:, None, None] + weight[:, None, None] * structure[None, :, :]


def channel_mask(xc: np.ndarray, yc: np.ndarray, size: float) -> np.ndarray:
    """Células cujo centro cai no canal sinuoso que cruza a malha de sul para norte."""
    axis = 0.45 * size + 0.12 * size * np.sin(2.0 * np.pi * yc / size)
    return np.abs(xc - axis) < 0.07 * size


def build(nx: int, ny: int, size: float) -> dict:
    nz = len(LAYERS)
    xs = np.linspace(0.0, size, nx + 1)
    ys = np.linspace(0.0, size, ny + 1)
    px, py = np.meshgrid(xs, ys)
    horizons = surfaces(px, py, size)

    bottom = float(horizons[-1].max()) + 1.0
    coord = np.stack([px, py, np.zeros_like(px), px, py, np.full_like(px, bottom)], axis=-1)

    # ZCORN[k, topo/base, j, dj, i, di]: pilares verticais, sem falhas.
    zcorn = np.empty((nz, 2, ny, 2, nx, 2))
    for k in range(nz):
        for t, horizon in enumerate((horizons[k], horizons[k + 1])):
            for dj in range(2):
                for di in range(2):
                    zcorn[k, t, :, dj, :, di] = horizon[dj:dj + ny, di:di + nx]

    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    cx, cy = np.meshgrid(xc, yc)
    channel = channel_mask(cx, cy, size)
    lithology = np.empty((nz, ny, nx), dtype=int)
    for k, (_, code) in enumerate(LAYERS):
        lithology[k] = code
        if k in CHANNEL_LAYERS:
            lithology[k, channel] = SANDSTONE
    return {"nx": nx, "ny": ny, "nz": nz, "coord": coord, "zcorn": zcorn, "lithology": lithology}


def _block(name: str, values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    lines = [" ".join(fmt.format(value) for value in flat[i:i + 8]) for i in range(0, flat.size, 8)]
    return f"{name}\n" + "\n".join(f"  {line}" for line in lines) + "\n/\n\n"


def write_grdecl(path: Path, model: dict) -> None:
    nx, ny, nz = model["nx"], model["ny"], model["nz"]
    text = [
        "-- Malha sintética de teste gerada por scripts/make_test_grid.py\n",
        "-- Profundidade (ZCORN) em metros abaixo do fundo do mar, positiva para baixo.\n\n",
        f"SPECGRID\n  {nx} {ny} {nz} 1 F /\n\n",
        _block("COORD", model["coord"], "{:.2f}"),
        _block("ZCORN", model["zcorn"], "{:.2f}"),
        f"ACTNUM\n  {nx * ny * nz}*1 /\n\n",
        _block("LITHOLOGY", model["lithology"], "{:d}"),
        "-- código  nome  coeficiente de ROP\nLITHTAB\n",
        *(f"  {code}  '{name}'  {coefficient:.2f} /\n" for code, (name, coefficient) in LITHOLOGIES.items()),
        "/\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--size", type=float, default=3000.0, help="lado da malha em planta (m)")
    args = parser.parse_args()

    model = build(args.nx, args.ny, args.size)
    write_grdecl(args.output, model)
    zcorn = model["zcorn"]
    print(f"{args.output}: {args.nx} x {args.ny} x {model['nz']} cells, depth {zcorn.min():.0f} to {zcorn.max():.0f} m")


if __name__ == "__main__":
    main()
