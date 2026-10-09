"""Gera a malha GRDECL realista usada nos testes (``tests/data/synthetic_basin.grdecl``).

A malha é só um arquivo de teste: o sistema não constrói malhas, ele as lê.
Ela segue o formato que o sistema espera de um usuário que já definiu as
litologias (código ``LITHOLOGY`` por célula e tabela ``LITHTAB`` com nome e
coeficiente de ROP) e imita um modelo geológico de bacia marinha, do fundo do
mar até abaixo do alvo padrão da GUI:

* 3 km × 3 km em planta, 20 × 20 colunas de 150 m;
* 16 formações até ~3,5 km, subdivididas em camadas de até 100 m no
  capeamento e de ~25 m nos reservatórios arenosos (refinamento proporcional);
* mergulho para leste e um anticlinal, mais fortes nas camadas profundas;
* uma falha normal norte-sul com rejeito que cresce com a profundidade
  (descontinuidade de ``ZCORN`` entre colunas vizinhas);
* um arenito que se acunha para oeste: onde a espessura vai a zero, as
  células ficam inativas (``ACTNUM`` = 0);
* um canal turbidítico arenoso dentro de duas formações de folhelho, ou seja,
  litologias diferentes na mesma profundidade.

Os coeficientes de ROP seguem a facilidade de perfuração: arenito 1,30,
siltito 1,00, marga 0,90, folhelho 0,80 e calcário 0,60.

Uso::

    python scripts/make_test_grid.py                  # tests/data/synthetic_basin.grdecl
    python scripts/make_test_grid.py --nx 40 --ny 40 -o outputs/malha_fina.grdecl

Com a malha padrão, a GUI roda com Wellhead = 1000, 1500, 0 e
Target = 2000, 1500, 3000 (o mesmo P3 = (1000, 3000) dos goldens); o alvo cai
no reservatório arenoso profundo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "synthetic_basin.grdecl"

# Código: (nome, coeficiente de ROP). Rochas mais friáveis têm coeficiente maior.
LITHOLOGIES = {
    1: ("Sandstone", 1.30),
    2: ("Siltstone", 1.00),
    3: ("Marl", 0.90),
    4: ("Shale", 0.80),
    5: ("Limestone", 0.60),
}
SANDSTONE, SILTSTONE, MARL, SHALE, LIMESTONE = 1, 2, 3, 4, 5

OVERBURDEN_LAYER_M = 100.0
RESERVOIR_LAYER_M = 25.0

# Formações do topo para a base: (nome, espessura nominal em m, litologia, espessura máxima das camadas).
FORMATIONS = [
    ("Marine marl", 200.0, MARL, OVERBURDEN_LAYER_M),
    ("Upper shale", 350.0, SHALE, OVERBURDEN_LAYER_M),
    ("Upper siltstone", 200.0, SILTSTONE, OVERBURDEN_LAYER_M),
    ("Turbidite shale", 120.0, SHALE, RESERVOIR_LAYER_M),
    ("Middle shale", 300.0, SHALE, OVERBURDEN_LAYER_M),
    ("Middle marl", 150.0, MARL, OVERBURDEN_LAYER_M),
    ("Upper limestone", 180.0, LIMESTONE, OVERBURDEN_LAYER_M),
    ("Channel shale", 250.0, SHALE, OVERBURDEN_LAYER_M),
    ("Wedge sandstone", 150.0, SANDSTONE, RESERVOIR_LAYER_M),
    ("Seal shale", 200.0, SHALE, OVERBURDEN_LAYER_M),
    ("Lower siltstone", 180.0, SILTSTONE, OVERBURDEN_LAYER_M),
    ("Lower limestone", 220.0, LIMESTONE, OVERBURDEN_LAYER_M),
    ("Intermediate sandstone", 200.0, SANDSTONE, RESERVOIR_LAYER_M),
    ("Deep shale", 300.0, SHALE, OVERBURDEN_LAYER_M),
    ("Main reservoir", 250.0, SANDSTONE, RESERVOIR_LAYER_M),
    ("Basal shale", 250.0, SHALE, OVERBURDEN_LAYER_M),
]
CHANNEL_FORMATIONS = ("Turbidite shale", "Channel shale")
"""Formações de folhelho cortadas pelo canal turbidítico arenoso."""
PINCH_OUT_FORMATION = "Wedge sandstone"
"""Arenito que se acunha para oeste."""

FAULT_X_FRACTION = 0.55
"""Posição da falha normal (fração do lado da malha); o bloco a leste é o abatido."""
FAULT_THROW_M = 90.0
"""Rejeito da falha na base da malha; cresce linearmente a partir do fundo do mar."""


def _structure(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    """Deslocamento estrutural (m) na base da malha: mergulho para leste e anticlinal."""
    dip = 150.0 * (x / size)
    anticline = -250.0 * np.exp(-((x - 0.55 * size) ** 2 + (y - 0.5 * size) ** 2) / (2 * (0.25 * size) ** 2))
    return dip + anticline


def _thickness_factor(name: str, x: np.ndarray, size: float) -> np.ndarray:
    """Fator de espessura da formação: 1, ou o acunhamento para oeste do arenito em cunha."""
    if name != PINCH_OUT_FORMATION:
        return np.ones_like(x)
    return np.clip((x - 0.15 * size) / (0.30 * size), 0.0, 1.0)


def horizons(x: np.ndarray, y: np.ndarray, size: float, downthrown: bool) -> tuple[np.ndarray, list[int]]:
    """Profundidade ``(nz + 1, ...)`` de todos os horizontes e a formação de cada camada.

    As fronteiras das formações ficam na profundidade nominal, exceto o topo do
    arenito em cunha: onde ele afina, a formação de cima desce até ele. Cada
    formação é dividida em camadas iguais (refinamento proporcional). A
    estrutura e o rejeito da falha deslocam cada horizonte na proporção da sua
    profundidade, então o fundo do mar é plano e horizontal.
    """
    boundaries = [np.full_like(x, depth) for depth in np.cumsum([0.0] + [formation[1] for formation in FORMATIONS])]
    for index, (name, thickness, _, _) in enumerate(FORMATIONS):
        if name == PINCH_OUT_FORMATION:
            boundaries[index] = boundaries[index + 1] - thickness * _thickness_factor(name, x, size)
    total = float(boundaries[-1].max())
    offset = _structure(x, y, size) + (FAULT_THROW_M if downthrown else 0.0)

    surfaces = [boundaries[0]]
    layer_formation = []
    for index, (_, thickness, _, max_layer) in enumerate(FORMATIONS):
        n_layers = max(1, int(np.ceil(thickness / max_layer)))
        for layer in range(1, n_layers + 1):
            surfaces.append(boundaries[index] + (layer / n_layers) * (boundaries[index + 1] - boundaries[index]))
            layer_formation.append(index)
    flat = np.stack(surfaces)
    return flat + (flat / total) * offset, layer_formation


def channel_mask(xc: np.ndarray, yc: np.ndarray, size: float) -> np.ndarray:
    """Células cujo centro cai no canal sinuoso que cruza a malha de sul para norte."""
    axis = 0.45 * size + 0.12 * size * np.sin(2.0 * np.pi * yc / size)
    return np.abs(xc - axis) < 0.07 * size


def build(nx: int, ny: int, size: float) -> dict:
    xs = np.linspace(0.0, size, nx + 1)
    ys = np.linspace(0.0, size, ny + 1)
    px, py = np.meshgrid(xs, ys)
    upthrown, layer_formation = horizons(px, py, size, downthrown=False)
    downthrown, _ = horizons(px, py, size, downthrown=True)
    nz = len(layer_formation)

    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    cx, cy = np.meshgrid(xc, yc)
    east_of_fault = xc > FAULT_X_FRACTION * size

    # ZCORN[k, topo/base, j, dj, i, di]: pilares verticais; cada coluna usa os
    # horizontes do seu bloco, então os cantos sobre a falha diferem entre vizinhos.
    zcorn = np.empty((nz, 2, ny, 2, nx, 2))
    for k in range(nz):
        for t in range(2):
            for dj in range(2):
                for di in range(2):
                    up = upthrown[k + t][dj:dj + ny, di:di + nx]
                    down = downthrown[k + t][dj:dj + ny, di:di + nx]
                    zcorn[k, t, :, dj, :, di] = np.where(east_of_fault[None, :], down, up)

    bottom = float(zcorn.max()) + 1.0
    coord = np.stack([px, py, np.zeros_like(px), px, py, np.full_like(px, bottom)], axis=-1)

    thickness = (zcorn[:, 1] - zcorn[:, 0]).max(axis=(2, 4))
    actnum = thickness > 0.01

    channel = channel_mask(cx, cy, size)
    lithology = np.empty((nz, ny, nx), dtype=int)
    for k, index in enumerate(layer_formation):
        name, _, code, _ = FORMATIONS[index]
        lithology[k] = code
        if name in CHANNEL_FORMATIONS:
            lithology[k, channel] = SANDSTONE
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "coord": coord,
        "zcorn": zcorn,
        "actnum": actnum,
        "lithology": lithology,
        "layer_formation": layer_formation,
    }


def _block(name: str, values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    lines = [" ".join(fmt.format(value) for value in flat[i:i + 8]) for i in range(0, flat.size, 8)]
    return f"{name}\n" + "\n".join(f"  {line}" for line in lines) + "\n/\n\n"


def _run_length_block(name: str, values) -> str:
    """Inteiros no formato compacto ``n*valor`` usado pelos exportadores."""
    flat = np.asarray(values).ravel()
    starts = np.flatnonzero(np.diff(flat, prepend=flat[0] - 1))
    lengths = np.diff(np.append(starts, flat.size))
    items = [f"{count}*{flat[start]}" if count > 1 else f"{flat[start]}" for start, count in zip(starts, lengths)]
    lines = [" ".join(items[i:i + 8]) for i in range(0, len(items), 8)]
    return f"{name}\n" + "\n".join(f"  {line}" for line in lines) + "\n/\n\n"


def write_grdecl(path: Path, model: dict) -> None:
    nx, ny, nz = model["nx"], model["ny"], model["nz"]
    formation_lines = []
    previous = None
    for k, index in enumerate(model["layer_formation"], start=1):
        if index != previous:
            name, thickness, code, _ = FORMATIONS[index]
            formation_lines.append(f"--   K = {k:>2}: {name} ({LITHOLOGIES[code][0]}, {thickness:g} m nominais)\n")
            previous = index
    text = [
        "-- Malha sintética realista de teste gerada por scripts/make_test_grid.py\n",
        "-- Profundidade (ZCORN) em metros abaixo do fundo do mar, positiva para baixo.\n",
        f"-- Falha normal norte-sul em x = {FAULT_X_FRACTION:g} × lado, bloco leste abatido até {FAULT_THROW_M:g} m.\n",
        "-- Formações (primeira camada K de cada uma):\n",
        *formation_lines,
        "\n",
        f"SPECGRID\n  {nx} {ny} {nz} 1 F /\n\n",
        _block("COORD", model["coord"], "{:.1f}"),
        _block("ZCORN", model["zcorn"], "{:.1f}"),
        _run_length_block("ACTNUM", model["actnum"].astype(int)),
        _run_length_block("LITHOLOGY", model["lithology"]),
        "-- código  nome  coeficiente de ROP\nLITHTAB\n",
        *(f"  {code}  '{name}'  {coefficient:.2f} /\n" for code, (name, coefficient) in LITHOLOGIES.items()),
        "/\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--size", type=float, default=3000.0, help="lado da malha em planta (m)")
    args = parser.parse_args()

    model = build(args.nx, args.ny, args.size)
    write_grdecl(args.output, model)
    zcorn = model["zcorn"]
    inactive = int((~model["actnum"]).sum())
    print(
        f"{args.output}: {args.nx} x {args.ny} x {model['nz']} cells ({inactive} inactive), "
        f"depth {zcorn.min():.0f} to {zcorn.max():.0f} m"
    )


if __name__ == "__main__":
    main()
