"""Leitura de malhas GRDECL e consulta de litologia ao longo do poço.

A malha geológica não é construída aqui: ela chega pronta, em um arquivo
GRDECL (formato Eclipse de corner-point), e o módulo só lê o que a
otimização usa.

* ``SPECGRID`` (ou ``DIMENS``), ``COORD`` e ``ZCORN``: geometria das células.
* ``ACTNUM`` (opcional): células com 0 são ignoradas.
* ``LITHOLOGY`` (ou ``FACIES``): código inteiro da litologia de cada célula.
* ``LITHTAB`` (opcional): nome e coeficiente de ROP de cada código, um
  registro por litologia::

      LITHTAB
      -- código  nome         coeficiente de ROP
         1      'Sandstone'   1.30 /
         3      'Shale'       0.75 /
      /

  Códigos fora da tabela viram ``Lithology <código>``, e um coeficiente
  omitido vale 1.

Parte-se de um usuário que já definiu as litologias da malha: o sistema não
as deduz de outras propriedades. Todo o resto (``SED1``..., ``BATHYMETRY``,
``THICKNESS``, ``MAPAXES``, ``INCLUDE``...) é
ignorado. As coordenadas são lidas como estão no arquivo, em metros, com a
profundidade (``ZCORN``) positiva para baixo — a mesma convenção do módulo
Minimization.

``GridGeology`` posiciona o poço na malha (cabeça do poço e alvo em XYZ) e
expõe ``segment_at(horizontal, depth)``, a mesma consulta que o cálculo de
tempo de broca faz na ``mesh`` de intervalos. O ROP base de cada elemento é
``base_rop × coeficiente da litologia``; fora da malha vale o coeficiente
``outside_rop_coefficient``.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OUTSIDE_GRID = "Outside grid"
"""Nome da litologia atribuída a pontos fora da malha ou em células inativas."""

_LITHOLOGY_KEYWORDS = ("LITHOLOGY", "FACIES")
_TABLE_KEYWORDS = {"LITHTAB"}
"""Keywords de vários registros: cada registro termina em ``/`` e uma ``/`` sozinha fecha o keyword."""
_NO_DATA_KEYWORDS = {"ECHO", "NOECHO", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SCHEDULE", "RUNSPEC", "END"}
_KEYWORDS_READ = {"SPECGRID", "DIMENS", "COORD", "ZCORN", "ACTNUM", *_LITHOLOGY_KEYWORDS}
_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


@dataclass
class CornerPointGrid:
    """Malha corner-point lida de um GRDECL.

    Attributes
    ----------
    nx, ny, nz : int
        Número de células em I, J e K.
    coord : numpy.ndarray
        Pilares ``(ny + 1, nx + 1, 6)``: ``x, y, z`` do topo e da base.
    zcorn : numpy.ndarray
        Profundidade dos cantos ``(nz, 2, ny, 2, nx, 2)``, indexada por
        ``[k, topo/base, j, j-local, i, i-local]``.
    actnum : numpy.ndarray
        ``(nz, ny, nx)`` booleano.
    lithology : numpy.ndarray
        ``(nz, ny, nx)`` com o índice em ``lithology_names``.
    lithology_names : list of str
        Nome de cada litologia, na ordem dos códigos.
    rop_coefficients : dict
        Coeficiente de ROP de cada litologia lido de ``LITHTAB``.
    source : str
        Caminho do arquivo lido.
    """

    nx: int
    ny: int
    nz: int
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: np.ndarray
    lithology: np.ndarray
    lithology_names: list[str]
    rop_coefficients: dict[str, float] = field(default_factory=dict)
    source: str = ""
    _digest: str = field(default="", init=False, repr=False)

    def pillar_xy(self, depth: float) -> np.ndarray:
        """Posição ``(ny + 1, nx + 1, 2)`` dos pilares na profundidade ``depth``."""
        top = self.coord[..., 0:3]
        bottom = self.coord[..., 3:6]
        dz = bottom[..., 2] - top[..., 2]
        safe_dz = np.where(np.abs(dz) > 1e-12, dz, 1.0)
        t = np.where(np.abs(dz) > 1e-12, (depth - top[..., 2]) / safe_dz, 0.0)
        return top[..., 0:2] + t[..., None] * (bottom[..., 0:2] - top[..., 0:2])

    def cell_corners(self, k: int, j: int, i: int) -> np.ndarray:
        """Os 8 cantos ``(2, 2, 2, 3)`` da célula, indexados por ``[topo/base, j-local, i-local]``."""
        corners = np.empty((2, 2, 2, 3))
        for t in range(2):
            for dj in range(2):
                for di in range(2):
                    z = self.zcorn[k, t, j, dj, i, di]
                    pillar = self.coord[j + dj, i + di]
                    dz = pillar[5] - pillar[2]
                    s = 0.0 if abs(dz) < 1e-12 else (z - pillar[2]) / dz
                    corners[t, dj, di, 0:2] = pillar[0:2] + s * (pillar[3:5] - pillar[0:2])
                    corners[t, dj, di, 2] = z
        return corners

    def extent(self) -> str:
        """Texto com a extensão XY e de profundidade da malha, para mensagens ao usuário."""
        xy = self.coord[..., 0:2].reshape(-1, 2)
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        z0, z1 = self.depth_range()
        return f"x {x0:g} to {x1:g} m, y {y0:g} to {y1:g} m, depth {z0:g} to {z1:g} m"

    def depth_range(self) -> tuple[float, float]:
        """Menor e maior profundidade de ``ZCORN``."""
        return float(self.zcorn.min()), float(self.zcorn.max())

    def lithology_counts(self) -> dict[str, int]:
        """Número de células ativas de cada litologia."""
        counts = np.bincount(self.lithology[self.actnum].ravel(), minlength=len(self.lithology_names))
        return {name: int(count) for name, count in zip(self.lithology_names, counts)}

    def digest(self) -> str:
        """Hash do conteúdo usado pela otimização (chave dos caches)."""
        if not self._digest:
            sha = hashlib.sha1()
            for array in (self.coord, self.zcorn, self.actnum, self.lithology):
                sha.update(np.ascontiguousarray(array).tobytes())
            sha.update("|".join(self.lithology_names).encode())
            self._digest = sha.hexdigest()
        return self._digest


def _tokens(text: str):
    text = re.sub(r"--[^\n]*", "", text)
    for token in _TOKEN.findall(text):
        if token != "/" and token.endswith("/"):
            yield token[:-1]
            yield "/"
        else:
            yield token


def _expand(values: list[str]) -> list[str]:
    expanded = []
    for value in values:
        if "*" in value:
            count, _, item = value.partition("*")
            expanded.extend([item] * int(count))
        else:
            expanded.append(value)
    return expanded


def _read_keywords(text: str) -> dict[str, list]:
    keywords: dict[str, list] = {}
    current = None
    values: list[str] = []
    records: list[list[str]] = []
    for token in _tokens(text):
        if current is None:
            name = token.upper()
            if name in _NO_DATA_KEYWORDS:
                continue
            current, values, records = name, [], []
        elif token != "/":
            values.append(token)
        elif current in _TABLE_KEYWORDS:
            if values:
                records.append(values)
                values = []
            else:
                keywords[current] = records
                current = None
        else:
            if current in _KEYWORDS_READ:
                keywords[current] = _expand(values)
            current = None
    return keywords


def _lithology_table(records: list[list[str]]) -> dict[int, tuple[str, float]]:
    """``{código: (nome, coeficiente de ROP)}`` a partir dos registros de ``LITHTAB``."""
    table: dict[int, tuple[str, float]] = {}
    for record in records:
        if len(record) not in (2, 3):
            raise ValueError(f"LITHTAB record {' '.join(record)!r} must be: code 'name' [ROP coefficient] /")
        code, name = int(record[0]), record[1].strip("'\"")
        coefficient = float(record[2]) if len(record) == 3 else 1.0
        if code in table:
            raise ValueError(f"LITHTAB defines code {code} twice.")
        if name in (entry[0] for entry in table.values()):
            raise ValueError(f"LITHTAB uses the name {name!r} twice.")
        if coefficient <= 0:
            raise ValueError(f"LITHTAB coefficient for {name!r} must be positive.")
        table[code] = (name, coefficient)
    return table


def _floats(keywords: dict, name: str, size: int) -> np.ndarray:
    values = np.array(keywords[name], dtype=float)
    if values.size != size:
        raise ValueError(f"GRDECL keyword {name} has {values.size} values; expected {size}.")
    return values


def read_grdecl(path: str | Path) -> CornerPointGrid:
    """Lê um arquivo GRDECL e identifica a litologia de cada célula.

    Parameters
    ----------
    path : str or Path
        Arquivo ``.grdecl``.

    Returns
    -------
    CornerPointGrid
        Geometria, células ativas e litologias.

    Raises
    ------
    ValueError
        Se faltar ``SPECGRID``/``DIMENS``, ``COORD``, ``ZCORN`` ou
        ``LITHOLOGY``/``FACIES``, se um array tiver tamanho incompatível com as
        dimensões ou se ``LITHTAB`` for inválida.
    """
    path = Path(path)
    keywords = _read_keywords(path.read_text(encoding="utf-8", errors="replace"))

    dims_key = "SPECGRID" if "SPECGRID" in keywords else "DIMENS"
    if dims_key not in keywords:
        raise ValueError("GRDECL file has no SPECGRID or DIMENS keyword.")
    nx, ny, nz = (int(value) for value in keywords[dims_key][:3])
    for name in ("COORD", "ZCORN"):
        if name not in keywords:
            raise ValueError(f"GRDECL file has no {name} keyword.")

    n_cells = nx * ny * nz
    coord = _floats(keywords, "COORD", 6 * (nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1, 6)
    zcorn = _floats(keywords, "ZCORN", 8 * n_cells).reshape(nz, 2, ny, 2, nx, 2)
    if "ACTNUM" in keywords:
        actnum = _floats(keywords, "ACTNUM", n_cells).reshape(nz, ny, nx) > 0
    else:
        actnum = np.ones((nz, ny, nx), dtype=bool)

    lithology_key = next((name for name in _LITHOLOGY_KEYWORDS if name in keywords), None)
    if lithology_key is None:
        raise ValueError("GRDECL file has no LITHOLOGY (or FACIES) keyword with the lithology code of each cell.")
    raw_codes = _floats(keywords, lithology_key, n_cells)
    if not np.all(raw_codes == np.round(raw_codes)):
        raise ValueError(f"GRDECL keyword {lithology_key} must contain integer lithology codes.")
    codes = raw_codes.astype(int).reshape(nz, ny, nx)
    table = _lithology_table(keywords.get("LITHTAB", []))

    # Litologias da tabela e as usadas pelas células ativas, na ordem dos códigos.
    all_codes = sorted(set(table) | set(codes[actnum].ravel().tolist()))
    lithology_names = [table[code][0] if code in table else f"Lithology {code}" for code in all_codes]
    rop_coefficients = {table[code][0]: table[code][1] for code in all_codes if code in table}
    lithology = np.searchsorted(all_codes, codes).clip(0, max(len(all_codes) - 1, 0))

    return CornerPointGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        lithology=lithology.astype(int),
        lithology_names=lithology_names,
        rop_coefficients=rop_coefficients,
        source=str(path),
    )


def _inverse_bilinear(p00, p10, p01, p11, point, iterations: int = 8) -> tuple[float, float]:
    """Coordenadas locais ``(u, v)`` de ``point`` no quadrilátero (u ao longo de I, v ao longo de J)."""
    u = v = 0.5
    for _ in range(iterations):
        position = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11
        du = (1 - v) * (p10 - p00) + v * (p11 - p01)
        dv = (1 - u) * (p01 - p00) + u * (p11 - p10)
        jacobian = np.column_stack([du, dv])
        if abs(np.linalg.det(jacobian)) < 1e-12:
            break
        step = np.linalg.solve(jacobian, point - position)
        u, v = u + step[0], v + step[1]
        if abs(step).max() < 1e-10:
            break
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


class GridGeology:
    """Malha GRDECL vista ao longo do plano vertical do poço.

    O poço Tipo 1 fica no plano vertical que passa pela cabeça do poço e pelo
    alvo. Na construção, esse plano é amostrado a cada ``section_step`` metros
    e, em cada amostra, guarda-se o intervalo de profundidade e a litologia de
    cada célula atravessada. ``segment_at`` só faz uma busca binária nesses
    intervalos.

    Parameters
    ----------
    grid : CornerPointGrid or None
        Malha lida por ``read_grdecl``. ``None`` deixa todo o poço "fora da malha".
    wellhead, target : sequence of float
        ``(x, y, z)`` nas coordenadas da malha, profundidade positiva para baixo.
    base_rop : float
        ROP de referência em m/h.
    rop_coefficients : dict, optional
        Coeficiente adimensional por litologia; sobrepõe o de ``LITHTAB``.
        Litologias sem coeficiente em nenhum dos dois valem 1.
    outside_rop_coefficient : float, optional
        Coeficiente fora da malha e em células inativas.
    section_step : float, optional
        Espaçamento horizontal das amostras do plano do poço, em metros.

    Raises
    ------
    ValueError
        Se o alvo não estiver mais fundo e afastado da cabeça do poço, ou se
        houver malha e o alvo não cair em uma célula ativa dela.

    Notes
    -----
    A coluna (I, J) de cada amostra é localizada com os pilares na
    profundidade média da malha; com pilares inclinados a coluna é aproximada.
    """

    def __init__(
        self,
        grid: CornerPointGrid | None,
        wellhead,
        target,
        base_rop: float,
        rop_coefficients: dict | None = None,
        outside_rop_coefficient: float = 1.0,
        section_step: float = 1.0,
    ) -> None:
        self.grid = grid
        self.wellhead = np.asarray(wellhead, dtype=float).reshape(3)
        self.target = np.asarray(target, dtype=float).reshape(3)
        delta = self.target - self.wellhead
        self.horizontal_distance = float(np.hypot(delta[0], delta[1]))
        self.vertical_depth = float(delta[2])
        if self.horizontal_distance <= 0 or self.vertical_depth <= 0:
            raise ValueError("The target must be deeper than the wellhead and horizontally displaced from it.")
        if base_rop <= 0:
            raise ValueError("'base_rop' must be positive.")
        if section_step <= 0:
            raise ValueError("'section_step' must be positive.")
        self.direction = delta[0:2] / self.horizontal_distance
        self.base_rop = float(base_rop)
        self.section_step = float(section_step)

        names = [] if grid is None else list(grid.lithology_names)
        coefficients = {name: grid.rop_coefficients.get(name, 1.0) for name in names}
        coefficients.update(rop_coefficients or {})
        coefficients[OUTSIDE_GRID] = float(outside_rop_coefficient)
        for name, value in coefficients.items():
            if value is None or value <= 0:
                raise ValueError(f"ROP coefficient for '{name}' must be positive.")
        self.rop_coefficients = {name: float(coefficients[name]) for name in names + [OUTSIDE_GRID]}
        self._segments = [{"lithology": name, "rop": self.base_rop * self.rop_coefficients[name]} for name in names]
        self._outside = {"lithology": OUTSIDE_GRID, "rop": self.base_rop * self.rop_coefficients[OUTSIDE_GRID]}
        self._build_section()
        if grid is not None and self.segment_at(*self.P3) is self._outside:
            raise ValueError(
                f"The target ({self.target[0]:g}, {self.target[1]:g}, {self.target[2]:g}) is not inside an active "
                f"cell of the grid ({grid.extent()})."
            )

    @property
    def P3(self) -> tuple[float, float]:
        """Alvo no plano do poço: ``(distância horizontal, profundidade)`` a partir da cabeça."""
        return (self.horizontal_distance, self.vertical_depth)

    @property
    def lithology_names(self) -> list[str]:
        return [] if self.grid is None else list(self.grid.lithology_names)

    def to_world(self, horizontal, depth) -> np.ndarray:
        """Converte ``(horizontal, depth)`` do plano do poço em ``(x, y, z)`` da malha."""
        horizontal = np.asarray(horizontal, dtype=float)
        depth = np.asarray(depth, dtype=float)
        xy = self.wellhead[0:2] + horizontal[..., None] * self.direction
        return np.concatenate([xy, (self.wellhead[2] + depth)[..., None]], axis=-1)

    def _build_section(self) -> None:
        n = max(1, int(np.ceil(self.horizontal_distance / self.section_step)))
        self.section_s = np.linspace(0.0, self.horizontal_distance, n + 1)
        self.section_columns: list[tuple[int, int] | None] = []
        self._section_starts: list[list[float]] = []
        self._section_cells: list[list[tuple[float, float, int, int]]] = []
        grid = self.grid
        if grid is None:
            self.section_columns = [None] * len(self.section_s)
            self._section_starts = [[] for _ in self.section_s]
            self._section_cells = [[] for _ in self.section_s]
            self._build_section_arrays()
            return

        z_ref = float(np.mean(grid.depth_range()))
        pillars = grid.pillar_xy(z_ref)
        p00 = pillars[:-1, :-1].reshape(-1, 2)
        p10 = pillars[:-1, 1:].reshape(-1, 2)
        p11 = pillars[1:, 1:].reshape(-1, 2)
        p01 = pillars[1:, :-1].reshape(-1, 2)
        quads = np.stack([p00, p10, p11, p01], axis=1)
        lower = quads.min(axis=1)
        upper = quads.max(axis=1)
        tol = 1e-9 * max(1.0, float(np.abs(pillars).max()))

        for xyz in self.to_world(self.section_s, np.zeros_like(self.section_s)):
            point = xyz[0:2]
            column = None
            candidates = np.nonzero(np.all((point >= lower - tol) & (point <= upper + tol), axis=1))[0]
            for index in candidates:
                quad = quads[index]
                edges = np.roll(quad, -1, axis=0) - quad
                rel = point - quad
                cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
                if np.all(cross >= -tol) or np.all(cross <= tol):
                    column = (int(index // grid.nx), int(index % grid.nx))
                    break
            self.section_columns.append(column)
            if column is None:
                self._section_starts.append([])
                self._section_cells.append([])
                continue

            j, i = column
            u, v = _inverse_bilinear(pillars[j, i], pillars[j, i + 1], pillars[j + 1, i], pillars[j + 1, i + 1], point)
            weights = np.array([[(1 - u) * (1 - v), u * (1 - v)], [(1 - u) * v, u * v]])
            faces = grid.zcorn[:, :, j, :, i, :]
            depths = np.einsum("ktab,ab->kt", faces, weights)
            cells = [
                (float(depths[k, 0]), float(depths[k, 1]), int(grid.lithology[k, j, i]), k)
                for k in range(grid.nz)
                if grid.actnum[k, j, i] and depths[k, 1] - depths[k, 0] > 1e-9
            ]
            cells.sort()
            self._section_cells.append(cells)
            self._section_starts.append([cell[0] for cell in cells])
        self._build_section_arrays()

    def _build_section_arrays(self) -> None:
        """Topos, bases e litologias das amostras em matrizes ``(amostra, célula)`` para ``segments_at``."""
        width = max([len(cells) for cells in self._section_cells] + [1])
        shape = (len(self._section_cells), width)
        self._tops = np.full(shape, np.inf)
        self._bottoms = np.full(shape, -np.inf)
        self._codes = np.full(shape, -1, dtype=int)
        for sample, cells in enumerate(self._section_cells):
            for position, (top, bottom, code, _) in enumerate(cells):
                self._tops[sample, position] = top
                self._bottoms[sample, position] = bottom
                self._codes[sample, position] = code

    def _sample_index(self, horizontal: float) -> int:
        index = int(round(float(horizontal) / self.section_step))
        return min(max(index, 0), len(self.section_s) - 1)

    def segment_at(self, horizontal: float, depth: float) -> dict:
        """Litologia e ROP base no ponto ``(horizontal, depth)`` do plano do poço.

        Parameters
        ----------
        horizontal : float
            Distância horizontal a partir da cabeça do poço, em direção ao alvo.
        depth : float
            Profundidade abaixo da cabeça do poço.

        Returns
        -------
        dict
            ``{"lithology": nome, "rop": base_rop × coeficiente}``.
        """
        sample = self._sample_index(horizontal)
        z = self.wellhead[2] + float(depth)
        starts = self._section_starts[sample]
        index = bisect.bisect_right(starts, z) - 1
        if index >= 0:
            top, bottom, code, _ = self._section_cells[sample][index]
            if z <= bottom:
                return self._segments[code]
        return self._outside

    def segments_at(self, horizontal, depth) -> list[dict]:
        """``segment_at`` para arrays de pontos (usado na integração do tempo de broca)."""
        horizontal = np.asarray(horizontal, dtype=float)
        z = self.wellhead[2] + np.asarray(depth, dtype=float)
        samples = np.rint(horizontal / self.section_step).astype(int).clip(0, len(self.section_s) - 1)
        position = (self._tops[samples] <= z[:, None]).sum(axis=1) - 1
        safe = position.clip(0, None)
        inside = (position >= 0) & (z <= self._bottoms[samples, safe])
        codes = np.where(inside, self._codes[samples, safe], -1)
        return [self._outside if code < 0 else self._segments[code] for code in codes.tolist()]

    def section_polygons(self) -> list[dict]:
        """Polígonos ``(horizontal, depth)`` de cada célula cortada pelo plano do poço.

        Amostras consecutivas na mesma coluna formam um único polígono por camada.
        """
        polygons = []
        start = 0
        columns = self.section_columns
        for end in range(1, len(columns) + 1):
            if end < len(columns) and columns[end] == columns[start]:
                continue
            if columns[start] is not None:
                samples = range(start, end)
                layers = {cell[3] for sample in samples for cell in self._section_cells[sample]}
                for k in sorted(layers):
                    rows = [
                        (self.section_s[sample], cell[0], cell[1], cell[2])
                        for sample in samples
                        for cell in self._section_cells[sample]
                        if cell[3] == k
                    ]
                    # Meia amostra para cada lado, para as colunas vizinhas se encostarem.
                    half = 0.5 * (self.section_s[1] - self.section_s[0]) if len(self.section_s) > 1 else 0.0
                    s_values = [float(row[0]) for row in rows]
                    s_values = [s_values[0] - half, *s_values, s_values[-1] + half]
                    tops = [float(row[1] - self.wellhead[2]) for row in rows]
                    bottoms = [float(row[2] - self.wellhead[2]) for row in rows]
                    tops = [tops[0], *tops, tops[-1]]
                    bottoms = [bottoms[0], *bottoms, bottoms[-1]]
                    polygons.append(
                        {
                            "lithology": self.grid.lithology_names[rows[0][3]],
                            "horizontal": s_values + s_values[::-1],
                            "depth": tops + bottoms[::-1],
                        }
                    )
            start = end
        return polygons

    def cells_around_well(self, margin_xy: float | None = None, margin_z: float | None = None) -> tuple[list, tuple[float, float]]:
        """Células ativas na caixa que envolve o poço, para a vista 3D.

        A caixa vai da cabeça do poço ao alvo, com ``margin_xy`` para cada lado
        em planta e ``margin_z`` abaixo do alvo. Entram todas as colunas cujo
        centro cai na caixa, para a vista não ter buracos.

        Parameters
        ----------
        margin_xy : float, optional
            Margem horizontal em metros; o padrão é o maior entre duas colunas
            e 15 % do afastamento horizontal do poço.
        margin_z : float, optional
            Margem abaixo do alvo em metros; o padrão é 5 % da profundidade do poço.

        Returns
        -------
        tuple
            Lista de ``(k, j, i)`` e o intervalo de profundidade ``(topo, base)`` da caixa.
        """
        if margin_z is None:
            margin_z = 0.05 * self.vertical_depth
        z_range = (float(self.wellhead[2]), float(self.target[2] + margin_z))
        grid = self.grid
        if grid is None:
            return [], z_range

        pillars = grid.pillar_xy(float(np.mean(grid.depth_range())))
        centers = 0.25 * (pillars[:-1, :-1] + pillars[:-1, 1:] + pillars[1:, :-1] + pillars[1:, 1:])
        if margin_xy is None:
            widths = np.concatenate(
                [np.linalg.norm(np.diff(pillars, axis=0), axis=-1).ravel(), np.linalg.norm(np.diff(pillars, axis=1), axis=-1).ravel()]
            )
            margin_xy = max(2.0 * float(np.median(widths)), 0.15 * self.horizontal_distance)
        lower = np.minimum(self.wellhead[0:2], self.target[0:2]) - margin_xy
        upper = np.maximum(self.wellhead[0:2], self.target[0:2]) + margin_xy
        columns = np.all((centers >= lower) & (centers <= upper), axis=-1)

        tops = grid.zcorn[:, 0].min(axis=(2, 4))
        bottoms = grid.zcorn[:, 1].max(axis=(2, 4))
        selected = grid.actnum & columns[None] & (tops < z_range[1]) & (bottoms > z_range[0]) & (bottoms > tops)
        cells = [tuple(int(index) for index in cell) for cell in np.argwhere(selected)]
        return cells, z_range

    def cache_signature(self) -> tuple:
        return (
            None if self.grid is None else self.grid.digest(),
            tuple(self.wellhead),
            tuple(self.target),
            self.base_rop,
            tuple(sorted(self.rop_coefficients.items())),
            self.section_step,
        )
