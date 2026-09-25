"""Modelo geológico heterogêneo tridimensional.

O modelo é uma pilha estratigráfica: ``K + 1`` horizontes (superfícies de
profundidade ``z(x, y)`` amostradas em um raster horizontal comum) delimitam
``K`` camadas. Cada camada tem um mapa de fácies, isto é, um código de
litologia por célula do raster, de modo que a litologia pode variar
lateralmente *dentro* de uma camada. Horizontes inclinados/dobrados e mapas de
fácies laterais permitem que duas litologias diferentes ocupem a mesma
profundidade ``z`` em posições ``(x, y)`` distintas, o que uma pilha de
paralelepípedos não consegue representar.

Todas as consultas por ponto são vetorizadas: recebem arrays numpy de
coordenadas e devolvem arrays numpy, porque o otimizador amostra milhões de
pontos.

Convenção de eixos: ``x`` e ``y`` horizontais, ``z`` profundidade positiva
para baixo (a mesma do ``DataSet``).
"""

from __future__ import annotations

import hashlib

import numpy as np


# O código 0 é reservado para "sem dado". As tabelas de propriedades são indexadas por estes códigos.
LITHOLOGY_NAMES = (
    "Undefined",
    "Shale",
    "Siltstone",
    "Sandstone",
    "Limestone",
    "Dolomite",
    "Evaporite",
)

LITHOLOGY_CODES = {name: code for code, name in enumerate(LITHOLOGY_NAMES)}

UNDEFINED_CODE = 0

LITHOLOGY_COLORS = {
    "Shale": "#8c8c8c",
    "Siltstone": "#bdb76b",
    "Sandstone": "#d8b365",
    "Limestone": "#f6e8c3",
    "Dolomite": "#5ab4ac",
    "Evaporite": "#c2a5cf",
    "Undefined": "#dddddd",
}

_SURFACE_TOL = 1.0e-6


def lithology_code(name) -> int:
    """Devolve o código inteiro de uma litologia a partir do nome (sem diferenciar maiúsculas)."""
    if isinstance(name, (int, np.integer)):
        code = int(name)
        if not 0 <= code < len(LITHOLOGY_NAMES):
            raise ValueError(f"Unknown lithology code: {code}")
        return code
    key = str(name).strip().capitalize()
    if key not in LITHOLOGY_CODES:
        raise ValueError(
            f"Unknown lithology '{name}'. Supported: {', '.join(LITHOLOGY_NAMES[1:])}."
        )
    return LITHOLOGY_CODES[key]


def lithology_name(code) -> str:
    """Devolve o nome da litologia a partir do código inteiro."""
    return LITHOLOGY_NAMES[int(code)]


class GeoGrid2D:
    """Raster horizontal regular compartilhado por todos os horizontes e mapas de fácies."""

    def __init__(self, x0: float, y0: float, dx: float, dy: float, nx: int, ny: int) -> None:
        nx, ny = int(nx), int(ny)
        if nx < 2 or ny < 2:
            raise ValueError("The horizontal grid needs at least 2 nodes in each direction.")
        if dx <= 0 or dy <= 0:
            raise ValueError("The grid spacings 'dx' and 'dy' must be positive.")
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.dx = float(dx)
        self.dy = float(dy)
        self.nx = nx
        self.ny = ny

    @classmethod
    def from_bounds(cls, x_range, y_range, nx: int = 2, ny: int = 2) -> "GeoGrid2D":
        """Cria o raster a partir dos limites ``(min, max)`` e do número de nós."""
        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Each grid bound must satisfy min < max.")
        nx, ny = int(nx), int(ny)
        return cls(x0, y0, (x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1), nx, ny)

    @classmethod
    def from_spacing(cls, x_range, y_range, spacing: float) -> "GeoGrid2D":
        """Cria o raster a partir dos limites e de um espaçamento aproximado em metros."""
        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
        nx = max(2, int(np.ceil((x1 - x0) / float(spacing))) + 1)
        ny = max(2, int(np.ceil((y1 - y0) / float(spacing))) + 1)
        return cls.from_bounds((x0, x1), (y0, y1), nx, ny)

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.nx, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.dy * np.arange(self.ny, dtype=float)

    @property
    def x_range(self) -> tuple[float, float]:
        return (self.x0, self.x0 + self.dx * (self.nx - 1))

    @property
    def y_range(self) -> tuple[float, float]:
        return (self.y0, self.y0 + self.dy * (self.ny - 1))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordenadas dos nós ``(X, Y)``, cada uma com forma ``(ny, nx)``."""
        return np.meshgrid(self.x, self.y)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=float)

    def broadcast(self, field) -> np.ndarray:
        """Converte um escalar ou array em um campo float ``(ny, nx)``."""
        array = np.asarray(field, dtype=float)
        if array.ndim == 0:
            return np.full(self.shape, float(array))
        if array.shape != self.shape:
            raise ValueError(
                f"Field shape {array.shape} does not match the grid shape {self.shape}."
            )
        return np.ascontiguousarray(array, dtype=float)

    def fractional_index(self, X, Y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mapeia ``(x, y)`` do mundo em índices fracionários de nó, limitados ao raster.

        Devolve ``(fx, fy, outside)``, em que ``outside`` marca os pontos que
        caíram fora do raster e por isso foram grampeados na borda.
        """
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        fx = (x - self.x0) / self.dx
        fy = (y - self.y0) / self.dy
        outside = (fx < 0.0) | (fx > self.nx - 1) | (fy < 0.0) | (fy > self.ny - 1)
        return np.clip(fx, 0.0, self.nx - 1), np.clip(fy, 0.0, self.ny - 1), outside

    def bilinear_stack(self, stack: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Interpola bilinearmente uma pilha ``(K, ny, nx)`` em índices fracionários.

        ``fx`` e ``fy`` têm forma ``(N,)``; o resultado tem forma ``(K, N)``.
        """
        ix0 = np.clip(np.floor(fx).astype(np.intp), 0, self.nx - 2)
        iy0 = np.clip(np.floor(fy).astype(np.intp), 0, self.ny - 2)
        tx = fx - ix0
        ty = fy - iy0
        f00 = stack[:, iy0, ix0]
        f10 = stack[:, iy0, ix0 + 1]
        f01 = stack[:, iy0 + 1, ix0]
        f11 = stack[:, iy0 + 1, ix0 + 1]
        return (
            f00 * (1.0 - tx) * (1.0 - ty)
            + f10 * tx * (1.0 - ty)
            + f01 * (1.0 - tx) * ty
            + f11 * tx * ty
        )

    def nearest_index(self, fx: np.ndarray, fy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Índices do nó mais próximo; usado em campos categóricos, que não podem ser interpolados."""
        ix = np.clip(np.rint(fx).astype(np.intp), 0, self.nx - 1)
        iy = np.clip(np.rint(fy).astype(np.intp), 0, self.ny - 1)
        return ix, iy

    def signature(self) -> tuple:
        return (self.x0, self.y0, self.dx, self.dy, self.nx, self.ny)

    def __repr__(self) -> str:
        return (
            f"GeoGrid2D(x={self.x_range}, y={self.y_range}, nx={self.nx}, ny={self.ny})"
        )


class HorizonModel:
    """Pilha de ``K`` camadas delimitadas por ``K + 1`` superfícies de horizonte.

    ``horizons[k]`` é o topo da camada ``k`` e ``horizons[k + 1]`` sua base,
    ambos em profundidade sobre ``grid``. As superfícies devem ser não
    decrescentes com ``k``; valores iguais são permitidos e representam um
    acunhamento (camada de espessura zero), que a consulta por ponto ignora.

    ``layers[k]`` é o mapa de fácies da camada ``k``: um único nome de
    litologia ou um array ``(ny, nx)`` de códigos de litologia, o que permite
    que a litologia mude lateralmente dentro de uma mesma unidade.

    Parameters
    ----------
    grid : GeoGrid2D
        Raster horizontal dos horizontes e mapas de fácies.
    horizons : sequence
        ``K + 1`` superfícies (escalar ou array ``(ny, nx)``), em metros.
    layers : sequence
        ``K`` mapas de fácies (nome, array de códigos/nomes ou dict com
        ``lithology`` e ``name``).
    rop_values, mu_values, wear_factors : dict or None, optional
        Propriedades por litologia: ROP (m/h), coeficiente de atrito e fator
        de desgaste de broca.
    default_rop, default_mu : float or None, optional
        Valores usados para litologias ausentes das tabelas.
    outside : {"clamp", "undefined", "error"}, optional
        Tratamento de pontos fora do raster horizontal.
    name : str or None, optional
        Rótulo livre do modelo.
    """

    def __init__(
        self,
        grid: GeoGrid2D,
        horizons,
        layers,
        rop_values: dict | None = None,
        mu_values: dict | None = None,
        wear_factors: dict | None = None,
        default_rop: float | None = None,
        default_mu: float | None = None,
        outside: str = "clamp",
        name: str | None = None,
    ) -> None:
        if not isinstance(grid, GeoGrid2D):
            raise TypeError("'grid' must be a GeoGrid2D instance.")
        if outside not in ("clamp", "undefined", "error"):
            raise ValueError("'outside' must be one of 'clamp', 'undefined' or 'error'.")

        horizons = list(horizons)
        layers = list(layers)
        if len(horizons) < 2:
            raise ValueError("At least two horizons are required to define one layer.")
        if len(layers) != len(horizons) - 1:
            raise ValueError(
                f"{len(horizons)} horizons define {len(horizons) - 1} layers, "
                f"but {len(layers)} facies maps were given."
            )

        self.grid = grid
        self.outside = outside
        self.name = name
        self.n_layers = len(layers)

        surfaces = np.stack([grid.broadcast(surface) for surface in horizons], axis=0)
        for k in range(surfaces.shape[0] - 1):
            if np.any(surfaces[k + 1] < surfaces[k] - _SURFACE_TOL):
                raise ValueError(
                    f"Horizon {k + 1} rises above horizon {k} somewhere on the grid. "
                    "Horizons must be non-decreasing with depth (equal values = pinch-out)."
                )
        self.horizons = np.ascontiguousarray(surfaces, dtype=float)

        codes = np.empty((self.n_layers,) + grid.shape, dtype=np.int8)
        self.layer_names: list[str] = []
        for k, layer in enumerate(layers):
            facies, label = self._normalize_layer(grid, layer, k)
            codes[k] = facies
            self.layer_names.append(label)
        self.layer_codes = codes

        self.present_codes = tuple(sorted(int(c) for c in np.unique(self.layer_codes)))
        self.lithologies = tuple(lithology_name(c) for c in self.present_codes)

        self.default_rop = None if default_rop is None else float(default_rop)
        self.default_mu = None if default_mu is None else float(default_mu)
        self._rop = self._property_table("ROP", rop_values, self.default_rop, positive=True)
        self._mu = self._property_table("mu", mu_values, self.default_mu, positive=False)
        self._wear = self._property_table("wear factor", wear_factors, 1.0, positive=True)
        self.has_mu = mu_values is not None or self.default_mu is not None

        for code in self.present_codes:
            if code == UNDEFINED_CODE:
                continue
            if np.isnan(self._rop[code]):
                raise ValueError(
                    f"Missing ROP for lithology '{lithology_name(code)}'. "
                    "Provide it in 'rop_values' or set 'default_rop'."
                )

        self.z_min = float(self.horizons[0].min())
        self.z_max = float(self.horizons[-1].max())
        self._signature = self._build_signature()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _normalize_layer(grid: GeoGrid2D, layer, index: int) -> tuple[np.ndarray, str]:
        label = f"Layer {index}"
        facies = layer
        if isinstance(layer, dict):
            label = str(layer.get("name", label))
            facies = layer.get("lithology", layer.get("facies"))
            if facies is None:
                raise ValueError(f"Layer {index} has no 'lithology' entry.")
        if isinstance(facies, str):
            code = lithology_code(facies)
            return np.full(grid.shape, code, dtype=np.int8), (
                label if isinstance(layer, dict) else facies
            )
        array = np.asarray(facies)
        if array.ndim == 0:
            return np.full(grid.shape, lithology_code(array.item()), dtype=np.int8), label
        if array.shape != grid.shape:
            raise ValueError(
                f"Facies map of layer {index} has shape {array.shape}, expected {grid.shape}."
            )
        if array.dtype.kind in "US":
            flat = np.array([lithology_code(v) for v in array.ravel()], dtype=np.int8)
            return flat.reshape(grid.shape), label
        codes = array.astype(np.int8)
        bad = np.unique(codes[(codes < 0) | (codes >= len(LITHOLOGY_NAMES))])
        if bad.size:
            raise ValueError(f"Layer {index} contains invalid lithology codes: {bad.tolist()}")
        return np.ascontiguousarray(codes), label

    @staticmethod
    def _property_table(what: str, values: dict | None, fallback, positive: bool) -> np.ndarray:
        table = np.full(len(LITHOLOGY_NAMES), np.nan, dtype=float)
        if fallback is not None:
            table[:] = float(fallback)
        if values is not None:
            for key, value in values.items():
                code = lithology_code(key)
                if value is None:
                    continue
                value = float(value)
                if positive and value <= 0:
                    raise ValueError(f"The {what} of '{key}' must be positive.")
                table[code] = value
        return table

    def _build_signature(self) -> tuple:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(self.horizons, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.layer_codes, dtype=np.int8).tobytes())
        return (
            "horizon",
            self.grid.signature(),
            self.n_layers,
            digest.hexdigest(),
            tuple(np.nan_to_num(self._rop, nan=-1.0).tolist()),
            tuple(np.nan_to_num(self._mu, nan=-1.0).tolist()),
            tuple(np.nan_to_num(self._wear, nan=-1.0).tolist()),
            self.default_rop,
            self.default_mu,
            self.outside,
        )

    def cache_signature(self) -> tuple:
        """Chave imutável usada pelos caches da otimização."""
        return self._signature

    # ------------------------------------------------------------------ query

    def horizon_depths_at(self, X, Y) -> np.ndarray:
        """Profundidade de cada horizonte em ``(x, y)``; forma ``(K + 1, N)``."""
        fx, fy, _ = self.grid.fractional_index(np.ravel(X), np.ravel(Y))
        return self.grid.bilinear_stack(self.horizons, fx, fy)

    def lithology_code_at_points(self, X, Y, Z) -> np.ndarray:
        """Código de litologia em cada ponto ``(x, y, z)`` (arrays com broadcast)."""
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        z = np.asarray(Z, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        xf = np.ascontiguousarray(np.broadcast_to(x, shape).ravel())
        yf = np.ascontiguousarray(np.broadcast_to(y, shape).ravel())
        zf = np.ascontiguousarray(np.broadcast_to(z, shape).ravel())

        fx, fy, lateral_outside = self.grid.fractional_index(xf, yf)
        if self.outside == "error" and lateral_outside.any():
            first = int(np.argmax(lateral_outside))
            raise ValueError(
                f"Point ({xf[first]:.3f}, {yf[first]:.3f}) lies outside the geological grid "
                f"{self.grid.x_range} x {self.grid.y_range}."
            )

        depths = self.grid.bilinear_stack(self.horizons, fx, fy)  # (K + 1, N)
        index = np.count_nonzero(depths <= zf[None, :], axis=0) - 1
        # Um ponto exatamente na base do modelo pertence à última camada.
        index = np.where(
            (index == self.n_layers) & np.isclose(zf, depths[self.n_layers]),
            self.n_layers - 1,
            index,
        )
        valid = (index >= 0) & (index < self.n_layers)
        if self.outside == "undefined":
            valid &= ~lateral_outside

        ix, iy = self.grid.nearest_index(fx, fy)
        safe_index = np.where(valid, index, 0)
        codes = np.where(
            valid, self.layer_codes[safe_index, iy, ix], UNDEFINED_CODE
        ).astype(np.int8)
        return codes.reshape(shape)

    def lithology_at_points(self, X, Y, Z) -> np.ndarray:
        """Nome da litologia em cada ponto ``(x, y, z)``."""
        codes = self.lithology_code_at_points(X, Y, Z)
        names = np.array(LITHOLOGY_NAMES, dtype=object)
        return names[codes]

    def _lookup(self, table: np.ndarray, what: str, codes: np.ndarray) -> np.ndarray:
        values = table[codes]
        missing = np.isnan(values)
        if np.any(missing):
            names = sorted({lithology_name(c) for c in np.unique(codes[missing])})
            raise ValueError(
                f"No {what} defined for lithology/lithologies {names} reached by the "
                f"trajectory. Provide the value or set a default."
            )
        return values

    def _sampled(self, table: np.ndarray, what: str, X, Y, Z) -> np.ndarray:
        return self._lookup(table, what, self.lithology_code_at_points(X, Y, Z))

    def sample(self, X, Y, Z, with_mu: bool = True, with_wear: bool = False) -> dict:
        """Consulta o ponto uma vez e devolve todas as propriedades derivadas dele.

        O otimizador amostra milhões de pontos; consultar ROP, atrito e desgaste
        separadamente triplicaria o custo da única etapa cara.

        Returns
        -------
        dict
            ``code``, ``lithology``, ``rop`` e, quando pedidos e disponíveis,
            ``mu`` e ``wear`` (``None`` caso contrário).
        """
        codes = self.lithology_code_at_points(X, Y, Z)
        names = np.array(LITHOLOGY_NAMES, dtype=object)
        result = {
            "code": codes,
            "lithology": names[codes],
            "rop": self._lookup(self._rop, "ROP", codes),
            "mu": None,
            "wear": None,
        }
        if with_mu and self.has_mu:
            result["mu"] = self._lookup(self._mu, "friction coefficient", codes)
        if with_wear:
            result["wear"] = self._lookup(self._wear, "wear factor", codes)
        return result

    def rop_at_points(self, X, Y, Z) -> np.ndarray:
        """ROP base (m/h) em cada ponto."""
        return self._sampled(self._rop, "ROP", X, Y, Z)

    def mu_at_points(self, X, Y, Z) -> np.ndarray:
        """Coeficiente de atrito em cada ponto; exige ``mu_values`` ou ``default_mu``."""
        if not self.has_mu:
            raise ValueError(
                "This geological model has no friction data. Build it with 'mu_values' "
                "(or 'default_mu') to use the lithology-dependent friction model."
            )
        return self._sampled(self._mu, "friction coefficient", X, Y, Z)

    def wear_at_points(self, X, Y, Z) -> np.ndarray:
        """Fator de desgaste de broca em cada ponto."""
        return self._sampled(self._wear, "wear factor", X, Y, Z)

    # ---------------------------------------------------------- scalar helpers

    def lithology_at(self, x: float, y: float, z: float) -> str:
        return lithology_name(int(self.lithology_code_at_points(x, y, z)))

    def rop_at_point(self, x: float, y: float, z: float) -> float:
        return float(self.rop_at_points(x, y, z))

    def mu_at_point(self, x: float, y: float, z: float) -> float:
        return float(self.mu_at_points(x, y, z))

    def segment_at_point(self, x: float, y: float, z: float) -> dict:
        """Consulta escalar no formato da antiga API de malha em blocos."""
        code = int(self.lithology_code_at_points(x, y, z))
        rop = self._rop[code]
        return {
            "lithology": lithology_name(code),
            "code": code,
            "rop": None if np.isnan(rop) else float(rop),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }

    def thickness_at(self, x: float, y: float) -> np.ndarray:
        """Espessura de cada camada na vertical de ``(x, y)``."""
        depths = self.horizon_depths_at(x, y)[:, 0]
        return np.diff(depths)

    def property_values(self, what: str) -> dict:
        """Tabela ``{litologia: valor}`` de ``"rop"``, ``"mu"`` ou ``"wear"`` definida no modelo."""
        tables = {"rop": self._rop, "mu": self._mu, "wear": self._wear}
        if what not in tables:
            raise ValueError("'what' must be one of 'rop', 'mu' or 'wear'.")
        table = tables[what]
        return {
            LITHOLOGY_NAMES[code]: float(table[code])
            for code in range(1, len(LITHOLOGY_NAMES))
            if not np.isnan(table[code])
        }

    @property
    def is_flat(self) -> bool:
        """``True`` quando todos os horizontes são planos e todas as fácies são uniformes."""
        flat_horizons = np.allclose(self.horizons, self.horizons[:, :1, :1], atol=_SURFACE_TOL)
        uniform_facies = bool(np.all(self.layer_codes == self.layer_codes[:, :1, :1]))
        return bool(flat_horizons and uniform_facies)

    def flat_intervals(self) -> list[dict]:
        """Intervalos de profundidade ``{lithology, start, end}`` de um modelo plano.

        É a forma que a tabela da GUI edita. Camadas ``Undefined`` (lacunas) e
        de espessura zero são omitidas.

        Raises
        ------
        ValueError
            Se o modelo tiver horizontes inclinados ou fácies laterais.
        """
        if not self.is_flat:
            raise ValueError("Only flat, laterally uniform models can be listed as depth intervals.")
        depths = self.horizons[:, 0, 0]
        intervals = []
        for k in range(self.n_layers):
            code = int(self.layer_codes[k, 0, 0])
            start, end = float(depths[k]), float(depths[k + 1])
            if code == UNDEFINED_CODE or end - start <= _SURFACE_TOL:
                continue
            intervals.append({"lithology": lithology_name(code), "start": start, "end": end})
        return intervals

    def as_dict(self) -> dict:
        return {
            "kind": "horizon",
            "name": self.name,
            "grid": self.grid.signature(),
            "n_layers": self.n_layers,
            "layer_names": list(self.layer_names),
            "lithologies": list(self.lithologies),
            "z_range": (self.z_min, self.z_max),
        }

    def __repr__(self) -> str:
        return (
            f"HorizonModel(layers={self.n_layers}, grid={self.grid.nx}x{self.grid.ny}, "
            f"z=[{self.z_min:.1f}, {self.z_max:.1f}], lithologies={list(self.lithologies)})"
        )


# --------------------------------------------------------------------------- #
# Construtores                                                                 #
# --------------------------------------------------------------------------- #

def _stacked_model(grid, top, layers, **kwargs) -> HorizonModel:
    horizons = [grid.broadcast(top)]
    facies = []
    for index, layer in enumerate(layers):
        if "base" not in layer:
            raise ValueError(f"Layer {index} must define its 'base' surface.")
        horizons.append(grid.broadcast(layer["base"]))
        facies.append(layer)
    return HorizonModel(grid, horizons, facies, **kwargs)


def _flat_layer_stack(intervals) -> tuple[list[float], list[str]]:
    """Converte triplas ``(z0, z1, nome)`` ordenadas e sem sobreposição em horizontes + camadas."""
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered[:-1], ordered[1:]):
        if current[0] < previous[1] - _SURFACE_TOL:
            raise ValueError(
                f"Overlapping depth intervals: [{previous[0]}, {previous[1]}] "
                f"({previous[2]}) and [{current[0]}, {current[1]}] ({current[2]})."
            )
    depths: list[float] = [ordered[0][0]]
    names: list[str] = []
    for z0, z1, name in ordered:
        if z0 > depths[-1] + _SURFACE_TOL:  # lacuna entre intervalos
            depths.append(z0)
            names.append("Undefined")
        depths.append(z1)
        names.append(name)
    return depths, names


def _flat_layers_classmethod(cls, **kwargs):
    rop_values = kwargs.pop("rop_values", None)
    mu_values = kwargs.pop("mu_values", None)
    wear_factors = kwargs.pop("wear_factors", None)
    default_rop = kwargs.pop("default_rop", None)
    default_mu = kwargs.pop("default_mu", None)
    x_range = kwargs.pop("x_range", None)
    y_range = kwargs.pop("y_range", None)
    outside = kwargs.pop("outside", "clamp")
    name = kwargs.pop("name", None)

    intervals: list[tuple[float, float, str]] = []
    xs: list[float] = []
    ys: list[float] = []
    for key, entries in kwargs.items():
        if entries is None:
            continue
        lithology = lithology_name(lithology_code(key))
        for entry in entries:
            values = [float(v) for v in entry]
            if len(values) == 2:
                z0, z1 = values
            elif len(values) == 6:
                x0, x1, y0, y1, z0, z1 = values
                xs.extend((x0, x1))
                ys.extend((y0, y1))
            else:
                raise ValueError(
                    f"Interval {entry} of '{key}' must be [z0, z1] or [x0, x1, y0, y1, z0, z1]."
                )
            if z1 <= z0:
                raise ValueError(f"Invalid interval {entry} of '{key}': z1 must exceed z0.")
            intervals.append((z0, z1, lithology))

    if not intervals:
        raise ValueError("No geological interval was provided.")

    if x_range is None:
        x_range = (min(xs), max(xs)) if xs else (-1.0e4, 1.0e4)
    if y_range is None:
        y_range = (min(ys), max(ys)) if ys else (-1.0e4, 1.0e4)

    depths, names = _flat_layer_stack(intervals)
    grid = GeoGrid2D.from_bounds(x_range, y_range, nx=2, ny=2)
    return cls(
        grid,
        [flat_surface(grid, z) for z in depths],
        names,
        rop_values=rop_values,
        mu_values=mu_values,
        wear_factors=wear_factors,
        default_rop=default_rop,
        default_mu=default_mu,
        outside=outside,
        name=name,
    )


def _layer_stack_classmethod(cls, grid, top, layers, **kwargs):
    """Cria o modelo a partir do topo e de uma lista de camadas com ``lithology`` e ``base``."""
    return _stacked_model(grid, top, layers, **kwargs)


_flat_layers_classmethod.__doc__ = (
    "Cria um modelo plano a partir de ``litologia=[[z0, z1], ...]`` "
    "(ou blocos ``[x0, x1, y0, y1, z0, z1]``)."
)

HorizonModel.from_layer_stack = classmethod(_layer_stack_classmethod)
HorizonModel.from_flat_layers = classmethod(_flat_layers_classmethod)


# --------------------------------------------------------------------------- #
# Construtores de superfícies sintéticas                                       #
# --------------------------------------------------------------------------- #

def flat_surface(grid: GeoGrid2D, z: float) -> np.ndarray:
    """Superfície horizontal na profundidade constante ``z``."""
    return np.full(grid.shape, float(z), dtype=float)


def dipping_surface(
    grid: GeoGrid2D,
    z0: float,
    dip_x: float = 0.0,
    dip_y: float = 0.0,
    x_ref: float | None = None,
    y_ref: float | None = None,
) -> np.ndarray:
    """Superfície plana com gradientes ``dip_x``/``dip_y`` em metros de profundidade por metro."""
    X, Y = grid.meshgrid()
    x_ref = grid.x0 if x_ref is None else float(x_ref)
    y_ref = grid.y0 if y_ref is None else float(y_ref)
    return float(z0) + float(dip_x) * (X - x_ref) + float(dip_y) * (Y - y_ref)


def dipping_surface_deg(
    grid: GeoGrid2D,
    z0: float,
    dip_deg: float,
    azimuth_deg: float = 0.0,
    x_ref: float | None = None,
    y_ref: float | None = None,
) -> np.ndarray:
    """Superfície plana dada por um mergulho verdadeiro e o azimute da direção de mergulho."""
    slope = np.tan(np.radians(float(dip_deg)))
    azimuth = np.radians(float(azimuth_deg))
    return dipping_surface(
        grid, z0, slope * np.cos(azimuth), slope * np.sin(azimuth), x_ref, y_ref
    )


def anticline(
    grid: GeoGrid2D,
    surface,
    x0: float,
    y0: float,
    amplitude: float,
    sigma_x: float,
    sigma_y: float | None = None,
) -> np.ndarray:
    """Eleva ``surface`` com um domo gaussiano (``amplitude`` positiva = alto estrutural)."""
    X, Y = grid.meshgrid()
    sigma_y = sigma_x if sigma_y is None else float(sigma_y)
    bulge = np.exp(
        -(((X - float(x0)) ** 2) / (2.0 * float(sigma_x) ** 2))
        - (((Y - float(y0)) ** 2) / (2.0 * sigma_y**2))
    )
    return grid.broadcast(surface) - float(amplitude) * bulge


def syncline(grid: GeoGrid2D, surface, x0, y0, amplitude, sigma_x, sigma_y=None) -> np.ndarray:
    """Baixo estrutural; a contraparte de sinal invertido de :func:`anticline`."""
    return anticline(grid, surface, x0, y0, -float(amplitude), sigma_x, sigma_y)


def _side_of_line(grid: GeoGrid2D, point, strike_deg: float) -> np.ndarray:
    """Distância com sinal até a reta que passa por ``point`` com a direção dada."""
    X, Y = grid.meshgrid()
    strike = np.radians(float(strike_deg))
    # Normal unitária à direção: para direção 90° (traço N-S) o lado positivo
    # é x > point[0], que é a leitura intuitiva.
    nx, ny = np.sin(strike), -np.cos(strike)
    return nx * (X - float(point[0])) + ny * (Y - float(point[1]))


def fault(
    grid: GeoGrid2D,
    surface,
    point,
    strike_deg: float,
    throw: float,
    smooth_m: float = 0.0,
) -> np.ndarray:
    """Desloca ``surface`` através de um plano de falha vertical.

    O traço da falha passa por ``point`` com direção ``strike_deg``; o lado
    para o qual a normal aponta é rebaixado de ``throw`` metros. ``smooth_m``
    suaviza o rejeito nessa distância para a superfície continuar contínua na
    interpolação.
    """
    distance = _side_of_line(grid, point, strike_deg)
    if smooth_m > 0.0:
        step = 0.5 * (1.0 + np.tanh(distance / float(smooth_m)))
    else:
        step = (distance > 0.0).astype(float)
    return grid.broadcast(surface) + float(throw) * step


def pinch_out(
    grid: GeoGrid2D,
    top,
    base,
    axis: str = "x",
    full_at: float | None = None,
    zero_at: float | None = None,
) -> np.ndarray:
    """Afina uma camada até espessura zero e devolve a superfície de base ajustada.

    A espessura é mantida em ``full_at`` e decresce linearmente até zero em
    ``zero_at`` ao longo de ``axis`` ('x' ou 'y'), como as camadas reais se
    acunham.
    """
    X, Y = grid.meshgrid()
    coordinate = X if axis.lower() == "x" else Y
    low, high = (grid.x_range if axis.lower() == "x" else grid.y_range)
    full_at = low if full_at is None else float(full_at)
    zero_at = high if zero_at is None else float(zero_at)
    if np.isclose(full_at, zero_at):
        raise ValueError("'full_at' and 'zero_at' must differ.")
    fraction = (coordinate - zero_at) / (full_at - zero_at)
    fraction = np.clip(fraction, 0.0, 1.0)
    top_surface = grid.broadcast(top)
    return top_surface + (grid.broadcast(base) - top_surface) * fraction


# --------------------------------------------------------------------------- #
# Construtores de mapas de fácies sintéticos                                   #
# --------------------------------------------------------------------------- #

def facies_map(grid: GeoGrid2D, lithology) -> np.ndarray:
    """Mapa de fácies uniforme."""
    return np.full(grid.shape, lithology_code(lithology), dtype=np.int8)


def facies_linear_boundary(
    grid: GeoGrid2D,
    lithology_a,
    lithology_b,
    point,
    strike_deg: float = 90.0,
) -> np.ndarray:
    """Duas fácies separadas por um limite retilíneo em planta.

    ``lithology_b`` ocupa o lado para o qual a normal do limite aponta. É o
    construtor do caso-alvo: uma profundidade, duas litologias em ``(x, y)``
    diferentes.
    """
    distance = _side_of_line(grid, point, strike_deg)
    codes = np.full(grid.shape, lithology_code(lithology_a), dtype=np.int8)
    codes[distance > 0.0] = lithology_code(lithology_b)
    return codes


def facies_channel(
    grid: GeoGrid2D,
    background,
    channel,
    path,
    width: float,
) -> np.ndarray:
    """Corpo de canal sinuoso de largura ``width`` imerso em ``background``."""
    X, Y = grid.meshgrid()
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("'path' must be a sequence of at least two (x, y) points.")
    distance = np.full(grid.shape, np.inf)
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length_sq = float(segment @ segment)
        if length_sq <= 0.0:
            continue
        t = ((X - start[0]) * segment[0] + (Y - start[1]) * segment[1]) / length_sq
        t = np.clip(t, 0.0, 1.0)
        dx = X - (start[0] + t * segment[0])
        dy = Y - (start[1] + t * segment[1])
        distance = np.minimum(distance, np.hypot(dx, dy))
    codes = np.full(grid.shape, lithology_code(background), dtype=np.int8)
    codes[distance <= 0.5 * float(width)] = lithology_code(channel)
    return codes


def facies_lens(grid: GeoGrid2D, background, lens, center, radius_x, radius_y=None) -> np.ndarray:
    """Lente elíptica de ``lens`` dentro de ``background``."""
    X, Y = grid.meshgrid()
    radius_y = radius_x if radius_y is None else float(radius_y)
    inside = ((X - float(center[0])) / float(radius_x)) ** 2 + (
        (Y - float(center[1])) / radius_y
    ) ** 2 <= 1.0
    codes = np.full(grid.shape, lithology_code(background), dtype=np.int8)
    codes[inside] = lithology_code(lens)
    return codes
