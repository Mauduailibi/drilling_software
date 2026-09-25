"""Modelos de domínio da otimização Tipo 1: dados mecânicos do poço e modelo geológico.

``DataSet`` guarda geometria (``P0``/``P3`` em 3D), densidades, diâmetros,
parâmetros de tempo de broca e o modelo de atrito. O modelo geológico em si
fica em :mod:`drilling.features.minimization.geology`; este módulo o
reexporta e mantém ``mesh`` como ponto de entrada histórico para camadas
planas. Nenhuma das classes implementa as fórmulas de tração ou de tempo;
essas ficam em ``minimal`` e ``auxiliaries``.
"""

import numpy as np

from drilling.features.minimization.geology import (
    GeoGrid2D,
    HorizonModel,
    LITHOLOGY_COLORS,
    LITHOLOGY_NAMES,
    lithology_code,
    lithology_name,
)

__all__ = [
    "DataSet",
    "GeoGrid2D",
    "HorizonModel",
    "LITHOLOGY_COLORS",
    "LITHOLOGY_NAMES",
    "lithology_code",
    "lithology_name",
    "mesh",
]


class _MechanicalDataSet:
    """Contêiner dos dados geométricos e mecânicos de entrada."""
    def __init__(
        self,
        P0: tuple,
        P3: tuple,
        ro_fluid: float,
        ro_command: float,
        ro_drillpipe: float,
        ro_heavypipe: float,
        diameters_command: tuple[float, float],
        diameters_drillpipe: tuple[float, float],
        diameters_heavypipe: tuple[float, float],
        lp: float,
        µ: float,
        z: float,
        max: float,
        radius: tuple[float, float],
    ) -> None:
        P0 = self._normalize_point(P0, "P0")
        P3 = self._normalize_point(P3, "P3")
        if P0 != (0.0, 0.0, 0.0):
            raise ValueError("The current Type-1 geometry implementation requires P0 = (0, 0, 0).")
        departure = float(np.hypot(P3[0], P3[1]))
        if departure <= 0 or P3[2] <= 0:
            raise ValueError("P3 must contain a positive horizontal offset (x, y) and a positive depth z.")
        if lp <= 0:
            raise ValueError("The heavy-pipe length 'lp' must be positive.")
        if µ < 0:
            raise ValueError("The friction coefficient 'µ' cannot be negative.")
        if max <= 0:
            raise ValueError("'max' must be positive.")
        if radius[0] <= 0 or radius[1] <= 0 or radius[0] > radius[1]:
            raise ValueError("The radius interval must satisfy 0 < min_radius <= max_radius.")

        self._validate_diameters("command", diameters_command)
        self._validate_diameters("drillpipe", diameters_drillpipe)
        self._validate_diameters("heavypipe", diameters_heavypipe)

        area_command = (np.pi / 4) * (diameters_command[0] ** 2 - diameters_command[1] ** 2)
        area_drill = (np.pi / 4) * (diameters_drillpipe[0] ** 2 - diameters_drillpipe[1] ** 2)
        area_heavy = (np.pi / 4) * (diameters_heavypipe[0] ** 2 - diameters_heavypipe[1] ** 2)

        self.P0 = P0
        self.P3 = P3
        self.departure = departure
        self.azimuth = float(np.arctan2(P3[1], P3[0]))
        self.g = 9.81
        self.ro_fluid = ro_fluid
        self.ro_command = ro_command
        self.ro_drillpipe = ro_drillpipe
        self.ro_heavypipe = ro_heavypipe
        self.µ = µ
        self.z = z
        self.area_command = area_command
        self.area_drill = area_drill
        self.area_heavy = area_heavy
        self.lambd_command = ro_command * area_command
        self.lambd_drill = ro_drillpipe * area_drill
        self.lambd_heavy = ro_heavypipe * area_heavy
        self.d_ext_drill = diameters_drillpipe[0]
        self.d_int_drill = diameters_drillpipe[1]
        self.d_ext_command = diameters_command[0]
        self.d_int_command = diameters_command[1]
        self.d_ext_heavy = diameters_heavypipe[0]
        self.d_int_heavy = diameters_heavypipe[1]
        self.lp = lp
        self.max = max
        self.min_l1 = 100.0
        self.min_radius = radius[0]
        self.max_radius = radius[1]
        self.angle_limit_deg = 52.0
        self.l1_step = 10.0
        self.radius_step = 50.0
        self.operational_parameters = None

    @staticmethod
    def _normalize_point(point, name: str) -> tuple[float, float, float]:
        try:
            values = tuple(float(value) for value in point)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of coordinates.") from exc
        if len(values) == 2:
            return (values[0], 0.0, values[1])
        if len(values) == 3:
            return values
        raise ValueError(f"{name} must have 2 coordinates (x, depth) or 3 coordinates (x, y, z).")

    @staticmethod
    def _validate_diameters(name: str, diameters: tuple[float, float]) -> None:
        if len(diameters) != 2:
            raise ValueError(f"The diameter tuple for '{name}' must have length 2.")
        d_ext, d_int = diameters
        if d_ext <= 0 or d_int <= 0:
            raise ValueError(f"The diameters for '{name}' must be positive.")
        if d_int >= d_ext:
            raise ValueError(f"The internal diameter for '{name}' must be smaller than the external diameter.")

    def cache_signature(self) -> tuple:
        return (
            self.P0, self.P3, self.ro_fluid, self.ro_command, self.ro_drillpipe, self.ro_heavypipe,
            self.µ, self.z, self.area_command, self.area_drill, self.area_heavy,
            self.d_ext_drill, self.d_int_drill, self.d_ext_command, self.d_int_command,
            self.d_ext_heavy, self.d_int_heavy, self.lp, self.max, self.min_l1,
            self.min_radius, self.max_radius, self.angle_limit_deg, self.l1_step, self.radius_step,
        )


class DataSetTimeMixin:
    def _init_drilling_time_parameters(self, drilling_time_parameters: dict | None = None):
        params = {
            "trajectory_step": 10.0,
            "min_inclination_factor": 0.85,
            "inclination_reduction": 0.15,
            "inclination_exponent": 1.00,
            "reference_dls_deg_per_30m": 3.0,
            "min_dls_factor": 0.50,
            "dls_reduction": 0.50,
            "dls_exponent": 1.00,
            "surface_wob": 1.60e5,
            "optimal_wob": 1.80e5,
            "min_wob_factor": 0.90,
            "wob_factor_exponent": 1.00,
            "drag_inclination_coeff": 0.55,
            "drag_dls_coeff": 0.22,
            "wob_transfer_exponent": 1.00,
            "torque_limit": 1.20e4,
            "min_torque_factor": 0.90,
            "torque_reduction": 0.10,
            "torque_exponent": 1.00,
            "bit_radius": None,
            "mesh_plot_margin_x": 100.0,
            "mesh_plot_alpha": 0.25,
        }
        if drilling_time_parameters is not None:
            params.update(drilling_time_parameters)

        if params["trajectory_step"] <= 0:
            raise ValueError("'trajectory_step' must be positive.")
        if params["reference_dls_deg_per_30m"] <= 0:
            raise ValueError("'reference_dls_deg_per_30m' must be positive.")
        if params["surface_wob"] <= 0 or params["optimal_wob"] <= 0:
            raise ValueError("'surface_wob' and 'optimal_wob' must be positive.")
        if params["torque_limit"] <= 0:
            raise ValueError("'torque_limit' must be positive.")
        if params["bit_radius"] is not None and params["bit_radius"] <= 0:
            raise ValueError("'bit_radius' must be positive when provided.")
        if params["bit_radius"] is None:
            params["bit_radius"] = 0.5 * self.d_ext_command

        self.drilling_time_parameters = params
        self.buoyed_linear_weight_command = (self.lambd_command - self.ro_fluid * self.area_command) * self.g
        self.buoyed_linear_weight_heavy = (self.lambd_heavy - self.ro_fluid * self.area_heavy) * self.g
        self.buoyed_linear_weight_drill = (self.lambd_drill - self.ro_fluid * self.area_drill) * self.g
        self.buoyed_linear_weight_avg = float(np.mean([self.buoyed_linear_weight_command, self.buoyed_linear_weight_heavy, self.buoyed_linear_weight_drill]))


class DataSet(_MechanicalDataSet, DataSetTimeMixin):
    """Dados mecânicos, parâmetros de tempo de broca e modelo de atrito de uma trajetória Tipo 1.

    ``P0`` e ``P3`` aceitam ``(x, profundidade)`` ou ``(x, y, z)``, com ``z``
    positivo para baixo; ``(x, profundidade)`` vira ``(x, 0, profundidade)``.
    O poço fica no plano vertical que passa por ``P0`` com o azimute de ``P3``
    (``departure`` é o afastamento horizontal e ``azimuth`` o azimute em rad).

    ``friction_model`` escolhe como o coeficiente de atrito é aplicado:

    ``"constant"``
        Usa ``Data.µ`` em todo o poço e a solução fechada de torque e arraste.
        É o comportamento histórico.
    ``"lithology"``
        Toma o coeficiente de atrito do modelo geológico em cada elemento e
        integra torque e arraste numericamente, de modo que os objetivos
        mecânicos respondem à rocha que o poço atravessa. Exige uma malha
        construída com ``mu_values``.
    """

    def __init__(
        self,
        *args,
        drilling_time_parameters: dict | None = None,
        operational_parameters: dict | None = None,
        friction_model: str = "constant",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if friction_model not in ("constant", "lithology"):
            raise ValueError("'friction_model' must be either 'constant' or 'lithology'.")
        self._init_drilling_time_parameters(drilling_time_parameters)
        self.operational_parameters = operational_parameters
        self.friction_model = friction_model

    def cache_signature(self) -> tuple:
        return super().cache_signature() + (
            tuple(sorted(self.drilling_time_parameters.items())),
            None if self.operational_parameters is None else tuple(sorted(self.operational_parameters.items())),
            self.friction_model,
        )


def mesh(**kwargs) -> HorizonModel:
    """Cria um modelo geológico a partir de intervalos de profundidade planos.

    Mantido como ponto de entrada histórico: aceita a forma
    ``sandstone=[[z0, z1], ...]`` (e os blocos ``[x0, x1, y0, y1, z0, z1]``),
    além de ``rop_values``, ``mu_values``, ``wear_factors`` e defaults, e
    devolve um :class:`~drilling.features.minimization.geology.HorizonModel`.
    Geologia com variação lateral é construída diretamente com
    ``HorizonModel.from_layer_stack`` e os construtores de superfícies/fácies.
    """
    return HorizonModel.from_flat_layers(**kwargs)
