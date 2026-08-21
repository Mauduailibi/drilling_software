
import bisect
import numpy as np


class _MechanicalDataSet:
    """Container for the geometric and mechanical input data."""
    def __init__(
        self,
        P0: tuple[float, float],
        P3: tuple[float, float],
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
        if tuple(P0) != (0, 0):
            raise ValueError("The current Type-1 geometry implementation requires P0 = (0, 0).")
        if P3[0] <= 0 or P3[1] <= 0:
            raise ValueError("P3 must contain positive horizontal distance and depth.")
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

        self.P0 = tuple(P0)
        self.P3 = tuple(P3)
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


class lithology:
    """Simple lithology descriptor holding only the ROP value."""
    def __init__(self, rop: float | None = None) -> None:
        self.rop = rop
    def shale(self, rop: int): self.rop = rop
    def siltstone(self, rop: int): self.rop = rop
    def sandstone(self, rop: int): self.rop = rop
    def limestone(self, rop: int): self.rop = rop
    def dolomite(self, rop: int): self.rop = rop
    def evaporite(self, rop: int): self.rop = rop


class _BaseMesh:
    def __init__(
        self,
        sandstone: list | None = None,
        limestone: list | None = None,
        dolomite: list | None = None,
        evaporite: list | None = None,
        shale: list | None = None,
        siltstone: list | None = None,
    ):
        self.intervals = {
            "Shale": self._normalize_intervals([] if shale is None else shale, "Shale"),
            "Siltstone": self._normalize_intervals([] if siltstone is None else siltstone, "Siltstone"),
            "Sandstone": self._normalize_intervals([] if sandstone is None else sandstone, "Sandstone"),
            "Limestone": self._normalize_intervals([] if limestone is None else limestone, "Limestone"),
            "Dolomite": self._normalize_intervals([] if dolomite is None else dolomite, "Dolomite"),
            "Evaporite": self._normalize_intervals([] if evaporite is None else evaporite, "Evaporite"),
        }
        self.segments = []
        for lithology_name, intervals in self.intervals.items():
            for start, end in intervals:
                self.segments.append({"lithology": lithology_name, "start": float(start), "end": float(end), "length": float(end - start)})
        self.segments.sort(key=lambda item: (item["start"], item["end"]))
        self.total_length = sum(segment["length"] for segment in self.segments)

    @staticmethod
    def _normalize_intervals(intervals: list, name: str) -> list[tuple[float, float]]:
        normalized = []
        for interval in intervals:
            if len(interval) != 2:
                raise ValueError(f"Each interval of '{name}' must have two values: [start, end].")
            start, end = float(interval[0]), float(interval[1])
            if end <= start:
                raise ValueError(f"Invalid interval in '{name}': end ({end}) must be greater than start ({start}).")
            normalized.append((start, end))
        return normalized

    def as_dict(self) -> dict:
        return {"intervals": self.intervals, "segments": self.segments, "total_length": self.total_length}


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
    def __init__(self, *args, drilling_time_parameters: dict | None = None, operational_parameters: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_drilling_time_parameters(drilling_time_parameters)
        self.operational_parameters = operational_parameters

    def cache_signature(self) -> tuple:
        return super().cache_signature() + (
            tuple(sorted(self.drilling_time_parameters.items())),
            None if self.operational_parameters is None else tuple(sorted(self.operational_parameters.items())),
        )


class mesh(_BaseMesh):
    def __init__(
        self,
        sandstone: list | None = None,
        limestone: list | None = None,
        dolomite: list | None = None,
        evaporite: list | None = None,
        shale: list | None = None,
        siltstone: list | None = None,
        rop_values: dict | None = None,
        default_rop: float | None = None,
    ):
        super().__init__(
            sandstone=sandstone,
            limestone=limestone,
            dolomite=dolomite,
            evaporite=evaporite,
            shale=shale,
            siltstone=siltstone,
        )
        self.rop_values = {"Shale": None, "Siltstone": None, "Sandstone": None, "Limestone": None, "Dolomite": None, "Evaporite": None}
        if rop_values is not None:
            for key, value in rop_values.items():
                if key not in self.rop_values:
                    raise ValueError(f"Unsupported lithology in 'rop_values': {key}")
                if value is None or value <= 0:
                    raise ValueError(f"ROP for '{key}' must be positive.")
                self.rop_values[key] = float(value)

        if default_rop is not None and default_rop <= 0:
            raise ValueError("'default_rop' must be positive when provided.")
        self.default_rop = None if default_rop is None else float(default_rop)

        self._check_overlaps()
        for segment in self.segments:
            lithology_name = segment["lithology"]
            rop = self.rop_values.get(lithology_name)
            if rop is None and self.default_rop is None:
                raise ValueError(f"Missing ROP for lithology '{lithology_name}'. Provide it in 'rop_values' or use 'default_rop'.")
            segment["rop"] = float(self.default_rop if rop is None else rop)

        self._starts = [segment["start"] for segment in self.segments]

    def _check_overlaps(self) -> None:
        ordered = sorted(self.segments, key=lambda item: (item["start"], item["end"]))
        for previous, current in zip(ordered[:-1], ordered[1:]):
            if current["start"] < previous["end"]:
                raise ValueError("The geological intervals overlap. Please provide a non-overlapping depth partition.")

    def segment_at_depth(self, depth: float) -> dict:
        d = float(depth)
        if not self.segments:
            raise ValueError("The geological mesh is empty.")
        idx = bisect.bisect_right(self._starts, d) - 1
        if 0 <= idx < len(self.segments):
            segment = self.segments[idx]
            if segment["start"] <= d < segment["end"]:
                return segment
        if np.isclose(d, self.segments[-1]["end"]):
            return self.segments[-1]
        if self.default_rop is None:
            raise ValueError(f"Depth {d:.3f} m is outside the geological mesh and no default ROP was provided.")
        return {"lithology": "Undefined", "start": d, "end": d, "length": 0.0, "rop": self.default_rop}

    def rop_at_depth(self, depth: float) -> float:
        return float(self.segment_at_depth(depth)["rop"])

    def cache_signature(self) -> tuple:
        seg_tuple = tuple((seg["lithology"], seg["start"], seg["end"], seg.get("rop")) for seg in self.segments)
        return (seg_tuple, tuple(sorted(self.rop_values.items())), self.default_rop)

    def as_dict(self) -> dict:
        data = super().as_dict()
        data["rop_values"] = self.rop_values
        data["default_rop"] = self.default_rop
        return data
