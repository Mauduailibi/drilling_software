from __future__ import annotations

from copy import deepcopy

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

import features.minimization.Auxiliaries as ax
from features.minimization.Data_base import DataSet, mesh
from features.minimization.Minimal import drilling_time_breakdown
from features.minimization.Operational import (
    DEFAULT_MECHANICAL_LIMITS,
    DEFAULT_OPERATIONAL_PARAMETERS,
    _best_constrained_time_candidate,
    _best_mechanical_candidates,
    _best_total_time_candidate,
    _series_best_metric_for_each_l1,
    _series_varying_l1_for_fixed_radius,
    _series_varying_radius_for_fixed_l1,
    evaluate_mechanical_limits,
    operational_time_breakdown,
)
from features.minimization.plot import (
    OBJECTIVE_STYLES,
    plot_global_curves,
    plot_time_breakdown,
    plot_trajectories,
    use_default_matplotlib_style,
)


LITHOLOGIES = ["Shale", "Siltstone", "Sandstone", "Limestone", "Dolomite", "Evaporite"]


def build_default_data():
    drilling_time_parameters = {
        "trajectory_step": 1.0,
        "reference_dls_deg_per_30m": 3.0,
        "surface_wob": 1.60e5,
        "optimal_wob": 1.80e5,
        "torque_limit": 1.20e4,
        "mesh_plot_alpha": 0.45,
    }
    operational_parameters = {
        "trip_fixed_time_h": 1.0,
        "trip_speed_drillpipe_mph": 500.0,
        "trip_speed_heavypipe_mph": 250.0,
        "trip_speed_command_mph": 150.0,
        "bit_run_length_limit_m": 900.0,
        "bit_run_time_limit_h": None,
        "routine_stop_every_m": 500.0,
        "routine_stop_time_h": 0.5,
        "fatigue_dls_threshold_deg_per_30m": 3.0,
        "fatigue_dls_multiplier": 0.30,
        "fatigue_torque_ratio_threshold": 0.75,
        "fatigue_torque_multiplier": 0.35,
        "bit_trip_on_lithology_change": True,
        "operation_merge_distance_m": 10.0,
        "casing_connection_length_m": 9.0,
        "casing_connection_time_h": 0.10,
        "casing_trip_speed_mph": 300.0,
        "casing_logging_time_h": 5.0,
        "cement_pumping_time_h": 2.5,
        "cement_curing_time_h": 12.0,
        "casing_events": [
            {
                "depth_m": 2000.0,
                "name": "Casing shoe / cementing",
                "fixed_time_h": 10.0,
                "include_trip": True,
            }
        ],
    }
    data = DataSet(
        (0, 0),
        (1000, 3000),
        1737.5,
        8000,
        8000,
        8000,
        (0.2032, 0.1143),
        (0.127, 0.1086104),
        (0.1524, 0.1143),
        36,
        0.23,
        (5000 * 8) * 4.44822,
        2300,
        (100, 600),
        drilling_time_parameters=drilling_time_parameters,
    )
    data.l1_step = 10.0
    data.radius_step = 50.0
    data.min_l1 = 100.0
    return data, operational_parameters


def build_default_mesh():
    return mesh(
        sandstone=[[0, 100], [400, 500], [900, 1600], [2200, 3000]],
        dolomite=[[100, 200], [1600, 2000]],
        evaporite=[[200, 300], [2000, 2200]],
        limestone=[[300, 400], [500, 900]],
        rop_values={
            "Sandstone": 18.0,
            "Limestone": 11.0,
            "Dolomite": 9.5,
            "Evaporite": 24.0,
        },
    )


def _round(value, digits=3):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return round(float(value), digits)


class OptimizationWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, data, geological_mesh, operational_parameters, mechanical_limits):
        super().__init__()
        self.data = data
        self.geological_mesh = geological_mesh
        self.operational_parameters = operational_parameters
        self.mechanical_limits = mechanical_limits

    def run(self):
        try:
            self.finished.emit(calculate_minimization(self.data, self.geological_mesh, self.operational_parameters, self.mechanical_limits))
        except Exception as exc:
            self.failed.emit(str(exc))


def calculate_minimization(data, geological_mesh, operational_parameters, mechanical_limits):
    best_force, best_torque = _best_mechanical_candidates(data)
    best_time = _best_constrained_time_candidate(data, geological_mesh, mechanical_limits=mechanical_limits)
    best_total = _best_total_time_candidate(
        data,
        geological_mesh,
        operational_parameters=operational_parameters,
        mechanical_limits=mechanical_limits,
    )

    selected = {
        "force": best_force,
        "torque": best_torque,
        "time": best_time,
        "total": best_total,
    }

    results = {}
    for key, candidate in selected.items():
        l1 = float(candidate["l1"])
        radius = float(candidate["R"])
        timing = candidate.get("timing") or drilling_time_breakdown(data, geological_mesh, l1, radius)
        operational = candidate.get("operational") or operational_time_breakdown(
            data,
            geological_mesh,
            l1,
            radius,
            drilling_timing=timing,
            operational_parameters=operational_parameters,
        )
        mechanical = evaluate_mechanical_limits(candidate["up_force_1"], candidate["torque"], mechanical_limits)
        results[key] = {
            **candidate,
            "l1": l1,
            "R": radius,
            "timing": timing,
            "operational": operational,
            "mechanical": mechanical,
            "drilling_time_h": float(timing["total_time_h"]),
            "operational_time_h": float(operational["total_operational_time_h"]),
            "total_time_h": float(operational["total_time_h"]),
        }

    series = {
        "radius": _series_varying_radius_for_fixed_l1(
            data,
            geological_mesh,
            l1_force=best_force["l1"],
            l1_torque=best_torque["l1"],
            l1_time=best_time["l1"],
            l1_total=best_total["l1"],
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
        "l1": _series_varying_l1_for_fixed_radius(
            data,
            geological_mesh,
            r_force=best_force["R"],
            r_torque=best_torque["R"],
            r_time=best_time["R"],
            r_total=best_total["R"],
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
        "best_per_l1": _series_best_metric_for_each_l1(
            data,
            geological_mesh,
            operational_parameters=operational_parameters,
            mechanical_limits=mechanical_limits,
        ),
    }
    return {"data": data, "mesh": geological_mesh, "operational_parameters": operational_parameters, "mechanical_limits": mechanical_limits, "results": results, "series": series}


class MinimizationView(QWidget):
    def __init__(self):
        super().__init__()
        use_default_matplotlib_style()
        self.current_payload = None
        self.thread = None
        self.worker = None
        self.setObjectName("MinimizationView")
        self.setup_ui()

    def setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        controls = self.build_controls()
        root.addWidget(controls, 0)

        results_panel = self.build_results_panel()
        root.addWidget(results_panel, 1)

    def build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(520)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        title = QLabel("Minimization")
        title.setObjectName("Title")
        subtitle = QLabel("Inputs, geological mesh, operations and mechanical limits")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.run_button = QPushButton("Run minimization")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.run_optimization)
        layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        layout.addWidget(self.geometry_group())
        layout.addWidget(self.mechanical_group())
        layout.addWidget(self.drilling_time_group())
        layout.addWidget(self.mesh_group())
        layout.addWidget(self.operations_group())
        layout.addWidget(self.limits_group())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def build_results_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.results_tabs = QTabWidget()

        self.summary_table = QTableWidget(0, 13)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Objective",
                "L1 (m)",
                "L2 (m)",
                "L3 (m)",
                "R (m)",
                "Angle (deg)",
                "Top axial force (N)",
                "Torque (N*m)",
                "Drilling time (h)",
                "Operations (h)",
                "Total time (h)",
                "Avg ROP (m/h)",
                "Valid",
            ]
        )
        self.setup_table(self.summary_table)
        self.results_tabs.addTab(self.summary_table, "Summary")

        self.trajectory_canvas = self.make_canvas(figsize=(9, 6))
        self.results_tabs.addTab(self.build_trajectory_tab(), "Trajectories")

        curves_tab = QWidget()
        curves_layout = QVBoxLayout(curves_tab)
        self.curve_selector = QComboBox()
        self.curve_selector.addItems(["Best L1: vary radius", "Best R: vary L1", "Best metric for each L1"])
        self.curve_selector.currentIndexChanged.connect(self.refresh_global_curves)
        curves_layout.addWidget(self.curve_selector)
        self.global_canvas = self.make_canvas(figsize=(10, 7))
        curves_layout.addWidget(self.global_canvas)
        self.results_tabs.addTab(curves_tab, "Global Curves")

        self.breakdown_canvas = self.make_canvas(figsize=(10, 5))
        self.results_tabs.addTab(self.wrap_canvas(self.breakdown_canvas), "Time Breakdown")

        self.details_table = QTableWidget(0, 4)
        self.details_table.setHorizontalHeaderLabels(["Objective", "Category", "Metric", "Value"])
        self.setup_table(self.details_table)
        self.results_tabs.addTab(self.details_table, "Details")

        layout.addWidget(self.results_tabs)
        return container

    def make_canvas(self, figsize):
        figure = Figure(figsize=figsize, dpi=100)
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(520)
        return canvas

    def wrap_canvas(self, canvas):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(canvas)
        return widget

    def setup_table(self, table):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(28)

    def build_trajectory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        controls = QGroupBox("Trajectory display")
        controls_layout = QHBoxLayout(controls)
        self.trajectory_objective_checks = {}
        for key in ["force", "torque", "time", "total"]:
            checkbox = QCheckBox(OBJECTIVE_STYLES[key]["label"])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.refresh_trajectory_plot)
            self.trajectory_objective_checks[key] = checkbox
            controls_layout.addWidget(checkbox)

        self.show_command_sections = QCheckBox("Show command sections")
        self.show_radius_lines = QCheckBox("Show curvature radius")
        self.show_command_sections.stateChanged.connect(self.refresh_trajectory_plot)
        self.show_radius_lines.stateChanged.connect(self.refresh_trajectory_plot)
        controls_layout.addWidget(self.show_command_sections)
        controls_layout.addWidget(self.show_radius_lines)
        controls_layout.addStretch()

        layout.addWidget(controls)
        layout.addWidget(self.trajectory_canvas)
        return widget

    def geometry_group(self):
        group = QGroupBox("General drilling geometry")
        form = QFormLayout(group)
        self.p0_input = QLineEdit("0, 0")
        self.p3_input = QLineEdit("1000, 3000")
        self.max_l1 = self.spin(2300, 1, 100000, 1)
        self.min_l1 = self.spin(100, 1, 100000, 1)
        self.min_radius = self.spin(100, 1, 100000, 1)
        self.max_radius = self.spin(600, 1, 100000, 1)
        self.l1_step = self.spin(10, 0.1, 10000, 1)
        self.radius_step = self.spin(50, 0.1, 10000, 1)
        self.angle_limit = self.spin(52, 1, 89, 1)
        form.addRow("P0 (x, y)", self.p0_input)
        form.addRow("P3 target (x, y)", self.p3_input)
        form.addRow("Max L1 (m)", self.max_l1)
        form.addRow("Min L1 (m)", self.min_l1)
        form.addRow("Min radius (m)", self.min_radius)
        form.addRow("Max radius (m)", self.max_radius)
        form.addRow("L1 step (m)", self.l1_step)
        form.addRow("Radius step (m)", self.radius_step)
        form.addRow("Angle limit (deg)", self.angle_limit)
        return group

    def mechanical_group(self):
        group = QGroupBox("Mechanical inputs")
        form = QFormLayout(group)
        self.ro_fluid = self.spin(1737.5, 0, 50000, 1)
        self.ro_command = self.spin(8000, 0, 50000, 1)
        self.ro_drillpipe = self.spin(8000, 0, 50000, 1)
        self.ro_heavypipe = self.spin(8000, 0, 50000, 1)
        self.diam_command = QLineEdit("0.2032, 0.1143")
        self.diam_drillpipe = QLineEdit("0.127, 0.1086104")
        self.diam_heavypipe = QLineEdit("0.1524, 0.1143")
        self.lp = self.spin(36, 0.1, 100000, 1)
        self.friction = self.spin(0.23, 0, 10, 0.01, decimals=4)
        self.z_force = self.spin((5000 * 8) * 4.44822, 0, 1e9, 100)
        form.addRow("Fluid density", self.ro_fluid)
        form.addRow("Command density", self.ro_command)
        form.addRow("Drillpipe density", self.ro_drillpipe)
        form.addRow("Heavypipe density", self.ro_heavypipe)
        form.addRow("Command diam. ext/int", self.diam_command)
        form.addRow("Drillpipe diam. ext/int", self.diam_drillpipe)
        form.addRow("Heavypipe diam. ext/int", self.diam_heavypipe)
        form.addRow("Heavy-pipe length lp (m)", self.lp)
        form.addRow("Friction coefficient", self.friction)
        form.addRow("Z force parameter", self.z_force)
        return group

    def drilling_time_group(self):
        group = QGroupBox("Drilling-time parameters")
        layout = QVBoxLayout(group)
        self.time_params_table = QTableWidget(0, 2)
        self.time_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.setup_table(self.time_params_table)
        self.time_params_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.time_params_table.setMinimumHeight(300)
        defaults = deepcopy(DataSet((0, 0), (1000, 3000), 1737.5, 8000, 8000, 8000, (0.2032, 0.1143), (0.127, 0.1086104), (0.1524, 0.1143), 36, 0.23, 1, 2300, (100, 600)).drilling_time_parameters)
        defaults.update(build_default_data()[0].drilling_time_parameters)
        self.fill_key_value_table(self.time_params_table, defaults)
        layout.addWidget(self.time_params_table)
        return group

    def mesh_group(self):
        group = QGroupBox("Geological mesh and ROP")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        add_button = QPushButton("Add interval")
        remove_button = QPushButton("Remove selected")
        add_button.clicked.connect(self.add_mesh_row)
        remove_button.clicked.connect(lambda: self.remove_selected_rows(self.mesh_table))
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        layout.addLayout(buttons)

        self.mesh_table = QTableWidget(0, 4)
        self.mesh_table.setHorizontalHeaderLabels(["Lithology", "Start depth (m)", "End depth (m)", "ROP (m/h)"])
        self.setup_table(self.mesh_table)
        self.mesh_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.mesh_table.setMinimumHeight(340)
        default_mesh = build_default_mesh()
        for segment in default_mesh.segments:
            self.add_mesh_row(segment["lithology"], segment["start"], segment["end"], segment["rop"])
        layout.addWidget(self.mesh_table)
        return group

    def operations_group(self):
        group = QGroupBox("Operational inputs")
        layout = QVBoxLayout(group)
        self.operation_params_table = QTableWidget(0, 2)
        self.operation_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.setup_table(self.operation_params_table)
        self.operation_params_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.operation_params_table.setMinimumHeight(300)
        operational_parameters = deepcopy(DEFAULT_OPERATIONAL_PARAMETERS)
        default_data, defaults = build_default_data()
        operational_parameters.update(defaults)
        simple_params = {key: value for key, value in operational_parameters.items() if key not in {"casing_events", "lithology_wear_factors"}}
        self.fill_key_value_table(self.operation_params_table, simple_params)
        layout.addWidget(self.operation_params_table)

        self.casing_table = QTableWidget(0, 4)
        self.casing_table.setHorizontalHeaderLabels(["Depth (m)", "Name", "Fixed time (h)", "Include trip"])
        self.setup_table(self.casing_table)
        self.casing_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.casing_table.setMinimumHeight(120)
        for event in defaults["casing_events"]:
            self.add_casing_row(event["depth_m"], event["name"], event["fixed_time_h"], event["include_trip"])
        layout.addWidget(QLabel("Casing / cementing events"))
        layout.addWidget(self.casing_table)

        casing_buttons = QHBoxLayout()
        add_casing = QPushButton("Add casing")
        remove_casing = QPushButton("Remove casing")
        add_casing.clicked.connect(self.add_casing_row)
        remove_casing.clicked.connect(lambda: self.remove_selected_rows(self.casing_table))
        casing_buttons.addWidget(add_casing)
        casing_buttons.addWidget(remove_casing)
        layout.addLayout(casing_buttons)

        self.wear_table = QTableWidget(0, 2)
        self.wear_table.setHorizontalHeaderLabels(["Lithology", "Wear factor"])
        self.setup_table(self.wear_table)
        self.wear_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.wear_table.setMinimumHeight(150)
        self.fill_key_value_table(self.wear_table, DEFAULT_OPERATIONAL_PARAMETERS["lithology_wear_factors"])
        layout.addWidget(QLabel("Lithology wear factors"))
        layout.addWidget(self.wear_table)
        return group

    def limits_group(self):
        group = QGroupBox("Mechanical limits")
        form = QFormLayout(group)
        self.max_top_force = QLineEdit("")
        self.max_torque = QLineEdit("")
        self.max_top_force.setPlaceholderText("Optional")
        self.max_torque.setPlaceholderText("Optional")
        defaults = DEFAULT_MECHANICAL_LIMITS
        if defaults["max_top_axial_force_N"] is not None:
            self.max_top_force.setText(str(defaults["max_top_axial_force_N"]))
        if defaults["max_torque_Nm"] is not None:
            self.max_torque.setText(str(defaults["max_torque_Nm"]))
        form.addRow("Max top axial force (N)", self.max_top_force)
        form.addRow("Max torque (N*m)", self.max_torque)
        return group

    def spin(self, value, minimum, maximum, step, decimals=3):
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(float(step))
        widget.setValue(float(value))
        return widget

    def parse_pair(self, text):
        parts = [float(value.strip()) for value in text.split(",")]
        if len(parts) != 2:
            raise ValueError("Use two comma-separated values.")
        return tuple(parts)

    def fill_key_value_table(self, table, data):
        table.setRowCount(0)
        for key, value in data.items():
            row = table.rowCount()
            table.insertRow(row)
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, key_item)
            table.setItem(row, 1, QTableWidgetItem("" if value is None else str(value)))

    def table_to_params(self, table):
        params = {}
        for row in range(table.rowCount()):
            key = table.item(row, 0).text().strip()
            value_text = table.item(row, 1).text().strip()
            if value_text == "":
                params[key] = None
            elif value_text.lower() in {"true", "false"}:
                params[key] = value_text.lower() == "true"
            else:
                params[key] = float(value_text)
        return params

    def add_mesh_row(self, lithology="Sandstone", start=0.0, end=100.0, rop=10.0):
        row = self.mesh_table.rowCount()
        self.mesh_table.insertRow(row)
        combo = QComboBox()
        combo.addItems(LITHOLOGIES)
        combo.setCurrentText(str(lithology))
        combo.setMinimumHeight(24)
        self.mesh_table.setCellWidget(row, 0, combo)
        self.mesh_table.setItem(row, 1, QTableWidgetItem(str(_round(start))))
        self.mesh_table.setItem(row, 2, QTableWidgetItem(str(_round(end))))
        self.mesh_table.setItem(row, 3, QTableWidgetItem(str(_round(rop))))

    def add_casing_row(self, depth=2000.0, name="Casing shoe / cementing", fixed_time=10.0, include_trip=True):
        row = self.casing_table.rowCount()
        self.casing_table.insertRow(row)
        self.casing_table.setItem(row, 0, QTableWidgetItem(str(_round(depth))))
        self.casing_table.setItem(row, 1, QTableWidgetItem(str(name)))
        self.casing_table.setItem(row, 2, QTableWidgetItem(str(_round(fixed_time))))
        self.casing_table.setItem(row, 3, QTableWidgetItem(str(bool(include_trip))))

    def remove_selected_rows(self, table):
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def build_data_from_inputs(self):
        p0 = self.parse_pair(self.p0_input.text())
        p3 = self.parse_pair(self.p3_input.text())
        time_params = self.table_to_params(self.time_params_table)
        operational_parameters = self.table_to_params(self.operation_params_table)

        casing_events = []
        for row in range(self.casing_table.rowCount()):
            casing_events.append(
                {
                    "depth_m": float(self.casing_table.item(row, 0).text()),
                    "name": self.casing_table.item(row, 1).text(),
                    "fixed_time_h": float(self.casing_table.item(row, 2).text()),
                    "include_trip": self.casing_table.item(row, 3).text().strip().lower() in {"true", "1", "yes", "sim"},
                }
            )
        operational_parameters["casing_events"] = casing_events
        operational_parameters["lithology_wear_factors"] = self.table_to_params(self.wear_table)

        data = DataSet(
            p0,
            p3,
            self.ro_fluid.value(),
            self.ro_command.value(),
            self.ro_drillpipe.value(),
            self.ro_heavypipe.value(),
            self.parse_pair(self.diam_command.text()),
            self.parse_pair(self.diam_drillpipe.text()),
            self.parse_pair(self.diam_heavypipe.text()),
            self.lp.value(),
            self.friction.value(),
            self.z_force.value(),
            self.max_l1.value(),
            (self.min_radius.value(), self.max_radius.value()),
            drilling_time_parameters=time_params,
        )
        data.l1_step = self.l1_step.value()
        data.radius_step = self.radius_step.value()
        data.angle_limit_deg = self.angle_limit.value()
        data.min_l1 = self.min_l1.value()

        intervals = {name: [] for name in LITHOLOGIES}
        rop_values_by_lithology = {}
        for row in range(self.mesh_table.rowCount()):
            lithology = self.mesh_table.cellWidget(row, 0).currentText()
            start = float(self.mesh_table.item(row, 1).text())
            end = float(self.mesh_table.item(row, 2).text())
            rop = float(self.mesh_table.item(row, 3).text())
            intervals[lithology].append([start, end])
            rop_values_by_lithology.setdefault(lithology, rop)
            if not np.isclose(rop_values_by_lithology[lithology], rop):
                raise ValueError(f"Current mesh model accepts one ROP per lithology. Check {lithology}.")

        rop_values = {name: rop_values_by_lithology[name] for name in LITHOLOGIES if name in rop_values_by_lithology}
        geological_mesh = mesh(
            sandstone=intervals["Sandstone"],
            limestone=intervals["Limestone"],
            dolomite=intervals["Dolomite"],
            evaporite=intervals["Evaporite"],
            shale=intervals["Shale"],
            siltstone=intervals["Siltstone"],
            rop_values=rop_values,
        )

        mechanical_limits = {
            "max_top_axial_force_N": self.optional_float(self.max_top_force.text()),
            "max_torque_Nm": self.optional_float(self.max_torque.text()),
        }
        return data, geological_mesh, operational_parameters, mechanical_limits

    def optional_float(self, text):
        stripped = text.strip()
        if not stripped:
            return None
        return float(stripped)

    def run_optimization(self):
        try:
            data, geological_mesh, operational_parameters, mechanical_limits = self.build_data_from_inputs()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid inputs", str(exc))
            return

        self.run_button.setEnabled(False)
        self.status_label.setText("Running minimization...")
        QApplication.processEvents()

        self.thread = QThread()
        self.worker = OptimizationWorker(data, geological_mesh, operational_parameters, mechanical_limits)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.failed.connect(self.on_optimization_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_optimization_finished(self, payload):
        self.current_payload = payload
        self.run_button.setEnabled(True)
        self.status_label.setText("Optimization complete.")
        self.populate_summary(payload["results"])
        self.populate_details(payload["results"])
        self.refresh_plots()

    def on_optimization_failed(self, message):
        self.run_button.setEnabled(True)
        self.status_label.setText("Optimization failed.")
        QMessageBox.critical(self, "Optimization error", message)

    def populate_summary(self, results):
        self.summary_table.setRowCount(0)
        for key, result in results.items():
            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)
            mechanical = result["mechanical"]
            values = [
                OBJECTIVE_STYLES[key]["label"],
                _round(result["l1"]),
                _round(result["l2"]),
                _round(result["l3"]),
                _round(result["R"]),
                _round(result["angle_deg"]),
                _round(result["up_force_1"]),
                _round(result["torque"]),
                _round(result["drilling_time_h"]),
                _round(result["operational_time_h"]),
                _round(result["total_time_h"]),
                _round(result["timing"]["average_rop_mph"]),
                "Yes" if mechanical["is_valid"] else "No",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row, col, item)
        self.summary_table.resizeColumnsToContents()

    def populate_details(self, results):
        rows = []
        for key, result in results.items():
            label = OBJECTIVE_STYLES[key]["label"]
            timing = result["timing"]
            operational = result["operational"]
            for lithology, values in timing["by_lithology"].items():
                rows.append([label, "Lithology", f"{lithology} length (m)", _round(values["length_m"])])
                rows.append([label, "Lithology", f"{lithology} time (h)", _round(values["time_h"])])
            for section, values in timing["by_section"].items():
                rows.append([label, "Section", f"{section} length (m)", _round(values["length_m"])])
                rows.append([label, "Section", f"{section} time (h)", _round(values["time_h"])])
            for category, values in operational["by_category"].items():
                rows.append([label, "Operations", f"{category} time (h)", _round(values["time_h"])])
                rows.append([label, "Operations", f"{category} count", int(values["count"])])

        self.details_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                self.details_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.details_table.resizeColumnsToContents()

    def refresh_plots(self):
        if not self.current_payload:
            return
        self.refresh_trajectory_plot()
        self.refresh_global_curves()
        plot_time_breakdown(self.breakdown_canvas.figure, self.current_payload["results"])
        self.breakdown_canvas.draw_idle()

    def refresh_trajectory_plot(self):
        if not self.current_payload:
            return
        visible_objectives = [
            key for key, checkbox in self.trajectory_objective_checks.items()
            if checkbox.isChecked()
        ]
        plot_trajectories(
            self.trajectory_canvas.figure,
            self.current_payload["data"],
            self.current_payload["mesh"],
            self.current_payload["results"],
            visible_objectives=visible_objectives,
            show_command_sections=self.show_command_sections.isChecked(),
            show_radius_lines=self.show_radius_lines.isChecked(),
        )
        self.trajectory_canvas.draw_idle()

    def refresh_global_curves(self):
        if not self.current_payload:
            return
        selected = ["radius", "l1", "best_per_l1"][self.curve_selector.currentIndex()]
        plot_global_curves(self.global_canvas.figure, self.current_payload["series"], selected)
        self.global_canvas.draw_idle()
