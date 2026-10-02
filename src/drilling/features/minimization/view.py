"""Aba Qt da otimização de trajetória Tipo 1.

Os widgets de entrada montam um ``DataSet`` e uma ``GridGeology`` (malha
GRDECL carregada pelo usuário, com cabeça do poço e alvo em XYZ); em seguida
um ``QThread`` chama ``optimize.calculate_minimization``. A plotagem
permanece em ``plot.py``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from drilling.core import Point3D, format_pair, parse_pair
from drilling.features.minimization.data_base import DataSet
from drilling.features.minimization.defaults import (
    DEFAULT_BASE_ROP,
    DEFAULT_TARGET,
    DEFAULT_WELLHEAD,
    DRILLING_TIME_FIELD_SPECS,
    OPERATIONAL_FIELD_SPECS,
    build_default_data,
)
from drilling.features.minimization.grid import OUTSIDE_GRID, GridGeology, read_grdecl
from drilling.features.minimization.optimize import calculate_minimization
from drilling.features.minimization.operational import DEFAULT_MECHANICAL_LIMITS
from drilling.features.minimization.plot import (
    OBJECTIVE_STYLES,
    plot_global_curves,
    plot_time_breakdown,
    plot_trajectories,
    plot_trajectories_3d,
    use_default_matplotlib_style,
)
from drilling.gui.loading_overlay import LoadingOverlay
from drilling.gui.param_form import ParamForm


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
    """Executa ``calculate_minimization`` fora da thread da GUI."""

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


class MinimizationView(QWidget):
    """Painel de entradas com rolagem e abas de resultado (resumo, trajetórias, curvas)."""

    def __init__(self):
        super().__init__()
        use_default_matplotlib_style()
        self.current_payload = None
        self.thread = None
        self.worker = None
        self.default_data, self.default_operational = build_default_data()
        self.grid = None
        self.setObjectName("MinimizationView")
        self.setup_ui()

    def setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self.controls_panel = self.build_controls()
        root.addWidget(self.controls_panel, 0)

        self.results_panel = self.build_results_panel()
        root.addWidget(self.results_panel, 1)

        self.loading_overlay = LoadingOverlay(self)

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
        subtitle = QLabel("Inputs, reservoir grid, operations and mechanical limits")
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
        layout.addWidget(self.geology_group())
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

        self.trajectory_3d_canvas = self.make_canvas(figsize=(9, 7))
        self.results_tabs.addTab(self.build_grid_3d_tab(), "3D Grid")

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

    def build_grid_3d_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        self.zoom_to_grid = QCheckBox("Zoom to the grid cells only (useful when the reservoir is thin)")
        self.zoom_to_grid.stateChanged.connect(self.refresh_trajectory_plot)
        layout.addWidget(self.zoom_to_grid)
        layout.addWidget(self.trajectory_3d_canvas)
        return widget

    def geometry_group(self):
        group = QGroupBox("General drilling geometry")
        form = QFormLayout(group)
        data = self.default_data
        self.wellhead_input = QLineEdit(Point3D(*DEFAULT_WELLHEAD).as_text())
        self.target_input = QLineEdit(Point3D(*DEFAULT_TARGET).as_text())
        self.max_l1 = self.spin(data.max, 1, 100000, 1)
        self.min_l1 = self.spin(data.min_l1, 1, 100000, 1)
        self.min_radius = self.spin(data.min_radius, 1, 100000, 1)
        self.max_radius = self.spin(data.max_radius, 1, 100000, 1)
        self.l1_step = self.spin(data.l1_step, 0.1, 10000, 1)
        self.radius_step = self.spin(data.radius_step, 0.1, 10000, 1)
        self.angle_limit = self.spin(data.angle_limit_deg, 1, 89, 1)
        form.addRow("Wellhead (x, y, z)", self.wellhead_input)
        form.addRow("Target (x, y, z)", self.target_input)
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
        data = self.default_data
        self.ro_fluid = self.spin(data.ro_fluid, 0, 50000, 1)
        self.ro_command = self.spin(data.ro_command, 0, 50000, 1)
        self.ro_drillpipe = self.spin(data.ro_drillpipe, 0, 50000, 1)
        self.ro_heavypipe = self.spin(data.ro_heavypipe, 0, 50000, 1)
        self.diam_command = QLineEdit(format_pair((data.d_ext_command, data.d_int_command)))
        self.diam_drillpipe = QLineEdit(format_pair((data.d_ext_drill, data.d_int_drill)))
        self.diam_heavypipe = QLineEdit(format_pair((data.d_ext_heavy, data.d_int_heavy)))
        self.lp = self.spin(data.lp, 0.1, 100000, 1)
        self.friction = self.spin(data.µ, 0, 10, 0.01, decimals=4)
        self.z_force = self.spin(data.z, 0, 1e9, 100)
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
        self.time_form = ParamForm(DRILLING_TIME_FIELD_SPECS, self.default_data.drilling_time_parameters)
        layout.addLayout(self.time_form.layout)
        return group

    def geology_group(self):
        group = QGroupBox("Reservoir grid (GRDECL) and ROP")
        layout = QVBoxLayout(group)
        load_button = QPushButton("Load GRDECL...")
        load_button.clicked.connect(self.load_grid)
        layout.addWidget(load_button)

        self.grid_label = QLabel()
        self.grid_label.setWordWrap(True)
        layout.addWidget(self.grid_label)

        form = QFormLayout()
        self.base_rop = self.spin(DEFAULT_BASE_ROP, 0.01, 10000, 0.5)
        form.addRow("Base ROP (m/h)", self.base_rop)
        layout.addLayout(form)

        self.lithology_table = QTableWidget(0, 4)
        self.lithology_table.setHorizontalHeaderLabels(["Lithology", "Active cells", "ROP coefficient", "Wear factor"])
        self.setup_table(self.lithology_table)
        self.lithology_table.setMinimumHeight(180)
        layout.addWidget(self.lithology_table)
        layout.addWidget(QLabel("Cell ROP = base ROP × lithology coefficient."))
        self.set_grid(None)
        return group

    def operations_group(self):
        group = QGroupBox("Operational inputs")
        layout = QVBoxLayout(group)
        operational = self.default_operational
        self.operation_form = ParamForm(OPERATIONAL_FIELD_SPECS, operational)
        layout.addLayout(self.operation_form.layout)

        self.casing_table = QTableWidget(0, 4)
        self.casing_table.setHorizontalHeaderLabels(["Depth (m)", "Name", "Fixed time (h)", "Include trip"])
        self.setup_table(self.casing_table)
        self.casing_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.casing_table.setMinimumHeight(120)
        for event in operational["casing_events"]:
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
        return parse_pair(text)

    def load_grid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load reservoir grid", "", "GRDECL (*.grdecl *.GRDECL);;All files (*)")
        if not path:
            return
        try:
            self.set_grid(read_grdecl(path))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid GRDECL", str(exc))

    def set_grid(self, grid):
        """Troca a malha carregada e refaz a tabela de litologias (coeficientes voltam a 1)."""
        self.grid = grid
        if grid is None:
            self.grid_label.setText("No grid loaded. Load a GRDECL file to run the optimization.")
            counts = {}
        else:
            self.grid_label.setText(f"{grid.source}\n{grid.nx} × {grid.ny} × {grid.nz} cells; {grid.extent()}")
            counts = grid.lithology_counts()

        self.lithology_table.setRowCount(0)
        for name in list(counts) + [OUTSIDE_GRID]:
            row = self.lithology_table.rowCount()
            self.lithology_table.insertRow(row)
            self.lithology_table.setItem(row, 0, QTableWidgetItem(name))
            self.lithology_table.setItem(row, 1, QTableWidgetItem(str(counts[name]) if name in counts else "-"))
            self.lithology_table.setCellWidget(row, 2, self.spin(1.0, 0.001, 1000, 0.05, decimals=3))
            self.lithology_table.setCellWidget(row, 3, self.spin(1.0, 0, 10, 0.01, decimals=4))

    def lithology_inputs(self):
        """Coeficientes de ROP e fatores de desgaste por litologia, lidos da tabela."""
        rop_coefficients, wear_factors = {}, {}
        for row in range(self.lithology_table.rowCount()):
            name = self.lithology_table.item(row, 0).text()
            rop_coefficients[name] = float(self.lithology_table.cellWidget(row, 2).value())
            wear_factors[name] = float(self.lithology_table.cellWidget(row, 3).value())
        return rop_coefficients, wear_factors

    def add_casing_row(self, depth=2000.0, name="Casing shoe / cementing", fixed_time=10.0, include_trip=True):
        row = self.casing_table.rowCount()
        self.casing_table.insertRow(row)
        self.casing_table.setItem(row, 0, QTableWidgetItem(str(_round(depth))))
        self.casing_table.setItem(row, 1, QTableWidgetItem(str(name)))
        self.casing_table.setItem(row, 2, QTableWidgetItem(str(_round(fixed_time))))
        trip_box = QCheckBox()
        trip_box.setChecked(bool(include_trip))
        self.casing_table.setCellWidget(row, 3, trip_box)

    def remove_selected_rows(self, table):
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def build_data_from_inputs(self):
        if self.grid is None:
            raise ValueError("Load a GRDECL grid before running the optimization.")
        rop_coefficients, wear_factors = self.lithology_inputs()
        geology = GridGeology(
            self.grid,
            Point3D.from_text(self.wellhead_input.text()).as_array(),
            Point3D.from_text(self.target_input.text()).as_array(),
            base_rop=self.base_rop.value(),
            rop_coefficients={name: value for name, value in rop_coefficients.items() if name != OUTSIDE_GRID},
            outside_rop_coefficient=rop_coefficients[OUTSIDE_GRID],
        )
        time_params = self.time_form.to_dict()
        operational_parameters = self.operation_form.to_dict()

        casing_events = []
        for row in range(self.casing_table.rowCount()):
            trip_widget = self.casing_table.cellWidget(row, 3)
            include_trip = trip_widget.isChecked() if trip_widget is not None else False
            casing_events.append(
                {
                    "depth_m": float(self.casing_table.item(row, 0).text()),
                    "name": self.casing_table.item(row, 1).text(),
                    "fixed_time_h": float(self.casing_table.item(row, 2).text()),
                    "include_trip": include_trip,
                }
            )
        operational_parameters["casing_events"] = casing_events
        operational_parameters["lithology_wear_factors"] = wear_factors

        data = DataSet(
            (0, 0),
            geology.P3,
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

        mechanical_limits = {
            "max_top_axial_force_N": self.optional_float(self.max_top_force.text()),
            "max_torque_Nm": self.optional_float(self.max_torque.text()),
        }
        return data, geology, operational_parameters, mechanical_limits

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

        self.set_running(True)

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

    def set_running(self, running: bool) -> None:
        """Bloqueia entradas e resultados e mostra a animação enquanto a otimização roda."""
        self.run_button.setEnabled(not running)
        self.controls_panel.setEnabled(not running)
        self.results_panel.setEnabled(not running)
        if running:
            self.status_label.setText("Running minimization...")
            self.loading_overlay.start("Running minimization...")
        else:
            self.loading_overlay.stop()

    def on_optimization_finished(self, payload):
        self.current_payload = payload
        self.loading_overlay.message = "Drawing results..."
        self.loading_overlay.repaint()
        self.populate_summary(payload["results"])
        self.populate_details(payload["results"])
        self.refresh_plots()
        self.set_running(False)
        self.status_label.setText("Optimization complete.")

    def on_optimization_failed(self, message):
        self.set_running(False)
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
        plot_trajectories_3d(
            self.trajectory_3d_canvas.figure,
            self.current_payload["data"],
            self.current_payload["mesh"],
            self.current_payload["results"],
            visible_objectives=visible_objectives,
            zoom_to_grid=self.zoom_to_grid.isChecked(),
        )
        self.trajectory_3d_canvas.draw_idle()

    def refresh_global_curves(self):
        if not self.current_payload:
            return
        selected = ["radius", "l1", "best_per_l1"][self.curve_selector.currentIndex()]
        plot_global_curves(self.global_canvas.figure, self.current_payload["series"], selected)
        self.global_canvas.draw_idle()
