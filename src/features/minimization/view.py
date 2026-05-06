from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QSplitter, QGroupBox, QTabWidget,
    QMessageBox, QProgressBar
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

MIN_DIR = Path(__file__).resolve().parent
if str(MIN_DIR) not in sys.path:
    sys.path.insert(0, str(MIN_DIR))

import Auxiliaries as ax
from Data_base import DataSet, mesh
from Minimal import (
    minimal_tension,
    minimal_torque,
    minimal_drilling_time,
    drilling_time_breakdown,
    _series_best_metric_for_each_l1,
)
from Operational import (
    minimal_total_time,
    operational_time_breakdown,
)


OPERATIONAL_PARAMETERS = {
    "trip_fixed_time_h": 2.0,
    "trip_time_per_meter_h": 0.0025,
    "bit_run_length_limit_m": 900.0,
    "bit_run_time_limit_h": 60.0,
    "routine_stop_every_m": 500.0,
    "routine_stop_time_h": 0.5,
    "min_spacing_between_bit_trips_m": 150.0,
    "fatigue_dls_threshold_deg_per_30m": 3.0,
    "fatigue_dls_multiplier": 0.30,
    "fatigue_torque_ratio_threshold": 0.75,
    "fatigue_torque_multiplier": 0.35,
    "abrupt_transition_threshold": 0.18,
    "abrupt_transition_extra_wear": 0.30,
    "casing_events": [
        {
            "depth_m": 2000.0,
            "name": "Casing shoe / cementing",
            "fixed_time_h": 10.0,
            "include_trip": True,
        }
    ],
}


def build_default_data():
    data = DataSet(
        P0=(0, 0),
        P3=(1000, 3000),
        ro_fluid=1737.5,
        ro_command=8000,
        ro_drillpipe=8000,
        ro_heavypipe=8000,
        diameters_command=(0.2032, 0.1143),
        diameters_drillpipe=(0.127, 0.1086104),
        diameters_heavypipe=(0.1524, 0.1143),
        µ=0.23,
        z=(5000 * 8) * 4.44822,
        lp=36,
        max=2300,
        radius=(100, 600),
        drilling_time_parameters={
            "trajectory_step": 1.0,
            "reference_dls_deg_per_30m": 3.0,
            "surface_wob": 1.60e5,
            "optimal_wob": 1.80e5,
            "torque_limit": 1.20e4,
            "mesh_plot_alpha": 0.45,
        },
    )

    geological_mesh = mesh(
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

    return data, geological_mesh


class OptimizationWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def run(self):
        try:
            data, geological_mesh = build_default_data()

            force_l1, force_r = minimal_tension(data)
            torque_l1, torque_r = minimal_torque(data)
            time_l1, time_r = minimal_drilling_time(data, geological_mesh)
            total_l1, total_r = minimal_total_time(
                data,
                geological_mesh,
                operational_parameters=OPERATIONAL_PARAMETERS,
            )

            objectives = {
                "Minimal axial force": self._build_objective(data, geological_mesh, force_l1, force_r, True),
                "Minimal torque": self._build_objective(data, geological_mesh, torque_l1, torque_r, True),
                "Minimal drilling time": self._build_objective(data, geological_mesh, time_l1, time_r, True),
                "Minimal total time": self._build_objective(data, geological_mesh, total_l1, total_r, True),
            }

            series = _series_best_metric_for_each_l1(data, geological_mesh)

            self.finished.emit({
                "data": data,
                "mesh": geological_mesh,
                "objectives": objectives,
                "series": series,
            })
        except Exception as exc:
            self.failed.emit(str(exc))

    def _build_objective(self, data, geological_mesh, l1, r, include_operational):
        config = ax.validate_configuration(data, l1, r)
        up = ax.up_tension(data, l1, r)
        down = ax.down_tension(data, l1, r)
        timing = drilling_time_breakdown(data, geological_mesh, l1, r)
        operational = operational_time_breakdown(
            data,
            geological_mesh,
            l1,
            r,
            operational_parameters=OPERATIONAL_PARAMETERS,
        ) if include_operational else None

        x, y = ax.points_coordinates(data, l1, r)

        return {
            "l1": float(l1),
            "R": float(r),
            "config": config,
            "up": up,
            "down": down,
            "timing": timing,
            "operational": operational,
            "x": x,
            "y": y,
        }


class MinimizationView(QWidget):
    def __init__(self):
        super().__init__()
        self.results = None
        self.thread = None
        self.worker = None

        self.setObjectName("MinimizationView")
        self.setup_ui()
        self.run_optimization()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("Trajectory Optimization")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel("Compare optimized Type-1 well trajectories by axial force, torque, drilling time and total time.")
        subtitle.setStyleSheet("color: #666; font-size: 13px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "All trajectories",
            "Minimal axial force",
            "Minimal torque",
            "Minimal drilling time",
            "Minimal total time",
        ])
        self.objective_combo.currentTextChanged.connect(self.refresh_plots)

        self.run_button = QPushButton("Run optimization")
        self.run_button.clicked.connect(self.run_optimization)

        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(QLabel("View:"))
        header.addWidget(self.objective_combo)
        header.addWidget(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        self.tabs = QTabWidget()

        self.summary_table = QTableWidget()
        self.summary_table.setAlternatingRowColors(True)

        self.canvas_trajectory = FigureCanvas(Figure(figsize=(9, 6)))
        self.canvas_metrics = FigureCanvas(Figure(figsize=(9, 8)))
        self.canvas_breakdown = FigureCanvas(Figure(figsize=(9, 6)))

        self.tabs.addTab(self.summary_table, "Summary")
        self.tabs.addTab(self.canvas_trajectory, "Trajectories")
        self.tabs.addTab(self.canvas_metrics, "Global curves")
        self.tabs.addTab(self.canvas_breakdown, "Time breakdown")

        root.addLayout(header)
        root.addWidget(self.progress)
        root.addWidget(self.tabs)

        self.setStyleSheet("""
            QWidget#MinimizationView {
                background: #f7f8fa;
            }
            QGroupBox, QTableWidget, QTabWidget::pane {
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
            }
            QPushButton {
                padding: 8px 14px;
                border-radius: 8px;
                background: #111827;
                color: white;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #9ca3af;
            }
            QComboBox {
                padding: 7px 12px;
                border-radius: 8px;
                border: 1px solid #ccc;
                background: white;
                min-width: 190px;
            }
        """)

    def run_optimization(self):
        self.run_button.setEnabled(False)
        self.progress.setVisible(True)

        self.thread = QThread()
        self.worker = OptimizationWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.failed.connect(self.on_optimization_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_optimization_finished(self, results):
        self.results = results
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.populate_summary()
        self.refresh_plots()

    def on_optimization_failed(self, message):
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Optimization error", message)

    def populate_summary(self):
        objectives = self.results["objectives"]

        rows = []
        for name, item in objectives.items():
            rows.append({
                "Objective": name,
                "L1 (m)": item["l1"],
                "R (m)": item["R"],
                "Angle (deg)": item["config"]["angle_deg"],
                "Top axial force (N)": item["up"][0],
                "Torque (N.m)": item["down"][3],
                "Drilling time (h)": item["timing"]["total_time_h"],
                "Total time (h)": item["operational"]["total_time_h"],
                "Average ROP (m/h)": item["timing"]["average_rop_mph"],
            })

        df = pd.DataFrame(rows)
        self.summary_table.setRowCount(len(df))
        self.summary_table.setColumnCount(len(df.columns))
        self.summary_table.setHorizontalHeaderLabels(df.columns.tolist())

        for row_idx, row in df.iterrows():
            for col_idx, col in enumerate(df.columns):
                value = row[col]
                if isinstance(value, float):
                    text = f"{value:,.3f}"
                else:
                    text = str(value)
                self.summary_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

        self.summary_table.resizeColumnsToContents()

    def refresh_plots(self):
        if not self.results:
            return

        self.plot_trajectories()
        self.plot_global_curves()
        self.plot_time_breakdown()

    def selected_objectives(self):
        selected = self.objective_combo.currentText()
        objectives = self.results["objectives"]

        if selected == "All trajectories":
            return objectives

        return {selected: objectives[selected]}

    def prepare_mesh_axes(self, ax_plot, data, geological_mesh, x_values, y_values):
        margin_x = float(data.drilling_time_parameters.get("mesh_plot_margin_x", 100.0))
        alpha = float(data.drilling_time_parameters.get("mesh_plot_alpha", 0.25))

        x_min = min(0.0, min(x_values) - 0.05 * max(data.P3[0], 1.0))
        x_max = max(max(x_values), data.P3[0]) + margin_x
        y_max = max([segment["end"] for segment in geological_mesh.segments] + [data.P3[1], max(y_values)])

        used_labels = set()
        for segment in geological_mesh.segments:
            color = ax.LITHOLOGY_COLORS.get(segment["lithology"], "#dddddd")
            label = segment["lithology"] if segment["lithology"] not in used_labels else None
            used_labels.add(segment["lithology"])

            rect = patches.Rectangle(
                (x_min, segment["start"]),
                x_max - x_min,
                segment["end"] - segment["start"],
                facecolor=color,
                edgecolor="white",
                alpha=alpha,
                linewidth=0.8,
                label=label,
                zorder=0,
            )
            ax_plot.add_patch(rect)

        ax_plot.set_xlim(x_min, x_max)
        ax_plot.set_ylim(0, y_max + 120)
        ax_plot.invert_yaxis()
        ax_plot.set_aspect("equal")
        ax_plot.set_xlabel("Horizontal distance (m)")
        ax_plot.set_ylabel("Depth (m)")
        ax_plot.grid(alpha=0.25)

    def plot_trajectories(self):
        fig = self.canvas_trajectory.figure
        fig.clear()
        ax_plot = fig.add_subplot(111)

        data = self.results["data"]
        geological_mesh = self.results["mesh"]
        objectives = self.selected_objectives()

        all_x = []
        all_y = []
        for item in objectives.values():
            all_x.extend(item["x"])
            all_y.extend(item["y"])

        self.prepare_mesh_axes(ax_plot, data, geological_mesh, all_x, all_y)

        for name, item in objectives.items():
            ax_plot.plot(item["x"], item["y"], linewidth=2.4, label=name, zorder=3)

        ax_plot.scatter([data.P0[0], data.P3[0]], [data.P0[1], data.P3[1]], s=45, zorder=4)
        ax_plot.annotate("P0", data.P0, xytext=(8, -12), textcoords="offset points")
        ax_plot.annotate("P3", data.P3, xytext=(8, -12), textcoords="offset points")
        ax_plot.set_title("Optimized trajectories over geological mesh")
        ax_plot.legend(loc="upper left", fontsize=9)

        fig.tight_layout()
        self.canvas_trajectory.draw()

    def plot_global_curves(self):
        fig = self.canvas_metrics.figure
        fig.clear()

        series = self.results["series"]
        axes = fig.subplots(3, 1)

        configs = [
            ("force", "Axial force", "Length L1 (m)"),
            ("torque", "Torque", "Length L1 (m)"),
            ("time", "Drilling time", "Length L1 (m)"),
        ]

        for ax_plot, (key, label, xlabel) in zip(axes, configs):
            x = np.asarray(series[key]["x"], dtype=float)
            y = np.asarray(series[key]["y"], dtype=float)
            ax_plot.plot(x, y, linewidth=2.0)
            best_idx = int(np.argmin(y))
            ax_plot.scatter([x[best_idx]], [y[best_idx]], s=45, zorder=5)
            ax_plot.set_title(series[key]["title"], fontsize=10)
            ax_plot.set_xlabel(xlabel)
            ax_plot.set_ylabel(series[key]["ylabel"])
            ax_plot.grid(alpha=0.3)

        fig.tight_layout()
        self.canvas_metrics.draw()

    def plot_time_breakdown(self):
        fig = self.canvas_breakdown.figure
        fig.clear()

        objectives = self.selected_objectives()
        ax_lith = fig.add_subplot(121)
        ax_total = fig.add_subplot(122)

        names = []
        drilling_times = []
        operational_times = []

        selected_name = next(iter(objectives.keys()))
        selected_item = objectives[selected_name]
        by_lithology = selected_item["timing"]["by_lithology"]

        lithologies = list(by_lithology.keys())
        lith_times = [by_lithology[lith]["time_h"] for lith in lithologies]

        ax_lith.bar(lithologies, lith_times)
        ax_lith.set_title(f"Time by lithology\n{selected_name}", fontsize=10)
        ax_lith.set_ylabel("Time (h)")
        ax_lith.tick_params(axis="x", rotation=35)
        ax_lith.grid(axis="y", alpha=0.3)

        for name, item in self.results["objectives"].items():
            names.append(name.replace("Minimal ", "Min. "))
            drilling_times.append(item["timing"]["total_time_h"])
            operational_times.append(item["operational"]["total_operational_time_h"])

        x = np.arange(len(names))
        ax_total.bar(x, drilling_times, label="Pure drilling")
        ax_total.bar(x, operational_times, bottom=drilling_times, label="Operational")
        ax_total.set_title("Drilling + operational time", fontsize=10)
        ax_total.set_ylabel("Time (h)")
        ax_total.set_xticks(x)
        ax_total.set_xticklabels(names, rotation=35, ha="right")
        ax_total.legend(fontsize=8)
        ax_total.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        self.canvas_breakdown.draw()