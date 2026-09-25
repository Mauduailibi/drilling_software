"""Aba Qt da correção de trajetória 3D.

A view monta um ``WellPathInput`` a partir dos campos XYZ, chama
``logic.solve_case*`` e entrega o dict a ``plot.plot_case_*``. Não contém
fórmulas de solver.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QLineEdit, QPushButton, QComboBox, QCheckBox,
                               QGroupBox, QMessageBox, QFrame, QTableWidget,
                               QTableWidgetItem, QHeaderView, QApplication)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

from drilling.core import ConstraintCheck, CorrectionResult, WellPathInput
from drilling.features.well_path.defaults import DEFAULT_WELL_PATH_INPUT

from .logic import solve_case1, solve_case2, solve_case3
from .plot import plot_case_1, plot_case_2, plot_case_3


class WellPathView(QWidget):
    """Entradas à esquerda e um canvas PyVista para os Casos 1–3."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        control_panel = QWidget()
        control_panel.setFixedWidth(380)
        control_layout = QVBoxLayout(control_panel)

        input_group = QGroupBox("Input Parameters (X, Y, Z)")
        form_layout = QFormLayout(input_group)

        defaults = DEFAULT_WELL_PATH_INPUT
        self.input_Pin = QLineEdit(defaults.pin.as_text())
        self.input_Pbd = QLineEdit(defaults.pbd.as_text())
        self.input_p1 = QLineEdit(defaults.p1.as_text())
        self.input_pt = QLineEdit(defaults.pt.as_text())
        self.input_v = QLineEdit(defaults.v.as_text())

        form_layout.addRow("Pin:", self.input_Pin)
        form_layout.addRow("Pbd:", self.input_Pbd)
        form_layout.addRow("p1:", self.input_p1)
        form_layout.addRow("pt:", self.input_pt)
        form_layout.addRow("Dir Vector (v):", self.input_v)

        control_layout.addWidget(input_group)

        config_group = QGroupBox("Settings")
        config_layout = QVBoxLayout(config_group)

        self.combo_case = QComboBox()
        self.combo_case.addItems(["Case 1", "Case 2", "Case 3"])
        config_layout.addWidget(self.combo_case)

        self.check_traj = QCheckBox("Show Project Trajectory")
        self.check_traj.setChecked(True)
        config_layout.addWidget(self.check_traj)

        self.check_coord = QCheckBox("Show Coordinates")
        self.check_coord.setChecked(False)
        config_layout.addWidget(self.check_coord)

        control_layout.addWidget(config_group)

        self.btn_calc = QPushButton("Calculate Trajectory")
        self.btn_calc.clicked.connect(self.run_calculation)
        self.btn_calc.setMinimumHeight(40)
        control_layout.addWidget(self.btn_calc)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        self.table_validation = QTableWidget(0, 4)
        self.table_validation.setHorizontalHeaderLabels(["Constraint", "Status", "Value", "Limit"])
        self.table_validation.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_validation.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_validation.setSelectionMode(QTableWidget.NoSelection)
        control_layout.addWidget(self.table_validation)

        plot_container = QFrame()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(15, 15, 15, 15)
        plot_layout.setSpacing(10)

        view_toolbar = QHBoxLayout()

        self.btn_view_iso = QPushButton("Isometric")
        self.btn_view_top = QPushButton("Top (XY)")
        self.btn_view_front = QPushButton("Front (XZ)")
        self.btn_view_side = QPushButton("Side (YZ)")

        self.btn_view_iso.clicked.connect(self.set_view_iso)
        self.btn_view_top.clicked.connect(self.set_view_top)
        self.btn_view_front.clicked.connect(self.set_view_front)
        self.btn_view_side.clicked.connect(self.set_view_side)

        view_toolbar.addWidget(self.btn_view_iso)
        view_toolbar.addWidget(self.btn_view_top)
        view_toolbar.addWidget(self.btn_view_front)
        view_toolbar.addWidget(self.btn_view_side)
        view_toolbar.addStretch()

        plot_layout.addLayout(view_toolbar)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()

        plot_layout.addWidget(self.plotter.interactor)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_container, stretch=1)

    def set_view_iso(self):
        self.plotter.view_isometric()
        self.plotter.update()

    def set_view_top(self):
        self.plotter.view_xy()
        self.plotter.update()

    def set_view_front(self):
        self.plotter.view_xz()
        self.plotter.update()

    def set_view_side(self):
        self.plotter.view_yz()
        self.plotter.update()

    def populate_validation_table(self, status_list):
        checks = [ConstraintCheck.from_item(item) for item in status_list]
        self.table_validation.setRowCount(len(checks))
        for row, check in enumerate(checks):
            item_name = QTableWidgetItem(check.name)
            item_status = QTableWidgetItem("OK" if check.ok else "FAILED")
            item_value = QTableWidgetItem(check.value)
            item_limit = QTableWidgetItem(check.limit)

            if not check.ok:
                item_status.setForeground(Qt.red)
            else:
                item_status.setForeground(Qt.darkGreen)

            item_name.setTextAlignment(Qt.AlignCenter)
            item_status.setTextAlignment(Qt.AlignCenter)
            item_value.setTextAlignment(Qt.AlignCenter)
            item_limit.setTextAlignment(Qt.AlignCenter)

            self.table_validation.setItem(row, 0, item_name)
            self.table_validation.setItem(row, 1, item_status)
            self.table_validation.setItem(row, 2, item_value)
            self.table_validation.setItem(row, 3, item_limit)

    def run_calculation(self):
        """Interpreta as entradas, executa o caso selecionado e atualiza a cena 3D."""
        try:
            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.add_text("Processing trajectory...", name="loading_msg", font_size=18, color="black")
            self.plotter.update()
            QApplication.processEvents()

            inputs = WellPathInput.from_text(
                self.input_Pin.text(),
                self.input_Pbd.text(),
                self.input_p1.text(),
                self.input_pt.text(),
                self.input_v.text(),
            )
            kwargs = inputs.solver_kwargs()

            show_traj = self.check_traj.isChecked()
            show_coords = self.check_coord.isChecked()
            selected_case = self.combo_case.currentIndex() + 1

            if selected_case == 1:
                raw = solve_case1(**kwargs)
            elif selected_case == 2:
                raw = solve_case2(**kwargs)
            else:
                raw = solve_case3(**kwargs)

            result = CorrectionResult.from_solver_dict(raw)
            self.populate_validation_table(result.status)

            self.plotter.remove_actor("loading_msg")

            if result.is_valid:
                payload = result.as_dict()
                if selected_case == 1:
                    plot_case_1(self.plotter, payload, show_traj, show_coords)
                elif selected_case == 2:
                    plot_case_2(self.plotter, payload, show_traj, show_coords)
                else:
                    plot_case_3(self.plotter, payload, show_traj, show_coords)
                self.plotter.update()
            else:
                self.plotter.clear()
                self.plotter.add_axes()
                self.plotter.add_text("Trajectory Invalid.\nPlease check parameters.", name="error_msg", font_size=18,
                                      color="red")
                self.plotter.update()
                QMessageBox.warning(self, "Validation Error",
                                    "The trajectory failed one or more constraints. Cannot plot the 3D model.")

        except Exception as e:
            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.update()
            QMessageBox.critical(self, "Calculation Error", str(e))