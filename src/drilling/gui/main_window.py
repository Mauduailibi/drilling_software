"""Janela principal: uma aba por módulo de produto."""
from PySide6.QtWidgets import QMainWindow, QTabWidget
from drilling.features.well_path.view import WellPathView
from drilling.features.minimization.view import MinimizationView

class MainWindow(QMainWindow):
    """Casca que hospeda Well Path Correction e Minimization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drilling Software - V0.1")
        self.resize(1920, 1080)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_tabs()

    def setup_tabs(self):
        """Anexa um widget Qt por módulo de produto."""
        tab_well = WellPathView()
        self.tabs.addTab(tab_well, "Well Path Correction")

        tab_minimization = MinimizationView()
        self.tabs.addTab(tab_minimization, "Minimization")
