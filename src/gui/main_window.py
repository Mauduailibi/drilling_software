from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drilling Software - V0.1")
        self.resize(1920, 1080)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_tabs()

    def setup_tabs(self):
        tab_well = QWidget()
        layout_well = QVBoxLayout(tab_well)

        layout_well.addWidget(QLabel("Coming soon..."), alignment=Qt.AlignCenter)
        self.tabs.addTab(tab_well, "Well Path Correction")

        tab_torque = QWidget()
        layout_torque = QVBoxLayout(tab_torque)
        layout_torque.addWidget(QLabel("Coming soon..."), alignment=Qt.AlignCenter)
        self.tabs.addTab(tab_torque, "Torque & Stress")