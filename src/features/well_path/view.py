import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QLineEdit, QPushButton, QComboBox, QCheckBox, QGroupBox, QMessageBox)
from pyvistaqt import QtInteractor

# Importe suas funções lógicas (ajuste os nomes conforme você salvou no seu logic.py e plot.py)
from .logic import solve_case1, solve_case2
from .plot import plot_case_1, plot_case_2

class WellPathView(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # ==================================================
        # PAINEL ESQUERDO: CONTROLES E INPUTS
        # ==================================================
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # Grupo de Inputs de Coordenadas
        input_group = QGroupBox("Parâmetros de Entrada (X, Y, Z)")
        form_layout = QFormLayout(input_group)

        # Campos de texto com valores padrão do seu main.py
        self.input_Pin = QLineEdit("0.0, 0.0, 0.0")
        self.input_Pbd = QLineEdit("0.0, 0.0, -2000.0")
        self.input_p1 = QLineEdit("50.0, 100.0, -1800.0")
        self.input_pt = QLineEdit("1000.0, 0.0, -3000.0")
        self.input_v = QLineEdit("0.2, 0.4, -1.0")

        form_layout.addRow("Pin:", self.input_Pin)
        form_layout.addRow("Pbd:", self.input_Pbd)
        form_layout.addRow("p1:", self.input_p1)
        form_layout.addRow("pt:", self.input_pt)
        form_layout.addRow("Vetor Dir (v):", self.input_v)
        
        control_layout.addWidget(input_group)

        # Grupo de Configurações
        config_group = QGroupBox("Configurações")
        config_layout = QVBoxLayout(config_group)
        
        self.combo_case = QComboBox()
        self.combo_case.addItems(["Case 1", "Case 2"])
        config_layout.addWidget(self.combo_case)

        self.check_traj = QCheckBox("Mostrar Trajetória do Projeto")
        self.check_traj.setChecked(True)
        config_layout.addWidget(self.check_traj)

        self.check_coord = QCheckBox("Mostrar Coordenadas")
        self.check_coord.setChecked(False)
        config_layout.addWidget(self.check_coord)

        control_layout.addWidget(config_group)

        # Botão de Calcular
        self.btn_calc = QPushButton("Calcular Trajetória")
        self.btn_calc.clicked.connect(self.run_calculation)
        self.btn_calc.setMinimumHeight(40)
        control_layout.addWidget(self.btn_calc)
        
        control_layout.addStretch() # Empurra tudo para cima

        # ==================================================
        # PAINEL DIREITO: GRÁFICO 3D (PyVista)
        # ==================================================
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.plotter.interactor) # Adiciona o widget do gráfico na tela

    def parse_vector(self, text):
        """Converte a string 'x, y, z' do QLineEdit para um numpy array"""
        try:
            return np.array([float(val.strip()) for val in text.split(",")])
        except ValueError:
            raise ValueError(f"Formato inválido: {text}. Use 'x, y, z'.")

    def run_calculation(self):
        try:
            # 1. Lendo os inputs da GUI
            Pin = self.parse_vector(self.input_Pin.text())
            Pbd = self.parse_vector(self.input_Pbd.text())
            p1 = self.parse_vector(self.input_p1.text())
            pt = self.parse_vector(self.input_pt.text())
            v = self.parse_vector(self.input_v.text())

            show_traj = self.check_traj.isChecked()
            show_coords = self.check_coord.isChecked()
            selected_case = self.combo_case.currentIndex() + 1 # Retorna 1 ou 2

            # 2. Executando os cálculos do seu módulo
            if selected_case == 1:
                result = solve_case1(Pin=Pin, Pbd=Pbd, p1=p1, pt=pt, v=v)
                # Passamos o self.plotter ao invés de criar um novo
                plot_case_1(self.plotter, result, show_traj, show_coords)
            else:
                result = solve_case2(Pin=Pin, Pbd=Pbd, p1=p1, pt=pt, v=v)
                plot_case_2(self.plotter, result, show_traj, show_coords)

            # Atualiza o widget 3D para exibir as mudanças
            self.plotter.update()

            # Opcional: Aqui você pode chamar uma função para atualizar
            # sua "Validation Window" ou jogar os dados dela em uma tabela na GUI.

        except Exception as e:
            QMessageBox.critical(self, "Erro de Cálculo", str(e))