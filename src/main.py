
# import os
#
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["QT_API"] = "pyside6"
# Descomente a linha abaixo se a tela ficar preta (força renderização por CPU)
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.main_window import MainWindow

def setup_dark_theme(app: QApplication):
    app.setStyle("Fusion")

def main():
    app = QApplication(sys.argv)
    setup_dark_theme(app)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()