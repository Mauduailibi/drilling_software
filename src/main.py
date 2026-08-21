import sys
import os

if sys.platform.startswith("win"):
    os.environ["QT_QPA_PLATFORM"] = "windows"
elif sys.platform == "darwin":
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["QT_API"] = "pyside6"
# Descomente a linha abaixo se a tela ficar preta (força renderização por CPU)
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from PySide6.QtWidgets import QApplication

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
