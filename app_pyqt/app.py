from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
import sys

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

