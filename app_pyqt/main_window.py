from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
from widgets.live_viewer import LiveViewerWidget
from widgets.log_panel import LogPanelWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forklift Detection System")
        self.resize(1200, 800)

        central_widget = QWidget()
        layout = QHBoxLayout()

        self.live_viewer = LiveViewerWidget()
        self.log_panel = LogPanelWidget()

        layout.addWidget(self.live_viewer, 3)
        layout.addWidget(self.log_panel, 1)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
