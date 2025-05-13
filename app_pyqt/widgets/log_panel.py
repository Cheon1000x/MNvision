from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget

class LogPanelWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)