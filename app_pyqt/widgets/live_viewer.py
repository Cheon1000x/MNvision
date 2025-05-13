from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

class LiveViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("[카메라 스트림 표시 영역]")
        layout.addWidget(self.label)
        self.setLayout(layout)
