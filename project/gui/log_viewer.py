from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog,  QPlainTextEdit
from PyQt5.QtCore import Qt
import os
import cv2

class LogViewer(QWidget):
    def __init__(self, log_dir="./resources/logs/", video_dir="./resources/videos/"):
        super().__init__()
        self.log_dir = log_dir
        self.video_dir = video_dir
        self.initUI()
        

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Event Logs"))
        
        self.setFixedSize(200, 420)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Event", "Play Video"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.onCellClicked)
        layout.addWidget(self.table)

        refresh_btn = QPushButton("🔄 Refresh Logs")
        refresh_btn.clicked.connect(self.loadLogs)
        layout.addWidget(refresh_btn)

        self.setLayout(layout)
        self.loadLogs()

    def loadLogs(self):
        self.table.setRowCount(0)
        if not os.path.exists(self.log_dir):
            return

        for i, filename in enumerate(sorted(os.listdir(self.log_dir))):
            if filename.endswith(".txt"):
                with open(os.path.join(self.log_dir, filename), "r") as f:
                    lines = f.readlines()
                timestamp = filename.replace(".txt", "")
                for line in lines:
                    event = line.strip()
                    self.table.insertRow(self.table.rowCount())
                    self.table.setItem(i, 0, QTableWidgetItem(timestamp))
                    self.table.setItem(i, 1, QTableWidgetItem(event))
                    self.table.setItem(i, 2, QTableWidgetItem("▶️"))

    def onCellClicked(self, row, column):
        if column == 2:
            timestamp = self.table.item(row, 0).text()
            video_path = os.path.join(self.video_dir, f"{timestamp}.mp4")
            if os.path.exists(video_path):
                self.playVideo(video_path)
            else:
                print(  f"Video not found: {video_path}")

    def playVideo(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Cannot open video:", path)
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Log Video Playback", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def append_log_text(self, text):
        self.logViewer.appendPlainText(text)
