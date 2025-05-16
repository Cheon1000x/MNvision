from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog,  QPlainTextEdit
from PyQt5.QtCore import Qt
import os
import cv2
import subprocess

class LogViewer(QWidget):
    """ 
    LogViewer 클래스
    이벤트 발생시 저장된 로그들을 확인하는 테이블과 버튼으로 구성
    """
    def __init__(self, log_dir="./resources/logs/", video_dir="./resources/videos/"):
        super().__init__()
        self.log_dir = log_dir
        self.video_dir = video_dir
        self.initUI()
        

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Event Logs"))
        layout.setContentsMargins(0,0,0,0)
        
        ## logvier 테이블 생성
        self.table = QTableWidget()
        self.table.setFixedWidth(400)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Cam", "Event", "Play"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.onCellClicked)
        layout.addWidget(self.table)

        ## 갱신refresh 버튼
        refresh_btn = QPushButton("🔄 Refresh Logs")
        refresh_btn.clicked.connect(self.loadLogs)
        refresh_btn.setFixedSize(400, 50)
        layout.addWidget(refresh_btn)
        
        
        ## 로그/비디오 버튼 레이아웃
        btnlayout = QWidget()
        main_layout = QHBoxLayout()
        btnlayout.setLayout(main_layout)
        main_layout.setContentsMargins(0,0,0,0)
        btnlayout.setFixedSize(400,50)

        # 로그 버튼
        log_btn = QPushButton("⏏️ Open Logs")
        log_btn.clicked.connect(lambda: self.openFolder(r'C:\Users\kdt\OneDrive\바탕 화면\PROJECT_MNV\App\resources\logs'))
        log_btn.setFixedSize(198, 50)
        main_layout.addWidget(log_btn)

        # 비디오 버튼
        video_btn = QPushButton("⏏️ Open Videos")
        video_btn.clicked.connect(lambda: self.openFolder(r'C:\Users\kdt\OneDrive\바탕 화면\PROJECT_MNV\App\resources\videos'))  # 각 버튼에 맞는 함수로 연결
        video_btn.setFixedSize(198, 50)
        main_layout.addWidget(video_btn)

        
        
        layout.addWidget(btnlayout)
        self.setLayout(layout)
        self.loadLogs()

    def loadLogs(self):
        """ 
        테이블 영역 내용 로드 메서드
        
        """
        self.table.setRowCount(0)
        if not os.path.exists(self.log_dir):
            return

        for i, filename in enumerate(sorted(os.listdir(self.log_dir))):
            if filename.endswith(".txt"):
                with open(os.path.join(self.log_dir, filename), "r") as f:
                    lines = f.readlines()
                _, date, timestamp = filename.replace(".txt", "").split('_')
                for line in lines:
                    texts = line.split('|')
                    cam = texts[1]
                    event = texts[2]
                    self.table.insertRow(self.table.rowCount())
                    item0 = QTableWidgetItem(f"{date[4:6]}/{date[6:8]} - {timestamp[-6:-4]}:{timestamp[-4:-2]}:{timestamp[-2:]}")
                    item0.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, 0, item0)
                    self.table.setColumnWidth(0, 120)  # 1 번째 열 너비 100으로 설정
                    
                    item1 = QTableWidgetItem(cam)
                    item1.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, 1, item1)
                    self.table.setColumnWidth(1, 40)  # 1 번째 열 너비 100으로 설정
                    
                    item2 = QTableWidgetItem(event)
                    self.table.setItem(i, 2, item2)
                    self.table.setColumnWidth(2, 150)  # 1 번째 열 너비 100으로 설정
                    
                    item3 = QTableWidgetItem("▶️")
                    item3.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, 3, item3)
                    self.table.setColumnWidth(3, 40)  # 1 번째 열 너비 100으로 설정
                    
    def openFolder(self, folder_path):
        """ 
        폴더 열기 메서드
        """
        folder_path = folder_path  # 여기에 열고 싶은 폴더 경로 입력

        if os.path.exists(folder_path):
            subprocess.Popen(f'explorer "{folder_path}"')
        else:
            print("경로가 존재하지 않습니다.")

    def onCellClicked(self, row, column):
        """ 
        테이블의 셀 클릭시 영상 재생하는 메서드
        """
        if column == 3:
            timestamp = self.table.item(row, 0).text()
            filename = sorted(os.listdir(self.log_dir))[row].replace('.txt','')
            video_path = os.path.join(self.video_dir, f"{filename}.mp4")
            if os.path.exists(video_path):
                self.playVideo(video_path)
            else:
                print(  f"Video not found: {video_path}")

    def playVideo(self, path):
        """  
        로그 비디오를 새 창에서 재생
        """
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
        
    # def append_log_text(self, text):
    #     self.logViewer.appendPlainText(text)
