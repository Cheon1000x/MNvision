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
    def __init__(self, cam_num, dir="./resources/logs/"):
        super().__init__()
        self.cam_num = cam_num
        self.dir = dir
        self.initUI()
        
        # self.setStyleSheet("background-color: green;")
        self.setContentsMargins(0,0,0,0)
        

    def initUI(self):
        lv_main = QWidget()
        layout = QHBoxLayout()
        
        layout.setSpacing(0)  # 위젯 사이 간격 제거
        layout.setContentsMargins(0,0,0,0)
        
        ## logvier 테이블 생성
        self.table = QTableWidget()
        self.table.setFixedWidth(750)

        self.table.setStyleSheet("margin:0; padding:0;")
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Cam", "Event", "Play"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.onCellClicked)
        layout.addWidget(self.table)
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                border-radius: 6px;
                background-color: #ffffff;
                gridline-color: #c6c9cc;
                font-size: 16px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                color: #000000;
            }

            QHeaderView::section {
                background-color: #42444e;
                color: white;
                padding: 4px 10px;
                border: none;
                font-weight: bold;
                font-size: 16px;
            }

            QTableWidget::item {
                padding: 4px 10px;
                border-right: 1px solid #c6c9cc;
                border-bottom: 1px solid #c6c9cc;
            }

            QTableWidget::item:first-child {
                border-left: 1px solid #c6c9cc;
            }

            QTableWidget::item:selected {
                background-color: #00b0ff;  /* 선택 시 파란색 강조 */
                color: white;
            }

            QTableWidget::item:alternate {
                background-color: #eaeaed;
            }

            QTableCornerButton::section {
                background-color: #42444e;
                border-top-left-radius: 6px;
            }
        """)

        # 격자선 제거 대신 border로 처리
        self.table.setShowGrid(False)

        # 행 색상 교차 설정
        self.table.setAlternatingRowColors(True)

        # 수평 헤더 고정 및 수직 헤더 숨김
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        btn_design = """
            background-color: 	#DCDCDC	;
            color: #000000;
            border: none;
            border-radius: 10px;
            font-size: 28px;
            font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
            font-weight: bold;
            
        """
        btn_hover = """
            QPushButton {
                background-color: 	#DCDCDC	;
                color: #000000;
                border: none;
                border-radius: 5px;
                font-size: 28px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #696969;
                color: #FFFFFF;
            }
        """
        
        ## 로그/비디오 버튼 레이아웃        
        btnlayout = QWidget()
        main_layout = QVBoxLayout()

        btnlayout.setLayout(main_layout)
        main_layout.setContentsMargins(0,0,0,0)
        btnlayout.setFixedSize(200,330)
        
        
        ## 갱신refresh 버튼
        refresh_btn = QPushButton("🔄 Refresh\nLogs")
        refresh_btn.clicked.connect(self.loadLogs)
        refresh_btn.setFixedSize(200, 105)
        main_layout.addWidget(refresh_btn)

        # 로그 버튼
        log_btn = QPushButton("⏏️ Open\nLogs")
        log_btn.clicked.connect(lambda: self.openFolder(r'C:\Users\kdt\OneDrive\바탕 화면\PROJECT_MNV\App\resources\logs'))
        log_btn.setFixedSize(200, 105)
        main_layout.addWidget(log_btn)

        # 비디오 버튼
        video_btn = QPushButton("⏏️ Open\nVideos")
        video_btn.clicked.connect(lambda: self.openFolder(r'C:\Users\kdt\OneDrive\바탕 화면\PROJECT_MNV\App\resources\logs'))  # 각 버튼에 맞는 함수로 연결
        video_btn.setFixedSize(200, 105)
        main_layout.addWidget(video_btn)

        refresh_btn.setStyleSheet(btn_design)
        log_btn.setStyleSheet(btn_design)
        video_btn.setStyleSheet(btn_design)
        
        refresh_btn.setStyleSheet(btn_hover)
        log_btn.setStyleSheet(btn_hover)
        video_btn.setStyleSheet(btn_hover)
        
        layout.addWidget(btnlayout)
        self.setLayout(layout)
        self.loadLogs()


    def loadLogs(self):
        """ 
        테이블 영역 내용 로드 메서드
        """
        self.table.setRowCount(0)
        if not os.path.exists(self.dir):
            return

        logList = [x for x in os.listdir(self.dir) if f"_{self.cam_num}_" in x]

        for filename in sorted(logList):
            if filename.endswith(".txt"):
                with open(os.path.join(self.dir, filename), "r") as f:
                    lines = f.readlines()
                try:
                    date, timestamp, cam_num, label = filename.replace(".txt", "").split('_')
                except ValueError:
                    print("파일 이름 포맷 오류:", filename)
                    continue

                for line in lines:
                    texts = line.strip().split('|')
                    if len(texts) < 3:
                        continue
                    cam = texts[1].strip()
                    event = texts[2].strip()

                    row = self.table.rowCount()
                    self.table.insertRow(row)

                    item0 = QTableWidgetItem(f"{date[4:6]}/{date[6:8]} - {timestamp[-6:-4]}:{timestamp[-4:-2]}:{timestamp[-2:]}")
                    item0.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, 0, item0)
                    self.table.setColumnWidth(0, 160)

                    item1 = QTableWidgetItem(cam)
                    item1.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, 1, item1)
                    self.table.setColumnWidth(1, 60)

                    item2 = QTableWidgetItem(event)
                    self.table.setItem(row, 2, item2)
                    self.table.setColumnWidth(2, 460)

                    item3 = QTableWidgetItem("▶️")
                    item3.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, 3, item3)
                    self.table.setColumnWidth(3, 50)


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
            # timestamp = self.table.item(row, 0).text()
            filename = sorted(os.listdir(self.dir))[row].replace('.txt','')
            video_path = os.path.join(self.dir, f"{filename}.mp4")
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
        
    def append_log_text(self, text):
        self.logViewer.appendPlainText(text)
