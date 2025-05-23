from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QPushButton, QFileDialog,  QPlainTextEdit, QMessageBox,
    QSizePolicy, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QGuiApplication
import os
import cv2
import subprocess
from utils.design import remove_custom_messagebox

class LogViewer(QWidget):
    """ 
    LogViewer 클래스
    이벤트 발생시 저장된 로그들을 확인하는 테이블과 버튼으로 구성
    """
    def __init__(self, cam_num, dir="./resources/logs"):
        super().__init__()
        self.cam_num = cam_num
        self.dir = dir+f'/{cam_num}/'
        screen = QGuiApplication.primaryScreen()
        self.screen_size = screen.availableGeometry()
        self.setContentsMargins(0,0,0,0)        
        ## UI 생성 선언
        self.initUI()

    def initUI(self):
        lv_main = QWidget()
        layout = QHBoxLayout()
        lv_main.setLayout(layout)
        
        layout.setSpacing(0)  # 위젯 사이 간격 제거
        
        # layout.setContentsMargins(10,10,10,10)
                        
        ## logvier 테이블 생성
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.table.setFixedWidth(int(self.size().width()-200))

        # self.table.setStyleSheet("margin:0; padding:0;")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed) 
        self.total_width = self.table.width()
        
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Cam", "Event", "Play"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.onCellClicked)
        layout.addWidget(self.table)
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                border-radius: 6px;
                background-color: #2b2b2b;
                gridline-color: transparent;
                font-size: 20px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                color: #000000;
            }

            QHeaderView::section {
                background-color: #252836;
                color: #fe8e52;
                padding: 4px 10px;
                border: none;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
                font-size: 20px;
            }

            QTableWidget::item {
                color: #dddddd;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
                padding: 4px 10px;
                border-right: 0px solid #c6c9cc;
                border-bottom: 1px solid #c6c9cc;
            }

            QTableWidget::item:first-child {
                border-left: 0px solid #c6c9cc;
            }

             QTableWidget::item:alternate {
                background-color: #0d0d0b;
            }
            
            QTableWidget::item:selected {
                background-color: #000000;  
                color: #ADFF2F;
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
        self.table.verticalHeader().setVisible(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        # self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.btn_design = """
            background-color: 	#DCDCDC	;
            color: #000000;
            border-right: 5px solid #a0a0a0;
            border-bottom: 5px solid #a0a0a0;
            border-radius: 10px;
            font-size: 28px;
            font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
            font-weight: bold;
            
        """
        
        self.btn_hover = """
            QPushButton {
                background-color: 	#000000	;
                color: #fe8e52;
                border: 1px solid #fe8e52;
                border-radius: 5px;
                font-size: 28px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f3a2f;
                border: 3px solid #fe8e52;
                color: #fe8e52;
                border-radius:5px;
            }
            QPushButton:pressed {
                background-color: #fe8e52;
                border: 3px solid #fe8e52;
                color: #000000;
                border-radius:5px;
            }
            
            
        """
        
        ## 로그/비디오 버튼 레이아웃        
        btnWidget = QWidget()
        main_layout = QVBoxLayout()

        btnWidget.setLayout(main_layout)
        main_layout.setContentsMargins(0,0,0,0)
        # btnWidget.setFixedSize(200,330)
        btnWidget.setFixedWidth(180)
        
        
        ## 갱신refresh 버튼
        refresh_btn = QPushButton("Refresh\nLogs")
        refresh_btn.clicked.connect(self.loadLogs)
        refresh_btn.setFixedWidth(180) # 너비만 고정
        refresh_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        main_layout.addWidget(refresh_btn, 1)

        # 로그 버튼
        log_btn = QPushButton("Open\nLogs")
        log_btn.clicked.connect(lambda: self.openFolder('resources/logs'))  # 각 버튼에 맞는 함수로 연결
        log_btn.setFixedWidth(180) # 너비만 고정
        log_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        main_layout.addWidget(log_btn, 1) # 1은 stretch 비율

        # 삭제 버튼
        remove_btn = QPushButton("remove\nlogs")
        remove_btn.clicked.connect(lambda: self.removeLogs('resources/logs'))
        remove_btn.setFixedWidth(180) # 너비만 고정
        remove_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        main_layout.addWidget(remove_btn, 1) # 1은 stretch 비율

        refresh_btn.setStyleSheet(self.btn_design)
        log_btn.setStyleSheet(self.btn_design)
        remove_btn.setStyleSheet(self.btn_design)
        
        refresh_btn.setStyleSheet(self.btn_hover)
        log_btn.setStyleSheet(self.btn_hover)
        remove_btn.setStyleSheet(self.btn_hover)
        
        layout.addWidget(btnWidget)
        self.setLayout(layout)
        self.loadLogs()


    def loadLogs(self):
        """ 
        테이블 영역 내용 로드 메서드
        """
        self.table.setRowCount(0)
        if not os.path.exists(self.dir):
            return

        column_ratios = [ 0.3, 0.1, 0.5, 0.1, 0 ]      # 퍼센트 비율
        min_widths = [150, 60, 250, 60, 0]         # 최소 너비
        
        logList = [x for x in os.listdir(self.dir)]
        # print(self.dir)
        # print(logList)
        index = 0
        for filename in sorted(logList):
            if filename.endswith(".txt"):
                # print(filename)
                with open(os.path.join(self.dir, filename), "r") as f:
                    lines = f.readlines()
                try:
                    date, timestamp, cam_num, label = filename.replace(".txt", "").split('_')
                except ValueError:
                    print("파일 이름 포맷 오류:", filename)
                    continue
                
                for line in lines:
                    # print(date, timestamp, cam_num, label)
                    
                    # index += 1
                    texts = line.strip().split(',')
                    if len(texts) < 4:
                        continue
                    cam = texts[1].strip()
                    event = texts[2].strip()

                    row = self.table.rowCount()
                    self.table.insertRow(row)

                    # 각 셀에 들어갈 텍스트 포맷 정의
                    cell_values = [
                        # f"{index}",
                        f"{date[4:6]}/{date[6:8]} - {timestamp[-6:-4]}:{timestamp[-4:-2]}:{timestamp[-2:]}",
                        cam,
                        event,
                        "▶️",
                        f"{filename}"
                    ]

                    # 텍스트 정렬 방식 (None은 정렬 생략)
                    text_aligns = [
                        # Qt.AlignCenter,
                        Qt.AlignCenter,
                        Qt.AlignCenter,
                        None,
                        Qt.AlignCenter,
                        Qt.AlignCenter
                    ]

                    for col, (value, align) in enumerate(zip(cell_values, text_aligns)):
                        header_item = QTableWidgetItem(str(row + 1))
                        header_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        self.table.setVerticalHeaderItem(row, header_item)
                        
                        item = QTableWidgetItem(value)
                        if align is not None:
                            item.setTextAlignment(align)
                        self.table.setItem(row, col, item)

                        # 열 너비 지정
                        desired_width = int(self.total_width * column_ratios[col])
                        final_width = max(desired_width, min_widths[col])
                        self.table.setColumnWidth(col, final_width)

    def openFolder(self, folder_path='../resources/logs'):
        """ 
        폴더 열기 메서드
        """
        folder_path = os.path.join(folder_path, str(self.cam_num))  # 여기에 열고 싶은 폴더 경로 입력

        if os.path.exists(folder_path):
            subprocess.Popen(["explorer", folder_path])
            print(f"{folder_path}")
        else:
            print(f"경로가 존재하지 않습니다. {folder_path}")
    
    def removeLogs(self, folder_path='../resources/logs'):
        """ 
        로그 삭제 메서드 
        """
        rfolder_path = os.path.join(folder_path, str(self.cam_num))  # 경로.
        
        reply = remove_custom_messagebox(self)
        
        if reply == QMessageBox.Yes:
            print(f'removeLogs cam{self.cam_num}')
            if os.path.exists(rfolder_path):
                for file in os.listdir(rfolder_path):
                    os.remove(rfolder_path+'/'+file)
                    self.loadLogs()
            else:
                print(f"경로가 존재하지 않습니다. {rfolder_path}")
        else:
            print(f'canceled: removeLogs cam{self.cam_num} ')



    def onCellClicked(self, row, column):
        """ 
        테이블의 셀 클릭시 영상 재생하는 메서드
        """
        if column == 4:
            # timestamp = self.table.item(row, 0).text()
            filename = sorted([x for x in os.listdir(self.dir) if x.endswith('mp4')])[row]
            video_path = os.path.join(self.dir, f"{filename}")
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
