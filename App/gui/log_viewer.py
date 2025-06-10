from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QPushButton, QFileDialog,  QPlainTextEdit, QMessageBox,
    QSizePolicy, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QGuiApplication
import os
import cv2
import subprocess
from utils.design import remove_custom_messagebox
# from utils.alert_manager import alert_manager


class LogViewer(QWidget):
    """ 
    LogViewer 클래스
    이벤트 발생시 저장된 로그들을 확인하는 테이블과 버튼으로 구성
    """
    mute_opt_TF = pyqtSignal(bool, int)
    
    def __init__(self, cam_num, dir="./resources/logs"):
        super().__init__()
        ## 캠 번호 구분을 위한 캠 객체 선언
        self.cam_num = cam_num
        self.dir = dir+f'/{cam_num}/'
        screen = QGuiApplication.primaryScreen()
        self.screen_size = screen.availableGeometry()
        self.setContentsMargins(0,0,0,0)        
        
        # ⭐ QPlainTextEdit 인스턴스를 멤버 변수로 생성 ⭐
        self.log_text_edit = QPlainTextEdit(self) 
        self.log_text_edit.setReadOnly(True) # 읽기 전용으로 설정
        self.log_text_edit.setFixedHeight(0) # 필요한 경우 최소 높이 설정
        
        self.mboxSS = """
        QLabel {
        color: white;
        font-size: 35px;
        margin: 10px 30px;
        }
        
        QMessageBox {
            width:300px;
            height:200px;
            color: white;
            font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
            font-weight: bold;
            }
        """
        ## UI 생성 선언
        self.mute_opt = False
        self.initUI()

    def initUI(self):
        lv_main = QWidget()
        layout = QHBoxLayout()
        lv_main.setLayout(layout)
        
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # 위젯 사이 간격 제거
        
        ## logvier 테이블 생성
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed) 
        self.total_width = self.table.width()
        
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time", "Cam", "Event", "Play"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.onCellClicked)
        layout.addWidget(self.table)
        self.table.setStyleSheet("""
            QTableWidget {
                border:none;
                border-radius: 16px;
                background-color: #161616;
                gridline-color: transparent;
                font-size: 20px;
                font-family: 'Koulen-Regular', 'Helvetica Neue', Arial, sans-serif;
                color: #E6E6E6;
            }

            QHeaderView::section {
                background-color: #040402;
                color: #E6E6E6;
                padding: 4px 10px;
                border: none;
                font-family: 'Koulen-Regular', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
                font-size: 23px;
            }

            QTableWidget::item {
                background-color: #161616;
                color: #E6E6E6;
                font-family: 'Koulen-Regular', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
                padding: 4px 10px;
                border-left: 1px dotted #e6e6e6;
                border-right: 1px dotted #e6e6e6;
            }

            QTableWidget::item:first-child {
                border-left: 0px solid #c6c9cc;
            }

             QTableWidget::item:alternate {
                background-color: #040402;
            }
            
            QTableWidget::item:selected {
                background-color: #1D2848;  
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
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        ## 버튼 레이아웃        
        btnWidget = QWidget()
        btn_layout = QVBoxLayout()

        btnWidget.setLayout(btn_layout)
        btn_layout.setContentsMargins(10,0,10,0)
        btn_layout.setSpacing(0)
        btnWidget.setFixedWidth(100)
        
        ## 갱신refresh 버튼
        sound_btn = QPushButton("")
        sound_btn.clicked.connect(self.lv_emit)
        sound_btn.setFixedSize(80, 80)
        sound_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.sound_btn = sound_btn
        btn_layout.addWidget(sound_btn, 1) # 1은 stretch 비율
        
        ## 갱신refresh 버튼
        refresh_btn = QPushButton("")
        refresh_btn.clicked.connect(self.loadLogs)
        refresh_btn.setFixedSize(80, 80)
        refresh_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_layout.addWidget(refresh_btn, 1) # 1은 stretch 비율
        
        # 로그 버튼
        log_btn = QPushButton("")
        log_btn.clicked.connect(lambda: self.openFolder())  # 각 버튼에 맞는 함수로 연결
        log_btn.setFixedSize(80, 80)
        log_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_layout.addWidget(log_btn, 1) # 1은 stretch 비율

        # 삭제 버튼
        remove_btn = QPushButton("")
        remove_btn.clicked.connect(lambda: self.removeLogs())
        remove_btn.setFixedSize(80, 80)
        remove_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_layout.addWidget(remove_btn, 1) # 1은 stretch 비율

        btn_layout.addStretch(1)
        
        if self.mute_opt == True:
            print("self.mute_opt css load", self.mute_opt)
            self.sound_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/mute.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-image: url(resources/icons/mute_c.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }} 
                QPushButton:pressed {{
                    background-image: url(resources/icons/mute_b.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
            """)  
        else: 
            print("self.mute_opt css load", self.mute_opt)
            self.sound_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/sound.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-image: url(resources/icons/sound_c.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }} 
                QPushButton:pressed {{
                    background-image: url(resources/icons/sound_b.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
            """)  

        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-image: url(resources/icons/refresh.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/refresh_c.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }} 
            QPushButton:pressed {{
                background-image: url(resources/icons/refresh_b.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
        """)        
        log_btn.setStyleSheet(f"""
            QPushButton {{
                background-image: url(resources/icons/folder.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/folder_c.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }} 
            QPushButton:pressed {{
                background-image: url(resources/icons/folder_b.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
        """)        
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-image: url(resources/icons/remove.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/remove_c.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }} 
            QPushButton:pressed {{
                background-image: url(resources/icons/remove_b.png);
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
            }}
        """)        
        
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
        min_widths = [180, 60, 300, 60, 0]         # 최소 너비
        
        logList = [x for x in os.listdir(self.dir)]
        # print('self.dir',self.dir)
        # print('logList',logList)
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
                    if len(texts) < 3:
                        continue
                    cam = texts[1].strip()
                    event = '  ' + texts[2].strip()

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
                        Qt.AlignCenter,
                        Qt.AlignCenter,
                        Qt.AlignCenter,
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

    def lv_emit(self):
        self.mute_opt_TF.emit(self.mute_opt, self.cam_num)

    def mute_control(self, mute_status: bool, camera_id: int):
        if mute_status:
            self.mute_opt = False
        else:
            self.mute_opt = True
        self.load_mute_btn()
        print('lv.mute_opt',self.mute_opt, self.cam_num)
        
    
    def load_mute_btn(self) :
        if self.mute_opt == True:
            self.sound_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/mute.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-image: url(resources/icons/mute_c.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }} 
                QPushButton:pressed {{
                    background-image: url(resources/icons/mute_b.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
            """)  
        else: 
            self.sound_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/sound.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-image: url(resources/icons/sound_c.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }} 
                QPushButton:pressed {{
                    background-image: url(resources/icons/sound_b.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    background-color: transparent;
                }}
            """)  
        

    def openFolder(self):
        """ 
        폴더 열기 메서드
        """
        folder_path = os.path.abspath('./resources/logs/')+f'\\{self.cam_num}'
        # print(folder_path)
        if os.path.exists(folder_path):
            subprocess.Popen(["explorer", folder_path])
            print(f"{folder_path}")
        else:
            print(f"경로가 존재하지 않습니다. {folder_path}")
    
    
    
    def removeLogs(self, folder_path='resources/logs'):
            """ 
            로그 삭제 메서드: 잠기지 않은 파일만 삭제하고 잠긴 파일은 건너뜁니다.
            """
            rfolder_path = os.path.join(folder_path, str(self.cam_num)) 
            
            reply = QMessageBox.No # 기본값을 No로 설정 (메시지 박스가 없는 경우 대비)
            # remove_custom_messagebox 함수 호출 (클래스 메서드이거나 전역 함수일 수 있음)
            if hasattr(self, 'remove_custom_messagebox'):
                reply = self.remove_custom_messagebox(self)
            elif 'remove_custom_messagebox' in globals():
                reply = remove_custom_messagebox(self)
            else:
                print("경고: remove_custom_messagebox 함수를 찾을 수 없습니다. 삭제를 중단합니다.")
                return # 사용자 확인 없이 삭제를 진행하지 않음

            if reply == QMessageBox.Yes:
                print(f'removeLogs cam{self.cam_num} 시작')
                if os.path.exists(rfolder_path):
                    # ⭐ 중요: 파일 삭제를 시도하기 전에 가능한 모든 VideoWriter를 해제해야 합니다. ⭐
                    # 이 함수는 removeLogs를 호출하는 곳에서 명시적으로 호출하거나,
                    # removeLogs 시작 시점에 자동으로 호출되도록 구성할 수 있습니다.
                    # self.release_all_video_writers() 

                    files_to_remove = os.listdir(rfolder_path)
                    deleted_count = 0
                    failed_to_delete = []
                    
                    for file_name in files_to_remove:
                        file_path = os.path.join(rfolder_path, file_name)
                        if os.path.isfile(file_path): # 파일인지 확인 (폴더는 삭제하지 않음)
                            try:
                                os.remove(file_path)
                                print(f"파일 삭제 성공: {file_name}")
                                deleted_count += 1
                            except PermissionError:
                                print(f"PermissionError: 파일 사용 중으로 삭제 실패: '{file_name}'")
                                failed_to_delete.append(file_name)
                                # 잠긴 파일은 건너뛰고 다음 파일로 진행
                            except Exception as e:
                                print(f"파일 삭제 중 예상치 못한 오류 발생: '{file_name}' - {e}")
                                failed_to_delete.append(file_name)
                        else:
                            print(f"경고: '{file_name}'은(는) 파일이 아니므로 건너뜁니다.")

                    # 모든 파일 삭제 시도 후 결과 요약
                    if deleted_count > 0:
                        print(f"총 {deleted_count}개의 파일을 삭제했습니다.")
                    if failed_to_delete:
                        print(f"다음 파일들은 삭제에 실패했습니다 (다른 프로그램이 사용 중일 수 있습니다):")
                        for f in failed_to_delete:
                            print(f"- {f}")
                    else:
                        pass
                    
                    # UI에 로그 목록을 새로고침 (클래스 내부 함수 또는 콜백)
                    self.loadLogs() 

            else:
                print(f'삭제 작업 취소: removeLogs cam{self.cam_num}')


    def onCellClicked(self, row, column):
        """ 
        테이블의 셀 클릭시 영상 재생하는 메서드
        """
        if column == 3:
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
        # self.logViewer.appendPlainText(text)
        # ⭐ 멤버 변수인 log_text_edit의 appendPlainText 메서드 사용 ⭐
        self.log_text_edit.appendPlainText(text)
        
        # 스크롤을 항상 최하단으로 내리는 옵션 (선택 사항, 사용자 편의성 증대)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())
        
        print(f"LogViewer: '{text}' 추가됨.")

