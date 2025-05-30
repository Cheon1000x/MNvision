from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTabWidget, QSizePolicy, QLabel, QGraphicsDropShadowEffect
)
import os, json
from PyQt5.QtGui import QFont, QGuiApplication, QCursor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer
import time
from collections import Counter
from utils import design

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ## 컨트롤할 변수들 생성
        self.video_widgets = {}
        self.roi_editors = {}
        self.info_widgets = {}
        self.log_viewers = {}
        self.roi_reset_buttons = {}
        
        screen = QGuiApplication.primaryScreen()
        self.size = screen.availableGeometry()
        # self.resize(int(size.width() * 0.8), int(size.height() * 0.8))
        self.setStyleSheet("background-color: #161616;")
        self.setWindowTitle("Forklift Detection")
        self.setMinimumSize(self.size.width(), self.size.height())
        # self.setMaximumSize(size.width(), size.height())
        self.showMaximized()
        self.setContentsMargins(0,0,0,0)
        
        ## 전체 영역 분할
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0)
        
        ## UI 영역 설정
        self.ui_area = QWidget()
        ui_layout = QHBoxLayout(self.ui_area)
        self.ui_area.setFixedHeight(55)
        self.ui_area.setStyleSheet("background-color: #161616;")
        self.ui_area.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.ui_area, alignment=Qt.AlignRight)

        self.btn_design = """
            background-color: 	#161616	;
            color: #000000;
            border-radius: 10px;
            font-size: 28px;
            font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
            font-weight: bold;
            
        """
        
        self.btn_hover = """
            QPushButton {
                background-color:  #161616	;
                color: #00D2B5;
                border: 3px solid #00D2B5;
                border-radius: 5px;
                font-size: 28px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #123332;
                border: 3px solid #00D2B5;
                color: #00D2B5;
                border-radius:5px;
            }
            QPushButton:pressed {
                background-color: #00D2B5;
                border: 3px solid #00D2B5;
                color: #000000;
                border-radius:5px;
            }
            
            
        """
        
        
        # Start 버튼

        ## 종료 버튼
        start_btn = QPushButton("")
        start_btn.clicked.connect(self.on_start)
        config_btn = QPushButton("")
        exit_btn = QPushButton("")
        exit_btn.clicked.connect(self.close)
        
        start_btn.setCursor(QCursor(Qt.PointingHandCursor))
        config_btn.setCursor(QCursor(Qt.PointingHandCursor))
        exit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        start_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/play_c.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-color: #123332;
                border: 3px solid #00D2B5;
                color: #00D2B5;
                background-image: url(resources/icons/play_c.png);
            }}
            QPushButton:pressed {{
                background-color: #00D2B5;
                border: 3px solid #00D2B5;
                color: #000000;
                background-image: url(resources/icons/play.png);
            }}
        """)

        # Config 버튼
        config_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/cog_gc.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-color: #123332;
                border: 3px solid #00D2B5;
                color: #00D2B5;
                background-image: url(resources/icons/cog_c.png);
            }}
            QPushButton:pressed {{
                background-color: #00D2B5;
                border: 3px solid #00D2B5;
                color: #000000;
                background-image: url(resources/icons/cog_bc.png);
            }}
        """)

        # Exit 버튼
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/Vector_c.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-color: #123332;
                border: 3px solid #00D2B5;
                color: #00D2B5;
                background-image: url(resources/icons/Vector_c.png);
            }}
            QPushButton:pressed {{
                background-color: #00D2B5;
                border: 3px solid #00D2B5;
                color: #000000;
                background-image: url(resources/icons/Vector.png);
            }}
        """)
        
        
        start_btn.setFixedSize(70,50)
        config_btn.setFixedSize(70,50)
        exit_btn.setFixedSize(70,50)
        ui_layout.addWidget(start_btn)
        ui_layout.addWidget(config_btn)
        ui_layout.addWidget(exit_btn)
        
        ## 비디오 영역
        self.video_area = QWidget()
        self.video_layout = QHBoxLayout()
        # self.video_layout.setContentsMargins(60,0,60,60)
        self.video_layout.setSpacing(60)
        self.video_area.setLayout(self.video_layout)
        
        main_layout.addWidget(self.video_area)

        
    def on_start(self):
        """ 
        시작시 cam의 vw, roi, log 생성
        """
        if self.video_widgets:
            return
        self.reset_timers = {}
        self.spotlights = {}
        self.onoff_labels = {}
        self.event_labels = {}  # 캠별 info1, info2 레이블 저장용 딕셔너리
        self.info_labels = {}  # 캠별 info3, info4 레이블 저장용 딕셔너리

        
        for cam_id in range(1, 3):  # CAM 1과 CAM 2
            cam_widget = QWidget()
            cam_layout = QVBoxLayout()
            cam_layout.setContentsMargins(20, 20, 0, 20)
            cam_layout.setSpacing(0)
            
            cam_widget.setStyleSheet(""" 
                background-color: #171D35;
                border-radius: 10px;
                                     """)
            cam_widget.setLayout(cam_layout)

            # CAM 이름 버튼
            cam_name_btn = QPushButton(f"CAM {cam_id}")
            cam_name_btn.setEnabled(False)
            cam_name_btn.setStyleSheet("""
                QPushButton {
                    border-bottom: none;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    background-color: #171D35;
                    color: #E6E6E6;
                    font-size: 28px;
                    font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                    font-weight: bold;
                }
            """)
            cam_name_btn.setFixedWidth(150)
            cam_layout.addWidget(cam_name_btn)

            # 비디오 위젯
            vw = VideoWidget(cam_num=cam_id, video_path=f"resources/videos/sample{cam_id}.avi")
            vw.setFixedSize(int((self.size.width()-200) * 0.5), int((self.size.width()-200) * 9/32))
            vw.setStyleSheet("border: 2px solid #000000;")
            self.video_widgets[cam_id] = vw

            self.create_roi_editor(cam_id, vw)
            polygon = self.load_roi_from_file(cam_id)
            if polygon:
                vw.set_roi_editor(self.roi_editors[cam_id])
                vw.set_roi(polygon)
                print(f'main_vw{cam_id}', polygon)

            # 정보 영역
            info_widget = QWidget()
            info_widget.setStyleSheet("""
                background-color: #1D2848; 
                color: #E6E6E6; 
                border-radius: 20px; 
                font: 20px 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                """)
            info_widget.setFixedSize(int((self.size.width()-200) * 0.5), 100)
            info_layout = QHBoxLayout(info_widget)
            self.info_widgets[cam_id] = info_widget

            #신호등
            
                   
            # app = design.QApplication()
            spotlight = design.CircleWidget()
            spotlight.setCircleColor(Qt.green)
            info_layout.addWidget(spotlight)
            self.spotlights[cam_id] = spotlight
            
            
            # 왼쪽 정보
            info_left = QVBoxLayout()
            info1 = QLabel(f"mute/on")
            info2 = QLabel(f"event_type")
            # info2 = QLabel(f"CAM {cam_id} Resolution: {self.size.width()}, {self.size.height()}")
            for lbl in (info1, info2):
                lbl.setStyleSheet("color: #E6E6E6;")
                info_left.addWidget(lbl)
            self.onoff_labels[cam_id] = info1
            self.event_labels[cam_id] = info2

            # 가운데 정보
            info_center = QVBoxLayout()
            info3 = QLabel("")
            info4 = QLabel("")
            for lbl in (info3, info4):
                lbl.setStyleSheet("color: #E6E6E6;")
                info_center.addWidget(lbl)
            self.info_labels[cam_id] = (info3, info4)
            
            # 버튼
            reset_btn = QPushButton("Reset\nROI")
            reset_btn.setStyleSheet(self.btn_design)
            reset_btn.setStyleSheet(self.btn_hover)
            reset_btn.setFixedWidth(150)
            reset_btn.clicked.connect(lambda _, cid=cam_id: self.reset_roi(cid))

            info_layout.addLayout(info_left)
            info_layout.addLayout(info_center)
            info_layout.addWidget(reset_btn)

            # LogViewer
            lv = LogViewer(cam_id)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setFixedWidth(int((self.size.width()-200) * 0.5))
            # lv.setStyleSheet("""
            #     background-color: white;
            #     border: 5px solid white;
            # """)
            
            # ROI 에디터 설정
            roi_editor = self.roi_editors.get(cam_id)
            if roi_editor and vw:
                roi_editor.setParent(vw)
                roi_editor.setGeometry(vw.rect())
                roi_editor.show()
                roi_editor.raise_()

            # 시그널 연결
            vw.vthread.on_triggered.connect(self.onoff_info)
            vw.vthread.mute_triggered.connect(self.onoff_info)
            vw.vthread.event_triggered.connect(self.event_info)
            vw.vthread.event_triggered.connect(self.lightControl)
            vw.vthread.event_triggered.connect(lambda: self.make_delayed_loader(lv)())
            vw.vthread.info_triggered.connect(self.handle_result)

            # 레이아웃 추가
            cam_layout.addWidget(vw)
            cam_layout.addWidget(info_widget)
            cam_layout.addWidget(lv)
            self.video_layout.addWidget(cam_widget)

     
    def lightControl(self, event_time, cam_num, label, iou):
        redlight = Qt.red
        greenlight = Qt.green

        # 빨간불로 설정
        target = self.spotlights.get(cam_num, None)
        target.setCircleColor(redlight)

        # 기존 타이머가 있다면 멈춤
        if cam_num in self.reset_timers:
            self.reset_timers[cam_num].stop()

        # 새로운 타이머 생성 또는 재사용
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: target.setCircleColor(greenlight))
        timer.timeout.connect(lambda: self.event_labels.get(cam_num, None).setText("no event"))
        timer.start(5000)  # 5초 (5000ms)

        # 타이머 저장
        self.reset_timers[cam_num] = timer
        
    
    def onoff_info(self, type, label, cam_num):
        text1 = f"{type}"

        info1 = self.onoff_labels.get(cam_num, None)
        if info1:
            info1.setText(text1)
            
    def event_info(self, event_time, cam_num, label, iou):
        text2 = f"{label}"

        info2 = self.event_labels.get(cam_num, None)
        if info2:
            info2.setText(text2)
            
    def handle_result(self, data, cam_num):
        class_names = [det['class_name'] for det in data]
        counts = Counter(class_names)
        class_count = sorted([[count, class_name] for class_name, count in counts.items()], reverse=True)

        print(class_count)

        if len(class_count) < 2:
            print("감지된 클래스가 2개 미만입니다.")
            return

        text1 = f"{'forklift' if class_count[0][1].startswith('fork') else 'person':10s}{class_count[0][0]}"
        text2 = f"{'forklift' if class_count[1][1].startswith('fork') else 'person':10s}{class_count[1][0]}"

        info3, info4 = self.info_labels.get(cam_num, (None, None))
        if info3 and info4:
            info3.setText(text1)
            info4.setText(text2)


    
    def make_delayed_loader(self, log_viewer):
        def loader(*args, **kwargs):
            QTimer.singleShot(3000, lambda: log_viewer.loadLogs())
        return loader

    
    def reset_roi(self, cam_id):
        """ 
        roi 리셋 버튼을 눌렀을 시 작동하는 함수
        video_widget의 clear_roi()
        같은 cam_id의 roi_editors를 제거함
        """
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.clear_roi()

        old_editor = self.roi_editors.pop(cam_id, None)
        if old_editor:
            old_editor.hide()
            old_editor.setParent(None)
            ## 참조 제거
            if hasattr(vw, 'roi_editor'):
                vw.roi_editor = None

            old_editor.deleteLater()

        self.create_roi_editor(cam_id, vw)
        if os.listdir('resources/config/'):
            for x in os.listdir('resources/config/'):
                os.remove('resources/config/'+x)
        print(f"카메라 {cam_id} ROI 초기화됨 및 새 ROIEditor 생성됨")

    def create_roi_editor(self, cam_id, vw):
        """ 
        ROI 에디터 객체 생성
        """
        editor = ROIEditor(vw, cam_id)
        editor.setParent(vw)
        editor.setGeometry(vw.rect())
        editor.roi_defined.connect(self.on_roi_defined)
        editor.show()
        editor.raise_()
        self.roi_editors[cam_id] = editor
        
    def adjust_video_size(self, vw, parent_height, num_videos):
        """ 
        영상 출력 사이즈 조절
        최소크기를 현재 창크기로부터 받아서 설정하고
        Policy로 남은 공간을 다 채우도록 함.
        """
        aspect_ratio = 1280 / 720
        if parent_height > 0 and num_videos > 0:
            target_height = parent_height / num_videos
            target_width = int(target_height * aspect_ratio)

            vw.setMinimumSize(target_width, int(target_height))
            vw.setMaximumSize(16777215, 16777215)  # 최대 크기 제한 없음 (Qt.WA_Unlimited)
            vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        else:
            vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


    def on_roi_defined(self, polygon, cam_id):
        """ 
        roi 확정시 이벤트 발생.
        """
        if len(polygon) < 3:
            print(f"카메라 {cam_id}에서 ROI는 최소 3개의 점이 필요합니다.")
            return
        print(f"카메라 {cam_id} ROI 확정:", polygon)
        # 1. VideoWidget에 ROI 설정
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.set_roi(polygon)

        # 2. ROI 설정 저장
        self.save_roi_to_file(polygon, cam_id)

    
    def save_roi_to_file(self, polygon, cam_id, base_dir="resources/config/"):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"roi_cam_{cam_id}.json")
        with open(filepath, "w") as f:
            json.dump({"cam_id": cam_id, "roi": polygon}, f)
    
    
    def load_roi_from_file(self, cam_id, base_dir="resources/config/"):
        filepath = os.path.join(base_dir, f"roi_cam_{cam_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            return data["roi"]
        return None
    
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        print('get resized')

        screen = QGuiApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        width = screen_size.width()

        if self.video_widgets:
            new_width = int((width - 200) * 0.5)
            new_height = int((width - 200) * 9 / 32)

            # if 1 in self.video_widgets:
            #     self.video_widgets[1].setFixedSize(new_width, new_height)
            # if 2 in self.video_widgets:
            #     self.video_widgets[2].setFixedSize(new_width, new_height)
            for cam_id, vw in self.video_widgets.items():
                vw.setFixedSize(new_width, new_height)
                roi_editor = self.roi_editors.get(cam_id)
                info_widget = self.info_widgets.get(cam_id)
                log_viewer = self.log_viewers.get(cam_id)
                if roi_editor:
                    roi_editor.setGeometry(vw.rect())  # 위치 및 크기 재조정
                if vw.vthread:
                    vw.vthread.set_ui_size(new_width, new_height)
                if log_viewer:
                    log_viewer.setFixedWidget(new_width)
                    
    def closeEvent(self, a0):
        if self.video_widgets[1]:
            self.save_roi_to_file(self.video_widgets[1].roi.tolist(), 1)
        if self.video_widgets[2]:
            self.save_roi_to_file(self.video_widgets[2].roi.tolist(), 2)
        return super().closeEvent(a0)