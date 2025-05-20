from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTabWidget, QSizePolicy, QLabel, QGraphicsDropShadowEffect
)
import os, json
from PyQt5.QtGui import QFont, QGuiApplication, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ## 컨트롤할 변수들 생성
        self.video_widgets = {}
        self.roi_editors = {}
        self.log_viewers = {}
        self.roi_reset_buttons = {}
        
        screen = QGuiApplication.primaryScreen()
        self.size = screen.availableGeometry()
        # self.resize(int(size.width() * 0.8), int(size.height() * 0.8))
        self.setStyleSheet("background-color: #C0C0C0;")
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


        ## 비디오 영역
        self.video_area = QWidget()
        self.video_layout = QHBoxLayout()
        self.video_layout.setContentsMargins(0,0,0,0)
        self.video_area.setLayout(self.video_layout)
        self.video_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.video_area)

        ## UI 영역 설정
        self.ui_area = QWidget()
        ui_layout = QHBoxLayout(self.ui_area)
        self.ui_area.setFixedSize(1920, 80)
        
        # self.ui_area.setStyleSheet("background-color: white;")
        self.ui_area.setContentsMargins(0,0,0,30)
        main_layout.addWidget(self.ui_area)


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
                background-color: 	#DCDCDC	;
                color: #000000;
                border-right: 5px solid #a0a0a0;
                border-bottom: 5px solid #a0a0a0;
                border-radius: 5px;
                font-size: 28px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #696969;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                border-left: 5px solid #696969;
                border-top: 5px solid #696969;
                border-right: 0px solid #a0a0a0;
                border-bottom: 0px solid #a0a0a0;
                background-color: #696969;
                color: #FFFFFF;
            }
            
            
        """
        ## 종료 버튼
        start_btn = QPushButton("시작")
        start_btn.clicked.connect(self.on_start)
        config_btn = QPushButton("설정")
        exit_btn = QPushButton("종료")
        exit_btn.clicked.connect(self.close)
        
        start_btn.setCursor(QCursor(Qt.PointingHandCursor))
        config_btn.setCursor(QCursor(Qt.PointingHandCursor))
        exit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        start_btn.setStyleSheet(self.btn_design)
        config_btn.setStyleSheet(self.btn_design)
        exit_btn.setStyleSheet(self.btn_design)
        
        #호버
        start_btn.setStyleSheet(self.btn_hover)
        config_btn.setStyleSheet(self.btn_hover)
        exit_btn.setStyleSheet(self.btn_hover)
        
        start_btn.setFixedSize(200,50)
        config_btn.setFixedSize(200,50)
        exit_btn.setFixedSize(200,50)
        ui_layout.addWidget(start_btn)
        ui_layout.addWidget(config_btn)
        ui_layout.addWidget(exit_btn)


    def on_start(self):
        """ 
        시작시 cam의 vw, roi, log 생성
        """
        if self.video_widgets:
            return
        ## cam1 설정
        cam1 = QWidget()
        cam1Layout = QVBoxLayout()  # ✅ 먼저 생성
        cam1Layout.setContentsMargins(0, 0, 0, 0)
        cam1Layout.setSpacing(0)

        cam1.setLayout(cam1Layout)  # ✅ 그다음 적용
        # cam1.setStyleSheet("background-color: white;")  # ✅ 문제 없음
        
        vw1 = VideoWidget(cam_num=1, video_path=f"resources/videos/sample1.avi")
        vw1.setFixedSize(int((self.size.width()-10) * 0.5), int((self.size.width()-10) * 9/32))
        self.video_widgets[1] = vw1
        self.create_roi_editor(1, vw1) # self.roi_editors[1]에 저장됨.
        
        polygon = self.load_roi_from_file(1) # roi 정보 불러옴.
        if polygon:
            vw1.set_roi_editor(self.roi_editors[1])
            vw1.set_roi(polygon)
            
        info_area1 = QVBoxLayout()
        cam_info1 = QLabel(f"CAM 1 Logs {polygon}{(int((self.size.width()-10) * 0.5), int((self.size.width()-10) * 9/32))}")
        # cam_info1.setStyleSheet("border: 2px solid black;    /* 두께 2px, 검은색 테두리 */")
        reset_roi_btn1 =  QPushButton("🔄️ Reset ROI ")
        reset_roi_btn1.setStyleSheet(self.btn_design)
        reset_roi_btn1.setStyleSheet(self.btn_hover)
        reset_roi_btn1.clicked.connect(lambda _, cid=1: self.reset_roi(cid))
        info_area1.addWidget(reset_roi_btn1)
        info_area1.addWidget(cam_info1)
        
        lv1 = LogViewer(1)
        
        roi_editor1 = self.roi_editors.get(1)
        if roi_editor1 and vw1:
            roi_editor1.setParent(vw1)
            roi_editor1.setGeometry(vw1.rect())
            
            roi_editor1.show()
            roi_editor1.raise_()
        
        cam1Layout.addWidget(vw1)
        # cam1Layout.addWidget(cre1)
        cam1Layout.addLayout(info_area1)
        cam1Layout.addWidget(lv1)
        self.video_layout.addWidget(cam1)
        
        
        ## cam2 설정
        cam2 = QWidget()

        cam2Layout = QVBoxLayout()  # ✅ 먼저 생성
        cam2Layout.setContentsMargins(0, 0, 0, 0)
        cam2Layout.setSpacing(0)

        cam2.setLayout(cam2Layout)  # ✅ 그다음 적용
        # cam2.setStyleSheet("background-color: white;")  # ✅ 문제 없음
        
        vw2 = VideoWidget(cam_num=2, video_path=f"resources/videos/sample2.avi")
        vw2.setFixedSize(int((self.size.width()-10) * 0.5), int((self.size.width()-10) * 9/32))
        self.video_widgets[2] = vw2
        self.create_roi_editor(2, vw2) # self.roi_editors[2]에 저장됨.
        
        polygon = self.load_roi_from_file(2)
        if polygon:
            vw2.set_roi_editor(self.roi_editors[2])
            vw2.set_roi(polygon)
            
        info_area2 = QVBoxLayout()
        # cam_info2 = QLabel(f"CAM 2 Logs {vw2.roi} {(int((self.size.width()-10) * 0.5), int((self.size.width()-10) * 9/32))} {vw2.size()}")
        cam_info2 = QLabel(f"CAM 2 Logs")
        # cam_info2.setStyleSheet("border: 2px solid black;    /* 두께 2px, 검은색 테두리 */")
        reset_roi_btn2 =  QPushButton("🔄️ Reset ROI")
        reset_roi_btn2.setStyleSheet(self.btn_design)
        reset_roi_btn2.setStyleSheet(self.btn_hover)
        
        reset_roi_btn2.clicked.connect(lambda _, cid=2: self.reset_roi(cid))
        info_area2.addWidget(reset_roi_btn2)
        info_area2.addWidget(cam_info2)
                
        lv2 = LogViewer(2)
       
        roi_editor2 = self.roi_editors.get(2)
        if roi_editor2 and vw2:
            roi_editor2.setParent(vw2)
            roi_editor2.setGeometry(vw2.rect())
            roi_editor2.show()
            roi_editor2.raise_()
            
        cam2Layout.addWidget(vw2)
        # cam2Layout.addWidget(cre2)
        cam2Layout.addLayout(info_area2)
        cam2Layout.addWidget(lv2)
        self.video_layout.addWidget(cam2)

        self.log_viewers[1] = lv1
        self.log_viewers[2] = lv2
        

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

    # def remove_video_and_editor(self, cam_id):
    #     """ 
    #     카메라 꺼질시 비디오와 에디터 객체 제거 메서드
    #     """
    #     vw = self.video_widgets.pop(cam_id, None)
    #     if vw:
    #         vw.setParent(None)

    #     editor = self.roi_editors.pop(cam_id, None)
    #     if editor:
    #         editor.setParent(None)
    #         editor.deleteLater()
   
        
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
            new_width = int((width - 10) * 0.5)
            new_height = int((width - 10) * 9 / 32)

            # if 1 in self.video_widgets:
            #     self.video_widgets[1].setFixedSize(new_width, new_height)
            # if 2 in self.video_widgets:
            #     self.video_widgets[2].setFixedSize(new_width, new_height)
            for cam_id, vw in self.video_widgets.items():
                vw.setFixedSize(new_width, new_height)
                roi_editor = self.roi_editors.get(cam_id)
                if roi_editor:
                    roi_editor.setGeometry(vw.rect())  # 위치 및 크기 재조정