from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTabWidget, QSizePolicy, QLabel
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forklift Detection")
        self.resize(600, 800)
        
        self.active_cameras = []
        self.video_widgets = {}
        self.roi_editors = {}
        self.camera_buttons = {}
        self.roi_reset_buttons = {}

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.video_area = QWidget()
        self.video_layout = QGridLayout()
        self.video_area.setLayout(self.video_layout)
        self.video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_area, stretch=3)

        self.ui_area = QWidget()
        ui_layout = QVBoxLayout(self.ui_area)
        main_layout.addWidget(self.ui_area, stretch=1)

        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedSize(400, 150)
        ui_layout.addWidget(self.tab_widget)
        self.init_tabs()

        ui_layout.addWidget(LogViewer(), stretch=1)
        ui_layout.addWidget(QPushButton("종료"))
        
        
    def get_current_window_size(self):
        size = self.size()
        width = size.width()
        height = size.height()
        print(f"현재 창 너비: {width}, 높이: {height}")
        return width, height
    
    
    def init_tabs(self):
        for tab_idx, cam_range in enumerate([(1, 3), (4, 6)]):
            tab = QWidget()
            # tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            layout = QVBoxLayout(tab)
            
            for cam_id in range(cam_range[0], cam_range[1] + 1):
                hbox = QHBoxLayout()

                toggle_btn = QPushButton(f"{cam_id}번 카메라")
                toggle_btn.setCheckable(True)
                toggle_btn.clicked.connect(self.on_camera_toggle)
                self.camera_buttons[cam_id] = toggle_btn
                hbox.addWidget(toggle_btn)

                reset_btn = QPushButton("ROI 리셋")
                reset_btn.clicked.connect(lambda _, cid=cam_id: self.reset_roi(cid))
                self.roi_reset_buttons[cam_id] = reset_btn
                hbox.addWidget(reset_btn)

                layout.addLayout(hbox)

            self.tab_widget.addTab(tab, f"Cam{tab_idx + 1}")

    def on_camera_toggle(self):
        button = self.sender()
        cam_id = int(button.text().split("번")[0])
        
        if button.isChecked():
            if cam_id not in self.active_cameras:
                self.active_cameras.append(cam_id)
                vw = VideoWidget(f"resources/videos/sample{cam_id}.avi")
                self.video_widgets[cam_id] = vw
                self.create_roi_editor(cam_id, vw)
        else:
            if cam_id in self.active_cameras:
                self.active_cameras.remove(cam_id)
                self.remove_video_and_editor(cam_id)
        self.update_grid_layout()

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
            old_editor.deleteLater()

        self.create_roi_editor(cam_id, vw)
        print(f"카메라 {cam_id} ROI 초기화됨 및 새 ROIEditor 생성됨")

    def create_roi_editor(self, cam_id, vw):
        editor = ROIEditor(vw, cam_id)
        editor.setParent(vw)
        editor.setGeometry(vw.rect())
        editor.roi_defined.connect(self.on_roi_defined)
        editor.show()
        editor.raise_()
        self.roi_editors[cam_id] = editor

    def remove_video_and_editor(self, cam_id):
        vw = self.video_widgets.pop(cam_id, None)
        if vw:
            vw.setParent(None)

        editor = self.roi_editors.pop(cam_id, None)
        if editor:
            editor.setParent(None)
            editor.deleteLater()

    def update_grid_layout(self):
        # 기존 레이아웃 제거
        for i in reversed(range(self.video_layout.count())):
            item = self.video_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                self.video_layout.removeWidget(widget)
                widget.setParent(None)

        # 새로운 레이아웃 구성
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_h_layout = QHBoxLayout()

        cam1_active = [cam_id for cam_id in sorted(self.active_cameras) if 1 <= cam_id <= 3]
        cam2_active = [cam_id for cam_id in sorted(self.active_cameras) if 4 <= cam_id <= 6]

        left_container = QWidget()
        left_container.setLayout(left_layout)
        main_h_layout.addWidget(left_container, stretch=1)

        right_container = QWidget()
        right_container.setLayout(right_layout)
        main_h_layout.addWidget(right_container, stretch=1)

        self.video_layout.addLayout(main_h_layout, 0, 0)

        for cam_id in cam1_active:
            if cam_id in self.video_widgets:
                vw = self.video_widgets[cam_id]
                vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred) # 선호하는 크기 유지하며 필요시 늘어남
                vw_layout = QVBoxLayout()
                vw_layout.addWidget(vw)
                left_layout.addLayout(vw_layout)
                
                self.adjust_video_size(vw, left_container.height(), len(cam1_active)) # 수정된 호출

                roi_editor = self.roi_editors.get(cam_id)
                if roi_editor and vw:
                    roi_editor.setParent(vw)
                    roi_editor.setGeometry(vw.rect())
                    roi_editor.show()
                    roi_editor.raise_()

        for cam_id in cam2_active:
            if cam_id in self.video_widgets:
                vw = self.video_widgets[cam_id]
                vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred) # 선호하는 크기 유지하며 필요시 늘어남
                vw_layout = QVBoxLayout()
                vw_layout.addWidget(vw)
                right_layout.addLayout(vw_layout)
                self.adjust_video_size(vw, right_container.height(), len(cam2_active)) # 수정된 호출

                roi_editor = self.roi_editors.get(cam_id)
                if roi_editor and vw:
                    roi_editor.setParent(vw)
                    roi_editor.setGeometry(vw.rect())
                    roi_editor.show()
                    roi_editor.raise_()
        
        
    from PyQt5.QtWidgets import QSizePolicy

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
        if len(polygon) < 3:
            print(f"카메라 {cam_id}에서 ROI는 최소 3개의 점이 필요합니다.")
            return
        print(f"카메라 {cam_id} ROI 확정:", polygon)
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.set_roi(polygon)
