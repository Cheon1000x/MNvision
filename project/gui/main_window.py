from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTabWidget, QSizePolicy, QLabel
)
from PyQt5.QtCore import Qt
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forklift Detection")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # [왼쪽] 비디오 레이아웃
        self.video_area = QWidget()
        self.video_layout = QGridLayout()
        self.video_area.setLayout(self.video_layout)
        self.video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_area, stretch=3)

        # [오른쪽] 기능 UI
        self.ui_area = QWidget()
        ui_layout = QVBoxLayout()
        self.ui_area.setLayout(ui_layout)
        main_layout.addWidget(self.ui_area, stretch=1)

        # 🔹 탭 위젯 추가
        self.tab_widget = QTabWidget()
        ui_layout.addWidget(self.tab_widget)

        self.camera_buttons = {}
        self.roi_reset_buttons = {}
        self.active_cameras = []
        self.video_widgets = {}
        self.roi_editors = {} 
        
        self.init_tabs()

        # 로그 뷰어 및 종료 버튼
        ui_layout.addWidget(LogViewer(), stretch=1)
        ui_layout.addWidget(QPushButton("종료"))

    def init_tabs(self):
        for tab_idx, cam_range in enumerate([(1, 3), (4, 6)]):  # Cam1 = 1~3, Cam2 = 4~6
            tab = QWidget()
            layout = QVBoxLayout()
            tab.setLayout(layout)

            for cam_id in range(cam_range[0], cam_range[1] + 1):
                hbox = QHBoxLayout()

                btn = QPushButton(f"{cam_id}번 카메라")
                btn.setCheckable(True)
                btn.clicked.connect(self.on_camera_toggle)
                self.camera_buttons[cam_id] = btn
                hbox.addWidget(btn)

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

                # 🔹 VideoWidget 생성
                vw = VideoWidget(f"resources/videos/sample{cam_id}.avi")
                self.video_widgets[cam_id] = vw

                # 🔹 ROIEditor 생성 (VideoWidget 생성 이후)
                roi_editor = ROIEditor(vw, cam_id)  # VideoWidget과 cam_id를 전달하여 ROIEditor 생성
                roi_editor.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                roi_editor.roi_defined.connect(self.on_roi_defined)
                roi_editor.setParent(vw)
                roi_editor.setGeometry(vw.rect())
                roi_editor.show()
                roi_editor.raise_()  # 비디오 위로 올림

                # 저장
                self.roi_editors[cam_id] = roi_editor  # cam_id를 키로 하여 roi_editor를 저장


        else:
            if cam_id in self.active_cameras:
                self.active_cameras.remove(cam_id)

                # 🔹 VideoWidget 및 ROIEditor 제거
                vw = self.video_widgets.pop(cam_id, None)
                if vw:
                    vw.setParent(None)

                roi_editor = self.roi_editors.pop(cam_id, None)
                if roi_editor:
                    roi_editor.setParent(None)

        self.update_grid_layout()


    def reset_roi(self, cam_id):
        # VideoWidget의 ROI 제거
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.clear_roi()

        # 기존 ROIEditor 제거
        roi_editor = self.roi_editors.get(cam_id)
        if roi_editor:
            roi_editor.roi = None
            roi_editor.points.clear()
            roi_editor.finished = False
            roi_editor.setParent(None)
            roi_editor.deleteLater()
            roi_editor.update()
            del self.roi_editors[cam_id]

        # 새로운 ROIEditor 생성 및 연결
        new_editor = ROIEditor(vw, cam_id=cam_id)
        new_editor.setParent(vw)
        new_editor.setGeometry(vw.rect())
        new_editor.show()
        new_editor.raise_()
        # new_editor.roi_defined.connect(self.on_roi_defined)
        new_editor.roi_defined.connect(lambda polygon, cid=cam_id: self.on_roi_defined(cid, polygon))
        new_editor.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        self.roi_editors[cam_id] = new_editor

        print(f"카메라 {cam_id} ROI 초기화됨 및 새 ROIEditor 생성됨")


    def update_grid_layout(self):
        # 기존 레이아웃 제거
        for i in reversed(range(self.video_layout.count())):
            widget = self.video_layout.itemAt(i).widget()
            self.video_layout.removeWidget(widget)
            widget.setParent(None)

        count = len(self.active_cameras)
        if count == 0:
            return

        cols = min(2, count)
        rows = (count + cols - 1) // cols

        for idx, cam_id in enumerate(sorted(self.active_cameras)):
            row = idx // cols
            col = idx % cols
            vw = self.video_widgets[cam_id]

            self.video_layout.addWidget(vw, row, col)
            # vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.adjust_video_size(vw)
            
            # ROIEditor 연결
            roi_editor = ROIEditor(vw, cam_id)  # 카메라 ID 전달
            roi_editor.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            roi_editor.roi_defined.connect(self.on_roi_defined)
            roi_editor.setParent(vw)
            roi_editor.setGeometry(vw.rect())
            roi_editor.show()
            roi_editor.raise_()  # 비디오 위로 올림
    
        for r in range(rows):
            self.video_layout.setRowStretch(r, 1)
        for c in range(cols):
            self.video_layout.setColumnStretch(c, 1)

    
    
    def adjust_video_size(self, vw):
        # 비디오 영역 크기와 비율을 맞추기 위한 계산
        video_rect = vw.rect()
        video_width = 1280
        video_height = 720
        aspect_ratio = video_width / video_height

        # 비디오 영역의 크기 비율에 맞게 크기 조정
        if video_rect.width() / video_rect.height() > aspect_ratio:
            # 가로가 더 긴 경우, 높이에 맞춰 너비를 조정
            new_width = video_rect.height() * aspect_ratio
            new_height = video_rect.height()
        else:
            # 세로가 더 긴 경우, 너비에 맞춰 높이를 조정
            new_width = video_rect.width()
            new_height = video_rect.width() / aspect_ratio

        # 비디오 위젯 크기 조정
        vw.setFixedSize(new_width, new_height)
                

    def on_roi_defined(self, polygon, cam_id):
        if len(polygon) < 3:
            print(f"카메라 {cam_id}에서 ROI는 최소 3개의 점이 필요합니다.")
            return
        print(f"카메라 {cam_id} ROI 확정:", polygon)
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.set_roi(polygon)  # 해당 카메라에 대한 ROI 설정