from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QPushButton, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forklift Detection")
        self.resize(920, 720)  # 기본 크기 설정

        # 중앙 위젯 설정 (전체 UI의 레이아웃)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 버티컬 레이아웃 (왼쪽 비디오, 오른쪽 UI)
        main_layout = QHBoxLayout(central_widget)

        # [왼쪽] 비디오들 (Grid)
        self.video_area = QWidget()
        self.video_layout = QGridLayout()
        self.video_layout.setSpacing(10)
        self.video_area.setLayout(self.video_layout)

        # 영상이 비율 유지하며 늘어나도록
        self.video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_area, stretch=3)
        
        # 기본 레이아웃 설정 (1x1 그리드)
        self.video_area.setLayout(self.video_layout)
        main_layout.addWidget(self.video_area, stretch=3)

        # [오른쪽] 기능 UI (카메라 토글 + 로그 + 종료)
        self.ui_area = QWidget()
        ui_layout = QVBoxLayout()
        self.ui_area.setLayout(ui_layout)
        main_layout.addWidget(self.ui_area, stretch=1)

        # 카메라 ON/OFF 버튼들
        # camera_btn_layout = QGridLayout()
        # camera_btn_layout
        self.camera_buttons = []
        self.active_cameras = []  # 활성화된 카메라 번호 저장
        self.create_camera_buttons(ui_layout)

        # 로그뷰어
        self.log_viewer = LogViewer()
        ui_layout.addWidget(self.log_viewer, stretch=1)

        # 종료 버튼
        ui_layout.addWidget(QPushButton("종료"))

        main_layout.addWidget(self.ui_area, stretch=1)

        # 비디오 위젯을 Grid에 배치
        self.video_widgets = []
        self.update_video_widgets()

        # ROI 오버레이 설정 (비디오 위젯에 대해 ROI 설정)
        self.roi_editor1 = ROIEditor(self.video_widgets[0] if self.video_widgets else None)
        self.roi_editor1.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.roi_editor1.roi_defined.connect(self.on_roi_defined)
        self.roi_editor1.raise_()  # 비디오 위로 올리기

    def create_camera_buttons(self, ui_layout):
        """카메라 버튼 생성 함수"""
        for i in range(1, 10):  # 카메라 1번 ~ 9번 버튼
            btn = QPushButton(f"{i}번 카메라")
            btn.setCheckable(True)  # 버튼 클릭 시 체크되도록 설정
            btn.clicked.connect(self.on_camera_button_clicked)
            self.camera_buttons.append(btn)
            ui_layout.addWidget(btn)

    def on_camera_button_clicked(self):
        """카메라 버튼 클릭 시 활성화된 카메라 수에 맞게 레이아웃 변경"""
        button = self.sender()
        camera_num = int(button.text().split('번')[0])  # "1번", "2번"에서 숫자 추출

        if button.isChecked():
            if camera_num not in self.active_cameras:
                self.active_cameras.append(camera_num)
        else:
            if camera_num in self.active_cameras:
                self.active_cameras.remove(camera_num)

        self.update_video_widgets()  # 비디오 위젯 업데이트

    def update_video_widgets(self):
        """활성화된 카메라 개수에 맞게 비디오 위젯과 레이아웃을 업데이트하는 함수"""
        # 기존 비디오 위젯 제거
        for widget in self.video_widgets:
            widget.setParent(None)
        self.video_widgets.clear()

        # 활성화된 카메라 수에 맞는 비디오 위젯 생성
        for camera_num in self.active_cameras:
            vw = VideoWidget(f"resources/videos/sample{camera_num}.avi")
            self.video_widgets.append(vw)

        # 그리드 레이아웃 업데이트
        self.update_grid_layout()

    def update_grid_layout(self):
        for i in reversed(range(self.video_layout.count())):
            widget = self.video_layout.itemAt(i).widget()
            self.video_layout.removeWidget(widget)
            widget.setParent(None)

        count = len(self.video_widgets)
        if count == 0:
            return

        cols = min(3, count)
        rows = (count + cols - 1) // cols

        for idx, vw in enumerate(self.video_widgets):
            row = idx // cols
            col = idx % cols
            self.video_layout.addWidget(vw, row, col)
            vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for r in range(rows):
            self.video_layout.setRowStretch(r, 1)
        for c in range(cols):
            self.video_layout.setColumnStretch(c, 1)


    def on_roi_defined(self, polygon):
        if len(polygon) < 3:
            print("ROI는 최소 3개의 점이 필요합니다.")
            return
        print("ROI 확정:", polygon)
        for vw in self.video_widgets:
            vw.set_roi(polygon)  # 모든 비디오 위젯에 ROI 설정
