from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QPlainTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QEvent
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

         # [왼쪽] 비디오들 (Grid)
        self.video_area = QWidget()
        self.video_layout = QGridLayout()
        
        # 비디오 영역의 크기 비율 설정
        self.video_layout.setRowStretch(0, 1)  # 첫 번째 행 (비디오 영역) 크기 비율 1
        self.video_layout.setColumnStretch(0, 1)  # 첫 번째 열 (비디오 영역) 크기 비율 1

        # 비디오 위젯 크기 고정 (예시: 최소 크기 설정)
        self.video_layout.setRowMinimumHeight(0, 300)  # 첫 번째 행의 최소 높이 설정
        self.video_layout.setColumnMinimumWidth(0, 400)  # 첫 번째 열의 최소 너비 설정
        
        self.video_area.setLayout(self.video_layout)
        main_layout.addWidget(self.video_area, stretch=3)

        # 오른쪽: UI 컨트롤
        self.ui_area = QWidget()
        ui_layout = QVBoxLayout()
        self.ui_area.setLayout(ui_layout)

        self.video_widgets = []
        self.roi_editors = []

        for i in range(2):  # 카메라 수만큼
            vw = VideoWidget(f"resources/videos/sample{i+1}.avi")
            vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_widgets.append(vw)
            row = i // 2
            col = i % 2
            self.video_layout.addWidget(vw, row, col)

            # ROIEditor 설정
            roi_editor = ROIEditor(vw)
            roi_editor.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            roi_editor.roi_defined.connect(lambda polygon, idx=i: self.on_roi_defined(polygon, idx))
            roi_editor.raise_()
            vw.installEventFilter(self)

            self.roi_editors.append(roi_editor)

            # 카메라 ON/OFF + ROI 리셋 버튼
            cam_btn_layout = QHBoxLayout()
            onoff_btn = QPushButton(f"카메라 {i+1} ON/OFF")
            reset_roi_btn = QPushButton("ROI 리셋")
            reset_roi_btn.clicked.connect(lambda _, idx=i: self.reset_roi(idx))
            cam_btn_layout.addWidget(onoff_btn)
            cam_btn_layout.addWidget(reset_roi_btn)
            ui_layout.addLayout(cam_btn_layout)

        # 로그뷰어
        self.log_viewer = LogViewer()
        ui_layout.addWidget(self.log_viewer, stretch=1)

        # 종료 버튼
        quit_btn = QPushButton("종료")
        quit_btn.clicked.connect(self.close)
        ui_layout.addWidget(quit_btn)

        main_layout.addWidget(self.ui_area, stretch=1)

    def eventFilter(self, watched, event):
        for vw, roi_editor in zip(self.video_widgets, self.roi_editors):
            if watched == vw and event.type() == QEvent.Resize:
                roi_editor.setGeometry(vw.rect())
        return super().eventFilter(watched, event)

    def on_roi_defined(self, polygon, idx):
        if len(polygon) < 3:
            print("ROI는 최소 3개의 점이 필요합니다.")
            return
        print(f"카메라 {idx+1} ROI 설정됨:", polygon)
        self.video_widgets[idx].set_roi(polygon)

    def reset_roi(self, idx):
        print(f"카메라 {idx+1} ROI 리셋")
        self.video_widgets[idx].clear_roi()
        self.roi_editors[idx].reset()
