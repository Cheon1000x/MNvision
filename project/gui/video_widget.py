from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import numpy as np
from datetime import datetime
import time
from recorder.video_buffer import VideoBuffer
from recorder.saver import VideoSaver
from detection.detector import Detector
from detection.postprocessor import PostProcessor
from gui.log_viewer import LogViewer

class VideoWidget(QLabel):
    def __init__(self, video_path="resources/videos/sample.avi"):
        super().__init__()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        self.detector = Detector()
        self.postprocessor = PostProcessor(conf_threshold=0.6)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 약 33ms = 30fps

        # 비디오 버퍼와 저장 설정
        self.video_buffer = VideoBuffer(fps=30, max_seconds=5)

        self.log_viewer = LogViewer()
        self.video_saver = VideoSaver(save_video_dir="resources/videos", save_log_dir="resources/logs")

        # 감지된 객체 목록과 ROI를 설정할 변수
        self.roi = None

        # QLabel 생성
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        
        # 이미지 사이즈를 내려받기위해서 설정.
        self.setMinimumSize(10, 10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)  # 중앙 정렬로 영상 표시

    def set_roi(self, roi_points):
        """ROI 영역을 설정하는 메소드 (폴리곤 형식으로 받기)"""
        # 전달된 ROI를 numpy 배열로 변환하고, dtype을 np.int32로 설정
        self.roi = np.array(roi_points, dtype=np.int32)
        print(f"ROI 설정됨: {self.roi}")

    def clear_roi(self):
        self.roi = None
        self.update()

    def update_frame(self):
        ret, frame = self.cap.read()
        widget_size = self.size()
        frame = cv2.resize(frame, (widget_size.width(), widget_size.height()))
        if ret:
            # 객체 감지
            results = self.detector.detect_objects(frame)
            filtered_objects = self.postprocessor.filter_results(results)

            # ROI 내 감지된 객체가 있으면
            for (x1, y1, x2, y2), conf, cls in filtered_objects:
                if self.roi is not None and self.is_within_roi(x1, y1, self.roi):
                    print(frame.shape, frame.dtype)
                    self.trigger_event(frame)  # 이벤트 처리 호출

            # 영상 버퍼에 프레임 추가
            self.video_buffer.add_frame(frame)

            # 객체 감지 결과 화면에 표시 (예시: 사각형 그리기)
            for (x1, y1, x2, y2), conf, cls in filtered_objects:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # ROI가 설정되어 있다면 폴리곤으로 표시
            if self.roi is not None:
                # ROI를 그리기 위해 polylines 사용
                cv2.polylines(frame, [self.roi], isClosed=True, color=(255, 0, 0), thickness=2)

            # 영상 표시
            self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # ✅ 비율 유지하여 QLabel 크기에 맞게 스케일 조절
        scaled_qimg = qimg.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(QPixmap.fromImage(scaled_qimg))

    def is_within_roi(self, x, y, roi):
        """주어진 점 (x, y)가 ROI 폴리곤 내부에 있는지 확인하는 함수"""
        point = (x, y)
        # OpenCV의 pointPolygonTest를 사용하여 점이 폴리곤 내에 있는지 확인
        result = cv2.pointPolygonTest(roi, point, False)
        return result >= 0  # 양수이면 내부에 있음

    def trigger_event(self, frame):
        """객체가 감지될 때 실행될 이벤트 처리 함수 (예: 로그 기록, 알람 등)"""
        event_time = time.time()  # 현재 시간을 이벤트 시간으로 설정
        print(f"ROI 내 객체 감지됨! 이벤트 시간: {event_time}")
        print("frame", frame.dtype, frame.shape)
        print("event_time", event_time)
        self.video_saver.save_clip(frames=frame, event_time=event_time)  # 이벤트 시간을 save_clip에 전달
        self.video_saver.save_logs(event_time=event_time)  # 이벤트 시간을 save_clip에 전달

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)
