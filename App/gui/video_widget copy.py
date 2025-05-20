from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
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

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    event_triggered = pyqtSignal(float)

    def __init__(self, video_path, detector, postprocessor, video_buffer):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.detector = detector
        self.postprocessor = postprocessor
        self.video_buffer = video_buffer
        self.roi = None
        self.running = True
        self.frame_count = 0

    def set_roi(self, roi):
        self.roi = np.array(roi, dtype=np.int32)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            self.video_buffer.add_frame(frame.copy())

            if self.frame_count % 3 != 0:
                continue

            results = self.detector.detect_objects(frame)
            filtered = self.postprocessor.filter_results(results)

            for (x1, y1, x2, y2), conf, cls in filtered:
                if self.roi is not None and cv2.pointPolygonTest(self.roi, (x1, y1), False) >= 0:
                    self.event_triggered.emit(time.time())

            self.frame_ready.emit(frame)

    def stop(self):
        self.running = False
        self.cap.release()

class VideoWidget(QLabel):
    """ 
    카메라 재생 객체
    영상 재생과 roi 표기
    
    """
    def __init__(self, cam_num, video_path="resources/videos/sample.avi"):
        self.cam_num = cam_num
        super().__init__()
        screen = QGuiApplication.primaryScreen()
        size = screen.availableGeometry()
        # self.resize(int(size.width() * 0.8), int(size.height() * 0.8))

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        self.detector = Detector()
        self.postprocessor = PostProcessor(conf_threshold=0.6)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 약 33ms = 30fps
        self.frame_count = 0
        
        # 비디오 버퍼와 저장 설정
        self.video_buffer = VideoBuffer(fps=30, max_seconds=5)

        self.log_viewer = LogViewer(cam_num=cam_num)
        self.video_saver = VideoSaver(cam_num=cam_num)

        # 감지된 객체 목록과 ROI를 설정할 변수
        self.roi = None

        # QThread 사용하여 최적화
        self.thread = VideoThread(video_path, self.detector, self.postprocessor, self.video_buffer)
        self.thread.frame_ready.connect(self.display_frame)
        self.thread.event_triggered.connect(self.trigger_event)
        self.thread.start()
        
        # QLabel 생성
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.resize(int((size.width()-10) * 0.5), int((size.width()-10) * 9/32))


    def set_roi(self, roi_points):
        """ROI 영역을 설정하는 메소드 (폴리곤 형식으로 받기)"""
        # 전달된 ROI를 numpy 배열로 변환하고, dtype을 np.int32로 설정
        self.roi = np.array(roi_points, dtype=np.int32)
        self.thread.set_roi(roi_points)
        print(f"ROI 설정됨: {self.roi}")


    def clear_roi(self):
        """ 
        roi 객체 갱신
        """
        self.roi = None
        self.update()


    def update_frame(self):
        """ 
        display_frame 전에 버퍼처리, 모델 상호작용
        """
        ret, frame = self.cap.read()
        widget_size = self.size()
        frame = cv2.resize(frame, (widget_size.width(), widget_size.height()))
        if ret:
            # 영상 버퍼에 프레임 추가
            self.video_buffer.add_frame(frame.copy())
            
            ## 성능문제로 3번쨰 프레임만 넣는중.
            self.frame_count += 1
            if self.frame_count % 3 != 0:
                return
            
            # 객체 감지
            results = self.detector.detect_objects(frame)
            filtered_objects = self.postprocessor.filter_results(results)

            # ROI 내 감지된 객체가 있으면
            for (x1, y1, x2, y2), conf, cls in filtered_objects:
                if self.roi is not None and self.is_within_roi(x1, y1, self.roi):
                    print(frame.shape, frame.dtype)
                    self.trigger_event(event_time=time.time())  # 이벤트 처리 호출
            
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
        """ 
        영상 출력하기.
        """
        ## QLabel의 pixmap 사용함
        
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


    def trigger_event(self, event_time=None):
        """객체가 감지될 때 실행될 이벤트 처리 함수 (예: 로그 기록, 알람 등)"""
        event_time = time.time()  # 이건 여전히 기준
        clip = self.video_buffer.get_clip(event_time)
        self.video_saver.save_clip(frames=clip, event_time=event_time)

        print(f"ROI 내 객체 감지됨! 이벤트 시간: {event_time}")
        print(f"[DEBUG] 추출된 프레임 수: {len(clip)}")
        # if clip:
        #     self.video_saver.save_clip(frames=clip, event_time=event_time)
        # else:
        #     print("[WARNING] 클립에 저장할 프레임이 없습니다.")

        self.video_saver.save_logs(event_time=event_time)


    def resizeEvent(self, event):
        """ 
        사이즈 변경시 호출되는 메서드
        """
        if hasattr(self, 'roi_editor'):
            self.roi_editor.setGeometry(self.rect())
        
    
    # def closeEvent(self, event):
    #     self.timer.stop()  # 타이머 멈춤
    #     self.cap.release()
    #     super().closeEvent(event)
        
    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        super().closeEvent(event)