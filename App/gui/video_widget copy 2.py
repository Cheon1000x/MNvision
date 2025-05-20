from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
import cv2, os, time, numpy as np
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
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.video_buffer.add_frame(frame.copy())
            if self.frame_count % 3 != 0:
                continue
            results = self.postprocessor.filter_results(self.detector.detect_objects(frame))
            for (x1, y1, x2, y2), _, _ in results:
                if self.roi is not None and cv2.pointPolygonTest(self.roi, (x1, y1), False) >= 0:
                    self.event_triggered.emit(time.time())
            self.frame_ready.emit(frame)

    def set_roi(self, roi_points):
        self.roi = np.array(roi_points, dtype=np.int32)
    
    def stop(self):
        self.running = False
        self.cap.release()

class VideoWidget(QLabel):
    def __init__(self, cam_num, video_path="resources/videos/sample.avi"):
        super().__init__()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        self.detector = Detector()
        self.postprocessor = PostProcessor(conf_threshold=0.6)
        self.video_buffer = VideoBuffer(fps=30, max_seconds=5)
        self.video_saver = VideoSaver(cam_num=cam_num)
        self.log_viewer = LogViewer(cam_num=cam_num)
        self.roi = None
        self.frame_count = 0

        self.vthread = VideoThread(video_path, self.detector, self.postprocessor, self.video_buffer)
        self.vthread.frame_ready.connect(self.display_frame)
        self.vthread.event_triggered.connect(self.trigger_event)
        self.vthread.start()

        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_roi_editor(self, roi_editor):
        """기존 ROIEditor 인스턴스를 등록함"""
        self.roi_editor = roi_editor

    def set_roi(self, roi):
        """ROI 설정 및 기존 ROIEditor를 활용한 시각화"""
        self.roi = np.array(roi, dtype=np.int32)
        self.vthread.set_roi(roi)
        
        if hasattr(self, 'roi_editor') and self.roi_editor:
            self.roi_editor.load_polygon(roi)  # 이미 만들어진 ROIEditor 활용
            print(f"[VideoWidget] ROIEditor에 ROI 반영 완료: {roi}")
    
    def clear_roi(self):
        self.roi = None
        self.update()

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def trigger_event(self, event_time=None):
        event_time = time.time()
        clip = self.video_buffer.get_clip(event_time)
        self.video_saver.save_clip(frames=clip, event_time=event_time)
        self.video_saver.save_logs(event_time=event_time)

    def is_within_roi(self, x, y, roi):
        return cv2.pointPolygonTest(roi, (x, y), False) >= 0

    def resizeEvent(self, event):
        if hasattr(self, 'roi_editor'):
            self.roi_editor.setGeometry(self.rect())

    def closeEvent(self, event):
        self.vthread.stop()
        self.vthread.wait()
        super().closeEvent(event)