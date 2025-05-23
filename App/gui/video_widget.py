from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2, os, time, numpy as np
from recorder.video_buffer import VideoBuffer
from recorder.saver import VideoSaver
from detection.detector import Detector
from detection.postprocessor import PostProcessor
from gui.log_viewer import LogViewer
from shapely.geometry import Polygon, box
from utils.alert_manager import alert_manager



class VideoThread(QThread):
    """ 
    멀티 쓰레딩을 위한 QThread 객체 VideoThread
    """
    
    ## 이벤트 시그널 발생기 정의
    frame_ready = pyqtSignal(np.ndarray)
    event_triggered = pyqtSignal(float, int, str, float) ## event_time, camnum, label, iou
    mute_triggered = pyqtSignal(str, str, int)
    on_triggered = pyqtSignal(str, str, KeyboardInterrupt)
    info_triggered = pyqtSignal(list, int)
    overlap_triggered = pyqtSignal(str)

    
    def __init__(self, video_path, detector, postprocessor, video_buffer, cam_num):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.detector = detector
        self.postprocessor = postprocessor
        self.video_buffer = video_buffer
        self.cam_num = cam_num
        self.roi = None
        self.running = True
        self.frame_count = 0
        
        # 녹화 쿨타임 관리
        self.cooldown_seconds = 5  # 쿨타임 10초s
        self.last_event_time = 0
    
    
    
    def set_ui_size(self, w, h):
        """ 
        UI 크기를 받고 정의하는 함수.
        size 동기화를 위해 필요
        """
        self.ui_width = w
        self.ui_height = h
        
        
    def can_trigger_event(self):
        """ 
        이벤트 발생 쿨다운 처리 함수.
        """
        now = time.time()
        if now - self.last_event_time > self.cooldown_seconds:
            self.last_event_time = now
            return True
        return False
   
    def run(self):
        """ 
        VThread 실행 영역
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # frame = cv2.resize(frame, (self.ui_width, self.ui_height))    
                
            self.frame_count += 1
            self.video_buffer.add_frame(frame.copy())

            ## 3프레임마다 1개씩 모델에 전달.
            if self.frame_count % 3 != 0:
                continue

            # ROI 시각화 (디버깅용)
            # if self.roi is not None and hasattr(self, 'ui_width') and self.ui_width > 0:
            #     ## 비디오 위젯의 크기를 받음.
            #     sx = frame.shape[1] / self.ui_width
            #     sy = frame.shape[0] / self.ui_height
            #     roi_scaled = np.array([[int(x * sx), int(y * sy)] for x, y in self.roi])
            #     # self.roi = roi_scaled
            #     cv2.polylines(frame, [roi_scaled], isClosed=True, color=(0, 255, 0), thickness=2)
                # cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

            # 객체 감지
            results = self.postprocessor.filter_results(self.detector.detect_objects(frame))
            
            self.info_triggered.emit(results, self.cam_num)    
                
            for det in results:
                x1, y1, x2, y2 = det['box']
                conf = det['conf']
                class_name = det['class_name']
                label = f"{class_name} {conf:.2f}"

                if class_name == 'forklift-left' or class_name == 'forklift-horizontal':
                    self.mute_triggered.emit('mute', label, self.cam_num)
                if class_name == 'forklift-vertical':
                    self.on_triggered.emit('on', label, self.cam_num)

                if det.get('polygons'):
                    color = (0, 255, 0) if class_name == 'person' else (205, 205, 0)
                    for poly in det['polygons']:
                        poly_np = np.array(poly, dtype=np.int32)
                        cv2.polylines(frame, [poly_np], isClosed=True, color=color, thickness=2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ## 쿨다운에 걸리지 않으면.
            if self.can_trigger_event():
                ## 사람 - ROI 영역 체크 or roi안에 외곽선 점이 있는지(잘 되는지 모름)
                person_roi_detected, person_roi_iou = self.check_person_roi_overlap(results)
                if person_roi_detected or self.is_within_roi(results):
                    alert_manager.on_alert_signal.emit("inroi")
                    self.event_triggered.emit(time.time(), self.cam_num, "person-roi overlap", person_roi_iou)
                
                ## 사람 - 지게차 IOU 체크
                overlap_detected, iou_val = self.check_person_forklift_overlap(results)
                if overlap_detected:
                    alert_manager.on_alert_signal.emit("overlap")
                    self.event_triggered.emit(time.time(), self.cam_num, "person-forklift overlap", iou_val)

            # print('inroi_result', self.is_within_roi(results))

            # 🔹 시각화된 frame 전달
            self.frame_ready.emit(frame)

    def set_roi(self, roi_points):
        """ 
        roi 좌표를 전달받아서 self.roi 설정.
        """
        self.roi = np.array(roi_points, dtype=np.int32)
        print('vt', self.roi)

    def stop(self):
        self.running = False
        self.cap.release()

    @staticmethod
    def calculate_iou(poly1, poly2):
        poly1 = Polygon(poly1)
        poly2 = Polygon(poly2)
        if not poly1.is_valid or not poly2.is_valid:
            return 0
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0
        return inter / union

    def check_person_forklift_overlap(self, detections, iou_threshold=0.001):
        person_polys = [Polygon(d['polygons'][0]) for d in detections if d['class_name'] == 'person']

        forklift_polys = [Polygon(d['polygons'][0]) for d in detections
                          if d['class_name'].startswith('forklift') and d.get('polygons')]

        for p_poly in person_polys:
            for f_poly in forklift_polys:
                if not p_poly.is_valid or not f_poly.is_valid:
                    continue
                iou = self.calculate_iou(p_poly.exterior.coords, f_poly.exterior.coords)
                if iou >= iou_threshold:
                    print(f"⚠️ 위험 감지: person-forklift IoU = {iou:.2f}")
                    return True, iou
        return False, 0.0
    

    def is_within_roi(self, detections):
        """
        ROI에 사람 폴리곤의 꼭짓점 중 하나라도 포함되면 True
        """
        if self.roi is None or len(self.roi) < 3:
            return False

        # OpenCV용 ROI 컨투어 (numpy array로 변환)
        roi_contour = np.array(self.roi, dtype=np.int32)

        for d in detections:
            if d['class_name'] != 'person':
                continue

            person_polygon = d.get('polygons', [[]])[0]
            for (x, y) in person_polygon:
                if cv2.pointPolygonTest(roi_contour, (x, y), False) >= 0:
                    return True  # 한 점이라도 ROI 안이면 True

        return False

    
    def check_person_roi_overlap(self, detections, iou_threshold=0.0001):
        person_polys = [Polygon(d['polygons'][0]) for d in detections if d['class_name'] == 'person']

        roi_poly = Polygon(self.roi)
        
        print(person_polys)
        print(roi_poly)
        
        if not roi_poly.is_valid:
            print("ROI 폴리곤이 유효하지 않습니다.")
            return False, 0.0

        for p_poly in person_polys:
            if not p_poly.is_valid:
                continue
            iou = self.calculate_iou(p_poly.exterior.coords, roi_poly.exterior.coords)
            # if iou >= iou_threshold:
            if iou :
                print(f"⚠️ 위험 감지: person-ROI IoU = {iou:.2f}")
                return True, iou

        return False, 0.0


class VideoWidget(QLabel):
    """ 
    영상 재생 클래스.
    """
    def __init__(self, cam_num, video_path="resources/videos/sample.avi"):
        super().__init__()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        self.detector = Detector()
        self.postprocessor = PostProcessor(conf_threshold=0.6)
        self.video_buffer = VideoBuffer(fps=30, max_seconds=5)
        self.video_saver = VideoSaver(cam_num=cam_num)
        self.log_viewer = LogViewer(cam_num=cam_num)
        self.cam_num = cam_num
        self.roi = None

        self.vthread = VideoThread(video_path, self.detector, self.postprocessor, self.video_buffer, cam_num)
        self.vthread.set_ui_size(self.width(), self.height())
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
        self.vthread.set_ui_size(self.width(), self.height())
        self.vthread.set_roi(roi)
        print('vw',roi)
        if hasattr(self, 'roi_editor') and self.roi_editor:
            self.roi_editor.load_polygon(roi)  # 이미 만들어진 ROIEditor 활용
            print(f"[VideoWidget] ROIEditor에 ROI 반영 완료: {roi}")
    
    def clear_roi(self):
        self.roi = None
        self.update()

    def display_frame(self, frame):
        """ 
        영상 재생 함수
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def trigger_event(self, event_time=None, cam_num = "", label='event', iou=""):
        # 이벤트 발생 시 영상 클립 저장 및 로그 저장
        event_time = time.time()
        start = time.time()
        clip = self.video_buffer.get_clip(event_time)
        print(f"[⏱️ get_clip] 프레임 수: {len(clip)}, 소요: {time.time() - start:.2f}초")
        # if iou:
        self.video_saver.save_event_async(frames=clip, event_time=event_time, label=label, iou=iou)
     
     
    def resizeEvent(self, event):
        """ 
        비디오위젯의 사이즈 변경시 호출되는 함수.   
        """
        self.vthread.set_ui_size(self.width(), self.height())
        if hasattr(self, 'roi_editor'):
            self.roi_editor.setGeometry(self.rect())

    def closeEvent(self, event):
        self.vthread.stop()
        self.vthread.wait()
        super().closeEvent(event)
