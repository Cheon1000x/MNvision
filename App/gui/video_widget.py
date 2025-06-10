from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2, os, time, numpy as np
from recorder.video_buffer import VideoBuffer
from recorder.saver import VideoSaver

# from detection.detector import Detector
from detection.detector_onnx import Detector
# from detection.detector_tensorRT import Detector

from detection.postprocessor import PostProcessor
# from gui.log_viewer import LogViewer
# from shapely.geometry import Polygon, box
from utils.alert_manager import AlertManager




class VideoThread(QThread):
    """ 
    멀티 쓰레딩을 위한 QThread 객체 VideoThread
    """
    ## 이벤트 시그널 발생기 정의
    frame_ready = pyqtSignal(np.ndarray)
    event_triggered = pyqtSignal(float, int, str) ## event_time, camnum, label
    mute_triggered = pyqtSignal(str, str, int)
    on_triggered = pyqtSignal(str, str, int)
    info_triggered = pyqtSignal(list, int)
    overlap_triggered = pyqtSignal(str)
    # super.roi_update.emit()
    
    def __init__(self, video_path, detector, postprocessor, video_buffer, cam_num):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.detector = detector
        self.postprocessor = postprocessor
        self.video_buffer = video_buffer
        self.cam_num = cam_num
        self.alert_manager = AlertManager(cam_num=cam_num)
        self.roi = None
        self.running = True
        self.frame_count = 0
        self.scaled_roi = None
        # 녹화 쿨타임 관리
        self.cooldown_seconds = 5  # 쿨타임 10초s
        self.last_event_time = 0

        self.label_visible = True
     
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
    
    def get_scaled_roi(self, frame):
        if self.roi is None or self.ui_width == 0 or self.ui_height == 0:
            return None

        sx = frame.shape[1] / self.ui_width
        sy = frame.shape[0] / self.ui_height
        self.scaled_roi = np.array([[int(x * sx), int(y * sy)] for x, y in self.roi], dtype=np.int32)
        return np.array([[int(x * sx), int(y * sy)] for x, y in self.roi], dtype=np.int32)
    
    
    def set_roi(self, roi_points):
        """ 
        roi 좌표를 전달받아서 self.roi 설정.
        """
        self.roi = np.array(roi_points, dtype=np.int32)
        # print('vt', self.roi)

    def stop(self):
        self.running = False 
        self.cap.release()

    # 1. 사람과 roi
    #   1-1 중심거리
    #   1-2 박스 좌표 겹침
    # 2. 사람과 forklift
    #   2-1 중심거리
    #   2-2 박스 좌표 겹침
    
    ## 거리계산 함수1
    
    def get_box_info(self, box):
        """
        바운딩 박스에서 중심점 (x, y), 너비 (w), 높이 (h)를 계산합니다.
        박스 형식: [x_min, y_min, x_max, y_max]
        """
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return (center_x, center_y, width, height)

    ## 거리계산 함수2
    def calculate_distance(self,p1, p2):
        """
        두 점 (x1, y1)과 (x2, y2) 사이의 유클리드 거리를 계산합니다.
        """
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    ## 1-1 사람 roi 중심거리
    # def check_proximity_person_roi(self, detections):
    #     """
    #     'forklift'과 'person' 객체의 중심점 간 거리가
    #     forklift의 가로 절반 또는 세로 절반보다 가까운지 확인합니다.
    #     """
    #     pass
    #     all_person_info = []   # (box_data, center_x, center_y, width, height) 튜플 저장

    #     if self.roi is not None:
    #         # 1. 'person'의 정보를 수집
    #         for d in detections:
    #             class_name = d['class_name']
    #             box = d.get('box')

    #             if box is None or len(box) != 4:
    #                 continue

    #             center_x, center_y, width, height = self.get_box_info(box)

    #             if class_name.startswith('person'):
    #                 all_person_info.append((box, center_x, center_y, width, height))

    #         # 2. roi와 'person' 쌍에 대해 거리 조건 확인
    #         f_cx, f_cy, f_w, f_h = self.get_box_info(self.roi)
    #         for p_box_raw, p_cx, p_cy, p_w, p_h in all_person_info:
                
    #             distance = self.calculate_distance((f_cx, f_cy), (p_cx, p_cy))
                
    #             # forklift의 가로 절반 또는 세로 절반
    #             roi_half_width = f_w / 2
    #             roi_half_height = f_h / 2

    #             # 조건: 두 중심점 간 거리가 forklift의 가로 절반보다 가깝거나 (작거나),
    #             #       또는 세로 절반보다 가깝다면 (작거나) True
    #             print('self.roi', self.roi)
    #             print('distance', distance)
    #             print('roi_half_width', roi_half_width)    
                
    #             # if distance < roi_half_width or distance < roi_half_height:
    #             #     print(f"사람 박스 {p_box_raw} (중심: {p_cx:.1f},{p_cy:.1f}) 와 "
    #             #         f"roi 박스 {self.roi} (중심: {f_cx:.1f},{f_cy:.1f}) 사이 거리: {distance:.2f}")
    #             #     print(f"  -> roi 가로 절반: {roi_half_width:.2f}, 세로 절반: {roi_half_height:.2f}")
    #             #     print("  -> 거리가 roi 가로 절반 또는 세로 절반보다 가깝습니다.")
    #             #     return True # 조건 만족 시 즉시 True 반환

    #     print("조건을 만족하는 'forklift'과 'person' 쌍이 없습니다.")
    #     return False # 모든 조합을 확인했지만 조건 만족 안 함
    
    ## 2-1 중심거리 계산
    def check_proximity_person_forklift(self, detections):
        """
        'forklift'과 'person' 객체의 중심점 간 거리가
        forklift의 가로 절반 또는 세로 절반보다 가까운지 확인합니다.
        """
        all_forklift_info = [] # (box_data, center_x, center_y, width, height) 튜플 저장
        all_person_info = []   # (box_data, center_x, center_y, width, height) 튜플 저장

        # 1. 모든 'forklift'과 'person'의 정보를 수집
        for d in detections:
            class_name = d['class_name']
            box = d.get('box')

            if box is None or len(box) != 4:
                continue

            center_x, center_y, width, height = self.get_box_info(box)

            if class_name.startswith('forklift'):
                all_forklift_info.append((box, center_x, center_y, width, height))
            elif class_name.startswith('person'):
                all_person_info.append((box, center_x, center_y, width, height))

        # 2. 각 'forklift'과 'person' 쌍에 대해 거리 조건 확인
        for f_box_raw, f_cx, f_cy, f_w, f_h in all_forklift_info:
            for p_box_raw, p_cx, p_cy, p_w, p_h in all_person_info:
                
                distance = self.calculate_distance((f_cx, f_cy), (p_cx, p_cy))
                
                # forklift의 가로 절반 또는 세로 절반
                forklift_half_width = f_w / 2
                forklift_half_height = f_h / 2

                # 조건: 두 중심점 간 거리가 forklift의 가로 절반보다 가깝거나 (작거나),
                #       또는 세로 절반보다 가깝다면 (작거나) True
                if distance < forklift_half_width or distance < forklift_half_height:
                    # print(f"사람 박스 {p_box_raw} (중심: {p_cx:.1f},{p_cy:.1f}) 와 "
                    #     f"포크리프트 박스 {f_box_raw} (중심: {f_cx:.1f},{f_cy:.1f}) 사이 거리: {distance:.2f}")
                    # print(f"  -> 포크리프트 가로 절반: {forklift_half_width:.2f}, 세로 절반: {forklift_half_height:.2f}")
                    # print("  -> 거리가 포크리프트 가로 절반 또는 세로 절반보다 가깝습니다.")
                    return True # 조건 만족 시 즉시 True 반환

        # print("조건을 만족하는 'forklift'과 'person' 쌍이 없습니다.")
        return False # 모든 조합을 확인했지만 조건 만족 안 함

    ## 1-2 박스 좌표겹침
    def is_within_roi(self, detections):
        """
        ROI에 사람 폴리곤의 꼭짓점 중 하나라도 포함되면 True
        """
        if self.scaled_roi is None or len(self.scaled_roi) < 3:
            return False

        # OpenCV용 ROI 컨투어 (numpy array로 변환)
        roi_contour = np.array(self.scaled_roi, dtype=np.int32)

        for d in detections:
            if d['class_name'] != 'person':
                continue
            
            # person_box = d.get('box', [[]])
            
            x1, y1, x2, y2 = d.get('box')
            person_box = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
            # print(person_box)
            for (x, y) in person_box:
                if cv2.pointPolygonTest(roi_contour, (x, y), False) >= 0:
                    return True  # 한 점이라도 ROI 안이면 True

        return False
    
    def is_box_inside(self, inner_box, outer_box):
        """
        inner_box가 outer_box 안에 완전히 포함되는지 확인합니다.
        박스 형식: [x_min, y_min, x_max, y_max]
        """
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box

        # return (inner_x1 >= outer_x1 and
        #         inner_y1 >= outer_y1 and
        #         inner_x2 <= outer_x2 and
        #         inner_y2 <= outer_y2)
        return (inner_x1 < outer_x2 and
            inner_x2 > outer_x1 and
            inner_y1 < outer_y2 and
            inner_y2 > outer_y1)

    
    ## 2-2 사람과 포크리프트 박스겹침
    def check_person_in_forklift_box(self, detections):
        """
        감지된 'forklift' 박스 중 하나라도 'person' 박스를 포함하는지 확인합니다.
        박스 형식: [x_min, y_min, x_max, y_max]
        """
        all_forklift_boxes = []
        all_person_boxes = []

        # 1. 모든 'forklift'과 'person' 박스를 수집
        for d in detections:
            class_name = d['class_name']
            box = d.get('box') # 'box' 키가 없을 경우 None 반환

            if box is None or len(box) != 4: # 박스 데이터가 유효한지 확인 (x_min, y_min, x_max, y_max 4개 값)
                continue

            if class_name.startswith('forklift'):
                all_forklift_boxes.append(box)
            elif class_name.startswith('person'):
                all_person_boxes.append(box)

        # 2. 'forklift' 박스와 'person' 박스를 서로 비교
        for forklift_box in all_forklift_boxes:
            for person_box in all_person_boxes:
                if self.is_box_inside(person_box, forklift_box):
                    # print(f"'{person_box}' (person) 박스가 '{forklift_box}' (forklift) 박스 안에 있습니다.")
                    return True # 조건 만족, 즉시 True 반환

        # print("어떤 'forklift' 박스도 'person' 박스를 포함하지 않습니다.")
        return False # 모든 조합을 확인했지만 조건 만족 안 함
    
    def run(self):
        """ 
        VThread 실행 영역
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 영상의 FPS 가져오기
        frame_interval = 1.0 / fps            # 프레임 간 시간 간격 (초)

        while self.running:
            start_time = time.time()          # 루프 시작 시간 기록

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            self.video_buffer.add_frame(frame.copy())

            elapsed = time.time() - start_time
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time) 
            
            ## fps 조절
            ## 3프레임마다 1개씩 모델에 전달.
            if self.frame_count % 5 != 0:
                continue
            
            self.scaled_roi = self.get_scaled_roi(frame)
            # if scaled_roi is not None:
            #     cv2.polylines(frame, [scaled_roi], isClosed=True, color=(0, 255, 0), thickness=2)
            #     # (선택적으로 원래 ROI도 표시)
            #     cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
                
            # # ROI 시각화 (디버깅용)
            # if self.roi is not None and hasattr(self, 'ui_width') and self.ui_width > 0:
                
            #     ## 비디오 위젯의 크기를 받음.
            #     sx = frame.shape[1] / self.ui_width
            #     sy = frame.shape[0] / self.ui_height
            #     roi_scaled = np.array([[int(x * sx), int(y * sy)] for x, y in self.roi])
            #     # self.roi = roi_scaled
            #     cv2.polylines(frame, [roi_scaled], isClosed=True, color=(0, 255, 0), thickness=2)
            #     cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

            # 객체 감지
            crt = time.time()
            
            results = self.postprocessor.filter_results(self.detector.detect_objects(frame))
            print(f"{time.time()-crt:.3f} ms")
        
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

                # if det.get('polygons'):
                #     color = (0, 255, 0) if class_name == 'person' else (205, 205, 0)
                #     for poly in det['polygons']:
                #         poly_np = np.array(poly, dtype=np.int32)
                #         cv2.polylines(frame, [poly_np], isClosed=True, color=color, thickness=2)
                #     cv2.putText(frame, label, (int(x1), int(y1) - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # 클래스 이름에 따라 색상 정의
            
                if class_name == 'person':
                    color = (0, 255, 0) # 초록색 (사람)
                elif class_name.startswith('forklift'):
                    color = (205, 205, 0) # 청록색 (지게차)
                else:
                    color = (0, 165, 255) # 주황색 (그 외 객체, 필요시 변경)
            

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                #  중심점 계산 및 그리기 추가 
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # 중심점 그리기 (예: 반지름 5픽셀의 원, 채워진 원)
                cv2.circle(frame, (center_x, center_y), 5, color, -1) # -1은 원을 채움
                
                # 라벨 텍스트 배경 그리기 (선택 사항, 텍스트 가독성 향상)
                # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                # cv2.rectangle(frame, (int(x1), int(y1) - text_size[1] - 10), 
                #               (int(x1) + text_size[0], int(y1)), color, -1) # 배경 채우기
                
                # 라벨 텍스트 넣기
                if self.label_visible:
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), # 박스 위쪽
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # 텍스트 색상을 박스 색상과 동일하게

            ## 사람 - ROI 영역 체크
            # if  self.check_proximity_person_roi(results) or self.is_within_roi(results):
            if  self.is_within_roi(results):
                self.alert_manager.on_alert_signal.emit("inroi", self.cam_num) 
                
                if self.can_trigger_event(): 
                    self.event_triggered.emit(time.time(), self.cam_num, "roi overlap")
            
            ## 사람 - 지게차 IOU 체크
            if  self.check_proximity_person_forklift(results) or self.check_person_in_forklift_box(results):
                self.alert_manager.on_alert_signal.emit("overlap", self.cam_num) 
                
                if self.can_trigger_event():
                    self.event_triggered.emit(time.time(), self.cam_num, "forklift overlap")

            # print('inroi_result', self.is_within_roi(results))
            self.frame_ready.emit(frame)


    def resizeEvent(self, event):
        """ 
        비디오 쓰레드 크기 변경시 호출되는 함수.   
        """
        pass
        

class VideoWidget(QLabel):
    """ 
    영상 재생 클래스.
    """
    roi_update = pyqtSignal(np.ndarray)
    def __init__(self, cam_num, video_path="resources/videos/sample.avi", conf_threshold = 0.65):
        super().__init__()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
        print('conf_threshold', conf_threshold)
        self.conf_threshold = conf_threshold
        self.detector = Detector(conf_threshold= conf_threshold)
        self.postprocessor = PostProcessor(conf_threshold= conf_threshold)
        self.video_buffer = VideoBuffer(fps=30, max_seconds=5)
        self.video_saver = VideoSaver(cam_num=cam_num)
        # self.log_viewer = LogViewer(cam_num=cam_num)
        
        self.cam_num = cam_num
        self.roi = None

        self.vthread = VideoThread(video_path, self.detector, self.postprocessor, self.video_buffer, cam_num)
        self.vthread.set_ui_size(self.width(), self.height())
        self.vthread.frame_ready.connect(self.display_frame)
        self.vthread.event_triggered.connect(self.trigger_event)
        self.vthread.start()

        self.setScaledContents(True)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_roi_editor(self, roi_editor):
        """기존 ROIEditor 인스턴스를 등록함"""
        self.roi_editor = roi_editor

    def set_roi(self, roi):
        """ROI 설정 및 기존 ROIEditor를 활용한 시각화"""
        self.roi = np.array(roi, dtype=np.int32)
        self.vthread.set_ui_size(self.width(), self.height())
        self.vthread.set_roi(roi)
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

    def trigger_event(self, event_time=None, cam_num = "", label='event'):
        # 이벤트 발생 시 영상 클립 저장 및 로그 저장
        event_time = time.time()
        start = time.time()
        clip = self.video_buffer.get_clip(event_time)
        print(f"[⏱️ get_clip] 프레임 수: {len(clip)}, 소요: {time.time() - start:.2f}초")
        
        self.video_saver.save_event_async(frames=clip, event_time=event_time, label=label)
     
     
    def resizeEvent(self, event):
        """ 
        비디오위젯의 사이즈 변경시 호출되는 함수.   
        """
        
        self.vthread.set_ui_size(self.width(), self.height())
        if self.roi is not None:
            self.vthread.set_roi(self.roi)
            
            # print(
            # f""" 
            
            # 비디오 쓰레드 크기 변경
            # {(self.width(), self.height())}
            # """)
        # self.vthread.set_roi(self.roi)
        if hasattr(self, 'roi_editor'):
            self.roi_editor.setGeometry(self.rect())

    def closeEvent(self, event):
        self.vthread.frame_ready.disconnect(self.display_frame) # 추가
        self.vthread.stop()
        self.vthread.wait()
        super().closeEvent(event)
