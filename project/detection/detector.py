from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_path='resources/models/yolov8n.pt'):
        # YOLOv8 모델 로드
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        """
        입력 프레임에서 객체 감지 실행
        반환 형식: [(x1, y1, x2, y2), confidence, class_name]
        """
        results = self.model(frame)[0]  # 첫 번째 결과만 사용

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            detections.append(((x1, y1, x2, y2), conf, class_name))
        return detections
