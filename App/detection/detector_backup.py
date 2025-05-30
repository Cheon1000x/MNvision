from ultralytics import YOLO
import torch
import numpy as np
import cv2
# from onnx_utils import ONNXProcessor
from detection.onnx_utils import ONNXProcessor

class Detector:
    # def __init__(self, model_path='resources/models/yolov8_continued_seg_ver3.pt'):
    def __init__(self, model_path='resources/models/yolov8_seg_custom.pt'):
    # def __init__(self, model_path='resources/models/yolov8n.pt'):
    # def __init__(self, model_path='resources/models/yolov5n.pt'):
    # def __init__(self, model_path='resources/models/best.onnx'):
        # YOLOv8 모델 로드
        self.model = YOLO(model_path)
        
        # YOLOv5n 모델 로드
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # yolov5n (nano) 모델 로드
        
        # self.processor = ONNXProcessor(model_path, conf_threshold=0.5)
        
        # 이미지 예측
                
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        


    def detect_objects(self, frame):
        """
        입력 프레임에서 객체 감지 실행
        반환 형식: [(x1, y1, x2, y2), confidence, class_name]
        """
        ## yolo 8n 코드 box
        # results = self.model.predict(frame, device=self.device, stream=False)[0]
        # detections = []
        # for box in results.boxes:
        #     x1, y1, x2, y2 = box.xyxy[0].tolist()
        #     conf = box.conf[0].item()
        #     cls_id = int(box.cls[0].item())
        #     class_name = self.model.names[cls_id]
        #     detections.append(((x1, y1, x2, y2), conf, class_name))
        # return detections
        
        ## yolo 8n 코드 polygon
        
        # results = self.processor(frame)
        results = self.model.predict(frame, device=self.device, stream=False)[0]
        detections = []

        # 원본 크기 기준으로 polygon 좌표 스케일링
        orig_h, orig_w = frame.shape[:2]
        input_h, input_w = results.orig_img.shape[:2]
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        if results.masks is not None:
            mask_polygons = results.masks.xy
        else:
            mask_polygons = [None] * len(results.boxes)

        for box, poly in zip(results.boxes, mask_polygons):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]

            scaled_polygons = []
            if poly is not None:
                scaled_polygons = [(int(x * scale_x), int(y * scale_y)) for x, y in poly]

            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': float(conf),
                'class_name': class_name,
                'polygons': [scaled_polygons] if scaled_polygons else []
            })

        return detections


        
        ## 5n 사용시 구조가 달라서 아래 코드 사용
        # detections = []
        # results = self.model(frame)
        # for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
        #     x1, y1, x2, y2 = box
        #     conf = float(conf)
        #     cls_id = int(cls_id)
        #     class_name = self.model.names[cls_id]
        #     detections.append(((x1, y1, x2, y2), conf, class_name))
        # return detections
