import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import onnx
import os

class Detector:
    # def __init__(self, onnx_path="resources/models/yolov8s-seg.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_onnx_ver1.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_onnx_ver1_opset11.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_onnx_ver1_opset13.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_onnx_ver1_opset20.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_custom.onnx"):
    def __init__(self, onnx_path="resources/models/yolov8_seg_custom_opset20.onnx"):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.conf_thres = 0.6
        self.iou_thres = 0.6
        
        # ONNX 모델의 입력 이름 가져오기
        self.input_name = self.session.get_inputs()[0].name
        
        # ONNX 모델의 입력 shape에서 height와 width 가져오기
        # get_inputs()[0].shape는 [batch_size, channels, height, width] 형태
        # 따라서 height는 인덱스 2, width는 인덱스 3
        # 만약 동적 shape로 변환되어 'height', 'width'와 같은 문자열이 나온다면,
        # 모델 변환 시 문제가 있었을 수 있습니다.
        # 일반적으로는 640, 640 같은 정수 값이 들어와야 합니다.
        # self.input_height = self.session.get_inputs()[0].shape[2] # 640
        # self.input_width = self.session.get_inputs()[0].shape[3]  # 640
        self.input_height = 640
        self.input_width = 640
        # print(self.session.get_inputs()[0].shape)
        # print(self.input_height)
        # print(self.input_width)
        try:
            if not os.path.exists(onnx_path):
                print(f"오류: ONNX 모델 파일이 다음 경로에 없습니다: {onnx_path}")
                raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")
            
            onnx.checker.check_model(onnx_path)
            print(f"ONNX 모델 '{onnx_path}'의 유효성 검사 성공.")
        except Exception as e:
            print(f"ONNX 모델 유효성 검사 중 오류 발생: {e}")
            raise 

        self.class_names = [
            "forklift-vertical",
            "forklift-left",
            "forklift-right",
            "forklift-horizontal",
            "forklift-vertical(cam2)",
            "forklift-left(cam2)",
            "forklift-right(cam2)",
            "forklift-horizontal(cam2)",
            "person",
            "object",
            "object"
        ]

    @staticmethod
    def sigmoid(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return torch.sigmoid(x)

    def preprocess(self, frame):
        # 모델의 입력 크기 (예: 640x640)
        input_h, input_w = self.input_height, self.input_width # 이제 정수 값이 할당됨

        # 원본 이미지 크기
        orig_h, orig_w = frame.shape[:2] # 720, 1280

        # 비율 계산 (가로/세로 중 더 큰 쪽을 기준으로 축소)
        scale = min(float(input_w) / float(orig_w), float(input_h) / float(orig_h))
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 리사이즈
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 패딩 추가
        pad_w = input_w - new_w
        pad_h = input_h - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # HWC -> CHW
        img_input = img_padded.transpose(2, 0, 1)  

        # 정규화: 0-255 -> 0.0-1.0 범위로 변환
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0
        
        return img_input, scale, (top, left) # 패딩의 top, left 값도 반환

    def detect_objects(self, frame, conf_thres=0.4, iou_thres=0.4):
        img_input, scale, (pad_top, pad_left) = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: img_input})
        
        # process_output에 스케일과 패딩 정보 전달
        # boxes, confs, masks, cls_ids = self.process_output(outputs[0], outputs[1], conf_thres, iou_thres, scale, (pad_top, pad_left))
        boxes, scores_max, labels, seg_masks = self.process_output(
                                                                        outputs[0], 
                                                                        outputs[1],
                                                                        conf_thres,
                                                                        iou_thres,
                                                                        640
                                                                    )

        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(conf_thres)
        # print(iou_thres)
        # if labels is not None:
        #     print('output', outputs[0].shape)
        #     print(boxes)
        #     print(labels)
        
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = scores_max[i]
            cls_id = labels[i]
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            mask = seg_masks[i]

            # 마스크에서 폴리곤 추출
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for cnt in contours:
                polygon = [(int(x), int(y)) for [[x, y]] in cnt]
                if len(polygon) > 2:
                    polygons.append(polygon)

            detections.append({
                'box': (float(x1), float(y1), float(x2), float(y2)),
                'conf': float(conf),
                'class_name': class_name,
                'polygons': polygons
            })
            # print(class_name)
        return detections

    
    # def process_output(self, output0, output1, conf_threshold=0.005, iou_threshold=0.5, input_size=640):
    #     """
    #     output0: (1, 47, 8400)
    #     output1: (1, 32, 160, 160)
    #     """
    #     # conf_threshold = self.conf_thres
    #     # iou_threshold = self.iou_thres
        
    #     pred = output0.squeeze(0)  # shape: (47, 8400)
    #     # pred = pred.cpu() 
    #     mask_features = output1.squeeze(0)  # shape: (32, 160, 160)
    #     print('pred', pred.shape)
    #     print("pred min/max:", pred.min().item(), pred.max().item())
        
        
    #     # 분리
    #     boxes = pred[0:4, :].T  # shape: (8400, 4)
    #     print('boxes', boxes)
    #     # print('process_output_boxes', boxes)
    
    #     objectness = self.sigmoid(pred[4, :])  # shape: (8400,)
    #     class_scores = self.sigmoid(pred[5:15, :].T)  # shape: (8400, 10)
    #     # objectness = pred[4, :]  # shape: (8400,)
    #     # class_scores = pred[5:15, :].T  # shape: (8400, 10)
    #     mask_coeffs = pred[15:, :].T  # shape: (8400, 32)


    #     print('process_output_boxes', boxes.shape)
    #     print('objectness shape:', objectness.shape)
    #     print('objectness sample:', objectness[:5])

    #     print('class_scores shape:', class_scores.shape)
    #     print('class_scores sample:', class_scores[:2])

    #     print('objectness[:, None] shape:', objectness[:, None].shape)

    #     scores = objectness[:, None] * class_scores
    #     print('scores shape:', scores.shape)
    #     print('scores sample:', scores[:2])
        
    #     print('class_scores min, max:', class_scores.min().item(), class_scores.max().item())
    #     print('objectness min, max:', objectness.min().item(), objectness.max().item())

        

    #     # 클래스 확률과 objectness 곱하기
    #     scores = objectness[:, None] * class_scores  # shape: (8400, 10)
    #     scores_max, labels = scores.max(dim=1)  # PyTorch 방식


    #     # 필터링
    #     mask = scores_max > conf_threshold
    #     boxes = boxes[mask]
    #     mask_coeffs = mask_coeffs[mask]
    #     scores_max = scores_max[mask]
    #     labels = labels[mask]

    #     # xywh → xyxy 변환
    #     boxes_xyxy = np.zeros_like(boxes)
    #     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    #     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    #     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    #     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        
    #     # print("objectness 평균:", objectness.mean().item())
    #     # print("class_scores 평균:", class_scores.mean().item())

    #     # print('01', boxes[:, 0], boxes[:, 2])
    #     print('xyxy', boxes_xyxy)
        
    #     # NMS
    #     keep = torchvision.ops.nms(torch.tensor(boxes_xyxy), torch.tensor(scores_max), iou_threshold)
    #     boxes_xyxy = boxes_xyxy[keep]
    #     scores_max = scores_max[keep]
    #     labels = labels[keep]
    #     mask_coeffs = mask_coeffs[keep]

    #     # 마스크 생성
    #     seg_masks = []
    #     for coeff in mask_coeffs:
    #         # 마스크 벡터 (32,) × feature map (32, H, W) → (H, W)
    #         mask = np.tensordot(coeff, mask_features, axes=(0, 0))  # shape: (160, 160)
    #         mask = self.sigmoid(mask)
    #         mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    #         seg_masks.append(mask)

    #     return boxes_xyxy, scores_max, labels, seg_masks
    
    
    def process_output(self, output0, output1, conf_threshold=0.005, iou_threshold=0.5, input_size=640):
        pred = output0.squeeze(0)
        mask_features = output1.squeeze(0)

        boxes = pred[0:4, :].T  # (8400, 4), numpy 변환
        objectness = self.sigmoid(pred[4, :])  # tensor
        class_scores = self.sigmoid(pred[5:15, :].T)  # tensor
        mask_coeffs = pred[15:, :].T  # tensor
        
        print("pred min/max:", pred.min().item(), pred.max().item())
        
        print("Raw objectness logits:", pred[4, :10])
        print("Raw class logits:", pred[5:15, :10])
        
        print('process_output_boxes', boxes.shape)
        print('objectness shape:', objectness.shape)
        print('objectness sample:', objectness[:5])

        print('class_scores shape:', class_scores.shape)
        print('class_scores sample:', class_scores[:2])

        print('objectness[:, None] shape:', objectness[:, None].shape)

        scores = objectness[:, None] * class_scores
        print('scores shape:', scores.shape)
        print('scores sample:', scores[:2])
        
        print('class_scores min, max:', class_scores.min().item(), class_scores.max().item())
        print('objectness min, max:', objectness.min().item(), objectness.max().item())

    
            
            
            
        scores = objectness[:, None] * class_scores
        scores_max, labels = scores.max(dim=1)

        mask = scores_max > conf_threshold
        boxes = boxes[mask]  # boolean indexing numpy
        mask_coeffs = mask_coeffs[mask]
        scores_max = scores_max[mask]
        labels = labels[mask]

        # xywh -> xyxy 변환
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_tensor, scores_max, iou_threshold)

        boxes_xyxy = boxes_tensor[keep]
        scores_max = scores_max[keep]
        labels = labels[keep]
        mask_coeffs = mask_coeffs[keep]

        # numpy용 sigmoid
        def sigmoid_np(x):
            return 1 / (1 + np.exp(-x))

        seg_masks = []
        mask_features_np = mask_features

        for coeff in mask_coeffs:
            coeff_np = coeff
            mask = np.tensordot(coeff_np, mask_features_np, axes=(0, 0))
            mask = sigmoid_np(mask)
            mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            seg_masks.append(mask)

        return boxes_xyxy, scores_max, labels, seg_masks