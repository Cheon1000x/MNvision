import cv2
import numpy as np
import onnxruntime as ort
import torch
# import torchvision
import onnx
import os
import torch.nn.functional as F
""" 
1280 720으로 입력받음.
736으로 패딩되기때문에 위아래로 8패딩해줘야함

"""


class Detector:
    # def __init__(self, onnx_path="resources/models/yolov8_custom_fixed.onnx", conf_threshold=0.6, iou_threshold=0.45):
    # def __init__(self, onnx_path="resources/models/yolov8_custom_fixed_0603.onnx", conf_threshold=0.6, iou_threshold=0.45):
    def __init__(self, onnx_path="resources/models/yolov8_custom_fixed_v2.onnx", conf_threshold=0.6, iou_threshold=0.45):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # ONNX 모델의 입력 이름 가져오기
        self.input_name = self.session.get_inputs()[0].name
        
        self.input_height = 192
        self.input_width = 320
        self.padding_color = 114 
        
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
            "forklift-right",
            "forklift-left",
            "forklift-vertical",
            "forklift-horizontal",
            "person",
            "object",
        ]
        self.num_classes = len(self.class_names)

    @staticmethod
    def sigmoid(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return torch.sigmoid(x)

    def preprocess(self, frame):
        ### 640사이즈로 변환 후 상하패딩추가하여넣음
        # 원본 프레임의 높이, 너비
        orig_h, orig_w = frame.shape[:2] # 예: 720, 1280 (16:9)

        # 1. BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. 16:9 비율을 유지하며 640xN 으로 리사이즈 (가로 640에 맞춤)
        target_width = self.input_width # 640
        target_height = int(target_width / (orig_w / orig_h)) # 640 / (1280/720) = 640 / 1.777 = 360
        
        # 리사이즈
        resized_frame = cv2.resize(rgb_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        scale = target_width / orig_w
        
        # 3. 상하 패딩 추가하여 640x640 만들기
        # 최종 목표 높이: 640
        # 현재 리사이즈된 높이: target_height (예: 360)
        # 필요한 패딩 양: 640 - 360 = 280
        
        # 상단과 하단에 균등하게 패딩을 분배합니다.
        pad_top = (self.input_height - target_height) // 2
        pad_bottom = self.input_height - target_height - pad_top
        # print(pad_top, pad_bottom)
        
        # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
        padded_img = cv2.copyMakeBorder(
            resized_frame,
            pad_top,
            pad_bottom,
            0,  # 좌측 패딩 없음
            0,  # 우측 패딩 없음
            cv2.BORDER_CONSTANT, # 단색으로 채우기
            value=(self.padding_color, self.padding_color, self.padding_color) # RGB 색상 (114, 114, 114)
        )

        # 4. HWC(높이, 너비, 채널) -> CHW(채널, 높이, 너비) 변환
        # ONNX 런타임에 입력할 텐서 형태 (C, H, W)로 변경합니다.
        img_input = padded_img.transpose(2, 0, 1) # (640, 640, 3) -> (3, 640, 640)

        # 5. 배치 차원 추가 및 정규화
        # ONNX 모델은 보통 (Batch, Channel, Height, Width) 형태를 요구합니다.
        # 또한 0-255 범위의 픽셀 값을 0.0-1.0 범위로 정규화합니다.
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

        # print(img_input.shape, scale, (pad_top, pad_bottom))
        return img_input, scale, (pad_top, pad_bottom)


    def detect_objects(self, frame, conf_thres=0.00001, iou_thres=0.4):
        img_input, scale, (pad_top, pad_bottom) = self.preprocess(frame) 
        ## img_input  (1, 3, 360, 640)
        # img_input = img_input.transpose(0, 1, 3, 2)
        # print('check', img_input.shape)
        
        original_h, original_w = frame.shape[:2] 
        
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # process_output에 스케일과 패딩 정보 전달
        boxes, scores, labels, masks = self.postprocess(
            outputs, original_height=original_h, original_width=original_w, scale=scale, padding=(pad_top, pad_bottom)
        )

        # print('out0',outputs[0].shape)
        # print('out1',outputs[1].shape)
       
        # if labels is not None:
        #     print('output', outputs[0].shape)
        #     print(boxes)
        #     print(labels)
        
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            cls_id = labels[i]
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            # mask = masks[i]

            # 마스크에서 폴리곤 추출
            # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            # for cnt in contours:
            #     polygon = [(int(x), int(y)) for [[x, y]] in cnt]
            #     if len(polygon) > 2:
            #         polygons.append(polygon)

            detections.append({
                'box': (float(x1), float(y1), float(x2), float(y2)),
                'conf': float(conf),
                'class_name': class_name,
                'polygons': polygons
            })
            # print(class_name)
        return detections

   
    def postprocess(self, outputs, original_width, original_height, scale, padding):
        # output0: (1, 47, 8400) -> (47, 8400)
        # output1: (1, 32, H, W) -> (32, H, W)

        pred_raw = outputs[0].squeeze(0)
        # print(f"DEBUG: pred_raw shape (after squeeze): {pred_raw.shape}")
        # print(f"DEBUG: mask_features shape: {mask_features.shape}")
        # DEBUG: pred_raw shape (after squeeze): (47, 19320)
        # DEBUG: mask_features shape: (32, 184, 320)
        
        pred = pred_raw.T 
        
        # 박스 좌표 (4개)
        boxes_raw = pred[:, 0:4] # (N_proposals, 4) - xywh

         # YOLOv8은 보통 objectness를 클래스 스코어와 통합하여 출력합니다.
        class_scores_logits_raw = pred[:, 4 : 4 + self.num_classes] # (N_proposals, num_classes)
        mask_coeffs_raw = pred[:, 4 + self.num_classes :]       # (N_proposals, num_mask_coefficients)

        # print(f"DEBUG: boxes_raw shape: {boxes_raw.shape}")
        # print(f"DEBUG: class_scores_logits_raw shape: {class_scores_logits_raw.shape}")
        # print(f"DEBUG: mask_coeffs_raw shape: {mask_coeffs_raw.shape}")

        # 로짓에 시그모이드 적용
        class_scores_np = 1 / (1 + np.exp(-class_scores_logits_raw)) # 시그모이드 직접 구현
        
        # 최대 스코어 및 레이블 추출
        scores_max = np.max(class_scores_np, axis=1)
        labels = np.argmax(class_scores_np, axis=1)
        
        # print(f"DEBUG: Scores max (after sigmoid) min/max: {scores_max.min()}/{scores_max.max()}")
        # print(f"DEBUG: Labels (first 5): {labels[:5]}") # 상위 5개 레이블 샘플
        
        keep_mask = scores_max > self.conf_threshold
        if keep_mask.sum() == 0:
            print(f"DEBUG: No objects after confidence threshold filtering (scores_max too low or conf_threshold {self.conf_threshold} too high).")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        boxes_filtered = boxes_raw[keep_mask]
        scores_filtered = scores_max[keep_mask]
        labels_filtered = labels[keep_mask]
        mask_coeffs_filtered = mask_coeffs_raw[keep_mask]
        
        # NMS를 위해 xywh를 xyxy로 변환
        boxes_xyxy = np.copy(boxes_filtered)
        boxes_xyxy[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3] / 2
        
        # NMSBoxes는 x,y,w,h 형태의 박스를 기대합니다.
        boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes_xyxy])
        
        indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores_filtered.tolist(), self.conf_threshold, self.iou_threshold)
        
        if len(indices) == 0:
            print("DEBUG: No objects after NMS.")
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        indices = indices.flatten()

        boxes_final = boxes_xyxy[indices]
        scores_final = scores_filtered[indices]
        labels_final = labels_filtered[indices]
        mask_coeffs_final = mask_coeffs_filtered[indices]
        
        
       # 패딩 및 스케일 역변환 
        pad_top, _ = padding # padding은 (pad_top, pad_bottom) 형태로 전달됨

        # 1. 패딩 역변환 (y 좌표만 해당)
        # y1, y2 좌표에서 상단 패딩만큼 빼줍니다.
        boxes_final[:, 1] -= pad_top 
        boxes_final[:, 3] -= pad_top

        # 2. 스케일 역변환 (x, y 좌표 모두 해당)
        # preprocess에서 적용된 스케일만큼 나눠줍니다. (scale_ratio는 640/orig_w)
        boxes_final /= scale 

        # 3. 클리핑하여 원본 이미지 경계 내에 있도록 합니다.
        boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)
        
        # 유효한 박스 크기 보장 (최소 1픽셀)
        boxes_final[:, 2] = np.maximum(boxes_final[:, 2], boxes_final[:, 0] + 1)
        boxes_final[:, 3] = np.maximum(boxes_final[:, 3], boxes_final[:, 1] + 1)

        ## seg 모델 잔재. 사용하진 않으나 다른 입력위치도 수정해야 하므로 유지
        final_masks = []
        
        return boxes_final, scores_final, labels_final, final_masks
            