import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import onnx
import os
import torch.nn.functional as F
""" 
1280 720으로 입력받음.
736으로 패딩되기때문에 위아래로 8패딩해줘야함

"""


class Detector:
    # def __init__(self, onnx_path="resources/models/ver1_test_13q.onnx", conf_threshold=0.65, iou_threshold=0.45):
    def __init__(self, onnx_path="resources/models/ver1_test_13_640.onnx", conf_threshold=0.65, iou_threshold=0.45):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_custom_opset20.onnx"):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
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
        self.padding_color = 114 
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
        self.num_classes = len(self.class_names)

    @staticmethod
    def sigmoid(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return torch.sigmoid(x)

    # def preprocess(self, frame):
    #     # input_h, input_w = self.input_height, self.input_width  # 보통 1280, 736
    #     # orig_h, orig_w = frame.shape[:2]  # 720, 1280

    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # # 720->736 패딩 8 추가
    #     # # 이미지 패딩 처리 및 다시 넘파이형태로 변환
    #     # padded_img = (F.pad(torch.Tensor(frame), pad=(0, 0, 8, 8), mode='constant', value=114)).numpy()  # YOLOv8 default padding color: 114
    #     input_h, input_w = self.input_height, self.input_width  # 예: 736, 1280
    #     orig_h, orig_w = frame.shape[:2]  # 720, 1280

    #     # frame을 tensor로 변환 + 차원 순서 변경 (HWC->CHW)
    #     frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # (3, H, W)

    #     # 패딩: pad=(left, right, top, bottom)
    #     # 여기서 좌우 패딩 0, 상하 패딩 8씩 추가
    #     padded_tensor = F.pad(frame_tensor, pad=(0, 0, 8, 8), mode='constant', value=114)

    #     # (C, H, W) → (H, W, C) 다시 넘파이로 변환하고 싶으면
    #     padded_img = padded_tensor.permute(1, 2, 0).numpy()

    #     # HWC → CHW, 정규화, 배치 차원 추가
    #     img_input = padded_img.transpose(2, 0, 1)
    #     img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

    #     return img_input
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
        
        # 3. 상하 패딩 추가하여 640x640 만들기
        # 최종 목표 높이: 640
        # 현재 리사이즈된 높이: target_height (예: 360)
        # 필요한 패딩 양: 640 - 360 = 280
        
        # 상단과 하단에 균등하게 패딩을 분배합니다.
        pad_top = (self.input_height - target_height) // 2
        pad_bottom = self.input_height - target_height - pad_top
        
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

        return img_input, (0.5, 8/9), (pad_top, pad_bottom)


    def detect_objects(self, frame, conf_thres=0.00001, iou_thres=0.4):
        img_input = self.preprocess(frame) 

        outputs = self.session.run(None, {self.input_name: img_input})
        
        # process_output에 스케일과 패딩 정보 전달
        boxes, scores, labels, masks = self.postprocess(
            outputs
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
            mask = masks[i]

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

   
    # def process_output(self, output0, output1, conf_threshold=0.00005, iou_threshold=0.5):
    # def postprocess(self, outputs, original_width, original_height, scale, padding):
    def postprocess(self, outputs, original_width=1280, original_height=736):
        # output0: (1, 47, 8400) -> (47, 8400)
        # output1: (1, 32, H, W) -> (32, H, W)

        pred_raw = outputs[0].squeeze(0)
        mask_features = outputs[1].squeeze(0)

        # print(f"DEBUG: pred_raw shape (after squeeze): {pred_raw.shape}")
        # print(f"DEBUG: mask_features shape: {mask_features.shape}")
        # DEBUG: pred_raw shape (after squeeze): (47, 19320)
        # DEBUG: mask_features shape: (32, 184, 320)
        
        pred = pred_raw.T 
        # pred는 (N_proposals, N_features) 형태
        # N_features = 4 (bbox) + num_classes (class_scores) + num_mask_coefficients (mask_coeffs)
        # pred[:, 0:4] = x, y, w, h         4
        # pred[:, 4 : 4 + num_classes] = class scores (objectness 포함)         14
        # pred[:, 4 + num_classes : ] = mask coefficients       14
        
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
        
        ### 현재 preprocess()에서 패드와 스케일정보를 반환하지 않음.
        ### 모델 변환시 요구 입력값이 1280*736사이즈로 나왔기때문에
        ### 패딩및 스케일 처리하지않고 상하패딩 넣은 채로 투입
        # # 패딩 및 스케일 역변환
        # top_pad, left_pad = padding

        # boxes_final[:, 0] -= left_pad
        # boxes_final[:, 1] -= top_pad
        # boxes_final[:, 2] -= left_pad
        # boxes_final[:, 3] -= top_pad

        # boxes_final /= scale

        # # 클리핑하여 이미지 경계 내에 있도록 합니다.
        # boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        # boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        # boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        # boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)
        
        # 유효한 박스 크기 보장 (최소 1픽셀)
        boxes_final[:, 2] = np.maximum(boxes_final[:, 2], boxes_final[:, 0] + 1)
        boxes_final[:, 3] = np.maximum(boxes_final[:, 3], boxes_final[:, 1] + 1)

        final_masks = []
        
        # 시그모이드 함수 정의 (Postprocess 내부에 정의)
        def sigmoid_np(x):
            return 1 / (1 + np.exp(-x))
        
        for i in range(mask_coeffs_final.shape[0]):
            coeff_np = mask_coeffs_final[i, :]
            
            # DEBUG: coeff_np의 실제 shape 확인
            # print(f'DEBUG: coeff_np shape for mask {i}: {coeff_np.shape}') # 너무 많은 출력으로 주석 처리

            # mask_features와 coeff_np의 차원 일치 확인 (중요)
            if coeff_np.shape[0] != mask_features.shape[0]:
                print(f"ERROR: Mask coefficient count mismatch! Expected {mask_features.shape[0]}, got {coeff_np.shape[0]}.")
                final_masks.append(np.zeros((original_height, original_width), dtype=np.uint8))
                continue

            # (N_coeffs,) @ (N_coeffs, H_mask * W_mask) -> (H_mask * W_mask)
            mask_logit = np.dot(coeff_np, mask_features.reshape(mask_features.shape[0], -1)).reshape(mask_features.shape[1], mask_features.shape[2])
            
            mask_prob_raw = sigmoid_np(mask_logit)
            
            # 마스크 크기를 모델 입력 크기로 리사이즈 (1280x736)
            mask_prob_input_size = cv2.resize(mask_prob_raw, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

            ## 패딩제거 및 스케일을 적용하지 않았으므로 이쪽 코드도 사용하지 않음.
            # # 원본 이미지 비율에 맞춰 패딩 제거
            # actual_scaled_h = int(original_height * scale)
            # actual_scaled_w = int(original_width * scale)

            # cropped_mask_padded_scale = mask_prob_input_size[top_pad : top_pad + actual_scaled_h,
            #                                                   left_pad : left_pad + actual_scaled_w]

            # 최종 마스크를 원본 이미지 크기로 리사이즈
            final_mask_for_object = cv2.resize(mask_prob_input_size, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            final_mask_for_object_binary = (final_mask_for_object > 0.5).astype(np.uint8) * 255

            final_masks.append(final_mask_for_object_binary)

        return boxes_final, scores_final, labels_final, final_masks
            