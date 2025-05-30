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
    def __init__(self, onnx_path="resources/models/ver1_test_13nq.onnx"):
    # def __init__(self, onnx_path="resources/models/yolov8_seg_custom_opset20.onnx"):
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
        self.input_height = 1280
        self.input_width = 736
        
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
        # input_h, input_w = self.input_height, self.input_width  # 보통 1280, 736
        # orig_h, orig_w = frame.shape[:2]  # 720, 1280

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # # 720->736 패딩 8 추가
        # # 이미지 패딩 처리 및 다시 넘파이형태로 변환
        # padded_img = (F.pad(torch.Tensor(frame), pad=(0, 0, 8, 8), mode='constant', value=114)).numpy()  # YOLOv8 default padding color: 114
        input_h, input_w = self.input_height, self.input_width  # 예: 736, 1280
        orig_h, orig_w = frame.shape[:2]  # 720, 1280

        # frame을 tensor로 변환 + 차원 순서 변경 (HWC->CHW)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # (3, H, W)

        # 패딩: pad=(left, right, top, bottom)
        # 여기서 좌우 패딩 0, 상하 패딩 8씩 추가
        padded_tensor = F.pad(frame_tensor, pad=(0, 0, 8, 8), mode='constant', value=114)

        # (C, H, W) → (H, W, C) 다시 넘파이로 변환하고 싶으면
        padded_img = padded_tensor.permute(1, 2, 0).numpy()

        # HWC → CHW, 정규화, 배치 차원 추가
        img_input = padded_img.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

        return img_input


    def detect_objects(self, frame, conf_thres=0.00001, iou_thres=0.4):
        img_input = self.preprocess(frame) 

        outputs = self.session.run(None, {self.input_name: img_input})
        
        # process_output에 스케일과 패딩 정보 전달
        # boxes, confs, masks, cls_ids = self.process_output(outputs[0], outputs[1], conf_thres, iou_thres, scale, (pad_top, pad_left))
        boxes, scores_max, labels, seg_masks = self.process_output(
                                                                        outputs[0], 
                                                                        outputs[1],
                                                                        conf_thres,
                                                                        iou_thres
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

   
    def process_output(self, output0, output1, conf_threshold=0.00005, iou_threshold=0.5):
        # output0: (1, 47, 8400) -> (47, 8400)
        # output1: (1, 32, H, W) -> (32, H, W)

        pred = output0.squeeze(0)
        mask_features = output1.squeeze(0)

        # 박스 좌표 (4개)
        boxes = pred[0:4, :].T  # (8400, 4)

        # 객체 신뢰도 (1개) - 이미 sigmoid 적용된 값이라고 가정 (0~1 범위)
        objectness = pred[4, :]  # tensor

        # 클래스 신뢰도 (10개) - 이미 sigmoid 적용된 값이라고 가정 (0~1 범위)
        # 47 = 4(bbox) + 1(obj) + 10(cls) + 32(mask_coeffs)
        # class_scores는 pred[5:15]까지 (10개)
        class_scores = pred[5:15, :].T  # (8400, 10)

        # 32 마스크
        mask_coeffs_raw = pred[15:, :].T  # (8400, 32 또는 33)

        print("pred min/max:", pred.min().item(), pred.max().item())
        print("Objectness values (no sigmoid applied here):", objectness[:10])
        print("Class scores values (no sigmoid applied here):", class_scores[:2])

        # 이전에 "Raw objectness logits"로 출력했던 값이 이제 실제 objectness score여야 합니다.
        print('objectness min, max (after parsing, before combining):', objectness.min().item(), objectness.max().item())
        print('class_scores min, max (after parsing, before combining):', class_scores.min().item(), class_scores.max().item())

        # 최종 신뢰도 점수 = 객체 존재 확률 * 클래스 확률
        scores = objectness[:, np.newaxis] * class_scores # (8400, 10) NumPy 배열
        
        # NumPy 배열의 max 함수는 'axis' 인자를 사용합니다.
        scores_max, labels = scores.max(axis=1), scores.argmax(axis=1) # 각 객체의 최고 신뢰도 점수와 해당 클래스 인덱스 (NumPy)

        # NMS를 위한 필터링 (conf_threshold 적용)
        mask = scores_max > conf_threshold
        
        # 필터링 후 남은 객체가 없으면 빈 배열 반환
        if mask.sum() == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        boxes = boxes[mask] 
        scores_max = scores_max[mask]
        labels = labels[mask]
        mask_coeffs_filtered = mask_coeffs_raw[mask] # 필터링된 마스크 계수

        # 마스크 계수 차원 불일치 처리 (33 -> 32)
        # 이전에 mask_coeffs_raw가 33개였고 mask_features가 32개였기 때문에 필요했습니다.
        # 현재 코드에서는 mask_coeffs_filtered의 두 번째 차원을 확인해야 합니다.
        if mask_coeffs_filtered.shape[1] > mask_features.shape[0]:
            print(f"경고: 필터링된 mask_coeffs ({mask_coeffs_filtered.shape[1]})와 mask_features ({mask_features.shape[0]}) 차원 불일치. mask_coeffs를 자릅니다.")
            mask_coeffs_final = mask_coeffs_filtered[:, :mask_features.shape[0]] # 33개에서 32개로 자름
        else:
            mask_coeffs_final = mask_coeffs_filtered
        
        print(f"Number of objects after conf_thres filtering: {boxes.shape[0]}") 

        # xywh -> xyxy 변환 (넘파이 배열로 변환된 boxes를 사용)
        boxes_xyxy = np.copy(boxes) # boxes가 넘파이므로 np.copy 사용 또는 torch.tensor 변환 후 작업
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # NMS (Non-Maximum Suppression)
        # boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        # keep = torchvision.ops.nms(boxes_tensor, scores_max, iou_threshold)
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(scores_max, dtype=torch.float32)  # <- 중요!

        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        # boxes_xyxy = boxes_tensor[keep].numpy() # 최종 결과는 넘파이로
        # scores_max = scores_max[keep].numpy()

        boxes_xyxy = boxes_tensor[keep].numpy() # 최종 결과는 넘파이로
        scores_max = scores_max[keep]
        labels = labels[keep]
        mask_coeffs_final = mask_coeffs_final[keep] # NMS 후 마스크 계수도 필터링

        # # 마스크 생성 (numpy 연산으로 변경)
        # seg_masks = []
        # mask_features_np = mask_features # PyTorch 텐서에서 넘파이로 변환

        # numpy용 sigmoid 함수
        def sigmoid_np(x):
            return 1 / (1 + np.exp(-x))
# --- 추가된 안전 검사: NMS 후에도 객체가 없을 경우 처리 ---
        if boxes_xyxy.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])


        seg_masks = []
        # mask_features = np.array(mask_features) # (32, H_proto, W_proto)

        H_proto, W_proto = mask_features.shape[1], mask_features.shape[2]

        for i in range(mask_coeffs_final.shape[0]):
            coeff_np = mask_coeffs_final[i]
            
            # 여기서 오류가 발생했으므로, coeff_np가 스칼라가 아닌지 확인합니다.
            # coeff_np는 (32,) 형태의 1차원 배열이어야 합니다.
            # matmul: Input operand 0 does not have enough dimensions (has 0
            # 이 오류는 coeff_np가 빈 배열이 되어 shape[0]이 0이고, 그 결과로 [i] 인덱싱 시 스칼라가 아닌
            # 어떤 비어있는 "객체"가 되어버렸을 때 발생할 수 있습니다.
            # 또는 단순히 mask_coeffs_final.shape[0]이 0이어서 루프가 돌지 않아야 하는데,
            # 어떤 이유로 루프가 돌게 되어 i가 유효하지 않은 인덱스에 접근했을 수도 있습니다.
            # 하지만 traceback은 루프 내부에서 오류가 났으니 mask_coeffs_final.shape[0]은 0이 아니었을 겁니다.
            # 따라서 mask_coeffs_final[i]가 문제인 것입니다.

            mask_logit = np.dot(coeff_np, mask_features.reshape(mask_features.shape[0], -1)).reshape(mask_features.shape[1], mask_features.shape[2])
            
            mask = self.sigmoid_np(mask_logit)

            resized_mask_to_model_input = cv2.resize(mask, 
                                                     (self.model_input_width, self.model_input_height), 
                                                     interpolation=cv2.INTER_LINEAR)
            
            final_mask = resized_mask_to_model_input[self.pad_top : self.model_input_height - self.pad_bottom, 
                                                     self.pad_left : self.model_input_width - self.pad_right]
            
            final_mask_uint8 = (final_mask > 0.5).astype(np.uint8) * 255

            seg_masks.append(final_mask_uint8)

        return boxes_xyxy, scores_max, labels, seg_masks
    
    
        # for i in range(mask_coeffs_final.shape[0]):
        #     coeff_np = mask_coeffs_final[i]
        #     # Tensordot 대신 (C, H, W)와 (C,)를 곱하여 (H, W) 생성
        #     # 마스크 계수와 프로토타입의 내적
        #     # mask_coeffs_final: (N, 32), mask_features_np: (32, 160, 160)
        #     # 결과: (N, 160, 160)
        #     mask_logit = np.dot(coeff_np, mask_features_np.reshape(mask_features_np.shape[0], -1)).reshape(mask_features_np.shape[1], mask_features_np.shape[2])
            
        #     mask = sigmoid_np(mask_logit)
        #     mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        #     seg_masks.append(mask)

        # # 결과 마스크를 이진화
        # final_masks = np.array(seg_masks) > 0.5 # 넘파이 배열로 변환 및 이진화

        # return boxes_xyxy, scores_max, final_masks, labels