import cv2
import numpy as np
import onnxruntime as ort
import time

class ONNXDetector:
    def __init__(self, onnx_model_path, input_size=(640, 384), conf_threshold=0.2, iou_threshold=0.3):
        """
        새로운 ONNX 객체 탐지 모델 초기화
        
        Args:
            onnx_model_path: ONNX 모델 파일 경로
            input_size: 모델 입력 크기 (width, height)
            conf_threshold: 신뢰도 임계값 (낮춤)
            iou_threshold: IoU 임계값 (NMS용, 강화)
        """
        self.onnx_model_path = onnx_model_path
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_width, self.input_height = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 수정된 6개 클래스 (공백 클래스 제거)
        self.class_names = [
            'forklift-right',    # 0
            'forklift-left',        # 1  
            'forklift-horizontal',       # 2
            'person',  # 3
            'forklift-vertical',               # 4
            'object',
            ''# 5
        ]
        
        print(f"✅ ONNX 모델 로드 완료: {onnx_model_path}")
        print(f"   입력 이름: {self.input_name}")
        print(f"   출력 이름: {self.output_name}")
        print(f"   입력 크기: {self.input_width}x{self.input_height}")
        print(f"   클래스 수: {len(self.class_names)}")
        print(f"   신뢰도 임계값: {self.conf_threshold}")
        print(f"   IoU 임계값: {self.iou_threshold}")

    def preprocess(self, image):
        """
        이미지 전처리: 리사이즈 + 패딩 + 정규화
        수정된 전처리 - ONNX 모델과 정확히 일치
        """
        original_h, original_w = image.shape[:2]
        
        print(f"🔍 전처리:")
        print(f"   원본 크기: {original_w}x{original_h}")
        
        # 비율 유지하면서 리사이즈
        scale = min(self.input_width / original_w, self.input_height / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        print(f"   리사이즈: {new_w}x{new_h}, 스케일: {scale:.3f}")

        # 패딩 추가 (중앙 정렬)
        top_pad = (self.input_height - new_h) // 2
        bottom_pad = self.input_height - new_h - top_pad
        left_pad = (self.input_width - new_w) // 2
        right_pad = self.input_width - new_w - left_pad

        padded_image = cv2.copyMakeBorder(
            resized_image,
            top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        print(f"   패딩: 상{top_pad} 하{bottom_pad} 좌{left_pad} 우{right_pad}")
        print(f"   패딩 후: {padded_image.shape}")

        # 정규화 및 차원 변환 (수정됨!)
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))    # HWC → CHW (수정!)
        input_tensor = np.expand_dims(input_tensor, axis=0)     # 배치 차원 추가

        print(f"   최종 텐서: {input_tensor.shape}")
        return input_tensor, scale, (top_pad, left_pad)

    def postprocess(self, output, original_width, original_height, scale, padding):
        """
        YOLO 출력 후처리: 향상된 디버깅 및 처리
        """
        pred = output.squeeze(0)  # (11, 5040) → 배치 차원 제거
        
        print(f"🔍 후처리:")
        print(f"   예측 출력: {pred.shape}")
        
        # YOLO 출력 파싱
        boxes_raw = pred[0:4, :].T      # (5040, 4) - cx, cy, w, h
        objectness_raw = pred[4, :]     # (5040,) - 객체성 점수 (raw)
        class_scores_raw = pred[5:11, :].T  # (5040, 6) - 클래스 점수 (raw)
        
        print(f"   박스: {boxes_raw.shape}")
        print(f"   객체성(raw): {objectness_raw.shape}")
        print(f"   클래스(raw): {class_scores_raw.shape}")

        # 시그모이드 적용 (중요!)
        objectness = self.sigmoid(objectness_raw)
        class_scores = self.sigmoid(class_scores_raw)
        
        print(f"   객체성 범위: {objectness.min():.3f} ~ {objectness.max():.3f}")
        print(f"   클래스 점수 범위: {class_scores.min():.3f} ~ {class_scores.max():.3f}")

        # 최종 신뢰도 계산
        scores = objectness[:, np.newaxis] * class_scores
        scores_max = np.max(scores, axis=1)
        labels = np.argmax(scores, axis=1)
        
        print(f"   최종 점수 범위: {scores_max.min():.3f} ~ {scores_max.max():.3f}")
        print(f"   상위 10개 점수: {np.sort(scores_max)[-10:]}")

        # 임계값 필터링
        keep_mask = scores_max > self.conf_threshold
        print(f"   임계값 {self.conf_threshold} 이상: {keep_mask.sum()}개")
        
        if keep_mask.sum() == 0:
            print("   ❌ 임계값 통과한 객체 없음")
            return np.array([]), np.array([]), np.array([])

        boxes_filtered = boxes_raw[keep_mask]
        scores_filtered = scores_max[keep_mask]
        labels_filtered = labels[keep_mask]

        # 중심점 형식 → 좌상단/우하단 형식 변환
        boxes_xyxy = np.copy(boxes_filtered)
        boxes_xyxy[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3] / 2  # y2
        
        # NMS용 형식 변환 (x, y, w, h)
        boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes_xyxy])
        
        # NMS 적용 (강화된 설정)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(), 
            scores_filtered.tolist(), 
            self.conf_threshold, 
            self.iou_threshold  # 더 강한 NMS
        )
        
        if len(indices) == 0:
            print("   ❌ NMS 후 남은 객체 없음")
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten()
        
        boxes_final = boxes_xyxy[indices]
        scores_final = scores_filtered[indices]
        labels_final = labels_filtered[indices]

        print(f"   NMS 후 최종: {len(boxes_final)}개")

        # 패딩 및 스케일링 역변환
        top_pad, left_pad = padding

        # 패딩 제거
        boxes_final[:, 0] -= left_pad
        boxes_final[:, 1] -= top_pad
        boxes_final[:, 2] -= left_pad
        boxes_final[:, 3] -= top_pad

        # 스케일링 역변환
        boxes_final /= scale

        # 이미지 경계 내로 클리핑
        boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)

        # 최소 크기 보장
        boxes_final[:, 2] = np.maximum(boxes_final[:, 2], boxes_final[:, 0] + 1)
        boxes_final[:, 3] = np.maximum(boxes_final[:, 3], boxes_final[:, 1] + 1)

        return boxes_final, scores_final, labels_final

    def sigmoid(self, x):
        """시그모이드 함수 (overflow 방지)"""
        x = np.clip(x, -500, 500)  # overflow 방지
        return 1 / (1 + np.exp(-x))

    def predict(self, image_path):
        """
        이미지에 대한 객체 탐지 수행
        """
        # 이미지 로드
        original_image_bgr = cv2.imread(image_path)
        if original_image_bgr is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
            return np.array([]), np.array([]), np.array([]), None

        # BGR → RGB 변환
        original_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        print(f"🖼️ 이미지 처리: {image_path}")
        print(f"   원본 크기: {original_width}x{original_height}")

        # 전처리
        input_tensor, scale, padding = self.preprocess(original_image)

        # ONNX 추론
        print(f"🔍 ONNX 추론 수행...")
        start_time = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000
        print(f"   추론 시간: {inference_time:.2f}ms")
        
        # 후처리
        boxes, scores, labels = self.postprocess(
            outputs[0], original_width, original_height, scale, padding
        )
        
        return boxes, scores, labels, original_image_bgr

    def visualize(self, image_bgr, boxes, scores, labels, save_path=None):
        """
        탐지 결과 시각화 (향상된 버전)
        """
        display_image = image_bgr.copy()

        # 클래스별 색상 정의
        colors = [
            (255, 0, 0),    # forklift-vertical - 빨강
            (0, 255, 0),    # forklift-left - 초록
            (0, 0, 255),    # forklift-right - 파랑
            (255, 255, 0),  # forklift-horizontal - 노랑
            (255, 0, 255),  # person - 마젠타
            (0, 255, 255),  # object - 시안
        ]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            label = labels[i]
            score = scores[i]
            
            class_name = self.class_names[label] if label < len(self.class_names) else f"Class {label}"
            color = colors[label] if label < len(colors) else (255, 255, 255)

            # 바운딩 박스 그리기 (두꺼운 선)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 3)
            
            # 라벨 텍스트
            text = f"{class_name}: {score:.3f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 텍스트 배경
            cv2.rectangle(display_image, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), color, -1)
            
            # 텍스트
            cv2.putText(display_image, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 결과 표시 또는 저장
        if save_path:
            cv2.imwrite(save_path, display_image)
            print(f"💾 결과 이미지 저장: {save_path}")
        else:
            cv2.imshow("ONNX Detection Result", display_image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return display_image

def main():
    """
    메인 테스트 함수
    """
    import time
    
    # ========================================
    # 🔧 새로운 ONNX 모델 설정
    # ========================================
    
    ONNX_MODEL_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\9.Quantization\quantized_models\model_static_int8.onnx"
    TEST_IMAGE_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106\frame_000001.jpg"
    
    # 모델 입력 크기 (ONNX 모델에 맞게)
    INPUT_SIZE = (320, 192)  # (width, height)
    
    # 탐지 임계값 (낮춰서 더 많은 후보 확인)
    CONF_THRESHOLD = 0.3     # 신뢰도 임계값 (낮춤)
    IOU_THRESHOLD = 0.1      # NMS IoU 임계값 (강화)
    
    # ========================================
    
    print("🚀 새로운 ONNX 객체 탐지 테스트 시작")
    print("=" * 60)
    
    # 탐지기 초기화
    detector = ONNXDetector(
        onnx_model_path=ONNX_MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    # 객체 탐지 수행
    print(f"\n🔍 이미지 탐지 시작:")
    boxes, scores, labels, original_image = detector.predict(TEST_IMAGE_PATH)
    
    # 결과 출력
    if len(boxes) > 0:
        print(f"\n🎯 탐지 결과: {len(boxes)}개 객체 발견")
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = detector.class_names[labels[i]]
            score = scores[i]
            print(f"   {i+1}. {class_name}: {score:.3f} - 박스: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # 결과 시각화
        detector.visualize(original_image, boxes, scores, labels)
    else:
        print("\n❌ 탐지된 객체가 없습니다.")
        print("🔧 문제 해결 방법:")
        print("1. 신뢰도 임계값을 더 낮춰보기 (0.1 or 0.05)")
        print("2. 시그모이드 적용이 올바른지 확인")
        print("3. 원본 PT 모델과 ONNX 출력 직접 비교")

if __name__ == "__main__":
    main()