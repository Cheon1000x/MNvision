"""
ONNX 모델을 위한 간편한 유틸리티
기존 PyQt 앱에서 최소한의 변경으로 ONNX 모델을 사용할 수 있게 해주는 래퍼 클래스
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Dict, Union
import time

class ONNXProcessor:
    """
    ONNX 모델을 위한 간편한 처리 클래스
    기존 YOLO 사용법과 최대한 유사하게 설계
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        ONNX 모델 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # ONNX 세션 초기화
        self.session = ort.InferenceSession(model_path)
        
        # 입출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 입력 크기 설정 (모델에서 자동 감지)
        if len(self.input_shape) == 4:  # [batch, channels, height, width]
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
        else:
            self.input_height = 640
            self.input_width = 640
        
        print(f"ONNX 모델 로드 완료: {model_path}")
        print(f"입력 크기: {self.input_width}x{self.input_height}")
        print(f"출력 개수: {len(self.output_names)}")
    
    def __call__(self, source: Union[str, np.ndarray], **kwargs) -> 'ONNXResults':
        """
        예측 수행 (YOLO 스타일 호출)
        
        Args:
            source: 이미지 경로 또는 numpy 배열
            **kwargs: 추가 설정 (conf, iou 등)
            
        Returns:
            ONNXResults 객체
        """
        # 설정 업데이트
        conf = kwargs.get('conf', self.conf_threshold)
        iou = kwargs.get('iou', self.iou_threshold)
        
        # 이미지 로드
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {source}")
        else:
            image = source.copy()
        
        # 예측 수행
        results = self._predict_single(image, conf, iou)
        
        return ONNXResults(results, image)
    
    def _predict_single(self, image: np.ndarray, conf: float, iou: float) -> Dict:
        """
        단일 이미지 예측
        """
        # 1. 전처리
        input_tensor, scale, padding = self._preprocess(image)
        
        # 2. 추론
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # 3. 후처리
        results = self._postprocess(outputs, scale, padding, image.shape[:2], conf, iou)
        results['inference_time'] = inference_time
        
        return results
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        이미지 전처리 (letterbox 적용)
        """
        original_height, original_width = image.shape[:2]
        
        # 비율 유지하며 리사이즈
        scale = min(self.input_width / original_width, self.input_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_width, new_height))
        
        # 패딩 (중앙 정렬)
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # 패딩된 이미지 생성
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # 정규화 및 차원 변경
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)    # BCHW
        
        return input_tensor, scale, (pad_x, pad_y)
    
    def _postprocess(self, outputs: List[np.ndarray], scale: float, 
                    padding: Tuple[int, int], original_shape: Tuple[int, int],
                    conf: float, iou: float) -> Dict:
        """
        후처리 (NMS, 좌표 변환 등)
        """
        # 검출 결과 파싱
        detection_output = outputs[0]  # [1, 84, 8400] 또는 [1, 8400, 84]
        
        # 차원 확인 및 조정
        if len(detection_output.shape) == 3:
            if detection_output.shape[1] > detection_output.shape[2]:
                detection_output = detection_output.transpose(0, 2, 1)
            predictions = detection_output[0]  # [8400, 84]
        else:
            predictions = detection_output
        
        # 좌표와 클래스 점수 분리
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # 최고 점수 클래스 찾기
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # 신뢰도 필터링
        mask = max_scores >= conf
        if not np.any(mask):
            return self._empty_results()
        
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        
        # 중심점 좌표를 모서리 좌표로 변환
        x_center, y_center, width, height = filtered_boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # NMS 적용
        boxes_for_nms = np.column_stack([x1, y1, x2, y2])
        keep_indices = self._nms(boxes_for_nms, filtered_scores, iou)
        
        if len(keep_indices) == 0:
            return self._empty_results()
        
        # 최종 결과
        final_boxes = boxes_for_nms[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_classes = filtered_classes[keep_indices]
        
        # 좌표 변환 (모델 좌표 -> 원본 좌표)
        final_boxes = self._convert_coordinates(final_boxes, scale, padding, original_shape)
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes,
            'count': len(final_boxes)
        }
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression
        """
        if len(boxes) == 0:
            return []
        
        # 좌표 추출
        x1, y1, x2, y2 = boxes.T
        
        # 면적 계산
        areas = (x2 - x1) * (y2 - y1)
        
        # 점수 기준 내림차순 정렬
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # 가장 높은 점수의 박스
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # 나머지 박스들과 IoU 계산
            other_indices = indices[1:]
            
            # 교집합 계산
            xx1 = np.maximum(x1[current], x1[other_indices])
            yy1 = np.maximum(y1[current], y1[other_indices])
            xx2 = np.minimum(x2[current], x2[other_indices])
            yy2 = np.minimum(y2[current], y2[other_indices])
            
            # 교집합 면적
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            
            # IoU 계산
            union = areas[current] + areas[other_indices] - intersection
            iou = intersection / union
            
            # IoU 임계값 이하인 박스들만 유지
            indices = other_indices[iou <= iou_threshold]
        
        return keep
    
    def _convert_coordinates(self, boxes: np.ndarray, scale: float, 
                           padding: Tuple[int, int], original_shape: Tuple[int, int]) -> np.ndarray:
        """
        모델 좌표를 원본 이미지 좌표로 변환
        """
        pad_x, pad_y = padding
        original_height, original_width = original_shape
        
        # 패딩 제거
        boxes[:, [0, 2]] -= pad_x  # x 좌표
        boxes[:, [1, 3]] -= pad_y  # y 좌표
        
        # 스케일 복원
        boxes /= scale
        
        # 경계 클리핑
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
        
        return boxes
    
    def _empty_results(self) -> Dict:
        """
        빈 결과 반환
        """
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'classes': np.array([]),
            'count': 0
        }


class ONNXResults:
    """
    ONNX 예측 결과를 담는 클래스 (YOLO Results와 유사한 인터페이스)
    """
    
    def __init__(self, results: Dict, original_image: np.ndarray):
        self.results = results
        self.original_image = original_image
        
        # YOLO 스타일 속성들
        self.boxes = ONNXBoxes(results)
        self.speed = {'inference': results.get('inference_time', 0) * 1000}  # ms
    
    def plot(self, **kwargs) -> np.ndarray:
        """
        결과를 이미지에 그리기
        """
        return self._draw_results(self.original_image, **kwargs)
    
    def save(self, filename: str, **kwargs):
        """
        결과 이미지 저장
        """
        result_image = self.plot(**kwargs)
        cv2.imwrite(filename, result_image)
    
    def _draw_results(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        검출 결과를 이미지에 그리기
        """
        image_copy = image.copy()
        
        if self.results['count'] == 0:
            return image_copy
        
        boxes = self.results['boxes']
        scores = self.results['scores']
        classes = self.results['classes']
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            cls = int(classes[i])
            
            # 바운딩 박스 그리기
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 텍스트
            label = f"Class {cls}: {score:.2f}"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(image_copy, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # 텍스트
            cv2.putText(image_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image_copy


class ONNXBoxes:
    """
    YOLO Boxes와 유사한 인터페이스를 제공하는 클래스
    """
    
    def __init__(self, results: Dict):
        self.results = results
        
        # YOLO 스타일 속성들
        if results['count'] > 0:
            self.xyxy = results['boxes']  # [x1, y1, x2, y2] 형태
            self.conf = results['scores']
            self.cls = results['classes']
        else:
            self.xyxy = None
            self.conf = None
            self.cls = None
    
    def __len__(self):
        return self.results['count']
    
    def __bool__(self):
        return self.results['count'] > 0


# 사용 예시 및 테스트 함수
if __name__ == "__main__":
    def test_onnx_processor():
        """
        ONNX 프로세서 테스트
        """
        # 모델 초기화
        processor = ONNXProcessor("best.onnx", conf_threshold=0.5)
        
        # 이미지 예측
        results = processor("test_image.jpg")
        
        # 결과 확인
        print(f"검출된 객체 수: {len(results.boxes)}")
        print(f"추론 시간: {results.speed['inference']:.2f} ms")
        
        # 결과 이미지 저장
        results.save("result.jpg")
        
        return results
    
    # 테스트 실행
    # test_onnx_processor()