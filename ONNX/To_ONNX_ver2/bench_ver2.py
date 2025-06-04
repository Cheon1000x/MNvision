# model_comparison_fixed.py
import time
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import onnxruntime as ort

class ModelComparator:
    def __init__(self, pt_model_path, onnx_model_path):
        # PyTorch 모델 로드 (CPU 강제)
        print("PyTorch 모델 로딩 (CPU)...")
        self.pt_model = YOLO(pt_model_path)
        # CPU 강제 설정
        torch.set_num_threads(1)  # CPU 스레드 수 제한
        
        # ONNX 모델 로드 (CPU)
        print("ONNX 모델 로딩 (CPU)...")
        self.onnx_session = ort.InferenceSession(
            onnx_model_path, 
            providers=['CPUExecutionProvider']  # CPU만 사용
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        
        # ONNX 출력 정보 확인
        self.check_onnx_outputs()
        
        # 워밍업 실행
        self.warmup_models()
        
    def check_onnx_outputs(self):
        """ONNX 모델 출력 구조 확인"""
        print("\n=== ONNX 모델 출력 정보 ===")
        print(f"입력 이름: {self.input_name}")
        print(f"입력 형태: {self.onnx_session.get_inputs()[0].shape}")
        
        print("출력 정보:")
        for i, output in enumerate(self.onnx_session.get_outputs()):
            print(f"  출력 {i}: 이름={output.name}, 형태={output.shape}, 타입={output.type}")

    def warmup_models(self):
        """모델 워밍업 (첫 추론 캐싱 효과 제거)"""
        print("\n모델 워밍업 중...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # PyTorch 워밍업 (3회)
        for i in range(3):
            _ = self.pt_model.predict(dummy_frame, device='cpu', verbose=False)
        
        # ONNX 워밍업 (3회)
        dummy_input = self.preprocess_for_onnx(dummy_frame)
        for i in range(3):
            _ = self.onnx_session.run(None, {self.input_name: dummy_input})
        
        print("워밍업 완료!")

    def preprocess_for_onnx(self, frame):
        """ONNX용 이미지 전처리"""
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def run_pt_inference_multiple(self, frame, num_runs=5):
        """PyTorch 모델 여러 번 추론 (CPU 강제)"""
        times = []
        results_info = None
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            results = self.pt_model.predict(frame, device='cpu', verbose=False)[0]
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
            
            # 첫 번째 결과만 저장
            if results_info is None:
                num_detections = len(results.boxes) if results.boxes is not None else 0
                num_masks = len(results.masks) if results.masks is not None else 0
                
                # 신뢰도 정보 추가
                confidences = []
                if results.boxes is not None and len(results.boxes) > 0:
                    confidences = results.boxes.conf.cpu().numpy()
                
                results_info = {
                    'num_detections': num_detections,
                    'num_masks': num_masks,
                    'avg_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
                    'max_confidence': float(np.max(confidences)) if len(confidences) > 0 else 0.0
                }
        
        return {
            'inference_time': np.mean(times),
            'time_std': np.std(times),
            'all_times': times,
            **results_info
        }

    def run_onnx_inference_multiple(self, frame, num_runs=5, conf_threshold=0.3):
        """ONNX 모델 여러 번 추론 및 실제 객체 개수 계산"""
        times = []
        outputs_info = None
        
        # 전처리
        input_data = self.preprocess_for_onnx(frame)
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            outputs = self.onnx_session.run(None, {self.input_name: input_data})
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
            
            # 첫 번째 결과만 저장하고 실제 객체 개수 계산
            if outputs_info is None:
                outputs_info = self._analyze_onnx_detections(outputs, conf_threshold)
        
        return {
            'inference_time': np.mean(times),
            'time_std': np.std(times),
            'all_times': times,
            **outputs_info
        }

    def _analyze_onnx_detections(self, outputs, conf_threshold=0.3):
        """ONNX 출력에서 실제 감지된 객체 분석 (YOLOv8 후처리 포함)"""
        try:
            print(f"\n--- ONNX 출력 상세 분석 ---")
            for i, output in enumerate(outputs):
                print(f"출력 {i}: 형태={output.shape}, 타입={output.dtype}")
            
            # YOLOv8 후처리 적용
            boxes, confidences, class_ids = self._decode_yolov8_boxes(outputs[0], conf_threshold)
            
            num_detections = len(boxes)
            avg_confidence = float(np.mean(confidences)) if num_detections > 0 else 0.0
            max_confidence = float(np.max(confidences)) if num_detections > 0 else 0.0
            
            # 마스크 정보 (나중에 처리)
            num_masks = 0
            if len(outputs) > 1:
                mask_output = outputs[1]
                if len(mask_output.shape) >= 3:
                    num_masks = mask_output.shape[1] if mask_output.shape[0] == 1 else mask_output.shape[0]
            
            print(f"후처리 결과: {num_detections}개 객체, 평균 신뢰도: {avg_confidence:.3f}")
            
            return {
                'num_detections': num_detections,
                'num_masks': num_masks,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'num_tensor_outputs': len(outputs),
                'output_shapes': [output.shape for output in outputs]
            }
            
        except Exception as e:
            print(f"ONNX 후처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'num_detections': 0,
                'num_masks': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'num_tensor_outputs': len(outputs),
                'output_shapes': [output.shape for output in outputs],
                'analysis_error': str(e)
            }
    def apply_nms(self, boxes, confidences, class_ids, nms_threshold=0.1):
        """NMS를 적용하여 중복된 박스 제거"""
        if len(boxes) == 0:
            return [], [], []
        
        # [x1,y1,x2,y2] → [x,y,w,h] 형태로 변환 (OpenCV NMS용)
        nms_boxes = []
        for box in boxes:
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            nms_boxes.append([x, y, w, h])
        
        # OpenCV NMS 적용
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), 0.0, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], confidences[indices], class_ids[indices]
        else:
            return [], [], []            
    
    def _decode_yolov8_boxes(self, raw_output, conf_threshold=0.95, input_size=640):
        """YOLOv8 원시 출력에서 박스 디코딩 (수정됨)"""
        # 배치 차원 제거: (1, 47, 8400) -> (47, 8400)
        output = raw_output[0]
        
        # 전치: (47, 8400) -> (8400, 47)
        predictions = output.transpose()
        
        # 박스 좌표 (처음 4개) 추출
        boxes = predictions[:, :4]  # (8400, 4) - cx, cy, w, h
        
        # 다시 원래 방식으로 (objectness 없이)
        class_probs = predictions[:, 4:]  # 4번째부터 43개 클래스
        max_class_probs = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)  # ← 이 줄 추가!
        max_class_probs_sigmoid = 1 / (1 + np.exp(-max_class_probs))  # 시그모이드 적용
        confidences = max_class_probs_sigmoid

        # 높은 임계값 사용
        conf_threshold = 0.8  # 0.3 → 0.8

        # 디버깅 출력 (수정됨)
        print(f"클래스 확률 범위: [{np.min(class_probs):.3f}, {np.max(class_probs):.3f}]")
        print(f"Max class probs 범위: [{np.min(max_class_probs):.3f}, {np.max(max_class_probs):.3f}]")
        print(f"최종 신뢰도 범위: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
        print(f"임계값 {conf_threshold} 이상인 개수: {np.sum(confidences > conf_threshold)}")
        
        # 신뢰도 필터링
        valid_mask = confidences > conf_threshold
        
        if not np.any(valid_mask):
            return [], [], []
        
        valid_boxes = boxes[valid_mask]
        valid_confidences = confidences[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        
        # 중심점 형태를 일반 박스 형태로 변환 (cx,cy,w,h -> x1,y1,x2,y2)
        x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
        y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
        x2 = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
        y2 = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
        
        decoded_boxes = np.column_stack([x1, y1, x2, y2])
        
        # NMS 적용 (새로 추가!)
        final_boxes, final_confidences, final_class_ids = self.apply_nms(
            decoded_boxes, valid_confidences, valid_class_ids, nms_threshold=0.4
        )
        
        print(f"NMS 전: {len(decoded_boxes)}개 → NMS 후: {len(final_boxes)}개")
        
        return final_boxes, final_confidences, final_class_ids
    
    def get_image_files(self, folder_path):
        """폴더에서 이미지 파일 목록 가져오기"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        return sorted(image_files)
    
    def save_comparison_graphs(self, pt_times, onnx_times, image_names, pt_detections, onnx_detections, save_dir="comparison_results"):
        """비교 결과를 그래프로 저장"""
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 추론 시간 비교 그래프
        plt.figure(figsize=(15, 10))
        
        # 상단 좌측: 이미지별 추론 시간 비교
        plt.subplot(2, 3, 1)
        x = range(len(image_names))
        plt.plot(x, pt_times, 'b-o', label='PyTorch (CPU)', linewidth=2, markersize=4)
        plt.plot(x, onnx_times, 'r-s', label='ONNX (CPU)', linewidth=2, markersize=4)
        plt.xlabel('Image Index')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time Comparison by Image')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 상단 중앙: 박스플롯
        plt.subplot(2, 3, 2)
        plt.boxplot([pt_times, onnx_times], labels=['PyTorch', 'ONNX'])
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time Distribution')
        plt.grid(True, alpha=0.3)
        
        # 상단 우측: 평균 추론 시간 막대 그래프
        plt.subplot(2, 3, 3)
        models = ['PyTorch', 'ONNX']
        avg_times = [np.mean(pt_times), np.mean(onnx_times)]
        std_times = [np.std(pt_times), np.std(onnx_times)]
        
        bars = plt.bar(models, avg_times, yerr=std_times, capsize=5, 
                        color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Average Inference Time (ms)')
        plt.title('Average Performance Comparison')
        
        # 막대 위에 값 표시
        for bar, avg_time in zip(bars, avg_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_times)*0.1,
                    f'{avg_time:.2f}ms', ha='center', va='bottom')
        
        # 하단 좌측: 객체 감지 수 비교
        plt.subplot(2, 3, 4)
        plt.plot(x, pt_detections, 'b-o', label='PyTorch', linewidth=2, markersize=4)
        plt.plot(x, onnx_detections, 'r-s', label='ONNX', linewidth=2, markersize=4)
        plt.xlabel('Image Index')
        plt.ylabel('Number of Detections')
        plt.title('Detection Count Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 하단 중앙: 감지 수 분포 비교
        plt.subplot(2, 3, 5)
        plt.hist(pt_detections, bins=10, alpha=0.7, label='PyTorch', color='blue')
        plt.hist(onnx_detections, bins=10, alpha=0.7, label='ONNX', color='red')
        plt.xlabel('Number of Detections')
        plt.ylabel('Frequency')
        plt.title('Detection Count Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 하단 우측: 속도 vs 정확도
        plt.subplot(2, 3, 6)
        plt.scatter(np.mean(pt_times), np.mean(pt_detections), 
                    s=100, c='blue', label='PyTorch', alpha=0.7)
        plt.scatter(np.mean(onnx_times), np.mean(onnx_detections), 
                    s=100, c='red', label='ONNX', alpha=0.7)
        plt.xlabel('Average Inference Time (ms)')
        plt.ylabel('Average Detections')
        plt.title('Speed vs Detection Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'detailed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n그래프가 '{save_dir}/detailed_comparison.png'에 저장되었습니다.")
        
    def visualize_detections(self, image_path, output_folder="visualization_results"):
        """ONNX와 PyTorch 모델의 감지 결과 시각화"""
        # 저장 폴더 생성
        os.makedirs(output_folder, exist_ok=True)
        
        # 이미지 로드
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"이미지 로드 실패: {image_path}")
            return
        
        # 원본 이미지 복사본 생성
        pt_image = frame.copy()
        onnx_image = frame.copy()
        
        # PyTorch 모델 추론
        pt_results = self.pt_model.predict(frame, device='cpu', verbose=False)[0]
        
        # PyTorch 결과 그리기
        for box in pt_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            # 박스 그리기 (녹색)
            cv2.rectangle(pt_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 정보 텍스트
            text = f"Class: {cls_id}, Conf: {conf:.2f}"
            cv2.putText(pt_image, text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ONNX 모델 추론
        input_data = self.preprocess_for_onnx(frame)
        outputs = self.onnx_session.run(None, {self.input_name: input_data})
        
        # ONNX 후처리 및 결과 그리기
        boxes, confidences, class_ids = self._decode_yolov8_boxes(outputs[0], conf_threshold=0.9)
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            # 박스 그리기 (빨간색)
            cv2.rectangle(onnx_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # 정보 텍스트
            text = f"Class: {cls_id}, Conf: {conf:.2f}"
            cv2.putText(onnx_image, text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 결과 이미지 병합
        h, w = frame.shape[:2]
        combined_image = np.zeros((h, 2*w, 3), dtype=np.uint8)
        combined_image[:, :w] = pt_image
        combined_image[:, w:] = onnx_image
        
        # 모델 레이블 추가
        cv2.putText(combined_image, "PyTorch", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "ONNX", (w+10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 감지된 객체 수 추가
        cv2.putText(combined_image, f"Detections: {len(pt_results.boxes)}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, f"Detections: {len(boxes)}", (w+10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 이미지 저장
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"comparison_{base_name}")
        cv2.imwrite(output_path, combined_image)
        print(f"시각화 이미지 저장됨: {output_path}")
        
        return output_path
    
    def compare_models(self, image_folder_path, max_images=10):
        """모델 성능 비교"""
        print(f"\n=== 모델 성능 비교 (CPU Only) ===")
        
        # 이미지 파일 목록 가져오기
        image_files = self.get_image_files(image_folder_path)
        
        if not image_files:
            print(f"이미지를 찾을 수 없습니다: {image_folder_path}")
            return
        
        # 최대 이미지 수 제한
        if len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"총 {len(image_files)}개 이미지로 테스트 (최대 {max_images}개 제한)")
        else:
            print(f"총 {len(image_files)}개 이미지로 테스트")
        
        # 전체 결과 저장
        all_pt_times = []
        all_onnx_times = []
        all_pt_detections = []
        all_onnx_detections = []
        image_names = []
        
        # 각 이미지에 대해 테스트 (5회씩 실행)
        for i, image_path in enumerate(image_files):
            print(f"\n--- 이미지 {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
            
                # 각 이미지에 대해 테스트 (5회씩 실행)
            for i, image_path in enumerate(image_files):
                print(f"\n--- 이미지 {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
                
                # 이미지 로드
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"이미지 로드 실패: {image_path}")
                    continue
                
                print(f"이미지 크기: {frame.shape}")
                
                # 시각화 결과 저장 (추가된 부분)
                vis_path = self.visualize_detections(image_path)
                print(f"시각화 결과: {vis_path}")
            
            # PyTorch 모델 테스트 (5회 평균)
            pt_result = self.run_pt_inference_multiple(frame, num_runs=5)
            all_pt_times.append(pt_result['inference_time'])
            all_pt_detections.append(pt_result['num_detections'])
            print(f"PyTorch: {pt_result['inference_time']:.2f}±{pt_result['time_std']:.2f}ms, "
                    f"감지: {pt_result['num_detections']}개, 마스크: {pt_result['num_masks']}개, "
                    f"평균신뢰도: {pt_result['avg_confidence']:.3f}")
            
            # ONNX 모델 테스트 (5회 평균)
            onnx_result = self.run_onnx_inference_multiple(frame, num_runs=5)
            all_onnx_times.append(onnx_result['inference_time'])
            all_onnx_detections.append(onnx_result['num_detections'])
            print(f"ONNX: {onnx_result['inference_time']:.2f}±{onnx_result['time_std']:.2f}ms, "
                    f"감지: {onnx_result['num_detections']}개, 마스크: {onnx_result['num_masks']}개, "
                    f"평균신뢰도: {onnx_result['avg_confidence']:.3f}")
            
            image_names.append(os.path.basename(image_path))
        
        # 전체 결과 분석 및 출력
        if all_pt_times and all_onnx_times:
            pt_avg_time = np.mean(all_pt_times)
            pt_std_time = np.std(all_pt_times)
            onnx_avg_time = np.mean(all_onnx_times)
            onnx_std_time = np.std(all_onnx_times)
            
            pt_avg_det = np.mean(all_pt_detections)
            onnx_avg_det = np.mean(all_onnx_detections)
            
            print(f"\n=== 전체 성능 비교 결과 (5회 평균) ===")
            print(f"PyTorch 모델 (CPU):")
            print(f"  평균 추론 시간: {pt_avg_time:.2f} ± {pt_std_time:.2f} ms")
            print(f"  최소 시간: {min(all_pt_times):.2f} ms")
            print(f"  최대 시간: {max(all_pt_times):.2f} ms")
            print(f"  평균 감지 수: {pt_avg_det:.1f}개")
            
            print(f"\nONNX 모델 (CPU):")
            print(f"  평균 추론 시간: {onnx_avg_time:.2f} ± {onnx_std_time:.2f} ms")
            print(f"  최소 시간: {min(all_onnx_times):.2f} ms")
            print(f"  최대 시간: {max(all_onnx_times):.2f} ms")
            print(f"  평균 감지 수: {onnx_avg_det:.1f}개")
            
            # 속도 및 정확도 비교
            speed_ratio = pt_avg_time / onnx_avg_time
            detection_ratio = onnx_avg_det / pt_avg_det if pt_avg_det > 0 else 0
            
            print(f"\n=== 종합 비교 ===")
            print(f"속도 개선율: {speed_ratio:.2f}x {'빠름' if speed_ratio > 1 else '느림'}")
            print(f"감지 정확도 비율: {detection_ratio:.2f}x (ONNX/PyTorch)")
            print(f"평균 속도 차이: {abs(pt_avg_time - onnx_avg_time):.2f} ms")
            print(f"평균 감지 수 차이: {abs(pt_avg_det - onnx_avg_det):.1f}개")
            
            # 그래프 저장
            self.save_comparison_graphs(all_pt_times, all_onnx_times, image_names, 
                                        all_pt_detections, all_onnx_detections)

def main():
   # 파일 경로 설정
   pt_model_path = r"C:\Users\K\Desktop\Group_6\ONNX\ONNX.pt"
   onnx_model_path = "model_seg.onnx"
   
   image_folder_path = input("테스트할 이미지 폴더 경로를 입력하세요: ").strip()
   max_images = input("최대 테스트 이미지 수 (기본값: 10): ").strip()
   max_images = int(max_images) if max_images.isdigit() else 10
   
   # 비교 실행
   comparator = ModelComparator(pt_model_path, onnx_model_path)
   comparator.compare_models(image_folder_path, max_images)

if __name__ == "__main__":
   main()