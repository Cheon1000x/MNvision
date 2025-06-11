"""
ONNX 모델 정확도 자동 비교 시스템 (완전 수정 버전)
원본 ONNX vs 양자화 ONNX 모델 간 정확도 비교

필요한 라이브러리:
pip install onnxruntime opencv-python numpy matplotlib scipy
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json

class ONNXAccuracyComparator:
    def __init__(self, original_model_path, quantized_model_path, input_size=(320, 192)):
        """
        ONNX 모델 정확도 비교기 초기화
        """
        self.original_model_path = original_model_path
        self.quantized_model_path = quantized_model_path
        self.input_width, self.input_height = input_size
        
        # 모델 세션 생성
        self.original_session = ort.InferenceSession(original_model_path)
        self.quantized_session = ort.InferenceSession(quantized_model_path)
        
        # 입/출력 정보
        self.input_name = self.original_session.get_inputs()[0].name
        self.output_name = self.original_session.get_outputs()[0].name
        
        # YOLO 클래스
        self.class_names = [
            'forklift-right',
            'forklift-left', 
            'forklift-horizontal',
            'person',
            'forklift-vertical',
            'object'
        ]
        
        print(f"🔄 ONNX 모델 정확도 비교 시스템 초기화")
        print(f"   원본 모델: {os.path.basename(original_model_path)}")
        print(f"   양자화 모델: {os.path.basename(quantized_model_path)}")
        print(f"   입력 크기: {input_size}")
    
    def preprocess_image(self, image_path):
        """이미지 전처리"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]
        
        # 비율 유지 리사이즈
        scale = min(self.input_width / original_w, self.input_height / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 패딩
        top_pad = (self.input_height - new_h) // 2
        bottom_pad = self.input_height - new_h - top_pad
        left_pad = (self.input_width - new_w) // 2
        right_pad = self.input_width - new_w - left_pad
        
        padded = cv2.copyMakeBorder(
            resized, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, (top_pad, left_pad), (original_w, original_h)
    
    def postprocess_yolo_output(self, output, scale, padding, original_size, conf_threshold=0.2):
        """YOLO 출력 후처리"""
        original_width, original_height = original_size
        top_pad, left_pad = padding
        
        pred = output.squeeze(0)
        
        boxes_raw = pred[0:4, :].T
        objectness_raw = pred[4, :]
        class_scores_raw = pred[5:11, :].T
        
        objectness = self.sigmoid(objectness_raw)
        class_scores = self.sigmoid(class_scores_raw)
        
        scores = objectness[:, np.newaxis] * class_scores
        scores_max = np.max(scores, axis=1)
        labels = np.argmax(scores, axis=1)
        
        keep_mask = scores_max > conf_threshold
        
        if keep_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])
        
        boxes_filtered = boxes_raw[keep_mask]
        scores_filtered = scores_max[keep_mask]
        labels_filtered = labels[keep_mask]
        
        # 중심점 → 좌상단/우하단 변환
        boxes_xyxy = np.copy(boxes_filtered)
        boxes_xyxy[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3] / 2
        
        # NMS 적용
        if len(boxes_xyxy) > 0:
            boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes_xyxy])
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(), 
                scores_filtered.tolist(), 
                conf_threshold, 
                0.3
            )
            
            if len(indices) == 0:
                return np.array([]), np.array([]), np.array([])
            
            indices = indices.flatten()
            boxes_final = boxes_xyxy[indices]
            scores_final = scores_filtered[indices]
            labels_final = labels_filtered[indices]
        else:
            return np.array([]), np.array([]), np.array([])
        
        # 패딩 및 스케일링 역변환
        boxes_final[:, 0] -= left_pad
        boxes_final[:, 1] -= top_pad
        boxes_final[:, 2] -= left_pad
        boxes_final[:, 3] -= top_pad
        boxes_final /= scale
        
        # 이미지 경계 클리핑
        boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)
        
        return boxes_final, scores_final, labels_final
    
    def sigmoid(self, x):
        """시그모이드 함수"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def compare_single_image(self, image_path, conf_threshold=0.2):
        """단일 이미지에 대한 모델 비교"""
        print(f"\n🔍 이미지 비교: {os.path.basename(image_path)}")
        
        input_tensor, scale, padding, original_size = self.preprocess_image(image_path)
        
        # 원본 모델 추론
        start_time = time.perf_counter()
        original_output = self.original_session.run([self.output_name], {self.input_name: input_tensor})[0]
        original_time = (time.perf_counter() - start_time) * 1000
        
        # 양자화 모델 추론
        start_time = time.perf_counter()
        quantized_output = self.quantized_session.run([self.output_name], {self.input_name: input_tensor})[0]
        quantized_time = (time.perf_counter() - start_time) * 1000
        
        # 후처리
        orig_boxes, orig_scores, orig_labels = self.postprocess_yolo_output(
            original_output, scale, padding, original_size, conf_threshold
        )
        
        quant_boxes, quant_scores, quant_labels = self.postprocess_yolo_output(
            quantized_output, scale, padding, original_size, conf_threshold
        )
        
        # 결과 비교
        comparison_result = {
            'image_path': image_path,
            'original': {
                'inference_time_ms': original_time,
                'detections': len(orig_boxes),
                'boxes': orig_boxes.tolist() if len(orig_boxes) > 0 else [],
                'scores': orig_scores.tolist() if len(orig_scores) > 0 else [],
                'labels': orig_labels.tolist() if len(orig_labels) > 0 else []
            },
            'quantized': {
                'inference_time_ms': quantized_time,
                'detections': len(quant_boxes),
                'boxes': quant_boxes.tolist() if len(quant_boxes) > 0 else [],
                'scores': quant_scores.tolist() if len(quant_scores) > 0 else [],
                'labels': quant_labels.tolist() if len(quant_labels) > 0 else []
            }
        }
        
        # 출력 텐서 유사도 계산
        output_correlation = self.calculate_tensor_similarity(original_output, quantized_output)
        comparison_result['tensor_similarity'] = output_correlation
        
        # 탐지 결과 유사도 계산
        detection_similarity = self.calculate_detection_similarity(
            orig_boxes, orig_scores, orig_labels,
            quant_boxes, quant_scores, quant_labels
        )
        comparison_result['detection_similarity'] = detection_similarity
        
        # 결과 출력
        print(f"   원본 모델: {original_time:.2f}ms, {len(orig_boxes)}개 탐지")
        print(f"   양자화 모델: {quantized_time:.2f}ms, {len(quant_boxes)}개 탐지")
        print(f"   텐서 상관관계: {output_correlation['correlation']:.4f}")
        print(f"   탐지 일치율: {detection_similarity['bbox_match_ratio']:.4f}")
        
        return comparison_result
    
    def calculate_tensor_similarity(self, tensor1, tensor2):
        """두 텐서 간 유사도 계산"""
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        correlation, p_value = pearsonr(flat1, flat2)
        mae = np.mean(np.abs(flat1 - flat2))
        mse = np.mean((flat1 - flat2) ** 2)
        max_error = np.max(np.abs(flat1 - flat2))
        
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'mae': float(mae),
            'mse': float(mse),
            'max_error': float(max_error),
            'rmse': float(np.sqrt(mse))
        }
    
    def calculate_detection_similarity(self, boxes1, scores1, labels1, boxes2, scores2, labels2):
        """탐지 결과 간 유사도 계산"""
        if len(boxes1) == 0 and len(boxes2) == 0:
            return {
                'bbox_match_ratio': 1.0,
                'score_correlation': 1.0,
                'label_accuracy': 1.0,
                'detection_count_diff': 0
            }
        
        if len(boxes1) == 0 or len(boxes2) == 0:
            return {
                'bbox_match_ratio': 0.0,
                'score_correlation': 0.0,
                'label_accuracy': 0.0,
                'detection_count_diff': abs(len(boxes1) - len(boxes2))
            }
        
        # IoU 기반 박스 매칭
        matched_pairs = []
        iou_threshold = 0.5
        
        for i, box1 in enumerate(boxes1):
            best_iou = 0
            best_j = -1
            
            for j, box2 in enumerate(boxes2):
                iou = self.calculate_iou(box1, box2)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_j = j
            
            if best_j != -1:
                matched_pairs.append((i, best_j, best_iou))
        
        bbox_match_ratio = len(matched_pairs) / max(len(boxes1), len(boxes2))
        
        if len(matched_pairs) > 1:
            matched_scores1 = [scores1[i] for i, j, iou in matched_pairs]
            matched_scores2 = [scores2[j] for i, j, iou in matched_pairs]
            score_correlation, _ = pearsonr(matched_scores1, matched_scores2)
        else:
            score_correlation = 0.0
        
        if len(matched_pairs) > 0:
            correct_labels = sum(1 for i, j, iou in matched_pairs if labels1[i] == labels2[j])
            label_accuracy = correct_labels / len(matched_pairs)
        else:
            label_accuracy = 0.0
        
        return {
            'bbox_match_ratio': float(bbox_match_ratio),
            'score_correlation': float(score_correlation),
            'label_accuracy': float(label_accuracy),
            'detection_count_diff': abs(len(boxes1) - len(boxes2)),
            'matched_pairs': len(matched_pairs)
        }
    
    def calculate_iou(self, box1, box2):
        """두 박스 간 IoU 계산"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def batch_comparison(self, image_dir, num_images=30, conf_threshold=0.2):
        """배치 이미지에 대한 모델 비교"""
        print(f"\n📊 배치 비교 시작")
        print(f"   이미지 디렉토리: {image_dir}")
        print(f"   테스트 이미지 수: {num_images}")
        print(f"   신뢰도 임계값: {conf_threshold}")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(image_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if len(image_files) == 0:
            print("❌ 테스트할 이미지를 찾을 수 없습니다.")
            return None
        
        selected_images = image_files[:min(num_images, len(image_files))]
        print(f"   선택된 이미지: {len(selected_images)}개")
        
        all_results = []
        
        for i, image_path in enumerate(selected_images):
            try:
                result = self.compare_single_image(image_path, conf_threshold)
                all_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   진행률: {i+1}/{len(selected_images)}")
                    
            except Exception as e:
                print(f"   이미지 처리 실패 ({os.path.basename(image_path)}): {e}")
                continue
        
        overall_stats = self.calculate_overall_statistics(all_results)
        self.print_comparison_summary(overall_stats)
        
        return {
            'individual_results': all_results,
            'overall_statistics': overall_stats
        }
    
    def calculate_overall_statistics(self, results):
        """전체 통계 계산"""
        if not results:
            return {}
        
        correlations = [r['tensor_similarity']['correlation'] for r in results]
        maes = [r['tensor_similarity']['mae'] for r in results]
        rmses = [r['tensor_similarity']['rmse'] for r in results]
        
        bbox_matches = [r['detection_similarity']['bbox_match_ratio'] for r in results]
        score_corrs = [r['detection_similarity']['score_correlation'] for r in results if not np.isnan(r['detection_similarity']['score_correlation'])]
        label_accs = [r['detection_similarity']['label_accuracy'] for r in results]
        
        orig_times = [r['original']['inference_time_ms'] for r in results]
        quant_times = [r['quantized']['inference_time_ms'] for r in results]
        
        orig_detections = [r['original']['detections'] for r in results]
        quant_detections = [r['quantized']['detections'] for r in results]
        
        return {
            'tensor_similarity': {
                'correlation_mean': np.mean(correlations),
                'correlation_std': np.std(correlations),
                'mae_mean': np.mean(maes),
                'rmse_mean': np.mean(rmses)
            },
            'detection_similarity': {
                'bbox_match_mean': np.mean(bbox_matches),
                'bbox_match_std': np.std(bbox_matches),
                'score_correlation_mean': np.mean(score_corrs) if score_corrs else 0.0,
                'label_accuracy_mean': np.mean(label_accs)
            },
            'performance': {
                'original_time_mean': np.mean(orig_times),
                'quantized_time_mean': np.mean(quant_times),
                'speed_ratio': np.mean(orig_times) / np.mean(quant_times) if np.mean(quant_times) > 0 else 0.0
            },
            'detection_counts': {
                'original_mean': np.mean(orig_detections),
                'quantized_mean': np.mean(quant_detections),
                'detection_diff_mean': np.mean([abs(o - q) for o, q in zip(orig_detections, quant_detections)])
            },
            'total_images': len(results)
        }
    
    def print_comparison_summary(self, stats):
        """비교 결과 요약 출력"""
        print(f"\n📈 전체 비교 결과 요약")
        print("=" * 60)
        
        print(f"📊 텐서 유사도:")
        print(f"   평균 상관관계: {stats['tensor_similarity']['correlation_mean']:.4f} ± {stats['tensor_similarity']['correlation_std']:.4f}")
        print(f"   평균 절대 오차: {stats['tensor_similarity']['mae_mean']:.6f}")
        print(f"   RMSE: {stats['tensor_similarity']['rmse_mean']:.6f}")
        
        print(f"\n🎯 탐지 성능:")
        print(f"   박스 매칭률: {stats['detection_similarity']['bbox_match_mean']:.4f} ± {stats['detection_similarity']['bbox_match_std']:.4f}")
        print(f"   점수 상관관계: {stats['detection_similarity']['score_correlation_mean']:.4f}")
        print(f"   라벨 정확도: {stats['detection_similarity']['label_accuracy_mean']:.4f}")
        
        print(f"\n⚡ 성능 비교:")
        print(f"   원본 모델: {stats['performance']['original_time_mean']:.2f}ms")
        print(f"   양자화 모델: {stats['performance']['quantized_time_mean']:.2f}ms")
        print(f"   속도 비율: {stats['performance']['speed_ratio']:.2f}x")
        
        print(f"\n📦 탐지 수:")
        print(f"   원본 평균: {stats['detection_counts']['original_mean']:.1f}개")
        print(f"   양자화 평균: {stats['detection_counts']['quantized_mean']:.1f}개")
        print(f"   평균 차이: {stats['detection_counts']['detection_diff_mean']:.1f}개")
        
        print(f"\n📋 총 테스트 이미지: {stats['total_images']}개")
    
    def save_results_with_graphs(self, results, output_dir="comparison_results"):
        """결과를 그래프로 시각화하여 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if 'overall_statistics' not in results:
            print("❌ 통계 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        stats = results['overall_statistics']
        individual_results = results['individual_results']
        
        # 그래프 생성
        self.create_comparison_graphs(stats, individual_results, output_path)
        
        print(f"📊 그래프가 '{output_dir}' 폴더에 저장되었습니다.")
    
    def create_comparison_graphs(self, stats, individual_results, output_path):
        """비교 그래프 생성"""
        # 1. 전체 성능 비교
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 텐서 유사도
        plt.subplot(2, 3, 1)
        metrics = ['Correlation', 'MAE\n(×1000)', 'RMSE\n(×1000)']
        values = [
            stats['tensor_similarity']['correlation_mean'],
            stats['tensor_similarity']['mae_mean'] * 1000,
            stats['tensor_similarity']['rmse_mean'] * 1000
        ]
        plt.bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)
        plt.title('Tensor Similarity Metrics')
        plt.ylabel('Value')
        plt.grid(axis='y', alpha=0.3)
        
        # 서브플롯 2: 탐지 성능
        plt.subplot(2, 3, 2)
        det_metrics = ['Box Match', 'Score Corr', 'Label Acc']
        det_values = [
            stats['detection_similarity']['bbox_match_mean'],
            stats['detection_similarity']['score_correlation_mean'],
            stats['detection_similarity']['label_accuracy_mean']
        ]
        plt.bar(det_metrics, det_values, color=['orange', 'purple', 'brown'], alpha=0.7)
        plt.title('Detection Performance')
        plt.ylabel('Score (0-1)')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # 서브플롯 3: 추론 시간
        plt.subplot(2, 3, 3)
        models = ['Original', 'Quantized']
        times = [stats['performance']['original_time_mean'], stats['performance']['quantized_time_mean']]
        plt.bar(models, times, color=['blue', 'red'], alpha=0.7)
        plt.title('Inference Time')
        plt.ylabel('Time (ms)')
        plt.grid(axis='y', alpha=0.3)
        
        # 서브플롯 4: 상관관계 분포
        plt.subplot(2, 3, 4)
        correlations = [r['tensor_similarity']['correlation'] for r in individual_results]
        plt.hist(correlations, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2)
        plt.title('Correlation Distribution')
        plt.xlabel('Correlation')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        # 서브플롯 5: 탐지 수 비교
        plt.subplot(2, 3, 5)
        orig_detections = [r['original']['detections'] for r in individual_results]
        quant_detections = [r['quantized']['detections'] for r in individual_results]
        plt.scatter(orig_detections, quant_detections, alpha=0.6, color='green')
        max_det = max(max(orig_detections), max(quant_detections))
        plt.plot([0, max_det], [0, max_det], 'r--', alpha=0.8)
        plt.title('Detection Count Comparison')
        plt.xlabel('Original Detections')
        plt.ylabel('Quantized Detections')
        plt.grid(alpha=0.3)
        
        # 서브플롯 6: 속도 비교
        plt.subplot(2, 3, 6)
        orig_times = [r['original']['inference_time_ms'] for r in individual_results]
        quant_times = [r['quantized']['inference_time_ms'] for r in individual_results]
        plt.boxplot([orig_times, quant_times], labels=['Original', 'Quantized'])
        plt.title('Inference Time Distribution')
        plt.ylabel('Time (ms)')
        plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle('ONNX Model Comparison Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 상세 분석 그래프
        plt.figure(figsize=(12, 8))
        
        # 박스 매칭률 vs 상관관계
        plt.subplot(2, 2, 1)
        bbox_matches = [r['detection_similarity']['bbox_match_ratio'] for r in individual_results]
        plt.scatter(correlations, bbox_matches, alpha=0.6, color='purple')
        plt.xlabel('Tensor Correlation')
        plt.ylabel('BBox Match Ratio')
        plt.title('Correlation vs Detection Match')
        plt.grid(alpha=0.3)
        
        # 추론 시간 산점도
        plt.subplot(2, 2, 2)
        plt.scatter(orig_times, quant_times, alpha=0.6, color='blue')
        max_time = max(max(orig_times), max(quant_times))
        plt.plot([0, max_time], [0, max_time], 'r--', alpha=0.8)
        plt.xlabel('Original Time (ms)')
        plt.ylabel('Quantized Time (ms)')
        plt.title('Inference Time Correlation')
        plt.grid(alpha=0.3)
        
        # MAE 분포
        plt.subplot(2, 2, 3)
        maes = [r['tensor_similarity']['mae'] for r in individual_results]
        plt.hist(maes, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(maes), color='red', linestyle='--', linewidth=2)
        plt.title('MAE Distribution')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        # 라벨 정확도 분포
        plt.subplot(2, 2, 4)
        label_accs = [r['detection_similarity']['label_accuracy'] for r in individual_results]
        plt.hist(label_accs, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(label_accs), color='red', linestyle='--', linewidth=2)
        plt.title('Label Accuracy Distribution')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.suptitle('Detailed Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """메인 실행 함수"""
    
    # ========================================
    # 🔧 설정 수정
    # ========================================
    
    ORIGINAL_MODEL_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver6(프루닝모델_Test7)\yolov8_custom_fixed_test7_pruned.onnx"
    QUANTIZED_MODEL_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\9.Quantization\Test1\model_static_int8.onnx"
    
    # 테스트 이미지 디렉토리
    TEST_IMAGE_DIR = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106"
    NUM_TEST_IMAGES = 30  # 테스트할 이미지 수
    
    # 단일 이미지 테스트
    SINGLE_TEST_IMAGE = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106\frame_000000.jpg"
    
    # 모델 입력 크기
    INPUT_SIZE = (320, 192)  # (width, height)
    
    # 신뢰도 임계값
    CONF_THRESHOLD = 0.2
    
    # ========================================
    
    print("🚀 ONNX 모델 정확도 비교 시작")
    print("=" * 60)
    
    try:
        # 비교기 초기화
        comparator = ONNXAccuracyComparator(
            original_model_path=ORIGINAL_MODEL_PATH,
            quantized_model_path=QUANTIZED_MODEL_PATH,
            input_size=INPUT_SIZE
        )
        
        # 1단계: 단일 이미지 테스트
        print(f"\n1️⃣ 단일 이미지 정확도 테스트")
        if os.path.exists(SINGLE_TEST_IMAGE):
            single_result = comparator.compare_single_image(SINGLE_TEST_IMAGE, CONF_THRESHOLD)
        else:
            print(f"❌ 테스트 이미지를 찾을 수 없습니다: {SINGLE_TEST_IMAGE}")
        
        # 2단계: 배치 이미지 테스트
        print(f"\n2️⃣ 배치 이미지 정확도 테스트")
        if os.path.exists(TEST_IMAGE_DIR):
            batch_results = comparator.batch_comparison(
                image_dir=TEST_IMAGE_DIR,
                num_images=NUM_TEST_IMAGES,
                conf_threshold=CONF_THRESHOLD
            )
            
            # 결과 저장 (그래프로 시각화)
            if batch_results:
                comparator.save_results_with_graphs(batch_results)
                
                print(f"\n🎉 정확도 비교 완료!")
                print(f"📊 주요 결과:")
                stats = batch_results['overall_statistics']
                print(f"   텐서 상관관계: {stats['tensor_similarity']['correlation_mean']:.4f}")
                print(f"   탐지 매칭률: {stats['detection_similarity']['bbox_match_mean']:.4f}")
                print(f"   라벨 정확도: {stats['detection_similarity']['label_accuracy_mean']:.4f}")
                print(f"   속도 비율: {stats['performance']['speed_ratio']:.2f}x")
                
                print(f"\n📊 그래프 파일:")
                print(f"   - model_comparison_results.png")
                print(f"   - detailed_analysis.png")
        else:
            print(f"❌ 테스트 이미지 디렉토리를 찾을 수 없습니다: {TEST_IMAGE_DIR}")
    
    except Exception as e:
        print(f"❌ 정확도 비교 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 정확도 비교 완료")


if __name__ == "__main__":
    main()