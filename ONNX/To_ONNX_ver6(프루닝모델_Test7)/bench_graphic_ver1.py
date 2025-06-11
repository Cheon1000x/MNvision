#!/usr/bin/env python3
"""
PT 모델 vs ONNX 모델 성능 벤치마크 비교 (고급 시각화 버전)
정확도, 속도, 탐지 결과를 종합적으로 비교하고 상세한 시각화 제공
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
from collections import defaultdict

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class PTModel:
    """원본 PyTorch 모델 래퍼"""
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = list(self.model.names.values())
        
    def predict(self, image_path):
        start_time = time.perf_counter()
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)[0]
        inference_time = (time.perf_counter() - start_time) * 1000
        
        boxes = []
        scores = []
        labels = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()
                
                boxes.append(xyxy)
                scores.append(conf)
                labels.append(cls_id)
        
        return np.array(boxes), np.array(scores), np.array(labels), inference_time

class ONNXModel:
    """ONNX 모델 래퍼"""
    def __init__(self, model_path, input_size=(320, 192), conf_threshold=0.3, iou_threshold=0.1):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_width, self.input_height = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 클래스 이름 (프로젝트에 맞게 수정)
        self.class_names = [
            'forklift-right',       # 0
            'forklift-left',        # 1  
            'forklift-horizontal',  # 2
            'person',               # 3
            'forklift-vertical',    # 4
            'object',               # 5
            ''                      # 6
        ]

    def preprocess(self, image):
        original_h, original_w = image.shape[:2]
        
        # 비율 유지하면서 리사이즈
        scale = min(self.input_width / original_w, self.input_height / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 패딩 추가
        top_pad = (self.input_height - new_h) // 2
        bottom_pad = self.input_height - new_h - top_pad
        left_pad = (self.input_width - new_w) // 2
        right_pad = self.input_width - new_w - left_pad

        padded_image = cv2.copyMakeBorder(
            resized_image, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # 정규화 및 차원 변환
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, scale, (top_pad, left_pad)

    def postprocess(self, output, original_width, original_height, scale, padding):
        pred = output.squeeze(0)
        
        # YOLO 출력 파싱
        boxes_raw = pred[0:4, :].T
        objectness_raw = pred[4, :]
        class_scores_raw = pred[5:11, :].T

        # 시그모이드 적용
        objectness = self.sigmoid(objectness_raw)
        class_scores = self.sigmoid(class_scores_raw)

        # 최종 신뢰도 계산
        scores = objectness[:, np.newaxis] * class_scores
        scores_max = np.max(scores, axis=1)
        labels = np.argmax(scores, axis=1)

        # 임계값 필터링
        keep_mask = scores_max > self.conf_threshold
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
        
        # NMS
        boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes_xyxy])
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(), scores_filtered.tolist(), 
            self.conf_threshold, self.iou_threshold
        )
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten()
        boxes_final = boxes_xyxy[indices]
        scores_final = scores_filtered[indices]
        labels_final = labels_filtered[indices]

        # 패딩 및 스케일링 역변환
        top_pad, left_pad = padding
        boxes_final[:, 0] -= left_pad
        boxes_final[:, 1] -= top_pad
        boxes_final[:, 2] -= left_pad
        boxes_final[:, 3] -= top_pad
        boxes_final /= scale

        # 클리핑
        boxes_final[:, 0] = np.clip(boxes_final[:, 0], 0, original_width)
        boxes_final[:, 1] = np.clip(boxes_final[:, 1], 0, original_height)
        boxes_final[:, 2] = np.clip(boxes_final[:, 2], 0, original_width)
        boxes_final[:, 3] = np.clip(boxes_final[:, 3], 0, original_height)

        return boxes_final, scores_final, labels_final

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def predict(self, image_path):
        # 이미지 로드
        original_image_bgr = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        # 전처리
        input_tensor, scale, padding = self.preprocess(original_image)

        # ONNX 추론
        start_time = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # 후처리
        boxes, scores, labels = self.postprocess(
            outputs[0], original_width, original_height, scale, padding
        )
        
        return boxes, scores, labels, inference_time

class EnhancedBenchmarkComparator:
    """고급 시각화 기능이 포함된 벤치마크 비교기"""
    
    def __init__(self, pt_model_path, onnx_model_path, conf_threshold=0.3):
        print("🔄 모델 로딩 중...")
        self.pt_model = PTModel(pt_model_path, conf_threshold)
        self.onnx_model = ONNXModel(onnx_model_path, conf_threshold=conf_threshold)
        
        # 결과 저장용
        self.results_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 클래스별 통계
        self.class_stats = defaultdict(lambda: {'pt_count': 0, 'onnx_count': 0, 'matched': 0})
        
        print(f"✅ PT 모델 로드 완료: {len(self.pt_model.class_names)}개 클래스")
        print(f"✅ ONNX 모델 로드 완료: {len(self.onnx_model.class_names)}개 클래스")
        print(f"📁 결과 저장 폴더: {self.results_dir}")
        
    def calculate_iou(self, box1, box2):
        """두 박스 간의 IoU 계산"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, pt_boxes, pt_labels, onnx_boxes, onnx_labels, iou_threshold=0.5):
        """PT와 ONNX 탐지 결과 매칭"""
        matches = []
        unmatched_pt = list(range(len(pt_boxes)))
        unmatched_onnx = list(range(len(onnx_boxes)))
        
        for i, pt_box in enumerate(pt_boxes):
            best_iou = 0
            best_match = -1
            
            for j, onnx_box in enumerate(onnx_boxes):
                if j in unmatched_onnx:
                    iou = self.calculate_iou(pt_box, onnx_box)
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_match = j
            
            if best_match != -1:
                # 클래스 ID를 정수로 변환하여 비교
                pt_class = int(pt_labels[i])
                onnx_class = int(onnx_labels[best_match])
                class_match = pt_class == onnx_class
                
                # 디버깅을 위한 출력 (필요시 주석 해제)
                # print(f"PT class: {pt_class}, ONNX class: {onnx_class}, Match: {class_match}")
                
                matches.append({
                    'pt_idx': i,
                    'onnx_idx': best_match,
                    'iou': float(best_iou),
                    'class_match': class_match,
                    'pt_class': pt_class,
                    'onnx_class': onnx_class
                })
                unmatched_pt.remove(i)
                unmatched_onnx.remove(best_match)
        
        return matches, unmatched_pt, unmatched_onnx

    def update_class_stats(self, pt_labels, onnx_labels, matches):
        """클래스별 통계 업데이트"""
        # PT 탐지 카운트
        for label in pt_labels:
            self.class_stats[label]['pt_count'] += 1
        
        # ONNX 탐지 카운트
        for label in onnx_labels:
            self.class_stats[label]['onnx_count'] += 1
        
        # 매칭 카운트
        for match in matches:
            if match['class_match']:
                self.class_stats[match['pt_class']]['matched'] += 1

    def create_detection_visualization(self, image_path, pt_boxes, pt_labels, pt_scores, 
                                     onnx_boxes, onnx_labels, onnx_scores, matches):
        """탐지 결과 시각화 이미지 생성"""
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # PT 결과
        axes[0].imshow(image)
        axes[0].set_title(f'PT Model ({len(pt_boxes)} detections)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        for i, (box, label, score) in enumerate(zip(pt_boxes, pt_labels, pt_scores)):
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                               fill=False, color='red', linewidth=2)
            axes[0].add_patch(rect)
            class_name = self.pt_model.class_names[label] if label < len(self.pt_model.class_names) else f"class_{label}"
            axes[0].text(box[0], box[1]-5, f'{class_name}: {score:.2f}', 
                        color='red', fontsize=8, fontweight='bold')
        
        # ONNX 결과
        axes[1].imshow(image)
        axes[1].set_title(f'ONNX Model ({len(onnx_boxes)} detections)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        for i, (box, label, score) in enumerate(zip(onnx_boxes, onnx_labels, onnx_scores)):
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                               fill=False, color='blue', linewidth=2)
            axes[1].add_patch(rect)
            class_name = self.onnx_model.class_names[label] if label < len(self.onnx_model.class_names) else f"class_{label}"
            axes[1].text(box[0], box[1]-5, f'{class_name}: {score:.2f}', 
                        color='blue', fontsize=8, fontweight='bold')
        
        # 매칭 결과
        axes[2].imshow(image)
        axes[2].set_title(f'Matched Results ({len(matches)} matches)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # 매칭된 탐지들
        for match in matches:
            pt_box = pt_boxes[match['pt_idx']]
            onnx_box = onnx_boxes[match['onnx_idx']]
            
            color = 'green' if match['class_match'] else 'orange'
            
            # PT 박스 (실선)
            rect_pt = plt.Rectangle((pt_box[0], pt_box[1]), pt_box[2]-pt_box[0], pt_box[3]-pt_box[1], 
                                  fill=False, color=color, linewidth=2, linestyle='-')
            axes[2].add_patch(rect_pt)
            
            # ONNX 박스 (점선)
            rect_onnx = plt.Rectangle((onnx_box[0], onnx_box[1]), onnx_box[2]-onnx_box[0], onnx_box[3]-onnx_box[1], 
                                    fill=False, color=color, linewidth=2, linestyle='--')
            axes[2].add_patch(rect_onnx)
            
            # IoU 표시
            center_x = (pt_box[0] + pt_box[2]) / 2
            center_y = (pt_box[1] + pt_box[3]) / 2
            axes[2].text(center_x, center_y, f'IoU: {match["iou"]:.2f}', 
                        color=color, fontsize=8, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.results_dir, f'detection_{base_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

    def benchmark_single_image(self, image_path, save_visualization=True, debug_classes=False):
        """단일 이미지에 대한 벤치마크"""
        print(f"\n📸 이미지 처리: {os.path.basename(image_path)}")
        
        # PT 모델 예측
        pt_boxes, pt_scores, pt_labels, pt_time = self.pt_model.predict(image_path)
        
        # ONNX 모델 예측
        onnx_boxes, onnx_scores, onnx_labels, onnx_time = self.onnx_model.predict(image_path)
        
        # 클래스 정보 디버깅 출력
        if debug_classes and (len(pt_labels) > 0 or len(onnx_labels) > 0):
            print(f"   디버그 - PT 클래스: {pt_labels.tolist() if len(pt_labels) > 0 else 'None'}")
            print(f"   디버그 - ONNX 클래스: {onnx_labels.tolist() if len(onnx_labels) > 0 else 'None'}")
            
            # 클래스 이름도 출력
            if len(pt_labels) > 0:
                pt_class_names = [self.pt_model.class_names[int(label)] if int(label) < len(self.pt_model.class_names) else f"unknown_{label}" for label in pt_labels]
                print(f"   디버그 - PT 클래스명: {pt_class_names}")
            
            if len(onnx_labels) > 0:
                onnx_class_names = [self.onnx_model.class_names[int(label)] if int(label) < len(self.onnx_model.class_names) else f"unknown_{label}" for label in onnx_labels]
                print(f"   디버그 - ONNX 클래스명: {onnx_class_names}")
        
        # 결과 매칭
        matches, unmatched_pt, unmatched_onnx = self.match_detections(
            pt_boxes, pt_labels, onnx_boxes, onnx_labels
        )
        
        # 매칭 결과 디버깅
        if debug_classes and matches:
            print(f"   디버그 - 매칭 결과:")
            for i, match in enumerate(matches):
                pt_class_name = self.pt_model.class_names[match['pt_class']] if match['pt_class'] < len(self.pt_model.class_names) else f"unknown_{match['pt_class']}"
                onnx_class_name = self.onnx_model.class_names[match['onnx_class']] if match['onnx_class'] < len(self.onnx_model.class_names) else f"unknown_{match['onnx_class']}"
                print(f"     매칭 {i+1}: PT({match['pt_class']}:{pt_class_name}) vs ONNX({match['onnx_class']}:{onnx_class_name}) = {match['class_match']}")
        
        # 클래스별 통계 업데이트
        self.update_class_stats(pt_labels, onnx_labels, matches)
        
        # 시각화 생성
        viz_path = None
        if save_visualization and (len(pt_boxes) > 0 or len(onnx_boxes) > 0):
            viz_path = self.create_detection_visualization(
                image_path, pt_boxes, pt_labels, pt_scores,
                onnx_boxes, onnx_labels, onnx_scores, matches
            )
        
        # 통계 계산
        total_pt = len(pt_boxes)
        total_onnx = len(onnx_boxes)
        matched_count = len(matches)
        class_match_count = sum(1 for m in matches if m['class_match'])
        
        results = {
            'image': os.path.basename(image_path),
            'image_path': image_path,
            'pt_detections': total_pt,
            'onnx_detections': total_onnx,
            'matched_detections': matched_count,
            'class_accuracy': class_match_count / matched_count if matched_count > 0 else 0,
            'pt_inference_time': pt_time,
            'onnx_inference_time': onnx_time,
            'speed_improvement': pt_time / onnx_time if onnx_time > 0 else 0,
            'unmatched_pt': len(unmatched_pt),
            'unmatched_onnx': len(unmatched_onnx),
            'avg_iou': np.mean([m['iou'] for m in matches]) if matches else 0,
            'visualization_path': viz_path,
            'matches': matches
        }
        
        # 상세 결과 출력
        print(f"   PT 탐지: {total_pt}개, ONNX 탐지: {total_onnx}개")
        print(f"   매칭: {matched_count}개, 클래스 정확도: {results['class_accuracy']:.1%}")
        print(f"   평균 IoU: {results['avg_iou']:.3f}")
        print(f"   속도: PT {pt_time:.1f}ms vs ONNX {onnx_time:.1f}ms ({results['speed_improvement']:.1f}x)")
        if viz_path:
            print(f"   시각화 저장: {os.path.basename(viz_path)}")
        
        return results

    def create_comprehensive_visualizations(self, all_results):
        """종합적인 시각화 생성"""
        df = pd.DataFrame(all_results)
        
        # 1. 전체 성능 비교 대시보드
        self.create_performance_dashboard(df)
        
        # 2. 클래스별 성능 분석
        self.create_class_analysis(df)
        
        # 3. 시간별 성능 추이
        self.create_time_analysis(df)
        
        # 4. 상세 통계 리포트
        self.create_detailed_report(df)
        
        # 5. IoU 분포 분석
        self.create_iou_analysis(df)

    def create_performance_dashboard(self, df):
        """성능 비교 대시보드"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 전체 탐지 수 비교
        ax1 = fig.add_subplot(gs[0, 0])
        total_pt = df['pt_detections'].sum()
        total_onnx = df['onnx_detections'].sum()
        bars = ax1.bar(['PT Model', 'ONNX Model'], [total_pt, total_onnx], 
                      color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Total Detections', fontweight='bold')
        ax1.set_ylabel('Number of Detections')
        for bar, value in zip(bars, [total_pt, total_onnx]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. 평균 추론 시간 비교
        ax2 = fig.add_subplot(gs[0, 1])
        avg_pt_time = df['pt_inference_time'].mean()
        avg_onnx_time = df['onnx_inference_time'].mean()
        bars = ax2.bar(['PT Model', 'ONNX Model'], [avg_pt_time, avg_onnx_time], 
                      color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Average Inference Time', fontweight='bold')
        ax2.set_ylabel('Time (ms)')
        for bar, value in zip(bars, [avg_pt_time, avg_onnx_time]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 3. 속도 향상 분포
        ax3 = fig.add_subplot(gs[0, 2])
        speed_improvements = df['speed_improvement'].values
        ax3.hist(speed_improvements, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        ax3.axvline(speed_improvements.mean(), color='red', linestyle='--', 
                   label=f'Mean: {speed_improvements.mean():.1f}x')
        ax3.set_title('Speed Improvement Distribution', fontweight='bold')
        ax3.set_xlabel('Speed Improvement (x)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. 클래스 정확도 분포
        ax4 = fig.add_subplot(gs[0, 3])
        class_accuracies = df['class_accuracy'].values
        ax4.hist(class_accuracies, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.axvline(class_accuracies.mean(), color='red', linestyle='--', 
                   label=f'Mean: {class_accuracies.mean():.2f}')
        ax4.set_title('Class Accuracy Distribution', fontweight='bold')
        ax4.set_xlabel('Class Accuracy')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. 이미지별 탐지 수 비교
        ax5 = fig.add_subplot(gs[1, :2])
        x = range(len(df))
        width = 0.35
        ax5.bar([i - width/2 for i in x], df['pt_detections'], width, 
               label='PT Model', color='#FF6B6B', alpha=0.8)
        ax5.bar([i + width/2 for i in x], df['onnx_detections'], width, 
               label='ONNX Model', color='#4ECDC4', alpha=0.8)
        ax5.set_title('Detections per Image', fontweight='bold')
        ax5.set_xlabel('Image Index')
        ax5.set_ylabel('Number of Detections')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 이미지별 추론 시간
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.plot(x, df['pt_inference_time'], 'o-', label='PT Model', 
                color='#FF6B6B', linewidth=2, markersize=6)
        ax6.plot(x, df['onnx_inference_time'], 's-', label='ONNX Model', 
                color='#4ECDC4', linewidth=2, markersize=6)
        ax6.set_title('Inference Time per Image', fontweight='bold')
        ax6.set_xlabel('Image Index')
        ax6.set_ylabel('Time (ms)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 매칭 성능 요약
        ax7 = fig.add_subplot(gs[2, :2])
        categories = ['Total PT', 'Total ONNX', 'Matched', 'Unmatched PT', 'Unmatched ONNX']
        values = [
            df['pt_detections'].sum(),
            df['onnx_detections'].sum(),
            df['matched_detections'].sum(),
            df['unmatched_pt'].sum(),
            df['unmatched_onnx'].sum()
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFB4B4', '#B4E4E1']
        bars = ax7.bar(categories, values, color=colors)
        ax7.set_title('Detection Matching Summary', fontweight='bold')
        ax7.set_ylabel('Count')
        for bar, value in zip(bars, values):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 8. 성능 메트릭 요약
        ax8 = fig.add_subplot(gs[2, 2:])
        metrics = ['Avg Speed\nImprovement', 'Avg Class\nAccuracy', 'Avg IoU', 'Match Rate']
        values = [
            df['speed_improvement'].mean(),
            df['class_accuracy'].mean(),
            df['avg_iou'].mean(),
            df['matched_detections'].sum() / df['pt_detections'].sum() if df['pt_detections'].sum() > 0 else 0
        ]
        colors = ['gold', 'lightcoral', 'lightblue', 'lightgreen']
        bars = ax8.bar(metrics, values, color=colors)
        ax8.set_title('Performance Metrics Summary', fontweight='bold')
        for bar, value in zip(bars, values):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('PT vs ONNX Model Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 저장
        save_path = os.path.join(self.results_dir, 'performance_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 성능 대시보드 저장: {save_path}")

    def create_class_analysis(self, df):
        """클래스별 성능 분석"""
        if not self.class_stats:
            return
            
        # 클래스별 데이터 준비
        class_data = []
        for class_id, stats in self.class_stats.items():
            class_name = (self.pt_model.class_names[class_id] 
                         if class_id < len(self.pt_model.class_names) 
                         else f"class_{class_id}")
            
            precision = stats['matched'] / stats['onnx_count'] if stats['onnx_count'] > 0 else 0
            recall = stats['matched'] / stats['pt_count'] if stats['pt_count'] > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_data.append({
                'class_id': class_id,
                'class_name': class_name,
                'pt_count': stats['pt_count'],
                'onnx_count': stats['onnx_count'],
                'matched': stats['matched'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
        
        class_df = pd.DataFrame(class_data)
        
        if len(class_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 클래스별 탐지 수 비교
        x = range(len(class_df))
        width = 0.35
        axes[0, 0].bar([i - width/2 for i in x], class_df['pt_count'], width, 
                      label='PT Model', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar([i + width/2 for i in x], class_df['onnx_count'], width, 
                      label='ONNX Model', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('Detections by Class', fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_df['class_name'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 클래스별 정밀도, 재현율, F1 점수
        x = range(len(class_df))
        width = 0.25
        axes[0, 1].bar([i - width for i in x], class_df['precision'], width, 
                      label='Precision', color='lightcoral', alpha=0.8)
        axes[0, 1].bar(x, class_df['recall'], width, 
                      label='Recall', color='lightblue', alpha=0.8)
        axes[0, 1].bar([i + width for i in x], class_df['f1_score'], width, 
                      label='F1 Score', color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Performance Metrics by Class', fontweight='bold')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_df['class_name'], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. 매칭 성공률
        match_rates = class_df['matched'] / class_df['pt_count']
        match_rates = match_rates.fillna(0)
        bars = axes[1, 0].bar(range(len(class_df)), match_rates, 
                             color='gold', alpha=0.8)
        axes[1, 0].set_title('Match Rate by Class', fontweight='bold')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Match Rate')
        axes[1, 0].set_xticks(range(len(class_df)))
        axes[1, 0].set_xticklabels(class_df['class_name'], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # 각 막대 위에 값 표시
        for bar, rate in zip(bars, match_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 클래스별 상세 통계 테이블
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = class_df[['class_name', 'pt_count', 'onnx_count', 'matched', 'f1_score']].round(3)
        table = axes[1, 1].table(cellText=table_data.values,
                               colLabels=['Class', 'PT', 'ONNX', 'Matched', 'F1'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Detailed Statistics by Class', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.results_dir, 'class_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 클래스 분석 저장: {save_path}")

    def create_time_analysis(self, df):
        """시간별 성능 추이 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. 누적 추론 시간
        cumulative_pt = df['pt_inference_time'].cumsum()
        cumulative_onnx = df['onnx_inference_time'].cumsum()
        
        axes[0, 0].plot(cumulative_pt, label='PT Model', color='#FF6B6B', linewidth=2)
        axes[0, 0].plot(cumulative_onnx, label='ONNX Model', color='#4ECDC4', linewidth=2)
        axes[0, 0].set_title('Cumulative Inference Time', fontweight='bold')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Cumulative Time (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 이동평균 (rolling average)
        window = min(5, len(df))
        rolling_pt = df['pt_inference_time'].rolling(window=window).mean()
        rolling_onnx = df['onnx_inference_time'].rolling(window=window).mean()
        
        axes[0, 1].plot(rolling_pt, label=f'PT Model ({window}-img avg)', 
                       color='#FF6B6B', linewidth=2)
        axes[0, 1].plot(rolling_onnx, label=f'ONNX Model ({window}-img avg)', 
                       color='#4ECDC4', linewidth=2)
        axes[0, 1].set_title('Rolling Average Inference Time', fontweight='bold')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 속도 향상 추이
        axes[1, 0].plot(df['speed_improvement'], 'o-', color='purple', 
                       linewidth=2, markersize=6)
        axes[1, 0].axhline(y=df['speed_improvement'].mean(), color='red', 
                          linestyle='--', alpha=0.7, label='Average')
        axes[1, 0].set_title('Speed Improvement Trend', fontweight='bold')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Speed Improvement (x)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 정확도 추이
        axes[1, 1].plot(df['class_accuracy'], 's-', color='green', 
                       linewidth=2, markersize=6)
        axes[1, 1].axhline(y=df['class_accuracy'].mean(), color='red', 
                          linestyle='--', alpha=0.7, label='Average')
        axes[1, 1].set_title('Class Accuracy Trend', fontweight='bold')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Class Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.results_dir, 'time_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⏱️ 시간 분석 저장: {save_path}")

    def create_iou_analysis(self, df):
        """IoU 분포 및 분석"""
        # 모든 매칭에서 IoU 값 추출
        all_ious = []
        for _, row in df.iterrows():
            if row['matches']:
                all_ious.extend([match['iou'] for match in row['matches']])
        
        if not all_ious:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. IoU 히스토그램
        axes[0, 0].hist(all_ious, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(all_ious), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_ious):.3f}')
        axes[0, 0].axvline(np.median(all_ious), color='green', linestyle='--', 
                          label=f'Median: {np.median(all_ious):.3f}')
        axes[0, 0].set_title('IoU Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. IoU 박스플롯
        axes[0, 1].boxplot(all_ious, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_title('IoU Box Plot', fontweight='bold')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 이미지별 평균 IoU
        avg_ious_per_image = df['avg_iou'].values
        axes[1, 0].plot(avg_ious_per_image, 'o-', color='orange', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=np.mean(avg_ious_per_image), color='red', linestyle='--', 
                          alpha=0.7, label='Overall Average')
        axes[1, 0].set_title('Average IoU per Image', fontweight='bold')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Average IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. IoU vs 클래스 정확도 상관관계
        valid_indices = (df['avg_iou'] > 0) & (df['class_accuracy'] >= 0)
        if valid_indices.sum() > 0:
            x_vals = df.loc[valid_indices, 'avg_iou']
            y_vals = df.loc[valid_indices, 'class_accuracy']
            
            axes[1, 1].scatter(x_vals, y_vals, alpha=0.6, color='purple', s=50)
            
            # 추세선 추가
            if len(x_vals) > 1:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                axes[1, 1].plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)
                
                # 상관계수 계산
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                               transform=axes[1, 1].transAxes, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            axes[1, 1].set_title('IoU vs Class Accuracy', fontweight='bold')
            axes[1, 1].set_xlabel('Average IoU')
            axes[1, 1].set_ylabel('Class Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.results_dir, 'iou_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🎯 IoU 분석 저장: {save_path}")

    def create_detailed_report(self, df):
        """상세 텍스트 리포트 생성"""
        report_path = os.path.join(self.results_dir, 'detailed_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PT vs ONNX 모델 성능 벤치마크 상세 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 테스트 이미지: {len(df)}개\n\n")
            
            # 전체 요약
            f.write("📊 전체 성능 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 PT 탐지: {df['pt_detections'].sum()}개\n")
            f.write(f"총 ONNX 탐지: {df['onnx_detections'].sum()}개\n")
            f.write(f"총 매칭: {df['matched_detections'].sum()}개\n")
            f.write(f"전체 매칭률: {df['matched_detections'].sum() / df['pt_detections'].sum() * 100:.1f}%\n")
            f.write(f"평균 클래스 정확도: {df['class_accuracy'].mean() * 100:.1f}%\n")
            f.write(f"평균 IoU: {df['avg_iou'].mean():.3f}\n\n")
            
            # 속도 성능
            f.write("⚡ 속도 성능\n")
            f.write("-" * 40 + "\n")
            f.write(f"평균 PT 추론 시간: {df['pt_inference_time'].mean():.1f}ms\n")
            f.write(f"평균 ONNX 추론 시간: {df['onnx_inference_time'].mean():.1f}ms\n")
            f.write(f"평균 속도 향상: {df['speed_improvement'].mean():.1f}x\n")
            f.write(f"최대 속도 향상: {df['speed_improvement'].max():.1f}x\n")
            f.write(f"최소 속도 향상: {df['speed_improvement'].min():.1f}x\n\n")
            
            # 클래스별 성능
            if self.class_stats:
                f.write("🎯 클래스별 성능\n")
                f.write("-" * 40 + "\n")
                for class_id, stats in self.class_stats.items():
                    class_name = (self.pt_model.class_names[class_id] 
                                 if class_id < len(self.pt_model.class_names) 
                                 else f"class_{class_id}")
                    
                    precision = stats['matched'] / stats['onnx_count'] if stats['onnx_count'] > 0 else 0
                    recall = stats['matched'] / stats['pt_count'] if stats['pt_count'] > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f.write(f"\n클래스: {class_name}\n")
                    f.write(f"  PT 탐지: {stats['pt_count']}개\n")
                    f.write(f"  ONNX 탐지: {stats['onnx_count']}개\n")
                    f.write(f"  매칭: {stats['matched']}개\n")
                    f.write(f"  정밀도: {precision:.3f}\n")
                    f.write(f"  재현율: {recall:.3f}\n")
                    f.write(f"  F1 점수: {f1_score:.3f}\n")
            
            # 이미지별 상세 결과
            f.write("\n📸 이미지별 상세 결과\n")
            f.write("-" * 40 + "\n")
            for idx, row in df.iterrows():
                f.write(f"\n{idx+1}. {row['image']}\n")
                f.write(f"   PT 탐지: {row['pt_detections']}개\n")
                f.write(f"   ONNX 탐지: {row['onnx_detections']}개\n")
                f.write(f"   매칭: {row['matched_detections']}개\n")
                f.write(f"   클래스 정확도: {row['class_accuracy']:.3f}\n")
                f.write(f"   평균 IoU: {row['avg_iou']:.3f}\n")
                f.write(f"   PT 시간: {row['pt_inference_time']:.1f}ms\n")
                f.write(f"   ONNX 시간: {row['onnx_inference_time']:.1f}ms\n")
                f.write(f"   속도 향상: {row['speed_improvement']:.1f}x\n")
        
        print(f"📝 상세 리포트 저장: {report_path}")

    def save_results_json(self, all_results):
        """결과를 JSON 형태로 저장"""
        # NumPy 타입을 Python 기본 타입으로 변환하는 함수
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 매칭 정보는 직렬화를 위해 제거하고 NumPy 타입 변환
        clean_results = []
        for result in all_results:
            clean_result = {}
            for k, v in result.items():
                if k != 'matches':  # matches는 복잡한 구조이므로 제외
                    clean_result[str(k)] = convert_numpy_types(v)
            clean_results.append(clean_result)
        
        # 클래스 통계도 NumPy 타입 변환
        converted_class_stats = {}
        for class_id, stats in self.class_stats.items():
            converted_class_stats[str(class_id)] = {
                str(k): convert_numpy_types(v) for k, v in stats.items()
            }
        
        # 요약 통계 계산 및 변환
        summary_stats = {
            'total_images': len(all_results),
            'total_pt_detections': sum(r['pt_detections'] for r in all_results),
            'total_onnx_detections': sum(r['onnx_detections'] for r in all_results),
            'total_matches': sum(r['matched_detections'] for r in all_results),
            'avg_speed_improvement': float(np.mean([r['speed_improvement'] for r in all_results])),
            'avg_class_accuracy': float(np.mean([r['class_accuracy'] for r in all_results])),
            'avg_iou': float(np.mean([r['avg_iou'] for r in all_results]))
        }
        
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary_stats,
            'class_statistics': converted_class_stats,
            'detailed_results': clean_results
        }
        
        json_path = os.path.join(self.results_dir, 'benchmark_results.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON 결과 저장: {json_path}")
        except Exception as e:
            print(f"⚠️ JSON 저장 중 오류 발생: {e}")
            # 오류 발생 시 간단한 버전으로 저장
            try:
                simple_data = {
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_images': len(all_results),
                        'avg_speed_improvement': float(np.mean([r['speed_improvement'] for r in all_results])),
                        'avg_class_accuracy': float(np.mean([r['class_accuracy'] for r in all_results])),
                        'avg_iou': float(np.mean([r['avg_iou'] for r in all_results]))
                    },
                    'note': 'Simplified version due to serialization error'
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_data, f, indent=2, ensure_ascii=False)
                print(f"💾 간단한 JSON 결과 저장: {json_path}")
            except Exception as e2:
                print(f"❌ JSON 저장 완전 실패: {e2}")

    def benchmark_multiple_images(self, image_paths, save_individual_viz=True):
        """여러 이미지에 대한 종합 벤치마크"""
        print("🚀 다중 이미지 벤치마크 시작")
        print("=" * 60)
        
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                print(f"\n진행률: {i+1}/{len(image_paths)}")
                
                # 개별 시각화는 처음 10개 이미지만 저장 (용량 절약)
                save_viz = save_individual_viz and i < 10
                
                result = self.benchmark_single_image(image_path, save_visualization=save_viz)
                all_results.append(result)
            else:
                print(f"⚠️ 이미지 파일 없음: {image_path}")
        
        if not all_results:
            print("❌ 처리할 이미지가 없습니다.")
            return
        
        # 종합 통계 출력
        self.print_summary_stats(all_results)
        
        # 종합 시각화 생성
        print(f"\n🎨 종합 시각화 생성 중...")
        self.create_comprehensive_visualizations(all_results)
        
        # 결과 저장
        self.save_results_json(all_results)
        
        print(f"\n✅ 모든 결과가 '{self.results_dir}' 폴더에 저장되었습니다.")
        
        return all_results

    def print_summary_stats(self, results):
        """벤치마크 결과 요약 통계 출력"""
        df = pd.DataFrame(results)
        
        print(f"\n📊 종합 벤치마크 결과 ({len(results)}개 이미지)")
        print("=" * 60)
        
        print(f"🎯 탐지 성능:")
        print(f"   총 PT 탐지: {df['pt_detections'].sum()}개")
        print(f"   총 ONNX 탐지: {df['onnx_detections'].sum()}개")
        print(f"   총 매칭: {df['matched_detections'].sum()}개")
        print(f"   전체 매칭률: {(df['matched_detections'].sum() / df['pt_detections'].sum() * 100):.1f}%")
        print(f"   평균 클래스 정확도: {df['class_accuracy'].mean():.1%}")
        print(f"   평균 IoU: {df['avg_iou'].mean():.3f}")
        
        print(f"\n⚡ 속도 성능:")
        print(f"   평균 PT 시간: {df['pt_inference_time'].mean():.1f}ms")
        print(f"   평균 ONNX 시간: {df['onnx_inference_time'].mean():.1f}ms")
        print(f"   평균 속도 향상: {df['speed_improvement'].mean():.1f}x")
        print(f"   최대 속도 향상: {df['speed_improvement'].max():.1f}x")
        
        print(f"\n📈 상세 통계:")
        print(f"   미매칭 PT: {df['unmatched_pt'].sum()}개")
        print(f"   미매칭 ONNX: {df['unmatched_onnx'].sum()}개")
        print(f"   속도 향상 표준편차: {df['speed_improvement'].std():.2f}")

def find_images_in_folder(folder_path):
    """폴더에서 이미지 파일들을 찾아서 반환"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if not os.path.exists(folder_path):
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        return []
    
    print(f"📁 폴더 스캔 중: {folder_path}")
    
    # 폴더 내 모든 파일 검색 (하위 폴더 포함)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in image_extensions:
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    print(f"✅ 발견된 이미지: {len(image_files)}개")
    
    # 파일명으로 정렬
    image_files.sort()
    
    # 처음 10개 파일명 출력
    if image_files:
        print("📋 발견된 이미지 (처음 10개):")
        for i, img_path in enumerate(image_files[:10]):
            rel_path = os.path.relpath(img_path, folder_path)
            print(f"   {i+1}. {rel_path}")
        if len(image_files) > 10:
            print(f"   ... 외 {len(image_files) - 10}개")
    
    return image_files

def get_user_inputs():
    """사용자로부터 입력을 받는 함수"""
    
    print("🔧 벤치마크 설정")
    print("=" * 40)
    
    # PT 모델 경로
    print("\n1️⃣ PT 모델 경로:")
    pt_default = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver3\yolov8_custom.pt"
    pt_path = input(f"PT 모델 경로 [{pt_default}]: ").strip()
    if not pt_path:
        pt_path = pt_default
    
    # ONNX 모델 경로  
    print("\n2️⃣ ONNX 모델 경로:")
    onnx_default = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver3\yolov8_custom_fixed.onnx"
    onnx_path = input(f"ONNX 모델 경로 [{onnx_default}]: ").strip()
    if not onnx_path:
        onnx_path = onnx_default
    
    # 이미지 폴더 경로
    print("\n3️⃣ 테스트 이미지 폴더:")
    print("폴더 내 모든 이미지 파일(.jpg, .png 등)이 자동으로 검색됩니다.")
    folder_default = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1"
    folder_path = input(f"이미지 폴더 경로 [{folder_default}]: ").strip()
    if not folder_path:
        folder_path = folder_default
    
    # 신뢰도 임계값
    print("\n4️⃣ 신뢰도 임계값:")
    conf_default = "0.3"
    conf_input = input(f"신뢰도 임계값 (0.0-1.0) [{conf_default}]: ").strip()
    try:
        conf_threshold = float(conf_input) if conf_input else float(conf_default)
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError
    except ValueError:
        print(f"⚠️ 잘못된 값입니다. 기본값 {conf_default} 사용")
        conf_threshold = float(conf_default)
    
    # 최대 이미지 수 제한
    print("\n5️⃣ 최대 테스트 이미지 수:")
    max_default = "10"
    max_input = input(f"최대 이미지 수 (0=전체) [{max_default}]: ").strip()
    try:
        max_images = int(max_input) if max_input else int(max_default)
        if max_images < 0:
            raise ValueError
    except ValueError:
        print(f"⚠️ 잘못된 값입니다. 기본값 {max_default} 사용")
        max_images = int(max_default)
    
    # 개별 이미지 시각화 저장 여부
    print("\n6️⃣ 개별 이미지 시각화:")
    print("각 이미지에 대한 탐지 결과 비교 이미지를 저장할지 선택합니다.")
    print("(주의: 많은 이미지의 경우 저장 공간을 많이 사용합니다)")
    viz_input = input("개별 시각화 저장 (y/n) [y]: ").strip().lower()
    save_individual_viz = viz_input != 'n'
    
    return pt_path, onnx_path, folder_path, conf_threshold, max_images, save_individual_viz

def create_summary_html_report(results_dir, summary_data=None):
    """HTML 형태의 요약 리포트 생성"""
    # 기본 메트릭 (JSON 파일에서 로드될 수 있음)
    default_metrics = {
        'avg_speed_improvement': 0.0,
        'avg_class_accuracy': 0.0,
        'avg_iou': 0.0,
        'match_rate': 0.0
    }
    
    if summary_data:
        default_metrics.update(summary_data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PT vs ONNX 벤치마크 결과</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .image-card {{
                text-align: center;
            }}
            .image-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #666;
            }}
            .alert {{
                background-color: #e7f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 PT vs ONNX 모델 성능 벤치마크</h1>
            <p>생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
        </div>
        
        <div class="card">
            <h2>📊 주요 성능 지표</h2>
            <div class="metric-grid">
                <div class="metric-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #FF6B6BAA 100%);">
                    <div class="metric-value">{default_metrics['avg_speed_improvement']:.1f}x</div>
                    <div>평균 속도 향상</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #4ECDC4AA 100%);">
                    <div class="metric-value">{default_metrics['avg_class_accuracy']:.1%}</div>
                    <div>평균 클래스 정확도</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #45B7D1 0%, #45B7D1AA 100%);">
                    <div class="metric-value">{default_metrics['avg_iou']:.3f}</div>
                    <div>평균 IoU</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #96CEB4 0%, #96CEB4AA 100%);">
                    <div class="metric-value">{default_metrics['match_rate']:.1%}</div>
                    <div>전체 매칭률</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 성능 분석 그래프</h2>
            <div class="alert">
                <strong>💡 참고:</strong> 아래 그래프들을 클릭하면 더 큰 크기로 볼 수 있습니다.
            </div>
            <div class="image-grid">
                <div class="image-card">
                    <h3>성능 대시보드</h3>
                    <a href="performance_dashboard.png" target="_blank">
                        <img src="performance_dashboard.png" alt="성능 대시보드">
                    </a>
                    <p>전체적인 성능 비교와 주요 메트릭을 한눈에 볼 수 있습니다.</p>
                </div>
                <div class="image-card">
                    <h3>클래스별 분석</h3>
                    <a href="class_analysis.png" target="_blank">
                        <img src="class_analysis.png" alt="클래스별 분석">
                    </a>
                    <p>각 클래스별 정밀도, 재현율, F1 점수를 분석합니다.</p>
                </div>
                <div class="image-card">
                    <h3>시간 추이 분석</h3>
                    <a href="time_analysis.png" target="_blank">
                        <img src="time_analysis.png" alt="시간 추이 분석">
                    </a>
                    <p>시간에 따른 성능 변화와 안정성을 확인합니다.</p>
                </div>
                <div class="image-card">
                    <h3>IoU 분포 분석</h3>
                    <a href="iou_analysis.png" target="_blank">
                        <img src="iou_analysis.png" alt="IoU 분포 분석">
                    </a>
                    <p>IoU 분포와 정확도 간의 상관관계를 분석합니다.</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📁 생성된 파일 목록</h2>
            <div class="alert">
                <strong>📋 파일 설명:</strong>
                <ul>
                    <li><strong>benchmark_results.json</strong> - 모든 벤치마크 결과 데이터 (JSON 형식)</li>
                    <li><strong>detailed_report.txt</strong> - 상세한 텍스트 리포트</li>
                    <li><strong>performance_dashboard.png</strong> - 종합 성능 대시보드</li>
                    <li><strong>class_analysis.png</strong> - 클래스별 성능 분석</li>
                    <li><strong>time_analysis.png</strong> - 시간 추이 및 안정성 분석</li>
                    <li><strong>iou_analysis.png</strong> - IoU 분포 및 상관관계 분석</li>
                    <li><strong>detection_*.png</strong> - 개별 이미지 탐지 결과 비교</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 결과 요약</h2>
            <table>
                <tr>
                    <th>항목</th>
                    <th>PT 모델</th>
                    <th>ONNX 모델</th>
                    <th>개선도</th>
                </tr>
                <tr>
                    <td>평균 추론 시간</td>
                    <td id="pt-time">-</td>
                    <td id="onnx-time">-</td>
                    <td id="speed-improvement">-</td>
                </tr>
                <tr>
                    <td>평균 탐지 수</td>
                    <td id="pt-detections">-</td>
                    <td id="onnx-detections">-</td>
                    <td id="detection-difference">-</td>
                </tr>
                <tr>
                    <td>매칭 정확도</td>
                    <td colspan="2" id="match-accuracy">-</td>
                    <td>-</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>🔬 이 리포트는 PT와 ONNX 모델의 성능을 종합적으로 비교한 결과입니다.</p>
            <p>📄 자세한 내용은 <a href="detailed_report.txt">detailed_report.txt</a> 파일을 참조하세요.</p>
            <p>💾 원시 데이터는 <a href="benchmark_results.json">benchmark_results.json</a> 파일에서 확인할 수 있습니다.</p>
        </div>
        
        <script>
            // JSON 데이터를 로드하여 동적으로 표시 (실제 구현에서는 fetch 사용)
            function loadBenchmarkData() {{
                // 실제로는 benchmark_results.json에서 데이터를 로드
                // 여기서는 예시 데이터 사용
                document.getElementById('pt-time').textContent = '-';
                document.getElementById('onnx-time').textContent = '-';
                document.getElementById('speed-improvement').textContent = '{default_metrics["avg_speed_improvement"]:.1f}x';
                document.getElementById('pt-detections').textContent = '-';
                document.getElementById('onnx-detections').textContent = '-';
                document.getElementById('detection-difference').textContent = '-';
                document.getElementById('match-accuracy').textContent = '{default_metrics["avg_class_accuracy"]:.1%}';
            }}
            
            // 페이지 로드 시 데이터 로드
            window.onload = loadBenchmarkData;
        </script>
    </body>
    </html>
    """
    
    html_path = os.path.join(results_dir, 'benchmark_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🌐 HTML 리포트 저장: {html_path}")
    return html_path

def main():
    """메인 벤치마크 실행"""
    
    print("🔬 PT vs ONNX 모델 벤치마크 비교 (고급 시각화 버전)")
    print("=" * 70)
    
    # 사용자 입력 받기
    pt_path, onnx_path, folder_path, conf_threshold, max_images, save_individual_viz = get_user_inputs()
    
    print(f"\n📋 설정 확인:")
    print(f"   PT 모델: {pt_path}")
    print(f"   ONNX 모델: {onnx_path}")
    print(f"   이미지 폴더: {folder_path}")
    print(f"   신뢰도 임계값: {conf_threshold}")
    print(f"   최대 이미지 수: {max_images if max_images > 0 else '전체'}")
    print(f"   개별 시각화 저장: {'예' if save_individual_viz else '아니오'}")
    
    # 파일 존재 확인
    if not os.path.exists(pt_path):
        print(f"❌ PT 모델 파일이 없습니다: {pt_path}")
        return
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 모델 파일이 없습니다: {onnx_path}")
        return
    
    # 이미지 파일 검색
    image_files = find_images_in_folder(folder_path)
    
    if not image_files:
        print("❌ 테스트할 이미지가 없습니다.")
        return
    
    # 이미지 수 제한
    if max_images > 0 and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"📊 테스트 이미지를 {max_images}개로 제한합니다.")
    
    # 실행 확인
    print(f"\n🚀 {len(image_files)}개 이미지로 벤치마크를 실행합니다.")
    print(f"💾 결과는 'benchmark_results_YYYYMMDD_HHMMSS' 폴더에 저장됩니다.")
    confirm = input("계속하시겠습니까? (y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("❌ 벤치마크가 취소되었습니다.")
        return
    
    print("\n" + "=" * 70)
    
    try:
        # 벤치마크 객체 생성
        comparator = EnhancedBenchmarkComparator(pt_path, onnx_path, conf_threshold)
        
        # 벤치마크 실행
        results = comparator.benchmark_multiple_images(image_files, save_individual_viz)
        
        # 요약 데이터 준비
        df = pd.DataFrame(results)
        summary_data = {
            'avg_speed_improvement': df['speed_improvement'].mean(),
            'avg_class_accuracy': df['class_accuracy'].mean(),
            'avg_iou': df['avg_iou'].mean(),
            'match_rate': df['matched_detections'].sum() / df['pt_detections'].sum() if df['pt_detections'].sum() > 0 else 0
        }
        
        # HTML 리포트 생성
        html_path = create_summary_html_report(comparator.results_dir, summary_data)
        
        print(f"\n🏁 벤치마크 완료!")
        print(f"📁 결과 폴더: {comparator.results_dir}")
        print("\n📋 생성된 주요 파일:")
        print(f"   • benchmark_report.html - 웹 브라우저용 종합 리포트")
        print(f"   • performance_dashboard.png - 성능 비교 대시보드")
        print(f"   • class_analysis.png - 클래스별 성능 분석")
        print(f"   • time_analysis.png - 시간 추이 분석")
        print(f"   • iou_analysis.png - IoU 분포 분석")
        print(f"   • detailed_report.txt - 상세 텍스트 리포트")
        print(f"   • benchmark_results.json - 원시 데이터 (JSON)")
        
        if save_individual_viz:
            detection_files = [f for f in os.listdir(comparator.results_dir) if f.startswith('detection_')]
            if detection_files:
                print(f"   • detection_*.png - 개별 이미지 탐지 결과 ({len(detection_files)}개)")
        
        print(f"\n🌐 웹 리포트 확인: {html_path}")
        print(f"   브라우저에서 위 파일을 열어 결과를 확인하세요.")
        
        # 간단한 요약 출력
        print(f"\n📊 간단 요약:")
        print(f"   • 총 처리 이미지: {len(results)}개")
        print(f"   • 평균 속도 향상: {summary_data['avg_speed_improvement']:.1f}x")
        print(f"   • 평균 클래스 정확도: {summary_data['avg_class_accuracy']:.1%}")
        print(f"   • 평균 IoU: {summary_data['avg_iou']:.3f}")
        print(f"   • 전체 매칭률: {summary_data['match_rate']:.1%}")
        
        # 추가 분석 제안
        print(f"\n💡 추가 분석 제안:")
        if summary_data['avg_speed_improvement'] > 2.0:
            print(f"   ✅ ONNX 모델이 PT 모델보다 {summary_data['avg_speed_improvement']:.1f}배 빠릅니다!")
        if summary_data['avg_class_accuracy'] > 0.8:
            print(f"   ✅ 높은 클래스 정확도({summary_data['avg_class_accuracy']:.1%})를 보입니다!")
        if summary_data['avg_iou'] > 0.7:
            print(f"   ✅ 우수한 IoU 점수({summary_data['avg_iou']:.3f})입니다!")
        if summary_data['match_rate'] < 0.7:
            print(f"   ⚠️ 매칭률({summary_data['match_rate']:.1%})이 낮습니다. 임계값 조정을 고려해보세요.")
        
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

def fix_openmp_issues():
    """OpenMP 관련 문제들을 사전에 해결하는 함수"""
    import os
    import warnings
    
    # 환경 변수 설정
    env_vars = {
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'CUDA_LAUNCH_BLOCKING': '1'  # CUDA 동기화 (GPU 사용시)
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # 경고 메시지 억제
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # PyTorch 설정
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
        print("✅ PyTorch 멀티스레딩 설정 완료")
    except Exception as e:
        print(f"⚠️ PyTorch 설정 중 오류: {e}")
    
    # OpenCV 설정
    try:
        import cv2
        cv2.setNumThreads(1)
        print("✅ OpenCV 멀티스레딩 설정 완료")
    except Exception as e:
        print(f"⚠️ OpenCV 설정 중 오류: {e}")
    
    # NumPy 설정
    try:
        import numpy as np
        if hasattr(np, '__config__') and hasattr(np.__config__, 'show'):
            # NumPy의 BLAS 라이브러리 정보 확인 (선택사항)
            pass
        print("✅ NumPy 설정 확인 완료")
    except Exception as e:
        print(f"⚠️ NumPy 설정 중 오류: {e}")

if __name__ == "__main__":
    # OpenMP 문제 사전 해결
    print("🔧 시스템 설정 최적화 중...")
    fix_openmp_issues()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류가 발생했습니다: {e}")
        
        # OpenMP 관련 오류인 경우 추가 안내
        if "libiomp5md.dll" in str(e) or "OpenMP" in str(e):
            print("\n🔧 OpenMP 오류 해결 방법:")
            print("1. Anaconda 사용시: conda install intel-openmp")
            print("2. pip 사용시: pip uninstall intel-openmp && pip install intel-openmp")
            print("3. 시스템 재시작 후 다시 실행")
            print("4. 가상환경을 새로 만들어서 실행")
        
        import traceback
        traceback.print_exc()