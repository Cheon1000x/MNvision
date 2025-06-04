#!/usr/bin/env python3
"""
PT 모델 vs ONNX 모델 성능 벤치마크 비교
정확도, 속도, 탐지 결과를 종합적으로 비교
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
    """ONNX 모델 래퍼 (기존 ONNXDetector 기반)"""
    def __init__(self, model_path, input_size=(320, 192), conf_threshold=0.3, iou_threshold=0.1):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_width, self.input_height = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 수정된 클래스 순서 (현재 버전)
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

class BenchmarkComparator:
    """PT와 ONNX 모델 벤치마크 비교기"""
    
    def __init__(self, pt_model_path, onnx_model_path, conf_threshold=0.3):
        print("🔄 모델 로딩 중...")
        self.pt_model = PTModel(pt_model_path, conf_threshold)
        self.onnx_model = ONNXModel(onnx_model_path, conf_threshold=conf_threshold)
        
        print(f"✅ PT 모델 로드 완료: {len(self.pt_model.class_names)}개 클래스")
        print(f"✅ ONNX 모델 로드 완료: {len(self.onnx_model.class_names)}개 클래스")
        
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
                matches.append({
                    'pt_idx': i,
                    'onnx_idx': best_match,
                    'iou': best_iou,
                    'class_match': pt_labels[i] == onnx_labels[best_match]
                })
                unmatched_pt.remove(i)
                unmatched_onnx.remove(best_match)
        
        return matches, unmatched_pt, unmatched_onnx

    def benchmark_single_image(self, image_path):
        """단일 이미지에 대한 벤치마크"""
        print(f"\n📸 이미지 처리: {os.path.basename(image_path)}")
        
        # PT 모델 예측
        pt_boxes, pt_scores, pt_labels, pt_time = self.pt_model.predict(image_path)
        
        # ONNX 모델 예측
        onnx_boxes, onnx_scores, onnx_labels, onnx_time = self.onnx_model.predict(image_path)
        
        # 결과 매칭
        matches, unmatched_pt, unmatched_onnx = self.match_detections(
            pt_boxes, pt_labels, onnx_boxes, onnx_labels
        )
        
        # 통계 계산
        total_pt = len(pt_boxes)
        total_onnx = len(onnx_boxes)
        matched_count = len(matches)
        class_match_count = sum(1 for m in matches if m['class_match'])
        
        results = {
            'image': os.path.basename(image_path),
            'pt_detections': total_pt,
            'onnx_detections': total_onnx,
            'matched_detections': matched_count,
            'class_accuracy': class_match_count / matched_count if matched_count > 0 else 0,
            'pt_inference_time': pt_time,
            'onnx_inference_time': onnx_time,
            'speed_improvement': pt_time / onnx_time if onnx_time > 0 else 0,
            'unmatched_pt': len(unmatched_pt),
            'unmatched_onnx': len(unmatched_onnx)
        }
        
        # 상세 결과 출력
        print(f"   PT 탐지: {total_pt}개, ONNX 탐지: {total_onnx}개")
        print(f"   매칭: {matched_count}개, 클래스 정확도: {results['class_accuracy']:.1%}")
        print(f"   속도: PT {pt_time:.1f}ms vs ONNX {onnx_time:.1f}ms ({results['speed_improvement']:.1f}x)")
        
        return results

    def benchmark_multiple_images(self, image_paths):
        """여러 이미지에 대한 종합 벤치마크"""
        print("🚀 다중 이미지 벤치마크 시작")
        print("=" * 60)
        
        all_results = []
        
        for image_path in image_paths:
            if os.path.exists(image_path):
                result = self.benchmark_single_image(image_path)
                all_results.append(result)
            else:
                print(f"⚠️ 이미지 파일 없음: {image_path}")
        
        if not all_results:
            print("❌ 처리할 이미지가 없습니다.")
            return
        
        # 종합 통계
        self.print_summary_stats(all_results)
        
        # 결과 시각화
        self.visualize_results(all_results)
        
        return all_results

    def print_summary_stats(self, results):
        """벤치마크 결과 요약 통계 출력"""
        df = pd.DataFrame(results)
        
        print(f"\n📊 종합 벤치마크 결과 ({len(results)}개 이미지)")
        print("=" * 60)
        
        print(f"🎯 탐지 성능:")
        print(f"   평균 PT 탐지: {df['pt_detections'].mean():.1f}개")
        print(f"   평균 ONNX 탐지: {df['onnx_detections'].mean():.1f}개")
        print(f"   평균 매칭률: {(df['matched_detections'] / df['pt_detections']).mean():.1%}")
        print(f"   평균 클래스 정확도: {df['class_accuracy'].mean():.1%}")
        
        print(f"\n⚡ 속도 성능:")
        print(f"   평균 PT 시간: {df['pt_inference_time'].mean():.1f}ms")
        print(f"   평균 ONNX 시간: {df['onnx_inference_time'].mean():.1f}ms")
        print(f"   평균 속도 향상: {df['speed_improvement'].mean():.1f}x")
        
        print(f"\n📈 상세 통계:")
        print(f"   총 PT 탐지: {df['pt_detections'].sum()}개")
        print(f"   총 ONNX 탐지: {df['onnx_detections'].sum()}개")
        print(f"   총 매칭: {df['matched_detections'].sum()}개")
        print(f"   미매칭 PT: {df['unmatched_pt'].sum()}개")
        print(f"   미매칭 ONNX: {df['unmatched_onnx'].sum()}개")

    def visualize_results(self, results):
        """벤치마크 결과 시각화"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PT vs ONNX 모델 성능 비교', fontsize=16, fontweight='bold')
        
        # 1. 탐지 수 비교
        axes[0, 0].bar(['PT', 'ONNX'], [df['pt_detections'].sum(), df['onnx_detections'].sum()])
        axes[0, 0].set_title('총 탐지 수 비교')
        axes[0, 0].set_ylabel('탐지 수')
        
        # 2. 추론 시간 비교
        x = range(len(results))
        axes[0, 1].plot(x, df['pt_inference_time'], 'o-', label='PT', linewidth=2)
        axes[0, 1].plot(x, df['onnx_inference_time'], 's-', label='ONNX', linewidth=2)
        axes[0, 1].set_title('이미지별 추론 시간')
        axes[0, 1].set_xlabel('이미지 인덱스')
        axes[0, 1].set_ylabel('시간 (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 클래스 정확도
        axes[1, 0].bar(range(len(results)), df['class_accuracy'])
        axes[1, 0].set_title('이미지별 클래스 정확도')
        axes[1, 0].set_xlabel('이미지 인덱스')
        axes[1, 0].set_ylabel('정확도')
        axes[1, 0].set_ylim(0, 1)
        
        # 4. 속도 향상 배수
        axes[1, 1].bar(range(len(results)), df['speed_improvement'])
        axes[1, 1].set_title('속도 향상 배수')
        axes[1, 1].set_xlabel('이미지 인덱스')
        axes[1, 1].set_ylabel('배수 (x)')
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='1x (동일)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 벤치마크 결과 그래프 저장: benchmark_results.png")
        plt.show()

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
    pt_default = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver5(320x180)\best.pt"
    pt_path = input(f"PT 모델 경로 [{pt_default}]: ").strip()
    if not pt_path:
        pt_path = pt_default
    
    # ONNX 모델 경로  
    print("\n2️⃣ ONNX 모델 경로:")
    onnx_default = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver5(320x180)\yolov8_custom_fixed.onnx"
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
    
    return pt_path, onnx_path, folder_path, conf_threshold, max_images

def main():
    """메인 벤치마크 실행"""
    
    print("🔬 PT vs ONNX 모델 벤치마크 비교")
    print("=" * 60)
    
    # 사용자 입력 받기
    pt_path, onnx_path, folder_path, conf_threshold, max_images = get_user_inputs()
    
    print(f"\n📋 설정 확인:")
    print(f"   PT 모델: {pt_path}")
    print(f"   ONNX 모델: {onnx_path}")
    print(f"   이미지 폴더: {folder_path}")
    print(f"   신뢰도 임계값: {conf_threshold}")
    print(f"   최대 이미지 수: {max_images if max_images > 0 else '전체'}")
    
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
    confirm = input("계속하시겠습니까? (y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("❌ 벤치마크가 취소되었습니다.")
        return
    
    print("\n" + "=" * 60)
    
    # 벤치마크 객체 생성
    comparator = BenchmarkComparator(pt_path, onnx_path, conf_threshold)
    
    # 벤치마크 실행
    results = comparator.benchmark_multiple_images(image_files)
    
    print(f"\n🏁 벤치마크 완료!")
    print("결과 파일:")
    print("- benchmark_results.png: 성능 비교 그래프")
    print(f"- 총 처리된 이미지: {len(image_files)}개")

if __name__ == "__main__":
    main()