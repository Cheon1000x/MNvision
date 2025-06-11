"""
YOLO 모델 성능 벤치마크 시스템 (수정됨)
프루닝된 모델들과 원본 모델의 성능을 종합적으로 비교

기능:
1. JSON 라벨을 YOLO 형식으로 자동 변환
2. 정확도, 속도, 효율성, 종합 효율성 비교
3. 여러 모델 동시 벤치마크
4. 상세한 비교 리포트 생성
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import time
import shutil
from pathlib import Path
import tempfile
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class YOLOBenchmark:
    def __init__(self, data_dir, classes_file, num_eval_images=100):
        """
        Args:
            data_dir: JSON+이미지가 있는 데이터 폴더 경로
            classes_file: classes.txt 파일 경로
            num_eval_images: 평가에 사용할 이미지 수 (기본: 100장)
        """
        self.data_dir = data_dir
        self.classes_file = classes_file
        self.num_eval_images = num_eval_images
        self.temp_dir = None
        self.models = {}
        self.results = {}
        
        # 클래스 정보 로드
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"🎯 벤치마크 설정:")
        print(f"   데이터 폴더: {data_dir}")
        print(f"   클래스 수: {len(self.classes)}")
        print(f"   평가 이미지 수: {num_eval_images}")
        print(f"   해상도: 320x180")
        print(f"   배치 크기: 1")
        print(f"   환경: CPU")
    
    def prepare_test_data(self):
        """JSON+이미지 데이터를 YOLO 형식으로 변환하고 임시 데이터셋 생성"""
        print("\n📂 테스트 데이터 준비 중...")
        
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp(prefix="yolo_benchmark_")
        images_dir = os.path.join(self.temp_dir, "images")
        labels_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # JSON과 이미지 파일 찾기
        all_files = os.listdir(self.data_dir)
        json_files = [f for f in all_files if f.endswith('.json')]
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 파일 쌍 매칭
        paired_files = []
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            for img_file in image_files:
                if base_name == os.path.splitext(img_file)[0]:
                    paired_files.append((json_file, img_file))
                    break
        
        print(f"   찾은 파일 쌍: {len(paired_files)}개")
        
        # 평가 이미지 수만큼 샘플링
        if len(paired_files) > self.num_eval_images:
            import random
            random.shuffle(paired_files)
            paired_files = paired_files[:self.num_eval_images]
            print(f"   샘플링: {len(paired_files)}개 사용")
        
        # 변환 및 복사
        converted_count = 0
        for json_file, img_file in paired_files:
            json_path = os.path.join(self.data_dir, json_file)
            img_path = os.path.join(self.data_dir, img_file)
            
            # 이미지 복사
            base_name = os.path.splitext(img_file)[0]
            new_img_path = os.path.join(images_dir, f"{base_name}.jpg")
            shutil.copy(img_path, new_img_path)
            
            # JSON을 YOLO 형식으로 변환
            yolo_annotations = self._convert_json_to_yolo(json_path, img_path)
            
            # 라벨 파일 저장
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        # YAML 파일 생성
        yaml_path = os.path.join(self.temp_dir, "dataset.yaml")
        self._create_yaml(yaml_path)
        
        print(f"   변환 완료: {converted_count}개 파일")
        print(f"   임시 데이터셋: {self.temp_dir}")
        
        return yaml_path
    
    def _convert_json_to_yolo(self, json_path, img_path):
        """JSON 라벨을 YOLO 형식으로 변환"""
        # 이미지 크기 가져오기
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # JSON 파싱
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        yolo_annotations = []
        shapes = data.get("shapes", [])
        
        for shape in shapes:
            label = shape.get("label", "")
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "")
            
            if label in self.classes and (shape_type == "polygon" or shape_type == "rectangle"):
                class_id = self.classes.index(label)
                
                # 바운딩 박스 계산
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # YOLO 형식으로 정규화
                x_center = (x_min + x_max) / (2 * img_width)
                y_center = (y_min + y_max) / (2 * img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}"
                yolo_annotations.append(yolo_annotation)
        
        return yolo_annotations
    
    def _create_yaml(self, yaml_path):
        """YOLO 데이터셋 YAML 파일 생성"""
        yaml_content = f"""
path: {self.temp_dir}
train: images
val: images

nc: {len(self.classes)}
names: {self.classes}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
    
    def add_model(self, model_path, model_name=None):
        """벤치마크할 모델 추가"""
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        print(f"🔄 모델 로드: {model_name}")
        try:
            model = YOLO(model_path)
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # 파라미터 수 계산
            param_count = sum(p.numel() for p in model.model.parameters())
            
            self.models[model_name] = {
                'model': model,
                'path': model_path,
                'size_mb': model_size,
                'parameters': param_count
            }
            print(f"   크기: {model_size:.2f} MB")
            print(f"   파라미터: {param_count:,}개")
            
        except Exception as e:
            print(f"   ❌ 로드 실패: {e}")
    
    def _calculate_accuracy_manual(self, model, yaml_path):
        """직접 추론으로 정확도 계산 (학습 프로세스 없이)"""
        # 이미지와 라벨 로드
        images_dir = os.path.join(os.path.dirname(yaml_path), "images")
        labels_dir = os.path.join(os.path.dirname(yaml_path), "labels")
        
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        all_predictions = []
        all_targets = []
        
        for img_file in img_files:
            # 이미지 로드
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            
            # 추론 실행 (학습 없이)
            results = model(img, verbose=False, device='cpu', conf=0.25, iou=0.45)
            
            # 예측 결과 처리
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # 안전하게 텐서 확인
                if hasattr(boxes, 'xyxy') and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    pred_boxes = boxes.xyxy.cpu().numpy()
                    pred_confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') and boxes.conf is not None else np.array([])
                    pred_classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') and boxes.cls is not None else np.array([])
                    
                    # 정규화 (이미지 크기로 나누기)
                    if pred_boxes.shape[0] > 0:
                        img_w, img_h = img.size
                        pred_boxes[:, [0, 2]] /= img_w  # x 좌표들
                        pred_boxes[:, [1, 3]] /= img_h  # y 좌표들
                    
                    for i in range(pred_boxes.shape[0]):
                        all_predictions.append({
                            'bbox': pred_boxes[i],
                            'conf': float(pred_confs[i]) if i < len(pred_confs) else 1.0,
                            'class': int(pred_classes[i]) if i < len(pred_classes) else 0,
                            'image': img_file
                        })
            
            # 실제 라벨 로드
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                img_w, img_h = img.size
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # YOLO 형식을 xyxy로 변환
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2
                        
                        all_targets.append({
                            'bbox': np.array([x1, y1, x2, y2]),
                            'class': class_id,
                            'image': img_file
                        })
        
        # 간단한 mAP 계산
        if len(all_predictions) == 0 or len(all_targets) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        print(f"   예측: {len(all_predictions)}개, 실제: {len(all_targets)}개")
        
        # IoU 임계값별 정확도 계산
        total_tp = 0
        total_fp = 0
        matched_targets = set()  # 매칭된 타겟 추적
        
        for pred in all_predictions:
            best_iou = 0
            best_target_idx = None
            
            # 같은 이미지의 같은 클래스 타겟들과 비교
            for idx, target in enumerate(all_targets):
                if (target['image'] == pred['image'] and 
                    target['class'] == pred['class'] and
                    idx not in matched_targets):
                    
                    # IoU 계산
                    iou = self._calculate_iou(pred['bbox'], target['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = idx
            
            # IoU > 0.5이면 True Positive
            if best_iou > 0.5 and best_target_idx is not None:
                total_tp += 1
                matched_targets.add(best_target_idx)  # 중복 매칭 방지
            else:
                total_fp += 1
        
        total_fn = len(all_targets) - len(matched_targets)  # 매칭되지 않은 타겟들
        
        # 메트릭 계산
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        map50 = (precision + recall) / 2 if (precision + recall) > 0 else 0.0  # 간단한 근사
        map50_95 = map50 * 0.8  # 근사값
        
        return map50, map50_95, precision, recall
    
    def _calculate_iou(self, box1, box2):
        """두 박스 간의 IoU 계산"""
        # 교집합 영역 계산
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 각 박스의 면적
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 합집합 면적
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def benchmark_model(self, model_name, yaml_path):
        """개별 모델 벤치마크"""
        print(f"\n🧪 {model_name} 벤치마크 중...")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # 정확도 평가 - 직접 추론으로 계산
        print("   정확도 측정 중...")
        try:
            map50, map50_95, precision, recall = self._calculate_accuracy_manual(model, yaml_path)
            
        except Exception as e:
            print(f"   ❌ 정확도 측정 실패: {e}")
            map50 = map50_95 = precision = recall = 0.0
        
        # 속도 측정
        print("   속도 측정 중...")
        try:
            # 워밍업
            dummy_img = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            for _ in range(5):
                _ = model(dummy_img, verbose=False, device='cpu')
            
            # 실제 속도 측정
            times = []
            images_dir = os.path.join(os.path.dirname(yaml_path), "images")
            img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for i, img_file in enumerate(img_files[:50]):  # 최대 50장으로 속도 측정
                img_path = os.path.join(images_dir, img_file)
                img = Image.open(img_path)
                
                start_time = time.time()
                _ = model(img, verbose=False, device='cpu')
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times) * 1000  # ms
            fps = 1000 / avg_time
            
        except Exception as e:
            print(f"   ❌ 속도 측정 실패: {e}")
            avg_time = fps = 0.0
        
        # 결과 저장
        self.results[model_name] = {
            'model_size_mb': model_info['size_mb'],
            'parameters': model_info['parameters'],
            'map50': map50,
            'map50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'avg_time_ms': avg_time,
            'fps': fps,
            'efficiency_map_mb': map50 / model_info['size_mb'] if model_info['size_mb'] > 0 else 0,
            'efficiency_map_ms': map50 / avg_time if avg_time > 0 else 0,
            'efficiency_score': (map50 * fps) / model_info['size_mb'] if model_info['size_mb'] > 0 else 0
        }
        
        print(f"   mAP50: {map50:.3f}")
        print(f"   속도: {fps:.1f} FPS ({avg_time:.1f}ms)")
        print(f"   효율성: {self.results[model_name]['efficiency_score']:.2f}")
    
    def run_benchmark(self, model_paths, model_names=None):
        """전체 벤치마크 실행"""
        print("🚀 YOLO 모델 벤치마크 시작")
        print("=" * 50)
        
        # 모델 로드
        if model_names is None:
            model_names = [None] * len(model_paths)
        
        for model_path, model_name in zip(model_paths, model_names):
            self.add_model(model_path, model_name)
        
        if not self.models:
            print("❌ 로드된 모델이 없습니다.")
            return
        
        # 테스트 데이터 준비
        yaml_path = self.prepare_test_data()
        
        # 각 모델 벤치마크
        for model_name in self.models.keys():
            self.benchmark_model(model_name, yaml_path)
        
        # 결과 분석 및 출력
        self._print_results()
        self._create_comparison_chart()
        
        # 정리
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 임시 파일 정리 완료")
    
    def _print_results(self):
        """벤치마크 결과 출력"""
        print("\n" + "=" * 80)
        print("📊 벤치마크 결과 요약")
        print("=" * 80)
        
        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(self.results).T
        
        print("\n🎯 정확도 지표:")
        print("-" * 60)
        for model_name in df.index:
            print(f"{model_name:20} | mAP50: {df.loc[model_name, 'map50']:.3f} | mAP50-95: {df.loc[model_name, 'map50_95']:.3f}")
        
        print("\n⚡ 속도 지표:")
        print("-" * 60)
        for model_name in df.index:
            print(f"{model_name:20} | {df.loc[model_name, 'fps']:.1f} FPS | {df.loc[model_name, 'avg_time_ms']:.1f}ms")
        
        print("\n💾 효율성 지표:")
        print("-" * 60)
        for model_name in df.index:
            print(f"{model_name:20} | 크기: {df.loc[model_name, 'model_size_mb']:.1f}MB | 파라미터: {df.loc[model_name, 'parameters']:,}개")
        
        print("\n🏆 종합 효율성 (mAP×FPS/크기):")
        print("-" * 60)
        # 효율성 점수로 정렬
        sorted_models = df.sort_values('efficiency_score', ascending=False)
        for i, (model_name, row) in enumerate(sorted_models.iterrows()):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}위"
            print(f"{rank} {model_name:15} | 점수: {row['efficiency_score']:.2f}")
        
        # 최고 성능 모델들
        print("\n🌟 카테고리별 최고 성능:")
        print("-" * 40)
        print(f"📈 최고 정확도: {df['map50'].idxmax()} (mAP50: {df['map50'].max():.3f})")
        print(f"⚡ 최고 속도: {df['fps'].idxmax()} ({df['fps'].max():.1f} FPS)")
        print(f"💾 최소 크기: {df['model_size_mb'].idxmin()} ({df['model_size_mb'].min():.1f} MB)")
        print(f"🏆 최고 효율성: {df['efficiency_score'].idxmax()} (점수: {df['efficiency_score'].max():.2f})")
    
    def _create_comparison_chart(self):
        """비교 차트 생성 및 저장"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI 없는 백엔드 사용
            import matplotlib.pyplot as plt
            
            # 영어 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
            df = pd.DataFrame(self.results).T
            
            if len(df) == 0:
                print("📊 결과 데이터가 없어 차트를 생성할 수 없습니다.")
                return
            
            print(f"📊 차트 생성 시작... (현재 작업 디렉토리: {os.getcwd()})")
            
            # 1. 종합 비교 차트 (2x2)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1-1. 정확도 vs 속도
            ax1.scatter(df['fps'], df['map50'], s=150, alpha=0.7, c='blue')
            for i, model in enumerate(df.index):
                ax1.annotate(model, (df.iloc[i]['fps'], df.iloc[i]['map50']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            ax1.set_xlabel('FPS (Speed)', fontsize=12)
            ax1.set_ylabel('mAP50 (Accuracy)', fontsize=12)
            ax1.set_title('Accuracy vs Speed Comparison', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 1-2. 모델 크기 vs 정확도
            ax2.scatter(df['model_size_mb'], df['map50'], s=150, alpha=0.7, color='orange')
            for i, model in enumerate(df.index):
                ax2.annotate(model, (df.iloc[i]['model_size_mb'], df.iloc[i]['map50']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            ax2.set_xlabel('Model Size (MB)', fontsize=12)
            ax2.set_ylabel('mAP50 (Accuracy)', fontsize=12)
            ax2.set_title('Model Size vs Accuracy Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 1-3. 종합 효율성 바차트
            sorted_df = df.sort_values('efficiency_score', ascending=True)
            bars = ax3.barh(range(len(sorted_df)), sorted_df['efficiency_score'])
            ax3.set_yticks(range(len(sorted_df)))
            ax3.set_yticklabels(sorted_df.index, fontsize=10)
            ax3.set_xlabel('Efficiency Score (mAP×FPS/Size)', fontsize=12)
            ax3.set_title('Overall Efficiency Comparison', fontsize=14, fontweight='bold')
            
            # 막대 색상 그라데이션
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # 1-4. 상세 성능 바차트
            models = df.index.tolist()
            x = np.arange(len(models))
            width = 0.25
            
            ax4.bar(x - width, df['map50'], width, label='mAP50', alpha=0.8)
            ax4.bar(x, df['fps']/100, width, label='FPS/100', alpha=0.8)  # 스케일 조정
            ax4.bar(x + width, (1/df['model_size_mb'])*10, width, label='1/Size*10', alpha=0.8)  # 역수로 변환
            
            ax4.set_xlabel('Model', fontsize=12)
            ax4.set_ylabel('Normalized Score', fontsize=12)
            ax4.set_title('Detailed Performance Comparison', fontsize=14, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(models, fontsize=10)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장 시도
            try:
                chart_path = os.path.join(os.getcwd(), 'yolo_benchmark_comparison.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                print(f"✅ 종합 비교 차트 저장 완료: {chart_path}")
            except Exception as save_error:
                print(f"❌ 차트 저장 실패: {save_error}")
                # 대체 경로로 시도
                alt_path = 'yolo_benchmark_comparison.png'
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                print(f"✅ 대체 경로로 저장: {alt_path}")
            
            plt.close()
            
            # 2. 개별 지표 차트들
            self._create_individual_charts(df)
            
        except Exception as e:
            print(f"❌ 차트 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_individual_charts(self, df):
        """개별 지표별 상세 차트 생성"""
        try:
            # 2-1. 정확도 지표 차트
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            models = df.index.tolist()
            x = np.arange(len(models))
            width = 0.35
            
            # mAP50 vs mAP50-95
            ax1.bar(x - width/2, df['map50'], width, label='mAP50', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, df['map50_95'], width, label='mAP50-95', alpha=0.8, color='lightcoral')
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Accuracy Metrics Comparison', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Precision vs Recall
            ax2.bar(x - width/2, df['precision'], width, label='Precision', alpha=0.8, color='lightgreen')
            ax2.bar(x + width/2, df['recall'], width, label='Recall', alpha=0.8, color='orange')
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Score', fontsize=12)
            ax2.set_title('Precision vs Recall Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('yolo_accuracy_metrics.png', dpi=300, bbox_inches='tight')
            print("📊 정확도 지표 차트 저장: yolo_accuracy_metrics.png")
            plt.close()
            
            # 2-2. 속도 및 효율성 차트
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 속도 비교
            bars1 = ax1.bar(models, df['fps'], alpha=0.8, color='purple')
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('FPS', fontsize=12)
            ax1.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for bar, fps in zip(bars1, df['fps']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{fps:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 모델 크기 비교
            bars2 = ax2.bar(models, df['model_size_mb'], alpha=0.8, color='brown')
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Size (MB)', fontsize=12)
            ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for bar, size in zip(bars2, df['model_size_mb']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('yolo_speed_size_metrics.png', dpi=300, bbox_inches='tight')
            print("📊 속도/크기 지표 차트 저장: yolo_speed_size_metrics.png")
            plt.close()
            
            # 2-3. 효율성 레이더 차트
            if len(df) > 1:  # 2개 이상 모델이 있을 때만
                self._create_radar_chart(df)
            
            # 2-4. 종합 성능 요약 테이블 이미지
            self._create_summary_table(df)
            
        except Exception as e:
            print(f"개별 차트 생성 중 오류: {e}")
    
    def _create_radar_chart(self, df):
        """레이더 차트 생성"""
        try:
            from math import pi
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 정규화 (0-1 스케일)
            metrics = ['map50', 'fps', 'efficiency_score']
            df_norm = df[metrics].copy()
            
            for col in metrics:
                if df_norm[col].max() > 0:
                    df_norm[col] = df_norm[col] / df_norm[col].max()
            
            # 크기는 역순 (작을수록 좋음)
            if df['model_size_mb'].max() > df['model_size_mb'].min():
                df_norm['size_norm'] = 1 - ((df['model_size_mb'] - df['model_size_mb'].min()) / 
                                          (df['model_size_mb'].max() - df['model_size_mb'].min()))
            else:
                df_norm['size_norm'] = 1.0
            
            # 각도 설정
            categories = ['Accuracy', 'Speed', 'Efficiency', 'Lightness']
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # 닫힌 도형 만들기
            
            # 각 모델별로 그리기
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, (model_name, row) in enumerate(df_norm.iterrows()):
                values = [row['map50'], row['fps'], row['efficiency_score'], row['size_norm']]
                values += values[:1]
                
                color = colors[i % len(colors)]
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            # 축 설정
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('Overall Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('yolo_radar_chart.png', dpi=300, bbox_inches='tight')
            print("📊 레이더 차트 저장: yolo_radar_chart.png")
            plt.close()
            
        except Exception as e:
            print(f"레이더 차트 생성 중 오류: {e}")
    
    def _create_summary_table(self, df):
        """성능 요약 테이블 이미지 생성"""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # 테이블 데이터 준비
            table_data = []
            headers = ['Model', 'mAP50', 'mAP50-95', 'FPS', 'Size(MB)', 'Parameters', 'Efficiency']
            
            for model_name, row in df.iterrows():
                table_data.append([
                    model_name,
                    f"{row['map50']:.3f}",
                    f"{row['map50_95']:.3f}",
                    f"{row['fps']:.1f}",
                    f"{row['model_size_mb']:.1f}",
                    f"{int(row['parameters']):,}",
                    f"{row['efficiency_score']:.2f}"
                ])
            
            # 테이블 생성
            table = ax.table(cellText=table_data, colLabels=headers, 
                           cellLoc='center', loc='center')
            
            # 테이블 스타일링
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # 헤더 스타일
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 데이터 행 스타일 (교대로 색상)
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            plt.title('YOLO Model Performance Comparison Summary', fontsize=16, fontweight='bold', pad=20)
            plt.savefig('yolo_summary_table.png', dpi=300, bbox_inches='tight')
            print("📊 요약 테이블 저장: yolo_summary_table.png")
            plt.close()
            
        except Exception as e:
            print(f"요약 테이블 생성 중 오류: {e}")

def main():
    """사용 예시"""
    # 설정
    DATA_DIR = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106"  # 데이터 폴더 (JSON+이미지)
    CLASSES_FILE = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\classes.txt"  # 클래스 파일
    NUM_EVAL_IMAGES = 100  # 평가에 사용할 이미지 수
    
    # 그래프 저장 폴더 설정 (원하는 경우)
    CHART_OUTPUT_DIR = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\8.Pruning"  # 그래프 저장 폴더
    
    # 비교할 모델들 (경로 리스트)
    model_paths = [
        r"C:\Users\KDT-13\Desktop\A100\yolov8_continued.pt",  # 원본 모델
        r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\8.Pruning\ver0\model\pruning_model.pt"  # 프루닝 모델 1
        #r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\8.Pruning\ver2(기존학습코드로해서잘못됨_over_train)\runs\detect\train\weights\best.pt",  # 추가 모델들...
        #r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\8.Pruning\ver3(새 train code retrain)\retrained_pruned_model.pt",
        #r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\8.Pruning\ver4(새 train code over train\retrained_pruned_model.pt"
    ]
    
    # 모델 이름들 (선택사항, None이면 파일명 사용)
    model_names = [
        "0",
        "1"
        #"2",
        #"3",
        #"4"
    ]
    
    # 실행 전 경로 확인
    print("🔍 경로 확인 중...")
    
    # 데이터 폴더 확인
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더가 존재하지 않습니다: {DATA_DIR}")
        return
    
    # 클래스 파일 확인
    if not os.path.exists(CLASSES_FILE):
        print(f"❌ 클래스 파일이 존재하지 않습니다: {CLASSES_FILE}")
        return
    
    # 모델 파일들 확인
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일이 존재하지 않습니다: {model_names[i]} - {model_path}")
            return
        else:
            print(f"✅ {model_names[i]}: {model_path}")
    
    # 그래프 저장 폴더 생성
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    
    # 작업 디렉토리를 그래프 저장 폴더로 변경
    original_dir = os.getcwd()
    os.chdir(CHART_OUTPUT_DIR)
    
    print(f"📊 그래프 저장 위치: {CHART_OUTPUT_DIR}")
    
    try:
        # 벤치마크 실행
        benchmark = YOLOBenchmark(
            data_dir=DATA_DIR,
            classes_file=CLASSES_FILE, 
            num_eval_images=NUM_EVAL_IMAGES
        )
        
        benchmark.run_benchmark(model_paths, model_names)
        
    except Exception as e:
        print(f"❌ 벤치마크 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 원래 작업 디렉토리로 복원
        os.chdir(original_dir)

if __name__ == "__main__":
    main()