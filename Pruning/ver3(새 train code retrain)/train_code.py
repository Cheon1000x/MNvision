"""
프루닝된 모델 재학습 완전판 코드
- 자동 YAML 생성 (JSON+이미지 → train/val 분할)
- 학습 결과 그래프 저장
- 성능 비교 시각화

사용법:
1. 프루닝된 모델 파일 (.pt)
2. JSON+이미지가 있는 데이터 폴더
3. classes.txt 파일
"""

import os
import json
import torch
import time
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

class CompletePrunedModelRetrainer:
    def __init__(self, pruned_model_path, data_dir, classes_file):
        """
        Args:
            pruned_model_path: 프루닝된 모델 파일 경로 (.pt)
            data_dir: JSON+이미지가 있는 데이터 폴더
            classes_file: classes.txt 파일 경로
        """
        self.pruned_model_path = pruned_model_path
        self.data_dir = data_dir
        self.classes_file = classes_file
        self.model = None
        self.results = None
        self.temp_dataset_dir = None
        self.dataset_yaml_path = None
        
        # 파일 존재 확인
        if not os.path.exists(pruned_model_path):
            raise FileNotFoundError(f"프루닝된 모델 파일이 없습니다: {pruned_model_path}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")
            
        if not os.path.exists(classes_file):
            raise FileNotFoundError(f"클래스 파일이 없습니다: {classes_file}")
        
        # 클래스 정보 로드
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"🚀 프루닝된 모델 재학습 완전판 시스템 초기화")
        print(f"   프루닝된 모델: {pruned_model_path}")
        print(f"   데이터 폴더: {data_dir}")
        print(f"   클래스 수: {len(self.classes)}")
        
        # matplotlib 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def prepare_dataset(self, train_ratio=0.8, max_samples=None):
        """JSON+이미지 데이터를 YOLO 형식으로 변환하고 train/val 분할"""
        print(f"\n📂 데이터셋 준비 중...")
        
        # 임시 디렉토리 생성
        self.temp_dataset_dir = tempfile.mkdtemp(prefix="retrain_dataset_")
        
        # 디렉토리 구조 생성
        train_images_dir = os.path.join(self.temp_dataset_dir, "images", "train")
        train_labels_dir = os.path.join(self.temp_dataset_dir, "labels", "train")
        val_images_dir = os.path.join(self.temp_dataset_dir, "images", "val")
        val_labels_dir = os.path.join(self.temp_dataset_dir, "labels", "val")
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 모든 하위 폴더에서 JSON과 이미지 파일 찾기
        print(f"   메인 데이터 폴더 스캔: {self.data_dir}")
        
        all_paired_files = []
        subfolders = [f for f in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, f))]
        
        print(f"   발견된 하위 폴더: {len(subfolders)}개")
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.data_dir, subfolder)
            print(f"   스캔 중: {subfolder}")
            
            try:
                # 각 하위 폴더에서 파일 찾기
                files_in_subfolder = os.listdir(subfolder_path)
                json_files = [f for f in files_in_subfolder if f.endswith('.json')]
                image_files = [f for f in files_in_subfolder if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # 파일 쌍 매칭
                subfolder_pairs = []
                for json_file in json_files:
                    base_name = os.path.splitext(json_file)[0]
                    for img_file in image_files:
                        if base_name == os.path.splitext(img_file)[0]:
                            # 전체 경로 저장
                            json_path = os.path.join(subfolder_path, json_file)
                            img_path = os.path.join(subfolder_path, img_file)
                            subfolder_pairs.append((json_path, img_path, f"{subfolder}_{base_name}"))
                            break
                
                all_paired_files.extend(subfolder_pairs)
                print(f"     찾은 파일 쌍: {len(subfolder_pairs)}개")
                
            except Exception as e:
                print(f"     ⚠️ 폴더 스캔 오류: {e}")
                continue
        
        paired_files = all_paired_files
        
        print(f"\n📊 전체 스캔 결과:")
        print(f"   총 파일 쌍: {len(paired_files)}개")
        print(f"   스캔한 폴더: {len(subfolders)}개")
        
        if len(paired_files) == 0:
            print("❌ 매칭되는 JSON+이미지 파일을 찾을 수 없습니다.")
            return False
        
        # 샘플 수 제한 (메모리 절약)
        if max_samples and len(paired_files) > max_samples:
            random.shuffle(paired_files)
            paired_files = paired_files[:max_samples]
            print(f"   사용할 샘플: {len(paired_files)}개 (제한됨)")
        
        # train/val 분할
        random.shuffle(paired_files)
        train_size = int(len(paired_files) * train_ratio)
        train_pairs = paired_files[:train_size]
        val_pairs = paired_files[train_size:]
        
        print(f"   Train: {len(train_pairs)}개, Val: {len(val_pairs)}개")
        
        # 데이터 변환 및 복사
        def process_pairs(pairs, img_dir, label_dir, desc):
            for json_path, img_path, unique_name in tqdm(pairs, desc=desc):
                # 이미지 복사 (고유한 이름 사용)
                new_img_path = os.path.join(img_dir, f"{unique_name}.jpg")
                shutil.copy(img_path, new_img_path)
                
                # JSON → YOLO 변환
                yolo_annotations = self._convert_json_to_yolo(json_path, img_path)
                
                # 라벨 파일 저장
                label_path = os.path.join(label_dir, f"{unique_name}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
        
        process_pairs(train_pairs, train_images_dir, train_labels_dir, "Train 데이터 처리")
        process_pairs(val_pairs, val_images_dir, val_labels_dir, "Val 데이터 처리")
        
        # YAML 파일 생성
        self.dataset_yaml_path = os.path.join(self.temp_dataset_dir, "dataset.yaml")
        self._create_dataset_yaml()
        
        print(f"✅ 데이터셋 준비 완료: {self.temp_dataset_dir}")
        return True
    
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
    
    def _create_dataset_yaml(self):
        """YOLO 데이터셋 YAML 파일 생성"""
        yaml_content = f"""path: {self.temp_dataset_dir}
train: images/train
val: images/val

nc: {len(self.classes)}
names: {self.classes}
"""
        with open(self.dataset_yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"   YAML 파일 생성: {self.dataset_yaml_path}")
    
    def load_pruned_model(self):
        """프루닝된 모델 로드 및 정보 확인"""
        print(f"\n📂 프루닝된 모델 로드 중...")
        
        try:
            self.model = YOLO(self.pruned_model_path)
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in self.model.model.parameters())
            file_size_mb = os.path.getsize(self.pruned_model_path) / (1024 * 1024)
            
            print(f"✅ 프루닝된 모델 로드 성공!")
            print(f"   파라미터 수: {total_params:,}개")
            print(f"   모델 크기: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def benchmark_speed(self, label="", num_runs=20):
        """속도 벤치마크"""
        if label:
            print(f"\n⚡ {label} 속도 측정...")
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        # 더미 이미지로 측정
        dummy_image = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        
        # 워밍업
        for _ in range(3):
            _ = self.model.predict(dummy_image, verbose=False)
        
        # 실제 측정
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = self.model.predict(dummy_image, verbose=False)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"   추론 속도: {avg_time:.2f}ms ({fps:.1f} FPS)")
        return {'avg_time_ms': avg_time, 'fps': fps}
    
    def retrain_model(self, epochs=30, batch_size=16, learning_rate=0.001, patience=15):
        """프루닝된 모델 재학습"""
        print(f"\n🔄 프루닝된 모델 재학습 시작")
        print(f"   에폭: {epochs}, 배치: {batch_size}, 학습률: {learning_rate}")
        
        if self.model is None or self.dataset_yaml_path is None:
            print("❌ 모델이나 데이터셋이 준비되지 않았습니다.")
            return False
        
        try:
            # 재학습 설정
            train_args = {
                'data': self.dataset_yaml_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': [180, 320],
                
                # 프루닝된 모델 재학습 최적화 설정
                'lr0': learning_rate,
                'lrf': learning_rate * 0.01,
                'patience': patience,
                'save_period': max(5, epochs//6),
                
                # 안정성 설정
                'warmup_epochs': 3,
                'cos_lr': True,
                'weight_decay': 0.0005,
                
                # 환경 설정
                'device': 0 if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'verbose': True,
                'val': True,
                'plots': True,
                
                # 구조 보존 설정
                'freeze': 0,
                'dropout': 0.0,
                'close_mosaic': 10,
            }
            
            print(f"🔥 재학습 시작...")
            start_time = time.time()
            
            # 재학습 실행
            self.results = self.model.train(**train_args)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            print(f"✅ 재학습 완료! (소요시간: {training_time/60:.1f}분)")
            return True
            
        except Exception as e:
            print(f"❌ 재학습 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self, label=""):
        """모델 평가"""
        if label:
            print(f"\n📊 {label} 평가...")
        
        if self.model is None or self.dataset_yaml_path is None:
            return None
        
        try:
            val_results = self.model.val(
                data=self.dataset_yaml_path,
                imgsz=[180, 320],
                verbose=True
            )
            
            # 메트릭 추출
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'Precision': float(val_results.box.p.mean()) if hasattr(val_results.box.p, 'mean') else float(val_results.box.p),
                'Recall': float(val_results.box.r.mean()) if hasattr(val_results.box.r, 'mean') else float(val_results.box.r),
            }
            
            if label:
                print(f"📈 {label} 성능:")
                for metric_name, value in metrics.items():
                    print(f"   {metric_name}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            return None
    
    def create_training_charts(self, save_dir="./"):
        """학습 결과 그래프 생성"""
        print(f"\n📊 학습 결과 그래프 생성 중...")
        
        if self.results is None:
            print("❌ 학습 결과가 없습니다.")
            return
        
        try:
            # YOLO 학습 결과에서 메트릭 추출 (results.csv 파일 사용)
            results_dir = None
            
            # runs 폴더에서 최신 실험 결과 찾기
            runs_dir = "runs/detect"
            if os.path.exists(runs_dir):
                train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
                if train_dirs:
                    latest_dir = max(train_dirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
                    results_dir = os.path.join(runs_dir, latest_dir)
            
            csv_file = None
            if results_dir and os.path.exists(os.path.join(results_dir, "results.csv")):
                csv_file = os.path.join(results_dir, "results.csv")
            
            if csv_file:
                self._create_charts_from_csv(csv_file, save_dir)
            else:
                self._create_basic_charts(save_dir)
            
        except Exception as e:
            print(f"❌ 그래프 생성 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_charts_from_csv(self, csv_file, save_dir):
        """CSV 파일에서 학습 곡선 그래프 생성"""
        import pandas as pd
        
        # CSV 데이터 로드
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()  # 공백 제거
        
        # 그래프 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # 1. Loss 곡선
        if 'train/box_loss' in df.columns:
            ax1.plot(epochs, df['train/box_loss'], label='Train Box Loss', color='blue')
        if 'val/box_loss' in df.columns:
            ax1.plot(epochs, df['val/box_loss'], label='Val Box Loss', color='red')
        ax1.set_title('Box Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. mAP 곡선
        if 'metrics/mAP50(B)' in df.columns:
            ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', color='green', linewidth=2)
        if 'metrics/mAP50-95(B)' in df.columns:
            ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', color='orange', linewidth=2)
        ax2.set_title('mAP Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision/Recall
        if 'metrics/precision(B)' in df.columns:
            ax3.plot(epochs, df['metrics/precision(B)'], label='Precision', color='purple')
        if 'metrics/recall(B)' in df.columns:
            ax3.plot(epochs, df['metrics/recall(B)'], label='Recall', color='brown')
        ax3.set_title('Precision & Recall Curves', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate
        if 'lr/pg0' in df.columns:
            ax4.plot(epochs, df['lr/pg0'], label='Learning Rate', color='red')
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 학습 곡선 저장: {chart_path}")
    
    def _create_basic_charts(self, save_dir):
        """기본 차트 생성 (CSV가 없는 경우)"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.text(0.5, 0.5, 'Training Completed\nDetailed charts available in runs/detect/train/', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Retraining Results', fontsize=18, fontweight='bold')
        
        chart_path = os.path.join(save_dir, 'training_completed.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 기본 차트 저장: {chart_path}")
    
    def create_comparison_charts(self, original_metrics, original_speed, 
                               retrained_metrics, retrained_speed, save_dir="./"):
        """재학습 전후 비교 차트 생성"""
        print(f"\n📊 성능 비교 차트 생성 중...")
        
        try:
            # 1. 성능 비교 차트
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 메트릭 비교
            if original_metrics and retrained_metrics:
                metrics = list(original_metrics.keys())
                original_values = list(original_metrics.values())
                retrained_values = list(retrained_metrics.values())
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax1.bar(x - width/2, original_values, width, label='Original (Pruned)', 
                       alpha=0.8, color='skyblue')
                ax1.bar(x + width/2, retrained_values, width, label='Retrained', 
                       alpha=0.8, color='lightcoral')
                
                ax1.set_xlabel('Metrics')
                ax1.set_ylabel('Score')
                ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(metrics)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 막대 위에 값 표시
                for i, (orig, retrain) in enumerate(zip(original_values, retrained_values)):
                    ax1.text(i - width/2, orig + 0.01, f'{orig:.3f}', 
                            ha='center', va='bottom', fontweight='bold')
                    ax1.text(i + width/2, retrain + 0.01, f'{retrain:.3f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            # 속도 비교
            if original_speed and retrained_speed:
                speed_labels = ['FPS', 'Inference Time (ms)']
                original_speed_values = [original_speed['fps'], original_speed['avg_time_ms']]
                retrained_speed_values = [retrained_speed['fps'], retrained_speed['avg_time_ms']]
                
                x = np.arange(len(speed_labels))
                
                ax2.bar(x - width/2, original_speed_values, width, label='Original (Pruned)', 
                       alpha=0.8, color='lightgreen')
                ax2.bar(x + width/2, retrained_speed_values, width, label='Retrained', 
                       alpha=0.8, color='orange')
                
                ax2.set_xlabel('Speed Metrics')
                ax2.set_ylabel('Value')
                ax2.set_title('Speed Comparison', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(speed_labels)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 막대 위에 값 표시
                for i, (orig, retrain) in enumerate(zip(original_speed_values, retrained_speed_values)):
                    ax2.text(i - width/2, orig + max(original_speed_values)*0.02, f'{orig:.1f}', 
                            ha='center', va='bottom', fontweight='bold')
                    ax2.text(i + width/2, retrain + max(retrained_speed_values)*0.02, f'{retrain:.1f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            comparison_path = os.path.join(save_dir, 'performance_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 성능 비교 차트 저장: {comparison_path}")
            
        except Exception as e:
            print(f"❌ 비교 차트 생성 실패: {e}")
    
    def save_retrained_model(self, output_path=None):
        """재학습된 모델 저장"""
        if output_path is None:
            base_name = Path(self.pruned_model_path).stem
            output_path = f"{base_name}_retrained.pt"
        
        print(f"\n💾 재학습된 모델 저장...")
        
        try:
            self.model.save(output_path)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ 저장 완료: {output_path}")
            print(f"   파일 크기: {file_size_mb:.2f} MB")
            return output_path
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return None
    
    def cleanup(self):
        """임시 파일 정리"""
        if self.temp_dataset_dir and os.path.exists(self.temp_dataset_dir):
            try:
                shutil.rmtree(self.temp_dataset_dir)
                print(f"🧹 임시 데이터셋 정리 완료")
            except Exception as e:
                print(f"⚠️ 임시 파일 정리 실패: {e}")
    
    def full_retrain_pipeline(self, epochs=30, batch_size=16, learning_rate=0.001, 
                            patience=15, output_path=None, save_charts=True):
        """전체 재학습 파이프라인 실행"""
        print("=" * 70)
        print("🚀 프루닝된 모델 재학습 완전판 파이프라인 시작")
        print("=" * 70)
        
        original_metrics = None
        original_speed = None
        retrained_metrics = None
        retrained_speed = None
        
        try:
            # 1. 데이터셋 준비
            if not self.prepare_dataset(max_samples=None):  # 메모리 절약을 위해 1000샘플로 제한
                return False
            
            # 2. 프루닝된 모델 로드
            if not self.load_pruned_model():
                return False
            
            # 3. 재학습 전 성능 측정
            print("\n" + "="*50)
            print("📊 재학습 전 성능 측정")
            print("="*50)
            original_metrics = self.evaluate_model("재학습 전")
            original_speed = self.benchmark_speed("재학습 전")
            
            # 4. 재학습 실행
            print("\n" + "="*50)
            print("🔥 재학습 실행")
            print("="*50)
            if not self.retrain_model(epochs, batch_size, learning_rate, patience):
                return False
            
            # 5. 재학습 후 성능 측정
            print("\n" + "="*50)
            print("📊 재학습 후 성능 측정")
            print("="*50)
            retrained_metrics = self.evaluate_model("재학습 후")
            retrained_speed = self.benchmark_speed("재학습 후")
            
            # 6. 모델 저장
            saved_path = self.save_retrained_model(output_path)
            
            # 7. 그래프 생성
            if save_charts:
                print("\n" + "="*50)
                print("📊 결과 그래프 생성")
                print("="*50)
                self.create_training_charts()
                if original_metrics and retrained_metrics:
                    self.create_comparison_charts(original_metrics, original_speed,
                                                retrained_metrics, retrained_speed)
            
            # 8. 최종 결과 요약
            print("\n" + "=" * 70)
            print("📊 재학습 완료 - 최종 결과 요약")
            print("=" * 70)
            
            if original_metrics and retrained_metrics:
                print(f"\n🎯 성능 비교:")
                print(f"{'메트릭':<15} {'재학습 전':<12} {'재학습 후':<12} {'개선율':<10}")
                print("-" * 50)
                for metric in original_metrics.keys():
                    orig_val = original_metrics[metric]
                    new_val = retrained_metrics[metric]
                    improvement = ((new_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                    print(f"{metric:<15} {orig_val:<12.4f} {new_val:<12.4f} {improvement:+7.1f}%")
            
            if original_speed and retrained_speed:
                print(f"\n⚡ 속도 비교:")
                orig_fps = original_speed['fps']
                new_fps = retrained_speed['fps']
                fps_change = ((new_fps - orig_fps) / orig_fps * 100) if orig_fps > 0 else 0
                
                print(f"   재학습 전: {orig_fps:.1f} FPS")
                print(f"   재학습 후: {new_fps:.1f} FPS")
                print(f"   속도 변화: {fps_change:+.1f}%")
            
            if saved_path:
                print(f"\n💾 최종 모델: {saved_path}")
            
            if save_charts:
                print(f"\n📊 생성된 그래프:")
                print(f"   - training_curves.png (학습 곡선)")
                print(f"   - performance_comparison.png (성능 비교)")
            
            print(f"\n✅ 재학습 파이프라인 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # 임시 파일 정리
            self.cleanup()


def main():
    """사용 예시"""
    
    # 설정 - 실제 경로로 수정하세요
    PRUNED_MODEL_PATH = r"C:\Users\K\Desktop\Group_6\0610_ver3\pruned_model.pt"
    DATA_DIR = r"C:\Users\K\Desktop\Group_6\0610_ver3\data"  # 메인 데이터 폴더 (하위 폴더들 포함)
    CLASSES_FILE = r"C:\Users\K\Desktop\Group_6\0602\data\classes.txt"
    
    # 재학습 설정
    EPOCHS = 30              # 재학습 에폭 수 (프루닝된 모델은 적게)
    BATCH_SIZE = 150          # 배치 크기
    LEARNING_RATE = 0.001    # 낮은 학습률 (안정적 재학습)
    PATIENCE = 15            # Early stopping 인내
    OUTPUT_PATH = "retrained_pruned_model.pt"  # 저장할 파일명
    
    # 경로 확인
    print("🔍 파일 및 폴더 확인 중...")
    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"❌ 프루닝된 모델 파일이 없습니다: {PRUNED_MODEL_PATH}")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더가 없습니다: {DATA_DIR}")
        return
    
    if not os.path.exists(CLASSES_FILE):
        print(f"❌ 클래스 파일이 없습니다: {CLASSES_FILE}")
        return
    
    # 데이터 폴더 내용 확인
    if os.path.exists(DATA_DIR):
        subfolders = [f for f in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, f))]
        
        total_json = 0
        total_img = 0
        
        print(f"📁 하위 폴더 스캔 결과:")
        for subfolder in subfolders[:5]:  # 처음 5개만 표시
            subfolder_path = os.path.join(DATA_DIR, subfolder)
            try:
                files = os.listdir(subfolder_path)
                json_count = len([f for f in files if f.endswith('.json')])
                img_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_json += json_count
                total_img += img_count
                print(f"   {subfolder}: JSON {json_count}개, 이미지 {img_count}개")
            except:
                continue
        
        if len(subfolders) > 5:
            print(f"   ... 및 기타 {len(subfolders)-5}개 폴더")
        
        print(f"\n📊 전체 통계 (예상):")
        print(f"   하위 폴더: {len(subfolders)}개")
        print(f"   총 JSON 파일: ~{total_json}개 이상")
        print(f"   총 이미지 파일: ~{total_img}개 이상")
    else:
        print(f"❌ 데이터 폴더가 없습니다: {DATA_DIR}")
        return
    
    try:
        # 재학습기 초기화
        retrainer = CompletePrunedModelRetrainer(
            PRUNED_MODEL_PATH, 
            DATA_DIR, 
            CLASSES_FILE
        )
        
        # 사용자 확인
        user_input = input(f"\n재학습을 시작하시겠습니까? (y/n) [y]: ").strip().lower()
        if user_input == 'n':
            print("재학습을 취소했습니다.")
            return
        
        # 전체 파이프라인 실행
        success = retrainer.full_retrain_pipeline(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            output_path=OUTPUT_PATH,
            save_charts=True
        )
        
        if success:
            print("\n" + "="*50)
            print("🎉 모든 과정이 성공적으로 완료되었습니다!")
            print("="*50)
            print("📁 생성된 파일들:")
            print(f"   - {OUTPUT_PATH} (재학습된 모델)")
            print(f"   - training_curves.png (학습 곡선)")
            print(f"   - performance_comparison.png (성능 비교)")
            print(f"   - runs/detect/train/ (상세 학습 결과)")
        else:
            print("\n❌ 재학습 과정에서 오류가 발생했습니다.")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()