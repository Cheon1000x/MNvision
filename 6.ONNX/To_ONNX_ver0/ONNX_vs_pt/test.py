# OpenMP 오류 해결 방법들

## 방법 1: 환경 변수 설정 (가장 간단함)
import os

# 프로그램 시작 부분에 추가 (다른 import보다 먼저)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP 스레드 수 제한

# 그 다음에 다른 라이브러리들 import
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import psutil
import cv2
from pathlib import Path
import json
from glob import glob
# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')

class ModelComparison:
    def __init__(self):
        self.results = {}
        
    def evaluate_model_on_images(self, model_path, model_name, test_images_folder, conf_threshold=0.5):
        """
        이미지 폴더를 사용해서 모델 평가
        """
        print(f"\n{'='*50}")
        print(f"{model_name} 모델 평가 중...")
        print(f"{'='*50}")
        
        # 모델 로드
        model = YOLO(model_path)
        
        # 테스트 이미지 수집
        test_images = self.get_test_images_from_folder(test_images_folder)
        print(f"테스트 이미지 수: {len(test_images)}개")
        
        if not test_images:
            print("테스트 이미지를 찾을 수 없습니다!")
            return None
        
        # 1. 추론 속도 평가
        print("추론 속도 평가 중...")
        speed_results = self.measure_inference_speed_on_images(model, test_images, conf_threshold)
        
        # 2. 검출 성능 평가 (정성적)
        print("검출 성능 평가 중...")
        detection_results = self.evaluate_detection_performance(model, test_images, conf_threshold)
        
        # 3. 메모리 사용량 평가
        print("메모리 사용량 측정 중...")
        memory_usage = self.measure_memory_usage_on_images(model, test_images[:5])  # 샘플만
        
        # 결과 저장
        self.results[model_name] = {
            'detection_metrics': detection_results,
            'speed_metrics': speed_results,
            'memory_metrics': memory_usage,
            'model_size': self.get_model_size(model_path)
        }
        
        print(f"{model_name} 평가 완료!")
        return self.results[model_name]
    
    def get_test_images_from_folder(self, folder_path):
        """
        폴더에서 이미지 파일들을 찾기
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            return []
        
        # 지원하는 이미지 확장자
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        images = []
        
        for ext in extensions:
            images.extend(glob(str(folder / ext)))
            images.extend(glob(str(folder / ext.upper())))
        
        return sorted(images)
    
    def measure_inference_speed_on_images(self, model, test_images, conf_threshold, num_iterations=None):
        """
        이미지들에 대한 추론 속도 측정
        """
        if num_iterations is None:
            num_iterations = min(len(test_images), 50)  # 최대 50장
        
        times = []
        
        # Warm-up
        print("  Warm-up 중...")
        for _ in range(3):
            _ = model(test_images[0], conf=conf_threshold, verbose=False)
        
        # 실제 측정
        print(f"  {num_iterations}장 이미지로 속도 측정 중...")
        for i in range(num_iterations):
            img_path = test_images[i % len(test_images)]
            
            start_time = time.time()
            results = model(img_path, conf=conf_threshold, verbose=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"    진행률: {i+1}/{num_iterations}")
        
        return {
            'avg_inference_time': np.mean(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'std_inference_time': np.std(times),
            'fps': 1 / np.mean(times),
            'total_images_tested': num_iterations
        }
    
    def evaluate_detection_performance(self, model, test_images, conf_threshold):
        """
        검출 성능 평가 (Ground Truth 없이)
        """
        print("  검출 통계 수집 중...")
        
        total_detections = []
        confidence_scores = []
        detection_counts = []
        
        # 샘플 이미지들에 대해 검출 수행
        sample_size = min(len(test_images), 20)
        
        for i, img_path in enumerate(test_images[:sample_size]):
            results = model(img_path, conf=conf_threshold, verbose=False)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                num_detections = len(boxes)
                detection_counts.append(num_detections)
                
                # 신뢰도 점수들 수집
                confidences = boxes.conf.cpu().numpy()
                confidence_scores.extend(confidences)
                
                # 검출된 클래스들 수집
                classes = boxes.cls.cpu().numpy()
                total_detections.extend(classes)
            else:
                detection_counts.append(0)
        
        # 통계 계산
        avg_detections_per_image = np.mean(detection_counts) if detection_counts else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        unique_classes = len(np.unique(total_detections)) if total_detections else 0
        
        return {
            'avg_detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'total_detections': len(total_detections),
            'unique_classes_detected': unique_classes,
            'max_detections_per_image': max(detection_counts) if detection_counts else 0,
            'images_with_detections': sum(1 for x in detection_counts if x > 0),
            'detection_rate': sum(1 for x in detection_counts if x > 0) / len(detection_counts) if detection_counts else 0
        }
    
    def measure_memory_usage_on_images(self, model, sample_images):
        """
        메모리 사용량 측정
        """
        import gc
        import torch
        
        # 가비지 컬렉션
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 메모리 사용량 측정 시작
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU 메모리 (사용 가능한 경우)
        gpu_memory_before = 0
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # 추론 실행
        for img_path in sample_images:
            _ = model(img_path, verbose=False)
        
        # 메모리 사용량 측정 종료
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        gpu_memory_after = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        return {
            'cpu_memory_usage': max(0, memory_after - memory_before),
            'gpu_memory_usage': max(0, gpu_memory_after - gpu_memory_before),
            'total_memory': memory_after
        }
    
    def get_model_size(self, model_path):
        """
        모델 파일 크기 (MB)
        """
        try:
            size_bytes = os.path.getsize(model_path)
            return size_bytes / 1024 / 1024  # MB
        except:
            return 0
    
    def visualize_comparison(self, save_path="model_comparison.png"):
        """
        비교 결과 시각화
        """
        if len(self.results) < 2:
            print("비교할 모델이 2개 미만입니다.")
            return
        
        # 그래프 설정
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PyTorch vs ONNX Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. 검출 성능 비교
        self.plot_detection_metrics(axes[0, 0])
        
        # 2. 속도 지표 비교  
        self.plot_speed_metrics(axes[0, 1])
        
        # 3. 메모리 사용량 비교
        self.plot_memory_metrics(axes[0, 2])
        
        # 4. 효율성 종합 비교 (속도 vs 검출률)
        self.plot_efficiency_comparison(axes[1, 0])
        
        # 5. 모델 크기 비교
        self.plot_model_size(axes[1, 1])
        
        # 6. 종합 점수
        self.plot_overall_score(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"비교 결과가 {save_path}에 저장되었습니다.")
    
    def plot_detection_metrics(self, ax):
        """검출 성능 지표 시각화"""
        models = list(self.results.keys())
        metrics = ['avg_detections_per_image', 'avg_confidence', 'detection_rate', 'unique_classes_detected']
        metric_names = ['Avg Detections\nper Image', 'Avg Confidence', 'Detection Rate', 'Unique Classes']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # 값들을 0-1 범위로 정규화
        values1 = []
        values2 = []
        
        for metric in metrics:
            val1 = self.results[models[0]]['detection_metrics'].get(metric, 0)
            val2 = self.results[models[1]]['detection_metrics'].get(metric, 0)
            
            if metric == 'avg_detections_per_image':
                # 평균 검출 수는 그대로 사용 (보통 0-10 범위)
                values1.append(val1)
                values2.append(val2)
            elif metric == 'avg_confidence':
                # 신뢰도는 이미 0-1 범위
                values1.append(val1)
                values2.append(val2)
            elif metric == 'detection_rate':
                # 검출률도 이미 0-1 범위
                values1.append(val1)
                values2.append(val2)
            else:  # unique_classes_detected
                # 클래스 수는 그대로 사용
                values1.append(val1)
                values2.append(val2)
        
        bars1 = ax.bar(x - width/2, values1, width, label=models[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, values2, width, label=models[1], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Detection Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def plot_speed_metrics(self, ax):
        """속도 지표 시각화"""
        models = list(self.results.keys())
        
        fps_values = [self.results[model]['speed_metrics']['fps'] for model in models]
        inference_times = [self.results[model]['speed_metrics']['avg_inference_time'] * 1000 for model in models]
        
        x = np.arange(len(models))
        
        # FPS 비교
        ax2 = ax.twinx()
        bars1 = ax.bar(x - 0.2, fps_values, 0.4, label='FPS', color='skyblue', alpha=0.8)
        bars2 = ax2.bar(x + 0.2, inference_times, 0.4, label='Inference Time (ms)', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('FPS', color='blue')
        ax2.set_ylabel('Inference Time (ms)', color='red')
        ax.set_title('Speed Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        
        # 범례
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 값 표시
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.5,
                   f'{fps_values[i]:.1f}', ha='center', va='bottom', fontsize=9, color='blue')
            ax2.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 1,
                    f'{inference_times[i]:.1f}', ha='center', va='bottom', fontsize=9, color='red')
    
    def plot_memory_metrics(self, ax):
        """메모리 사용량 시각화"""
        models = list(self.results.keys())
        
        cpu_memory = [self.results[model]['memory_metrics']['cpu_memory_usage'] for model in models]
        model_sizes = [self.results[model]['model_size'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cpu_memory, width, label='Runtime Memory (MB)', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x + width/2, model_sizes, width, label='Model Size (MB)', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    def plot_efficiency_comparison(self, ax):
        """효율성 비교 (속도 vs 검출률)"""
        models = list(self.results.keys())
        
        fps_values = [self.results[model]['speed_metrics']['fps'] for model in models]
        detection_rates = [self.results[model]['detection_metrics']['detection_rate'] for model in models]
        
        colors = ['blue', 'red']
        
        for i, model in enumerate(models):
            ax.scatter(fps_values[i], detection_rates[i], 
                      s=200, c=colors[i], alpha=0.7, label=model)
            ax.annotate(model, (fps_values[i], detection_rates[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('FPS (Higher is Better)')
        ax.set_ylabel('Detection Rate (Higher is Better)')
        ax.set_title('Efficiency Comparison (Speed vs Detection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def plot_model_size(self, ax):
        """모델 크기 비교"""
        models = list(self.results.keys())
        sizes = [self.results[model]['model_size'] for model in models]
        
        colors = ['lightblue', 'lightgreen']
        bars = ax.bar(models, sizes, color=colors, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size Comparison')
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f} MB', ha='center', va='bottom', fontsize=10)
    
    def plot_overall_score(self, ax):
        """종합 점수 (속도 + 검출성능 + 효율성)"""
        models = list(self.results.keys())
        
        # 정규화된 점수 계산 (0-1 범위)
        scores = {}
        max_fps = max([self.results[model]['speed_metrics']['fps'] for model in models])
        
        for model in models:
            # 속도 점수 (정규화)
            speed_score = self.results[model]['speed_metrics']['fps'] / max_fps
            
            # 검출 성능 점수
            detection_score = self.results[model]['detection_metrics']['detection_rate']
            
            # 신뢰도 점수
            confidence_score = self.results[model]['detection_metrics']['avg_confidence']
            
            # 효율성 점수 (모델 크기가 작을수록 좋음)
            max_size = max([self.results[m]['model_size'] for m in models])
            efficiency_score = 1 - (self.results[model]['model_size'] / max_size) if max_size > 0 else 0.5
            
            # 종합 점수 (가중평균)
            overall_score = (speed_score * 0.3 + detection_score * 0.3 + 
                           confidence_score * 0.2 + efficiency_score * 0.2)
            scores[model] = overall_score
        
        colors = ['gold', 'silver']
        bars = ax.bar(models, list(scores.values()), color=colors, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Overall Score (0-1)')
        ax.set_title('Overall Performance Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    def save_results(self, filename="comparison_results.json"):
        """결과를 JSON 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"결과가 {filename}에 저장되었습니다.")
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("모델 비교 결과 요약")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n[{model_name}]")
            print(f"  검출 성능:")
            print(f"    - 이미지당 평균 검출 수: {results['detection_metrics']['avg_detections_per_image']:.2f}")
            print(f"    - 평균 신뢰도: {results['detection_metrics']['avg_confidence']:.3f}")
            print(f"    - 검출률: {results['detection_metrics']['detection_rate']:.3f}")
            print(f"  속도:")
            print(f"    - 평균 추론시간: {results['speed_metrics']['avg_inference_time']*1000:.2f} ms")
            print(f"    - FPS: {results['speed_metrics']['fps']:.1f}")
            print(f"  효율성:")
            print(f"    - 모델 크기: {results['model_size']:.1f} MB")
            print(f"    - CPU 메모리: {results['memory_metrics']['cpu_memory_usage']:.1f} MB")

def main():
    print("모델 성능 비교 도구 (이미지 폴더 버전)")
    print("PyTorch 모델과 ONNX 모델의 성능을 비교합니다.")
    
    # 사용자 입력
    pt_model_path = input("PyTorch 모델 경로 (.pt): ").strip()
    onnx_model_path = input("ONNX 모델 경로 (.onnx): ").strip()
    test_images_folder = input("테스트 이미지 폴더 경로: ").strip()
    
    conf_str = input("신뢰도 임계값 (기본값 0.5): ").strip() or "0.5"
    conf_threshold = float(conf_str)
    
    # 비교 실행
    comparator = ModelComparison()
    
    try:
        # PyTorch 모델 평가
        comparator.evaluate_model_on_images(pt_model_path, "PyTorch", test_images_folder, conf_threshold)
        
        # ONNX 모델 평가
        comparator.evaluate_model_on_images(onnx_model_path, "ONNX", test_images_folder, conf_threshold)
        
        # 결과 시각화
        comparator.visualize_comparison()
        
        # 결과 요약 출력
        comparator.print_summary()
        
        # 결과 저장
        comparator.save_results()
        
        print("\n✅ 모든 비교 작업이 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()