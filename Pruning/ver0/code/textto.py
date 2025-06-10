"""
프루닝 모델 벤치마크 비교 (핵심 기능만)
원본 vs 프루닝 모델의 성능, 속도, 크기 비교 + 자동 그래프 생성
"""

import torch
import time
import numpy as np
from ultralytics import YOLO
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PruningBenchmark:
    def __init__(self, original_model_path, pruned_model_path):
        """
        Args:
            original_model_path: 원본 모델 경로
            pruned_model_path: 프루닝된 모델 경로
        """
        print("🔄 모델 로딩 중...")
        self.original_model = YOLO(original_model_path)
        self.pruned_model = YOLO(pruned_model_path)
        print("✅ 모델 로딩 완료")
        
    def get_model_info(self, model, label):
        """모델 정보 추출"""
        # 파라미터 수
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # 모델 크기 (MB)
        param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'label': label,
            'params': total_params,
            'size_mb': size_mb
        }
    
    def benchmark_speed(self, model, num_runs=100, img_size=(192, 320)):
        """추론 속도 벤치마크"""
        model.model.eval()
        
        # 더미 이미지 생성
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
        
        # GPU 사용 가능하면 GPU로
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.model.to(device)
        dummy_input = dummy_input.to(device)
        
        # 워밍업 (5회)
        with torch.no_grad():
            for _ in range(5):
                _ = model.model(dummy_input)
        
        # 실제 벤치마크
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        # 결과 계산
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        fps = 1000 / avg_time_ms
        
        return {
            'avg_time_ms': avg_time_ms,
            'fps': fps,
            'total_time': total_time
        }
    
    def benchmark_memory(self, model):
        """메모리 사용량 벤치마크"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 더미 추론
            dummy_input = torch.randn(1, 3, 192, 320).cuda()
            model.model.cuda()
            
            with torch.no_grad():
                _ = model.model(dummy_input)
            
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()
            
            return {'gpu_memory_mb': memory_mb}
        else:
            # CPU 메모리 (대략적)
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return {'cpu_memory_mb': memory_mb}
    
    def create_benchmark_graphs(self, original_info, pruned_info, original_speed, pruned_speed, 
                               original_memory, pruned_memory, accuracy_results=None):
        """벤치마크 결과를 자동으로 그래프로 생성"""
        
        # 개선율 계산
        param_reduction = (1 - pruned_info['params'] / original_info['params']) * 100
        size_reduction = (1 - pruned_info['size_mb'] / original_info['size_mb']) * 100
        speed_improvement = pruned_speed['fps'] / original_speed['fps']
        time_reduction = (1 - pruned_speed['avg_time_ms'] / original_speed['avg_time_ms']) * 100
        
        memory_key = 'gpu_memory_mb' if torch.cuda.is_available() else 'cpu_memory_mb'
        memory_reduction = (1 - pruned_memory[memory_key] / original_memory[memory_key]) * 100
        
        # 그래프 생성
        if accuracy_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['#3498db', '#e74c3c']
        models = ['Original', 'Pruned']
        
        # 1. 모델 크기 및 파라미터 비교
        params = [original_info['params']/1000000, pruned_info['params']/1000000]  # 백만 단위
        sizes = [original_info['size_mb'], pruned_info['size_mb']]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, params, width, label='Parameters (M)', color=colors[0], alpha=0.8)
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, sizes, width, label='Size (MB)', color=colors[1], alpha=0.8)
        
        ax1.set_title('Model Size & Parameters', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Parameters (Millions)', fontsize=12)
        ax1_twin.set_ylabel('Size (MB)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, (param, size) in enumerate(zip(params, sizes)):
            ax1.text(i - width/2, param + 0.05, f'{param:.2f}M', ha='center', va='bottom', fontweight='bold')
            ax1_twin.text(i + width/2, size + 0.1, f'{size:.2f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 감소율 표시
        ax1.text(0.5, 0.9, f'Params: {param_reduction:.1f}% ↓\nSize: {size_reduction:.1f}% ↓', 
                transform=ax1.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. 속도 비교
        fps_values = [original_speed['fps'], pruned_speed['fps']]
        time_values = [original_speed['avg_time_ms'], pruned_speed['avg_time_ms']]
        
        bars3 = ax2.bar(x - width/2, fps_values, width, label='FPS', color='#2ecc71', alpha=0.8)
        ax2_twin = ax2.twinx()
        bars4 = ax2_twin.bar(x + width/2, time_values, width, label='Time (ms)', color='#f39c12', alpha=0.8)
        
        ax2.set_title('Speed Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('FPS', fontsize=12)
        ax2_twin.set_ylabel('Inference Time (ms)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, (fps, time_val) in enumerate(zip(fps_values, time_values)):
            ax2.text(i - width/2, fps + 1, f'{fps:.1f}', ha='center', va='bottom', fontweight='bold')
            ax2_twin.text(i + width/2, time_val + 0.2, f'{time_val:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 향상률 표시
        ax2.text(0.5, 0.9, f'Speed: {speed_improvement:.2f}x ↑\nTime: {time_reduction:.1f}% ↓', 
                transform=ax2.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # 3. 종합 성능 레이더 차트
        if accuracy_results:
            # 정확도 포함된 레이더 차트
            categories = ['Speed\n(FPS)', 'Efficiency\n(Params)', 'Size\n(MB)', 'Accuracy\n(mAP50)', 'Memory\n(MB)']
            
            # 정규화된 값 (0-1 범위)
            original_values = [
                original_speed['fps'] / 100,  # FPS 정규화
                1.0,  # 원본을 1.0으로 기준
                1.0,  # 원본을 1.0으로 기준  
                accuracy_results['original_map50'],
                1.0   # 원본을 1.0으로 기준
            ]
            
            pruned_values = [
                pruned_speed['fps'] / 100,
                pruned_info['params'] / original_info['params'],
                pruned_info['size_mb'] / original_info['size_mb'],
                accuracy_results['pruned_map50'],
                pruned_memory[memory_key] / original_memory[memory_key]
            ]
        else:
            # 정확도 없는 레이더 차트
            categories = ['Speed\n(FPS)', 'Efficiency\n(Params)', 'Size\n(MB)', 'Memory\n(MB)']
            
            original_values = [
                original_speed['fps'] / 100,
                1.0,
                1.0,
                1.0
            ]
            
            pruned_values = [
                pruned_speed['fps'] / 100,
                pruned_info['params'] / original_info['params'],
                pruned_info['size_mb'] / original_info['size_mb'],
                pruned_memory[memory_key] / original_memory[memory_key]
            ]
        
        # 레이더 차트 생성
        ax_radar = ax3 if not accuracy_results else ax3
        if accuracy_results:
            ax_radar = plt.subplot(2, 2, 3, projection='polar')
        else:
            ax_radar = plt.subplot(1, 3, 3, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 닫힌 도형을 위해
        
        original_values += [original_values[0]]  # 닫힌 도형
        pruned_values += [pruned_values[0]]
        
        ax_radar.plot(angles, original_values, 'o-', linewidth=2, label='Original', color='#3498db')
        ax_radar.fill(angles, original_values, alpha=0.25, color='#3498db')
        ax_radar.plot(angles, pruned_values, 'o-', linewidth=2, label='Pruned', color='#e74c3c')
        ax_radar.fill(angles, pruned_values, alpha=0.25, color='#e74c3c')
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax_radar.grid(True)
        
        # 4. 정확도 비교 (있는 경우)
        if accuracy_results:
            metrics = ['mAP50', 'mAP50-95']
            original_acc = [accuracy_results['original_map50'], accuracy_results['original_map']]
            pruned_acc = [accuracy_results['pruned_map50'], accuracy_results['pruned_map']]
            
            x_acc = np.arange(len(metrics))
            bars5 = ax4.bar(x_acc - width/2, original_acc, width, label='Original', color='#3498db', alpha=0.8)
            bars6 = ax4.bar(x_acc + width/2, pruned_acc, width, label='Pruned', color='#e74c3c', alpha=0.8)
            
            ax4.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Accuracy Score', fontsize=12)
            ax4.set_xticks(x_acc)
            ax4.set_xticklabels(metrics)
            ax4.grid(axis='y', alpha=0.3)
            ax4.legend()
            
            # 값 표시
            for i, (orig, prun) in enumerate(zip(original_acc, pruned_acc)):
                ax4.text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom', fontweight='bold')
                ax4.text(i + width/2, prun + 0.01, f'{prun:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 변화율 표시
            map50_change = accuracy_results['pruned_map50'] - accuracy_results['original_map50']
            map_change = accuracy_results['pruned_map'] - accuracy_results['original_map']
            ax4.text(0.5, 0.9, f'mAP50: {map50_change:+.3f}\nmAP: {map_change:+.3f}', 
                    transform=ax4.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('pruning_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("📊 벤치마크 결과 그래프가 'pruning_benchmark_results.png'에 저장되었습니다!")
        plt.show()
        
        return fig
    
    def validate_accuracy(self, data_path=None):
        """정확도 검증 (데이터셋이 있는 경우)"""
        if not data_path or not os.path.exists(data_path):
            print("⚠️  검증 데이터셋이 없어 정확도 비교를 건너뜁니다.")
            return None, None
        
        print("📊 정확도 검증 중...")
        
        # 데이터 타입 확인 (YAML 파일 vs 폴더)
        if os.path.isfile(data_path) and data_path.endswith('.yaml'):
            # YAML 파일인 경우
            data_source = data_path
            print(f"   📁 YAML 데이터셋 사용: {data_path}")
        elif os.path.isdir(data_path):
            # 폴더인 경우
            data_source = data_path
            print(f"   📁 폴더 데이터셋 사용: {data_path}")
        else:
            print(f"⚠️  지원하지 않는 데이터 형식: {data_path}")
            return None, None
        
        try:
            # 원본 모델 검증
            print("   🔄 원본 모델 검증 중...")
            original_results = self.original_model.val(data=data_source, verbose=False)
            
            # 프루닝 모델 검증  
            print("   🔄 프루닝 모델 검증 중...")
            pruned_results = self.pruned_model.val(data=data_source, verbose=False)
            
            return {
                'original_map50': float(original_results.box.map50),
                'original_map': float(original_results.box.map),
                'pruned_map50': float(pruned_results.box.map50), 
                'pruned_map': float(pruned_results.box.map)
            }, None
            
        except Exception as e:
            print(f"⚠️  정확도 검증 중 오류: {e}")
            return None, None
        """정확도 검증 (데이터셋이 있는 경우)"""
        if not data_path or not os.path.exists(data_path):
            print("⚠️  검증 데이터셋이 없어 정확도 비교를 건너뜁니다.")
            return None, None
        
        print("📊 정확도 검증 중...")
        
        # 데이터 타입 확인 (YAML 파일 vs 폴더)
        if os.path.isfile(data_path) and data_path.endswith('.yaml'):
            # YAML 파일인 경우
            data_source = data_path
            print(f"   📁 YAML 데이터셋 사용: {data_path}")
        elif os.path.isdir(data_path):
            # 폴더인 경우
            data_source = data_path
            print(f"   📁 폴더 데이터셋 사용: {data_path}")
        else:
            print(f"⚠️  지원하지 않는 데이터 형식: {data_path}")
            return None, None
        
        try:
            # 원본 모델 검증
            print("   🔄 원본 모델 검증 중...")
            original_results = self.original_model.val(data=data_source, verbose=False)
            
            # 프루닝 모델 검증  
            print("   🔄 프루닝 모델 검증 중...")
            pruned_results = self.pruned_model.val(data=data_source, verbose=False)
            
            return {
                'original_map50': float(original_results.box.map50),
                'original_map': float(original_results.box.map),
                'pruned_map50': float(pruned_results.box.map50), 
                'pruned_map': float(pruned_results.box.map)
            }, None
            
        except Exception as e:
            print(f"⚠️  정확도 검증 중 오류: {e}")
            return None, None
    
    def run_benchmark(self, data_path=None, num_runs=100):
        """전체 벤치마크 실행"""
        print("\n🚀 프루닝 모델 벤치마크 시작")
        print("=" * 50)
        
        # 1. 모델 정보 비교
        print("\n1️⃣ 모델 정보:")
        original_info = self.get_model_info(self.original_model, "원본")
        pruned_info = self.get_model_info(self.pruned_model, "프루닝")
        
        print(f"📊 {original_info['label']} 모델:")
        print(f"   파라미터: {original_info['params']:,}개")
        print(f"   크기: {original_info['size_mb']:.2f} MB")
        
        print(f"📊 {pruned_info['label']} 모델:")
        print(f"   파라미터: {pruned_info['params']:,}개")
        print(f"   크기: {pruned_info['size_mb']:.2f} MB")
        
        # 개선율 계산
        param_reduction = (1 - pruned_info['params'] / original_info['params']) * 100
        size_reduction = (1 - pruned_info['size_mb'] / original_info['size_mb']) * 100
        
        print(f"\n📉 감소율:")
        print(f"   파라미터: {param_reduction:.1f}% 감소")
        print(f"   모델 크기: {size_reduction:.1f}% 감소")
        
        # 2. 속도 벤치마크
        print(f"\n2️⃣ 속도 벤치마크 ({num_runs}회 평균):")
        
        print("⏱️  원본 모델 속도 측정 중...")
        original_speed = self.benchmark_speed(self.original_model, num_runs)
        
        print("⏱️  프루닝 모델 속도 측정 중...")
        pruned_speed = self.benchmark_speed(self.pruned_model, num_runs)
        
        print(f"🔄 원본 모델:")
        print(f"   추론 시간: {original_speed['avg_time_ms']:.2f}ms")
        print(f"   FPS: {original_speed['fps']:.1f}")
        
        print(f"🔄 프루닝 모델:")
        print(f"   추론 시간: {pruned_speed['avg_time_ms']:.2f}ms") 
        print(f"   FPS: {pruned_speed['fps']:.1f}")
        
        # 속도 향상 계산
        speed_improvement = pruned_speed['fps'] / original_speed['fps']
        time_reduction = (1 - pruned_speed['avg_time_ms'] / original_speed['avg_time_ms']) * 100
        
        print(f"\n⚡ 속도 향상:")
        print(f"   FPS 향상: {speed_improvement:.2f}배")
        print(f"   시간 단축: {time_reduction:.1f}%")
        
        # 3. 메모리 사용량
        print(f"\n3️⃣ 메모리 사용량:")
        
        original_memory = self.benchmark_memory(self.original_model)
        pruned_memory = self.benchmark_memory(self.pruned_model)
        
        memory_key = 'gpu_memory_mb' if torch.cuda.is_available() else 'cpu_memory_mb'
        memory_type = 'GPU' if torch.cuda.is_available() else 'CPU'
        
        print(f"💾 원본 모델 ({memory_type}): {original_memory[memory_key]:.1f} MB")
        print(f"💾 프루닝 모델 ({memory_type}): {pruned_memory[memory_key]:.1f} MB")
        
        memory_reduction = (1 - pruned_memory[memory_key] / original_memory[memory_key]) * 100
        print(f"💾 메모리 절약: {memory_reduction:.1f}%")
        
        # 4. 정확도 비교 (옵션)
        if data_path:
            accuracy_results, _ = self.validate_accuracy(data_path)
            if accuracy_results:
                print(f"\n4️⃣ 정확도 비교:")
                print(f"📈 원본 모델:")
                print(f"   mAP50: {accuracy_results['original_map50']:.3f}")
                print(f"   mAP50-95: {accuracy_results['original_map']:.3f}")
                
                print(f"📈 프루닝 모델:")
                print(f"   mAP50: {accuracy_results['pruned_map50']:.3f}")
                print(f"   mAP50-95: {accuracy_results['pruned_map']:.3f}")
                
                # 정확도 변화
                map50_change = accuracy_results['pruned_map50'] - accuracy_results['original_map50']
                map_change = accuracy_results['pruned_map'] - accuracy_results['original_map']
                
                print(f"📊 정확도 변화:")
                print(f"   mAP50: {map50_change:+.3f}")
                print(f"   mAP50-95: {map_change:+.3f}")
        
        # 5. 종합 요약
        print(f"\n🎯 종합 요약:")
        print("=" * 30)
        print(f"⚡ 속도 향상: {speed_improvement:.2f}배")
        print(f"📉 파라미터 감소: {param_reduction:.1f}%")
        print(f"💾 크기 감소: {size_reduction:.1f}%")
        print(f"🧠 메모리 절약: {memory_reduction:.1f}%")
        
        # 효율성 점수 계산
        efficiency_score = (speed_improvement * (param_reduction/100) * (size_reduction/100)) * 100
        print(f"🏆 효율성 점수: {efficiency_score:.1f}/100")
        
        # 자동 그래프 생성
        print(f"\n📊 벤치마크 결과 그래프 생성 중...")
        self.create_benchmark_graphs(
            original_info, pruned_info, 
            original_speed, pruned_speed,
            original_memory, pruned_memory,
            accuracy_results
        )
        
        print(f"\n✅ 벤치마크 완료!")


def main():
    # 모델 경로 설정
    ORIGINAL_MODEL = r"C:\Users\KDT34\Desktop\Group6\original_model.pt"  # 원본 모델
    PRUNED_MODEL = r"C:\Users\KDT34\Desktop\Group6\pruning_model.pt"  # 프루닝 모델
    
    # 검증 데이터 설정 - YAML 파일 직접 사용
    DATA_YAML = r"C:\Users\KDT34\Desktop\Group6\temp_validation\dataset.yaml"  # 변환된 YAML 파일
    
    # 벤치마크 실행
    benchmark = PruningBenchmark(ORIGINAL_MODEL, PRUNED_MODEL)
    
    # YAML 파일로 정확도 비교
    benchmark.run_benchmark(data_path=DATA_YAML, num_runs=50)


if __name__ == "__main__":
    main()