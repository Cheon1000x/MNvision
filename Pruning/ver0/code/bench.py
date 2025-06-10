"""
프루닝 모델 벤치마크 비교 (핵심 기능만)
원본 vs 프루닝 모델의 성능, 속도, 크기 비교
"""

import torch
import time
import numpy as np
from ultralytics import YOLO
import psutil
import os

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
        
        print(f"\n✅ 벤치마크 완료!")


def main():
    # 모델 경로 설정
    ORIGINAL_MODEL = r"C:\Users\KDT34\Desktop\Group6\original_model.pt"  # 원본 모델
    PRUNED_MODEL = r"C:\Users\KDT34\Desktop\Group6\pruning_model.pt"  # 프루닝 모델
    
    # 검증 데이터 설정 - 변환된 YOLO 형식 폴더 사용
    DATA_FOLDER = r"C:\Users\KDT34\Desktop\Group6\temp_validation"  # 변환된 YOLO 데이터셋
    
    # 벤치마크 실행
    benchmark = PruningBenchmark(ORIGINAL_MODEL, PRUNED_MODEL)
    
    # 변환된 YOLO 데이터셋으로 정확도 비교
    benchmark.run_benchmark(data_path=DATA_FOLDER, num_runs=50)


if __name__ == "__main__":
    main()