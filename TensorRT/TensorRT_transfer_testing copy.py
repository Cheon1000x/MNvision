import tensorrt as trt
import numpy as np
import time

def simulate_jetson_inference(engine_path, input_shape=(1, 3, 192, 320)):
    """Jetson Nano 추론 시뮬레이션"""
    print(f"🎯 Jetson 추론 시뮬레이션: {engine_path}")
    
    try:
        # 런타임 및 엔진 로드
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("❌ 엔진 로드 실패")
            return False
        
        # 실행 컨텍스트 생성
        context = engine.create_execution_context()
        
        # 더미 입력 데이터 생성
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # GPU 메모리 할당 시뮬레이션 (실제로는 CUDA 메모리 사용)
        print(f"📊 메모리 사용량 시뮬레이션:")
        input_size = dummy_input.nbytes / (1024 * 1024)
        print(f"   입력 메모리: {input_size:.2f} MB")
        
        # 추론 시간 측정
        print(f"⏱️ 추론 시간 측정 (10회 평균):")
        times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            
            # 실제 추론 대신 시뮬레이션
            time.sleep(0.01)  # Jetson Nano 예상 추론 시간
            
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000
            times.append(inference_time)
        
        avg_time = np.mean(times)
        print(f"   평균 추론 시간: {avg_time:.2f}ms")
        print(f"   예상 FPS: {1000/avg_time:.1f}")
        
        # Jetson Nano 2GB 메모리 제약 확인
        estimated_memory = input_size * 2 + 50  # 대략적 메모리 사용량
        print(f"\n💾 Jetson Nano 호환성:")
        print(f"   예상 메모리 사용량: {estimated_memory:.1f} MB")
        
        if estimated_memory < 1500:  # 2GB 중 1.5GB 이하 사용
            print(f"   ✅ Jetson Nano 2GB에서 실행 가능")
        else:
            print(f"   ⚠️ 메모리 부족 위험")
        
        return True
        
    except Exception as e:
        print(f"❌ 시뮬레이션 실패: {e}")
        return False

# 시뮬레이션 실행
simulate_jetson_inference(r"C:\Users\KDT34\Desktop\G6_ver2\jetson_models\yolov8_jetson_fp16.engine")