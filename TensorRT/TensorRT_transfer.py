import tensorrt as trt
import numpy as np
import cv2
import os
from pathlib import Path
import glob

def build_jetson_compatible_engine(onnx_path, output_engine_path, precision="fp16"):
    """
    Jetson Nano 2GB 호환 TensorRT 엔진 빌드 (플래그 호환성 개선)
    PC에서 빌드 후 Jetson으로 전송하여 사용
    
    Args:
        onnx_path: 원본 ONNX 모델 경로
        output_engine_path: 출력 엔진 파일 경로  
        precision: "fp16" 또는 "fp32"
    """
    
    print(f"🎯 Jetson Nano 호환 TensorRT {precision.upper()} 엔진 빌드")
    print("=" * 70)
    print("⚠️  주의: 이 엔진은 Jetson Nano 2GB 전용입니다!")
    print("")
    
    # TensorRT 초기화
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # 네트워크 생성
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # 🔧 Jetson Nano 2GB 최적화 설정
    print("🔧 Jetson Nano 최적화 설정 적용:")
    
    # 1. 매우 제한적인 메모리 설정 (128MB)
    jetson_workspace = 128 << 20  # 128MB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, jetson_workspace)
    print(f"   작업 메모리: {jetson_workspace / (1024*1024):.0f}MB")
    
    # 2. Jetson 호환성 플래그 (버전별 호환성 처리)
    print("🔧 호환성 플래그 설정:")
    
    # STRICT_TYPES 플래그 확인 및 설정
    if hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        print("   ✅ STRICT_TYPES 플래그 설정")
    else:
        print("   ⚠️  STRICT_TYPES 플래그 없음 (TensorRT 버전 호환)")
    
    # GPU_FALLBACK 플래그 확인 및 설정
    if hasattr(trt.BuilderFlag, 'GPU_FALLBACK'):
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print("   ✅ GPU_FALLBACK 플래그 설정")
    else:
        print("   ⚠️  GPU_FALLBACK 플래그 없음 (TensorRT 버전 호환)")
    
    # 3. 배치 크기 제한 (반드시 1)
    max_batch_size = 1
    print(f"   최대 배치 크기: {max_batch_size}")
    
    # 4. 정밀도 설정
    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   정밀도: FP16 (Jetson Nano 호환)")
        else:
            print("   ⚠️  FP16이 지원되지 않습니다. FP32로 빌드합니다.")
    elif precision.lower() == "int8":
        print("   ⚠️  INT8은 Jetson Nano에서 메모리 부족 가능성 높음")
        print("   FP16으로 변경하는 것을 권장합니다.")
        return False
    else:
        print("   정밀도: FP32 (기본)")
    
    # 5. 추가 Jetson 최적화 플래그들
    print("🔧 추가 최적화 설정:")
    
    # DLA 관련 설정 (Jetson에서만 의미 있음)
    if hasattr(trt.BuilderFlag, 'PREFER_PRECISION_CONSTRAINTS'):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print("   ✅ PREFER_PRECISION_CONSTRAINTS 설정")
    
    # 메모리 최적화
    if hasattr(config, 'set_tactic_sources'):
        # 전술 소스 제한 (메모리 절약)
        config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        print("   ✅ 전술 소스 제한 (메모리 절약)")
    
    print(f"   타겟 아키텍처: Maxwell (Jetson Nano GPU)")
    
    # ONNX 파싱
    print("\n🔄 ONNX 모델 파싱...")
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 파일을 찾을 수 없습니다: {onnx_path}")
        return False
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ ONNX 파싱 실패')
            for error in range(parser.num_errors):
                print(f"   오류: {parser.get_error(error)}")
            return False
    
    print("✅ ONNX 파싱 완료")
    
    # 입력/출력 정보 확인
    print(f"\n📊 모델 정보:")
    print(f"   입력 수: {network.num_inputs}")
    print(f"   출력 수: {network.num_outputs}")
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"   입력 {i}: {input_tensor.name}, 형태: {input_tensor.shape}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"   출력 {i}: {output_tensor.name}, 형태: {output_tensor.shape}")
    
    # 엔진 빌드
    print(f"\n🔄 Jetson 호환 엔진 빌드 중... ({precision.upper()})")
    print("   💡 이 과정은 5-20분 소요될 수 있습니다.")
    print("   ☕ 커피 한 잔 하고 오세요...")
    
    try:
        # 버전별 호환성 처리
        if hasattr(builder, 'build_serialized_network'):
            print("   빌드 방법: build_serialized_network")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if not serialized_engine:
                print("❌ 엔진 빌드 실패")
                return False
            
            # 엔진 저장
            os.makedirs(os.path.dirname(output_engine_path), exist_ok=True)
            with open(output_engine_path, "wb") as f:
                f.write(serialized_engine)
        
        elif hasattr(builder, 'build_engine'):
            print("   빌드 방법: build_engine")
            engine = builder.build_engine(network, config)
            
            if not engine:
                print("❌ 엔진 빌드 실패")
                return False
            
            # 엔진 저장
            os.makedirs(os.path.dirname(output_engine_path), exist_ok=True)
            with open(output_engine_path, "wb") as f:
                f.write(engine.serialize())
            
            del engine
        
        else:
            print("❌ 지원되지 않는 TensorRT 버전")
            return False
    
    except Exception as e:
        print(f"❌ 엔진 빌드 중 오류: {e}")
        print(f"   오류 상세: {type(e).__name__}")
        return False
    
    print("✅ Jetson 호환 엔진 빌드 완료!")
    
    # 파일 정보
    engine_size = os.path.getsize(output_engine_path) / (1024 * 1024)
    print(f"\n📦 빌드 결과:")
    print(f"   엔진 파일: {output_engine_path}")
    print(f"   파일 크기: {engine_size:.1f} MB")
    print(f"   정밀도: {precision.upper()}")
    print(f"   타겟: Jetson Nano 2GB")
    
    # Jetson으로 전송 방법 안내
    print(f"\n📤 Jetson Nano로 전송 방법:")
    print(f"   1. SCP 사용:")
    print(f"      scp {output_engine_path} jetson@<IP>:/home/jetson/models/")
    print(f"   2. USB/SD카드로 복사")
    print(f"   3. 원격 개발 환경에서 직접 복사")
    
    # 메모리 정리
    del network
    del config  
    del parser
    del builder
    
    return True

def create_jetson_engines():
    """여러 정밀도로 Jetson 엔진 생성"""
    
    # ========================================
    # 🔧 설정 수정
    # ========================================
    
    ONNX_MODEL_PATH = r"C:\Users\KDT34\Desktop\G6_ver2\yolov8_custom_fixed_test7_pruned.onnx"
    
    # Jetson용 출력 경로
    JETSON_FP32_ENGINE = "jetson_models/yolov8_jetson_fp32.engine"
    JETSON_FP16_ENGINE = "jetson_models/yolov8_jetson_fp16.engine"
    
    # ========================================
    
    print("🎯 Jetson Nano 호환 엔진 생성")
    print("=" * 50)
    print(f"📋 TensorRT 버전: {trt.__version__}")
    
    # 엔진 옵션
    engines_to_build = []
    
    print("빌드할 엔진을 선택하세요:")
    print("1. FP32 (가장 안전, 큰 용량)")
    print("2. FP16 (권장, 절반 용량)")  
    print("3. 둘 다")
    
    choice = input("선택 (1/2/3) [2]: ").strip() or "2"
    
    if choice in ["1", "3"]:
        engines_to_build.append(("fp32", JETSON_FP32_ENGINE))
    
    if choice in ["2", "3"]:
        engines_to_build.append(("fp16", JETSON_FP16_ENGINE))
    
    # 엔진 빌드 실행
    success_count = 0
    
    for precision, output_path in engines_to_build:
        print(f"\n{'='*50}")
        print(f"🔄 {precision.upper()} 엔진 빌드 시작")
        
        success = build_jetson_compatible_engine(
            onnx_path=ONNX_MODEL_PATH,
            output_engine_path=output_path,
            precision=precision
        )
        
        if success:
            success_count += 1
            print(f"✅ {precision.upper()} 엔진 빌드 성공!")
        else:
            print(f"❌ {precision.upper()} 엔진 빌드 실패!")
    
    # 최종 결과
    print(f"\n🎉 Jetson 엔진 빌드 완료!")
    print(f"   성공: {success_count}/{len(engines_to_build)}개")
    
    if success_count > 0:
        print(f"\n📋 다음 단계:")
        print(f"   1. 생성된 .engine 파일을 Jetson Nano로 전송")
        print(f"   2. Jetson에서 TensorRT 추론 코드 실행")
        print(f"   3. 성능 테스트 및 최적화")
        
        print(f"\n💡 Jetson에서 사용 예시:")
        print(f"   import tensorrt as trt")
        print(f"   runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))")
        print(f"   with open('yolov8_jetson_fp16.engine', 'rb') as f:")
        print(f"       engine = runtime.deserialize_cuda_engine(f.read())")

if __name__ == "__main__":
    create_jetson_engines()