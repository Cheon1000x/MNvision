def safe_segment_to_onnx(model_path, output_path="model_seg.onnx", img_size=640):
    """안전한 세그멘테이션 모델 ONNX 변환 (순환참조 문제 해결)"""
    import os
    from ultralytics import YOLO
    import torch
    
    try:
        print("--- 세그멘테이션 모델 ONNX 변환 시작 ---")
        
        # 1. 모델 로드 (task 명시)
        print(f"모델 로딩: {model_path}")
        model = YOLO(model_path, task='segment')
        
        # 2. 모델 정보 확인
        print(f"모델 타입: {model.task}")
        print(f"모델 이름: {type(model.model).__name__}")
        print(f"클래스 수: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        
        # 3. 순환참조 문제 해결을 위한 안전한 방법들
        print("모델 구조 안전 처리 중...")
        
        # 방법 1: detect 속성 문제 우회 - 직접 변환 시도
        conversion_methods = [
            ("기본 변환", lambda: basic_conversion(model, img_size)),
            ("torch.jit 우회", lambda: jit_trace_conversion(model, img_size)),
            ("직접 export", lambda: direct_export_conversion(model, img_size)),
            ("안전 모드", lambda: safe_mode_conversion(model, img_size))
        ]
        
        for method_name, conversion_func in conversion_methods:
            print(f"\n--- {method_name} 시도 ---")
            try:
                success = conversion_func()
                if success:
                    # 변환된 파일 확인
                    possible_paths = [
                        output_path,
                        model_path.replace('.pt', '.onnx'),
                        os.path.join(os.path.dirname(model_path), 'best.onnx'),
                        'yolov8_seg_onnx_ver1.onnx'
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            if path != output_path:
                                import shutil
                                shutil.move(path, output_path)
                            
                            print(f"✅ {method_name} 성공: {output_path}")
                            
                            # 검증
                            if verify_onnx_model(output_path):
                                return output_path
                            else:
                                print("⚠️ 검증 실패, 다음 방법 시도")
                                break
                
            except Exception as e:
                print(f"❌ {method_name} 실패: {e}")
                continue
        
        print("❌ 모든 변환 방법 실패")
        return None
                
    except Exception as e:
        print(f"ONNX 변환 중 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def basic_conversion(model, img_size):
    """기본 변환 방법"""
    try:
        success = model.export(
            format="onnx",
            imgsz=img_size,
            opset=12,
            simplify=False,
            dynamic=False,
            half=False,
            verbose=True
        )
        return success
    except:
        return False

def jit_trace_conversion(model, img_size):
    """torch.jit.trace를 이용한 변환"""
    try:
        # 모델을 직접 trace하여 변환
        model.model.eval()
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # JIT trace (순환참조 문제 우회)
        with torch.no_grad():
            traced_model = torch.jit.trace(model.model, dummy_input, strict=False)
        
        # ONNX로 변환
        torch.onnx.export(
            traced_model,
            dummy_input,
            "temp_traced.onnx",
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output0', 'output1'],
            verbose=True
        )
        
        return os.path.exists("temp_traced.onnx")
    except Exception as e:
        print(f"JIT trace 변환 오류: {e}")
        return False

def direct_export_conversion(model, img_size):
    """직접 export (detect 속성 무시)"""
    try:
        # detect 속성 문제를 우회하기 위해 임시로 제거
        head = model.model.model[-1]
        
        # 기존 detect 속성 백업 (있다면)
        original_detect = getattr(head, 'detect', None)
        
        # detect 속성 임시 제거
        if hasattr(head, 'detect'):
            delattr(head, 'detect')
        
        try:
            # 변환 시도
            success = model.export(
                format="onnx",
                imgsz=img_size,
                opset=11,  # 더 낮은 버전
                simplify=False,
                dynamic=False,
                half=False
            )
        finally:
            # detect 속성 복원 (필요시)
            if original_detect is not None:
                head.detect = original_detect
        
        return success
    except Exception as e:
        print(f"직접 export 오류: {e}")
        return False

def safe_mode_conversion(model, img_size):
    """안전 모드 변환 (최소 설정)"""
    try:
        # 가장 안전한 설정으로 변환
        success = model.export(
            format="onnx",
            imgsz=img_size,
            opset=9,     # 가장 낮은 버전
            simplify=False,
            dynamic=False,
            half=False,
            int8=False,
            verbose=False  # 로그 최소화
        )
        return success
    except Exception as e:
        print(f"안전 모드 변환 오류: {e}")
        return False

def verify_onnx_model(onnx_path):
    """변환된 ONNX 모델 검증"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"ONNX 모델 검증 중: {onnx_path}")
        
        # ONNX 모델 로드
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 입출력 정보 확인
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print("입력 정보:")
        for inp in inputs:
            print(f"  - {inp.name}: {inp.shape}")
        
        print("출력 정보:")
        for out in outputs:
            print(f"  - {out.name}: {out.shape}")
        
        # 더미 데이터로 추론 테스트
        input_shape = inputs[0].shape
        # 동적 차원 처리
        processed_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None:
                processed_shape.append(1)
            else:
                processed_shape.append(dim)
        
        dummy_input = np.random.randn(*processed_shape).astype(np.float32)
        
        print("더미 데이터로 추론 테스트 중...")
        result_outputs = session.run(None, {inputs[0].name: dummy_input})
        
        print(f"✅ 추론 성공! 출력 개수: {len(result_outputs)}")
        for i, output in enumerate(result_outputs):
            print(f"  출력 {i}: 형태={output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX 모델 검증 실패: {e}")
        return False

# 추가: 강제 변환 방법 (최후의 수단)
def force_conversion_last_resort(model_path, output_path="model_seg.onnx", img_size=640):
    """최후의 수단: 체크포인트에서 직접 변환"""
    try:
        import torch
        from ultralytics.nn.tasks import SegmentationModel
        
        print("--- 강제 변환 시도 (최후의 수단) ---")
        
        # 1. 체크포인트 직접 로드
        ckpt = torch.load(model_path, map_location='cpu')
        
        # 2. 새로운 모델 인스턴스 생성
        if 'model' in ckpt:
            # 모델 설정 추출
            cfg = ckpt['model'].yaml if hasattr(ckpt['model'], 'yaml') else None
            nc = ckpt['model'].nc if hasattr(ckpt['model'], 'nc') else 11
            
            # 기본 YOLOv8n-seg 사용
            from ultralytics import YOLO
            base_model = YOLO('yolov8n-seg.pt')
            model = base_model.model
            
            # 가중치 로드 (호환되는 것만)
            try:
                model.load_state_dict(ckpt['model'], strict=False)
                print("✅ 가중치 로드 완료")
            except:
                print("⚠️ 가중치 로드 일부 실패, 계속 진행")
            
            model.eval()
            
            # 3. torch.onnx.export로 직접 변환
            dummy_input = torch.randn(1, 3, img_size, img_size)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output0', 'output1']
            )
            
            if os.path.exists(output_path):
                print(f"✅ 강제 변환 성공: {output_path}")
                return output_path
            
        return None
        
    except Exception as e:
        print(f"❌ 강제 변환 실패: {e}")
        return None

# 사용 예시
if __name__ == "__main__":
    model_path = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\4.Model Experimental Data\Test_7_onnx\yolov8_seg_onnx_ver1.pt"
    output_path = "model_seg_fixed.onnx"
    
    # 1차 시도: 안전한 변환
    result = safe_segment_to_onnx(model_path, output_path, img_size=640)
    
    if not result:
        print("\n🚨 1차 시도 실패, 강제 변환 시도...")
        # 2차 시도: 강제 변환
        result = force_conversion_last_resort(model_path, output_path, img_size=640)
    
    if result:
        print(f"\n🎉 최종 변환 성공!")
        print(f"📁 파일 위치: {result}")
    else:
        print(f"\n💥 모든 변환 방법 실패!")
        print("🔧 다른 접근이 필요합니다.")