# ONNX_seg_converter.py
import os
from ultralytics import YOLO

def convert_seg_to_onnx(model_path, output_path="model_seg.onnx", img_size=640):
    """
    YOLOv8 세그멘테이션 모델을 ONNX로 변환
    """
    print(f"모델 로딩: {model_path}")
    model = YOLO(model_path)
    
    print("ONNX 변환 시작...")
    success = model.export(
        format="onnx",
        imgsz=img_size,
        opset=16,
        simplify=True
    )
    
    if success:
        # 기본 저장 경로에서 원하는 경로로 이동
        default_onnx = model_path.replace('.pt', '.onnx')
        if os.path.exists(default_onnx) and default_onnx != output_path:
            os.rename(default_onnx, output_path)
            onnx_path = output_path
        else:
            onnx_path = default_onnx
            
        print(f"ONNX 변환 완료: {onnx_path}")
        return onnx_path
    else:
        print("ONNX 변환 실패")
        return None

def test_onnx_model(onnx_path):
    """
    변환된 ONNX 모델 테스트
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        # ONNX 세션 생성
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 입력/출력 정보 확인
        input_info = session.get_inputs()[0]
        print(f"입력 형태: {input_info.shape}")
        print(f"입력 타입: {input_info.type}")
        
        print("출력 정보:")
        for i, output in enumerate(session.get_outputs()):
            print(f"  출력 {i}: {output.name}, 형태: {output.shape}")
        
        # 더미 입력으로 테스트
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(None, {input_info.name: dummy_input})
        print(f"테스트 성공! 출력 개수: {len(outputs)}")
        
        return True
    except Exception as e:
        print(f"테스트 실패: {e}")
        return False

if __name__ == "__main__":
    model_path = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\4.Model Experimental Data\Test_7_onnx\yolov8_seg_onnx_ver1.pt"
    output_path = "model_seg.onnx"
    
    # ONNX 변환
    onnx_path = convert_seg_to_onnx(model_path, output_path)
    
    # 테스트
    if onnx_path:
        test_onnx_model(onnx_path)