#!/usr/bin/env python3
"""
수정된 YOLO 모델 ONNX 변환 및 검증 코드
올바른 해상도와 전처리로 변환
"""

import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import time

def convert_to_onnx_fixed(pt_model_path, test_image_path, output_onnx_path=None):
    """
    올바른 설정으로 YOLO 모델을 ONNX로 변환하고 검증
    """
    
    print("🔄 수정된 YOLO → ONNX 변환 시작")
    print("=" * 60)
    
    # 1. 파일 존재 확인
    if not os.path.exists(pt_model_path):
        print(f"❌ PT 모델 파일을 찾을 수 없습니다: {pt_model_path}")
        return False
        
    if not os.path.exists(test_image_path):
        print(f"❌ 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
        return False
    
    # 2. 출력 경로 설정
    if output_onnx_path is None:
        base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
        output_onnx_path = f"{base_name}_fixed.onnx"
    
    # 3. 올바른 이미지 크기 설정 (실제 전처리와 일치)
    IMG_WIDTH = 320   # 리사이즈된 실제 width
    IMG_HEIGHT = 180  # 리사이즈된 실제 height
    PADDED_WIDTH = 320   # 패딩 후 width
    PADDED_HEIGHT = 192  # 패딩 후 height
    
    print(f"📂 입력 모델: {pt_model_path}")
    print(f"📂 테스트 이미지: {test_image_path}")
    print(f"📂 출력 ONNX: {output_onnx_path}")
    print(f"📏 리사이즈 크기: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📏 패딩 후 크기: {PADDED_WIDTH}x{PADDED_HEIGHT}")
    
    # 4. 원본 PT 모델 로드 및 테스트
    print("\n🔵 1단계: 원본 PT 모델 테스트")
    print("-" * 40)
    
    try:
        pt_model = YOLO(pt_model_path)
        print(f"✅ PT 모델 로드 성공")
        print(f"   모델 태스크: {pt_model.task}")
        print(f"   클래스 수: {len(pt_model.names)}")
        print(f"   클래스 이름: {list(pt_model.names.values())}")
        
        # 원본 모델로 추론 (패딩된 크기로)
        print(f"\n🔍 원본 모델 추론 테스트... (크기: {PADDED_WIDTH}x{PADDED_HEIGHT})")
        start_time = time.perf_counter()
        pt_results = pt_model(test_image_path, imgsz=[PADDED_HEIGHT, PADDED_WIDTH], verbose=False)
        pt_inference_time = (time.perf_counter() - start_time) * 1000
        
        pt_result = pt_results[0]
        pt_detections = len(pt_result.boxes) if pt_result.boxes is not None else 0
        
        print(f"   ⏱️ 추론 시간: {pt_inference_time:.2f}ms")
        print(f"   📦 탐지된 객체: {pt_detections}개")
        
        if pt_detections > 0:
            print("   📋 탐지 결과:")
            for i, box in enumerate(pt_result.boxes):
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                cls_name = pt_model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                print(f"      {i+1}. 클래스: {cls_name}({cls_id}), 신뢰도: {conf:.3f}, 박스: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
    except Exception as e:
        print(f"❌ PT 모델 테스트 실패: {e}")
        return False
    
    # 5. ONNX 변환 (패딩된 크기로)
    print("\n🟡 2단계: ONNX 변환")
    print("-" * 40)
    
    try:
        print(f"📦 ONNX 변환 시작... (해상도: {PADDED_WIDTH}x{PADDED_HEIGHT})")
        
        # ONNX 변환 수행 - 패딩된 크기로 변환
        success = pt_model.export(
            format="onnx",
            imgsz=[PADDED_HEIGHT, PADDED_WIDTH],  # [height, width] 순서 - 패딩된 크기
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            int8=False,
            nms=False,          # NMS 제외 (중요!)
            agnostic_nms=False,
            device='cpu',
            verbose=True
        )
        
        # 변환된 파일 찾기
        auto_generated_path = pt_model_path.replace('.pt', '.onnx')
        
        if os.path.exists(auto_generated_path):
            if auto_generated_path != output_onnx_path:
                import shutil
                shutil.move(auto_generated_path, output_onnx_path)
            print(f"✅ ONNX 변환 완료: {output_onnx_path}")
        else:
            print(f"❌ ONNX 파일을 찾을 수 없습니다: {auto_generated_path}")
            return False
            
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return False
    
    # 6. ONNX 모델 검증
    print("\n🟢 3단계: ONNX 모델 검증")
    print("-" * 40)
    
    try:
        # ONNX 모델 로드
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 모델 구조 검증 완료")
        
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(output_onnx_path)
        
        # 입력/출력 정보
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"📊 ONNX 모델 정보:")
        print(f"   입력 이름: {input_info.name}")
        print(f"   입력 형태: {input_info.shape}")
        print(f"   입력 타입: {input_info.type}")
        print(f"   출력 이름: {output_info.name}")
        print(f"   출력 형태: {output_info.shape}")
        print(f"   출력 타입: {output_info.type}")
        
        file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
        print(f"   파일 크기: {file_size_mb:.1f} MB")
        
        # 예상 형태와 비교
        expected_input = f"[1, 3, {PADDED_HEIGHT}, {PADDED_WIDTH}]"
        actual_input = str(input_info.shape)
        print(f"   예상 입력: {expected_input}")
        print(f"   실제 입력: {actual_input}")
        print(f"   입력 일치: {'✅' if actual_input == expected_input else '❌'}")
        
    except Exception as e:
        print(f"❌ ONNX 모델 검증 실패: {e}")
        return False
    
    # 7. 동일한 전처리로 ONNX 추론 테스트
    print("\n🔴 4단계: ONNX 추론 테스트 (동일한 전처리)")
    print("-" * 40)
    
    try:
        # 이미지 전처리 (detector 클래스와 동일하게)
        print("🖼️ 이미지 전처리...")
        img = cv2.imread(test_image_path)
        original_shape = img.shape[:2]  # (height, width)
        print(f"   원본 이미지 크기: {original_shape[1]}x{original_shape[0]} (width x height)")
        
        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 리사이즈 (비율 유지)
        scale = min(PADDED_WIDTH / original_shape[1], PADDED_HEIGHT / original_shape[0])
        new_w, new_h = int(original_shape[1] * scale), int(original_shape[0] * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"   리사이즈 후: {new_w}x{new_h}, 스케일: {scale:.3f}")
        
        # 패딩 추가
        top_pad = (PADDED_HEIGHT - new_h) // 2
        bottom_pad = PADDED_HEIGHT - new_h - top_pad
        left_pad = (PADDED_WIDTH - new_w) // 2
        right_pad = PADDED_WIDTH - new_w - left_pad
        
        img_padded = cv2.copyMakeBorder(
            img_resized,
            top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        print(f"   패딩: 상{top_pad} 하{bottom_pad} 좌{left_pad} 우{right_pad}")
        print(f"   패딩 후: {img_padded.shape}")
        
        # 정규화 및 텐서 변환
        img_tensor = img_padded.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC → CWH (ONNX 형태에 맞춤)
        img_batch = np.expand_dims(img_tensor, axis=0)
        
        print(f"   최종 입력 텐서 형태: {img_batch.shape}")
        
        # ONNX 추론
        print("🔍 ONNX 추론 수행...")
        start_time = time.perf_counter()
        onnx_outputs = session.run([output_info.name], {input_info.name: img_batch})
        onnx_inference_time = (time.perf_counter() - start_time) * 1000
        
        onnx_output = onnx_outputs[0]
        print(f"   ⏱️ ONNX 추론 시간: {onnx_inference_time:.2f}ms")
        print(f"   📤 ONNX 출력 형태: {onnx_output.shape}")
        print(f"   📈 출력 값 범위: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
        
        # 간단한 후처리로 탐지 확인
        pred = onnx_output.squeeze(0).T  # (5040, 11)
        
        # 객체성 점수와 클래스 점수 확인
        boxes = pred[:, 0:4]
        objectness_raw = pred[:, 4]
        class_scores_raw = pred[:, 5:11]
        
        # 시그모이드 적용
        objectness = 1 / (1 + np.exp(-objectness_raw))
        class_scores = 1 / (1 + np.exp(-class_scores_raw))
        
        print(f"   📊 ONNX 출력 분석:")
        print(f"      박스: {boxes.shape}")
        print(f"      객체성 범위: {objectness.min():.3f} ~ {objectness.max():.3f}")
        print(f"      클래스 점수 범위: {class_scores.min():.3f} ~ {class_scores.max():.3f}")
        
        # 최종 점수 계산
        final_scores = objectness[:, np.newaxis] * class_scores
        max_scores = np.max(final_scores, axis=1)
        
        high_conf_count = (max_scores > 0.3).sum()
        print(f"      신뢰도 0.3 이상: {high_conf_count}개")
        print(f"      최고 신뢰도: {max_scores.max():.3f}")
        
    except Exception as e:
        print(f"❌ ONNX 추론 테스트 실패: {e}")
        return False
    
    # 8. 결과 비교
    print("\n📊 5단계: 결과 비교")
    print("-" * 40)
    
    print(f"🏁 변환 및 검증 완료!")
    print(f"   원본 PT 추론 시간: {pt_inference_time:.2f}ms")
    print(f"   ONNX 추론 시간: {onnx_inference_time:.2f}ms")
    
    if onnx_inference_time > 0 and pt_inference_time > 0:
        speed_ratio = pt_inference_time / onnx_inference_time
        if speed_ratio > 1:
            print(f"   🚀 ONNX가 {speed_ratio:.2f}배 빠름")
        else:
            print(f"   🐌 ONNX가 {1/speed_ratio:.2f}배 느림")
    
    print(f"   원본 PT 탐지: {pt_detections}개")
    print(f"   ONNX 고신뢰도 후보: {high_conf_count}개")
    print(f"   ONNX 파일: {output_onnx_path}")
    
    return True

def main():
    """
    메인 실행 함수
    """
    
    # ========================================
    # 🔧 설정 수정
    # ========================================
    
    PT_MODEL_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver5(320x180)\best.pt"
    TEST_IMAGE_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231130081944_min\frame_000000.jpg"
    OUTPUT_ONNX_PATH = "yolov8_custom_fixed.onnx"
    
    # ========================================
    
    # 변환 및 검증 실행
    success = convert_to_onnx_fixed(
        pt_model_path=PT_MODEL_PATH,
        test_image_path=TEST_IMAGE_PATH,
        output_onnx_path=OUTPUT_ONNX_PATH
    )
    
    if success:
        print("\n🎉 성공! 다음 단계:")
        print("1. 새로 생성된 ONNX 파일로 detector 테스트")
        print("2. INPUT_SIZE를 (640, 384)로 유지")
        print("3. 정확도 향상 확인")
    else:
        print("\n❌ 실패! 오류를 확인하고 다시 시도하세요.")

if __name__ == "__main__":
    main()