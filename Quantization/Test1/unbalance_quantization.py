"""
ONNX 모델 양자화 시스템 (버전 호환성 수정)
프루닝된 YOLO 모델을 INT8로 양자화하여 크기 및 속도 최적화

필요한 라이브러리:
pip install onnxruntime onnx onnxruntime-tools
"""

import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import quantize_static, CalibrationDataReader
import time
import glob
from pathlib import Path
import shutil

class YOLOCalibrationDataReader(CalibrationDataReader):
    """YOLO 모델용 Calibration 데이터 리더"""
    
    def __init__(self, calibration_data, input_name):
        self.data = calibration_data
        self.input_name = input_name
        self.current_index = 0
    
    def get_next(self):
        if self.current_index < len(self.data):
            input_data = {self.input_name: self.data[self.current_index]}
            self.current_index += 1
            return input_data
        else:
            return None
    
    def rewind(self):
        self.current_index = 0

class ONNXQuantizer:
    def __init__(self, original_model_path, output_dir="quantized_models"):
        """
        ONNX 모델 양자화 시스템 초기화
        
        Args:
            original_model_path: 원본 ONNX 모델 경로
            output_dir: 양자화된 모델 저장 디렉토리
        """
        self.original_model_path = original_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 입력 이름 확인
        session = ort.InferenceSession(original_model_path)
        self.input_name = session.get_inputs()[0].name
        
        # 모델 정보 확인
        self.verify_original_model()
        
        print(f"🚀 ONNX 양자화 시스템 초기화")
        print(f"   원본 모델: {original_model_path}")
        print(f"   입력 이름: {self.input_name}")
        print(f"   출력 디렉토리: {output_dir}")
    
    def verify_original_model(self):
        """원본 모델 검증"""
        if not os.path.exists(self.original_model_path):
            raise FileNotFoundError(f"원본 모델을 찾을 수 없습니다: {self.original_model_path}")
        
        # 모델 크기 확인
        size_mb = os.path.getsize(self.original_model_path) / (1024 * 1024)
        print(f"📊 원본 모델 크기: {size_mb:.2f} MB")
        
        # ONNX 모델 로드 테스트
        try:
            model = onnx.load(self.original_model_path)
            onnx.checker.check_model(model)
            print("✅ 원본 모델 검증 완료")
        except Exception as e:
            raise RuntimeError(f"원본 모델 검증 실패: {e}")
    
    def dynamic_quantization(self):
        """
        동적 양자화 수행 (INT8) - 버전 호환성 개선
        Calibration 데이터 불필요, 빠르고 간단
        """
        print(f"\n🔄 동적 양자화 시작 (INT8)")
        print("=" * 50)
        
        output_path = self.output_dir / "model_dynamic_int8.onnx"
        
        try:
            start_time = time.perf_counter()
            
            # 최소한의 파라미터로 동적 양자화 수행
            quantize_dynamic(
                model_input=str(self.original_model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            # 결과 확인
            if output_path.exists():
                original_size = os.path.getsize(self.original_model_path) / (1024 * 1024)
                quantized_size = os.path.getsize(output_path) / (1024 * 1024)
                compression_ratio = original_size / quantized_size
                
                print(f"✅ 동적 양자화 완료!")
                print(f"   소요 시간: {elapsed_time:.2f}초")
                print(f"   원본 크기: {original_size:.2f} MB")
                print(f"   양자화 후: {quantized_size:.2f} MB")
                print(f"   압축 비율: {compression_ratio:.2f}배")
                print(f"   크기 감소: {(1-quantized_size/original_size)*100:.1f}%")
                print(f"   저장 경로: {output_path}")
                
                return str(output_path)
            else:
                raise RuntimeError("양자화된 모델 파일이 생성되지 않았습니다.")
                
        except Exception as e:
            print(f"❌ 동적 양자화 실패: {e}")
            return None
    
    def create_calibration_dataset(self, image_dir, num_images=100, input_size=(320, 192)):
        """
        Calibration 데이터셋 생성 (정적 양자화용)
        
        Args:
            image_dir: 이미지 디렉토리 경로
            num_images: 사용할 이미지 수
            input_size: 모델 입력 크기 (width, height)
        """
        print(f"\n📁 Calibration 데이터셋 생성")
        print(f"   이미지 디렉토리: {image_dir}")
        print(f"   사용할 이미지 수: {num_images}")
        print(f"   입력 크기: {input_size}")
        
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(image_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_dir}")
        
        # 필요한 수만큼 선택
        selected_images = image_files[:min(num_images, len(image_files))]
        print(f"   발견된 이미지: {len(image_files)}개")
        print(f"   선택된 이미지: {len(selected_images)}개")
        
        # 전처리된 데이터 생성
        calibration_data = []
        input_width, input_height = input_size
        
        print(f"   전처리 중...")
        for i, img_path in enumerate(selected_images):
            try:
                # 이미지 로드 및 전처리 (detecting_ver3.py와 동일)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # BGR → RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_h, original_w = img_rgb.shape[:2]
                
                # 비율 유지 리사이즈
                scale = min(input_width / original_w, input_height / original_h)
                new_w, new_h = int(original_w * scale), int(original_h * scale)
                resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 패딩
                top_pad = (input_height - new_h) // 2
                bottom_pad = input_height - new_h - top_pad
                left_pad = (input_width - new_w) // 2
                right_pad = input_width - new_w - left_pad
                
                padded = cv2.copyMakeBorder(
                    resized, top_pad, bottom_pad, left_pad, right_pad,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
                
                # 정규화 및 텐서 변환
                tensor = padded.astype(np.float32) / 255.0
                tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
                tensor = np.expand_dims(tensor, axis=0)   # 배치 차원
                
                calibration_data.append(tensor)
                
                if (i + 1) % 20 == 0:
                    print(f"      진행률: {i+1}/{len(selected_images)}")
                    
            except Exception as e:
                print(f"      이미지 처리 실패 ({img_path}): {e}")
                continue
        
        print(f"   ✅ Calibration 데이터 준비 완료: {len(calibration_data)}개")
        return calibration_data
    
    def static_quantization(self, image_dir, num_images=100, input_size=(320, 192)):
        """
        정적 양자화 수행 (INT8) - 임시 파일 처리 개선
        Calibration 데이터 필요, 더 정확한 양자화
        """
        print(f"\n🔄 정적 양자화 시작 (INT8)")
        print("=" * 50)
        
        output_path = self.output_dir / "model_static_int8.onnx"
        
        try:
            # Calibration 데이터 생성
            calibration_data = self.create_calibration_dataset(
                image_dir, num_images, input_size
            )
            
            if len(calibration_data) == 0:
                raise RuntimeError("Calibration 데이터가 생성되지 않았습니다.")
            
            # Calibration 데이터 리더 생성
            data_reader = YOLOCalibrationDataReader(calibration_data, self.input_name)
            
            print(f"   정적 양자화 수행 중...")
            start_time = time.perf_counter()
            
            # 임시 파일 경로 (안전한 이름으로)
            temp_model_path = self.output_dir / "temp_model_for_quantization.onnx"
            shutil.copy2(self.original_model_path, temp_model_path)
            
            # 최소한의 파라미터로 정적 양자화 수행
            quantize_static(
                model_input=str(temp_model_path),
                model_output=str(output_path),
                calibration_data_reader=data_reader
            )
            
            # 임시 파일 제거
            if temp_model_path.exists():
                temp_model_path.unlink()
            
            elapsed_time = time.perf_counter() - start_time
            
            # 결과 확인
            if output_path.exists():
                original_size = os.path.getsize(self.original_model_path) / (1024 * 1024)
                quantized_size = os.path.getsize(output_path) / (1024 * 1024)
                compression_ratio = original_size / quantized_size
                
                print(f"✅ 정적 양자화 완료!")
                print(f"   소요 시간: {elapsed_time:.2f}초")
                print(f"   사용된 이미지: {len(calibration_data)}개")
                print(f"   원본 크기: {original_size:.2f} MB")
                print(f"   양자화 후: {quantized_size:.2f} MB")
                print(f"   압축 비율: {compression_ratio:.2f}배")
                print(f"   크기 감소: {(1-quantized_size/original_size)*100:.1f}%")
                print(f"   저장 경로: {output_path}")
                
                return str(output_path)
            else:
                raise RuntimeError("양자화된 모델 파일이 생성되지 않았습니다.")
                
        except Exception as e:
            print(f"❌ 정적 양자화 실패: {e}")
            return None
    
    def benchmark_models(self, test_image_path, num_runs=50):
        """
        원본 vs 양자화 모델 성능 비교
        """
        print(f"\n📊 모델 성능 벤치마크")
        print("=" * 50)
        
        results = {}
        
        # 테스트할 모델들
        models_to_test = [
            ("원본", self.original_model_path),
        ]
        
        # 양자화된 모델들 추가
        dynamic_path = self.output_dir / "model_dynamic_int8.onnx"
        if dynamic_path.exists():
            models_to_test.append(("동적 INT8", str(dynamic_path)))
        
        static_path = self.output_dir / "model_static_int8.onnx"
        if static_path.exists():
            models_to_test.append(("정적 INT8", str(static_path)))
        
        # 각 모델 테스트
        for model_name, model_path in models_to_test:
            print(f"\n🔍 {model_name} 모델 테스트:")
            
            try:
                # 세션 생성
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                # 테스트 이미지 전처리
                img = cv2.imread(test_image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_h, original_w = img_rgb.shape[:2]
                
                # 전처리 (detecting_ver3.py와 동일)
                scale = min(320 / original_w, 192 / original_h)
                new_w, new_h = int(original_w * scale), int(original_h * scale)
                resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                top_pad = (192 - new_h) // 2
                bottom_pad = 192 - new_h - top_pad
                left_pad = (320 - new_w) // 2
                right_pad = 320 - new_w - left_pad
                
                padded = cv2.copyMakeBorder(
                    resized, top_pad, bottom_pad, left_pad, right_pad,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
                
                tensor = padded.astype(np.float32) / 255.0
                tensor = np.transpose(tensor, (2, 0, 1))
                tensor = np.expand_dims(tensor, axis=0)
                
                # 워밍업
                for _ in range(5):
                    _ = session.run([output_name], {input_name: tensor})
                
                # 벤치마크
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    output = session.run([output_name], {input_name: tensor})
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / num_runs * 1000
                fps = 1000 / avg_time
                
                # 모델 크기
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                results[model_name] = {
                    'time_ms': avg_time,
                    'fps': fps,
                    'size_mb': size_mb,
                    'output': output[0]
                }
                
                print(f"   ⏱️ 평균 추론 시간: {avg_time:.2f}ms")
                print(f"   🚀 FPS: {fps:.1f}")
                print(f"   💾 모델 크기: {size_mb:.2f} MB")
                
            except Exception as e:
                print(f"   ❌ 테스트 실패: {e}")
                results[model_name] = None
        
        # 비교 결과 출력
        self.print_comparison(results)
        return results
    
    def print_comparison(self, results):
        """성능 비교 결과 출력"""
        print(f"\n📈 성능 비교 결과")
        print("=" * 70)
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) < 2:
            print("비교할 모델이 부족합니다.")
            return
        
        # 기준 모델 (원본)
        baseline = valid_results.get("원본")
        if baseline is None:
            baseline = list(valid_results.values())[0]
            baseline_name = list(valid_results.keys())[0]
        else:
            baseline_name = "원본"
        
        print(f"기준 모델: {baseline_name}")
        print("-" * 70)
        print(f"{'모델':<12} {'크기(MB)':<10} {'시간(ms)':<10} {'FPS':<8} {'속도 향상':<10} {'크기 감소'}")
        print("-" * 70)
        
        for name, result in valid_results.items():
            if result is None:
                continue
            
            speed_improvement = baseline['time_ms'] / result['time_ms']
            size_reduction = (1 - result['size_mb'] / baseline['size_mb']) * 100
            
            print(f"{name:<12} {result['size_mb']:<10.2f} {result['time_ms']:<10.2f} "
                  f"{result['fps']:<8.1f} {speed_improvement:<10.2f}x {size_reduction:<6.1f}%")


def main():
    """
    메인 실행 함수
    """
    
    # ========================================
    # 🔧 설정 수정
    # ========================================
    
    ORIGINAL_MODEL_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\6.ONNX\To_ONNX_ver6(프루닝모델_Test7)\yolov8_custom_fixed_test7_pruned.onnx"
    OUTPUT_DIR = "quantized_models"
    
    # Calibration 이미지 디렉토리 (정적 양자화용)
    CALIBRATION_IMAGE_DIR = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106"
    NUM_CALIBRATION_IMAGES = 100  # 사용할 이미지 수
    
    # 테스트 이미지 (성능 비교용)
    TEST_IMAGE_PATH = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam1\20231214062106\frame_000000.jpg"
    
    # 모델 입력 크기
    INPUT_SIZE = (320, 192)  # (width, height)
    
    # ========================================
    
    print("🚀 ONNX 모델 양자화 시작")
    print("=" * 60)
    
    try:
        # 양자화 시스템 초기화
        quantizer = ONNXQuantizer(ORIGINAL_MODEL_PATH, OUTPUT_DIR)
        
        # 1단계: 동적 양자화 (빠르고 간단)
        print(f"\n1️⃣ 동적 양자화 수행")
        dynamic_model = quantizer.dynamic_quantization()
        
        if dynamic_model:
            print(f"✅ 동적 양자화 성공: {dynamic_model}")
        else:
            print(f"❌ 동적 양자화 실패")
        
        # 2단계: 정적 양자화 (더 정확, calibration 필요)
        print(f"\n2️⃣ 정적 양자화 수행")
        
        if os.path.exists(CALIBRATION_IMAGE_DIR):
            static_model = quantizer.static_quantization(
                image_dir=CALIBRATION_IMAGE_DIR,
                num_images=NUM_CALIBRATION_IMAGES,
                input_size=INPUT_SIZE
            )
            
            if static_model:
                print(f"✅ 정적 양자화 성공: {static_model}")
            else:
                print(f"❌ 정적 양자화 실패")
        else:
            print(f"❌ Calibration 이미지 디렉토리를 찾을 수 없습니다: {CALIBRATION_IMAGE_DIR}")
            print(f"   정적 양자화를 건너뛰고 동적 양자화만 수행합니다.")
        
        # 3단계: 성능 비교
        print(f"\n3️⃣ 성능 벤치마크")
        
        if os.path.exists(TEST_IMAGE_PATH):
            results = quantizer.benchmark_models(TEST_IMAGE_PATH, num_runs=50)
            
            if results:
                print(f"\n🎉 양자화 완료!")
                print(f"📂 생성된 모델들:")
                
                # 생성된 파일들 확인
                output_dir = Path(OUTPUT_DIR)
                for model_file in output_dir.glob("*.onnx"):
                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    print(f"   {model_file.name}: {size_mb:.2f} MB")
                
                print(f"\n💡 사용 권장사항:")
                print(f"   - 속도 우선: 동적 양자화 모델 사용")
                print(f"   - 정확도 우선: 정적 양자화 모델 사용")
                print(f"   - detecting_ver3.py에서 모델 경로만 변경하여 테스트")
        else:
            print(f"❌ 테스트 이미지를 찾을 수 없습니다: {TEST_IMAGE_PATH}")
            print(f"   성능 비교를 건너뛰고 양자화만 완료했습니다.")
    
    except Exception as e:
        print(f"❌ 양자화 과정에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 양자화 프로세스 완료")


if __name__ == "__main__":
    main()