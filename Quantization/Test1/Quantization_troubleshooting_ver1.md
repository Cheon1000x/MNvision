# ONNX 모델 양자화 트러블슈팅 리포트

## 📋 프로젝트 개요
- **목표**: 프루닝된 YOLOv8 ONNX 모델의 INT8 양자화
- **원본 모델**: yolov8_custom_fixed_test7_pruned.onnx (10.58MB)
- **최종 결과**: 정적 양자화 성공, 동적 양자화 실패

## 📊 결과 요약

### 성능 비교 결과

| 모델 | 크기(MB) | 추론시간(ms) | FPS | 크기 감소 | 속도 변화 |
|------|----------|-------------|-----|----------|----------|
| 원본 | 10.58 | 7.88 | 127.0 | 0% | 기준 |
| 정적 INT8 | 2.90 | 16.48 | 60.7 | -72.6% | -52.0% |

### 양자화 성과

| 방법 | 결과 | 압축비율 | 소요시간 | 상태 |
|------|------|----------|----------|------|
| 동적 양자화 | 실패 | - | - | ❌ |
| 정적 양자화 | 성공 | 3.65배 | 2.53초 | ✅ |

---

## ❌ 발생한 문제들

### 1차 문제: optimize_model 파라미터 오류
**오류 내용:**
```
quantize_dynamic() got an unexpected keyword argument 'optimize_model'
```

**원인:**
- 사용 중인 onnxruntime 버전에서 `optimize_model` 파라미터 미지원
- 코드에서 버전 호환성을 고려하지 않음

**해결 방법:**
```python
# 기존 코드
quantize_dynamic(
    model_input=str(self.original_model_path),
    model_output=str(output_path),
    weight_type=QuantType.QInt8,
    optimize_model=True,  # 이 부분 제거
    per_channel=False,
    reduce_range=False,
    extra_options={}
)

# 수정된 코드
quantize_dynamic(
    model_input=str(self.original_model_path),
    model_output=str(output_path),
    weight_type=QuantType.QInt8  # 최소한의 파라미터만 사용
)
```

### 2차 문제: inferred.onnx 파일 오류
**오류 내용:**
```
[Errno 2] No such file or directory: 'yolov8_custom_fixed_test7_pruned-inferred.onnx'
```

**원인:**
- onnxruntime 양자화 과정에서 자동 생성되는 임시 파일을 찾지 못함
- 원본 파일명에 특수문자나 긴 경로로 인한 문제

**해결 방법:**
```python
# 임시 파일 경로 (안전한 이름으로)
temp_model_path = self.output_dir / "temp_model_for_quantization.onnx"
shutil.copy2(self.original_model_path, temp_model_path)

# 정적 양자화 수행
quantize_static(
    model_input=str(temp_model_path),  # 임시 파일 사용
    model_output=str(output_path),
    calibration_data_reader=data_reader
)

# 임시 파일 제거
if temp_model_path.exists():
    temp_model_path.unlink()
```

### 3차 문제: 동적 양자화 지속 실패
**오류 내용:**
```
[Errno 2] No such file or directory: 'yolov8_custom_fixed_test7_pruned-inferred.onnx'
```

**원인:**
- 동적 양자화에서도 동일한 임시 파일 문제 발생
- 현재 onnxruntime 버전과 YOLOv8 모델 구조 간 호환성 이슈

**현재 상태:**
- 해결 시도했으나 여전히 실패
- 정적 양자화는 임시 파일 복사 방식으로 해결됨

---

## ✅ 해결 성과

### 정적 양자화 성공
**구현 사항:**
- Calibration 데이터셋 100개 이미지로 생성
- 전처리 파이프라인을 detecting_ver3.py와 동일하게 구성
- 임시 파일 복사 방식으로 경로 문제 해결

**결과:**
- 원본: 10.58MB → 양자화: 2.90MB
- 압축 비율: 3.65배
- 크기 감소: 72.6%
- 처리 시간: 2.53초

### Calibration 데이터 처리
**처리 과정:**
1. 이미지 디렉토리에서 600개 이미지 발견
2. 100개 이미지 선택하여 전처리
3. 320x192 해상도로 리사이즈 및 패딩
4. 정규화 및 텐서 변환 완료

**전처리 단계:**
- BGR → RGB 변환
- 비율 유지 리사이즈
- 중앙 정렬 패딩 (114, 114, 114)
- [0, 1] 정규화
- HWC → CHW 차원 변환

---

## 📊 성능 분석

### 모델 크기 최적화
- **목표 달성**: 72.6% 크기 감소로 목표 달성
- **압축 효율**: 3.65배 압축으로 우수한 결과
- **메모리 효율**: 7.68MB 메모리 사용량 감소

### 추론 속도 변화
- **속도 저하**: 127.0 FPS → 60.7 FPS (52% 감소)
- **원인**: CPU에서 INT8 연산이 FP32보다 느릴 수 있음
- **실용성**: 60.7 FPS는 실시간 처리에 충분함

### 트레이드오프 평가
- **장점**: 모델 크기 대폭 감소, 메모리 효율성 향상
- **단점**: 추론 속도 저하
- **적용 분야**: 배포 환경에서 메모리 제약이 중요한 경우 유리

---

## 💡 기술적 인사이트

### 양자화 방법별 특성
1. **동적 양자화**: Calibration 불필요하지만 현재 모델에서 호환성 문제
2. **정적 양자화**: Calibration 필요하지만 더 안정적이고 높은 압축률

### onnxruntime 버전 이슈
- 사용 중인 환경에서 일부 파라미터 미지원
- 최소한의 필수 파라미터만 사용하는 것이 안전함
- 임시 파일 처리 방식이 모델에 따라 다름

### YOLOv8 모델 특성
- 프루닝된 모델도 양자화 적용 가능
- 복잡한 구조로 인해 일부 양자화 방법에서 호환성 문제 발생
- Calibration 데이터 품질이 최종 성능에 중요한 영향

---

## 🔧 해결 방법론

### 점진적 파라미터 제거
1. 전체 파라미터로 시작 → 오류 발생
2. 비필수 파라미터 제거 → 호환성 확보
3. 최소 파라미터 세트 확정 → 안정성 달성

### 임시 파일 관리
1. 원본 파일 직접 사용 → 경로 문제 발생
2. 임시 파일 복사 방식 → 문제 해결
3. 처리 후 정리 → 디스크 공간 절약

### 전처리 파이프라인 일치
1. 기존 detecting_ver3.py와 동일한 전처리 적용
2. 해상도, 패딩, 정규화 방식 통일
3. Calibration 데이터 품질 확보

---

## 🎯 현재 상태

### 완료된 작업
- [x] 정적 양자화 구현 및 성공
- [x] Calibration 데이터셋 생성
- [x] 성능 벤치마크 완료
- [x] 양자화 모델 생성 (2.90MB)

### 미완료된 작업
- [ ] 동적 양자화 호환성 문제 해결
- [ ] 양자화 모델 정확도 검증
- [ ] GPU 환경에서의 성능 테스트

### 생성된 파일
- `quantized_models/model_static_int8.onnx`: 2.90MB (정적 양자화)
- 원본 모델 대비 72.6% 크기 감소 달성