# ONNX 변환 성공 - ver3의 핵심 차이점

## 🚀 성공 요인: To_ONNX_ver3.py가 성공한 이유

### 1. **정확한 이미지 해상도 설정**

#### ❌ ver1, ver2: 부정확한 크기
```python
# ver1 - 단순한 크기만 지정
success = model.export(format="onnx", imgsz=img_size)

# ver2 - 여전히 정사각형 크기
success = model.export(format="onnx", imgsz=img_size, opset=11)
```

#### ✅ ver3: 실제 사용하는 정확한 크기
```python
# 실제 전처리와 정확히 일치하는 크기 설정
IMG_WIDTH = 640   # 리사이즈된 실제 width
IMG_HEIGHT = 360  # 리사이즈된 실제 height
PADDED_WIDTH = 640   # 패딩 후 width
PADDED_HEIGHT = 384  # 패딩 후 height

# ONNX 변환 시 패딩된 크기로 변환
success = pt_model.export(
    format="onnx",
    imgsz=[PADDED_HEIGHT, PADDED_WIDTH],  # [384, 640] - 실제 입력 크기
    opset=12,
    nms=False  # NMS 제외가 핵심!
)
```

---

### 2. **NMS 제외 설정 (가장 중요!)**

#### ❌ ver1, ver2: NMS 포함으로 변환
```python
# NMS가 기본적으로 포함됨 - 순환참조 문제 발생
success = model.export(format="onnx", ...)
```

#### ✅ ver3: NMS 제외
```python
success = pt_model.export(
    format="onnx",
    imgsz=[PADDED_HEIGHT, PADDED_WIDTH],
    opset=12,
    nms=False,          # 🔥 핵심! NMS 제외
    agnostic_nms=False,
    device='cpu'
)
```

**왜 중요한가?**
- NMS가 포함되면 ONNX 변환 시 그래프 복잡도 증가
- 순환참조 및 동적 연산 문제 발생
- Raw 출력을 받아서 별도로 후처리하는 것이 안정적

---

### 3. **체계적인 검증 시스템**

#### ❌ ver1, ver2: 기본적인 테스트
```python
# ver1 - 단순 더미 입력 테스트
def test_onnx_model(onnx_path):
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {input_info.name: dummy_input})

# ver2 - 여러 변환 시도하지만 검증 부족
```

#### ✅ ver3: 5단계 체계적 검증
```python
# 1단계: PT 모델 원본 테스트
pt_results = pt_model(test_image_path, imgsz=[PADDED_HEIGHT, PADDED_WIDTH])

# 2단계: ONNX 변환

# 3단계: ONNX 모델 구조 검증
onnx.checker.check_model(onnx_model)

# 4단계: 동일한 전처리로 ONNX 추론 테스트
# 실제 이미지 → 리사이즈 → 패딩 → 정규화 → ONNX 추론

# 5단계: 결과 비교 (추론 시간, 탐지 수 등)
```

---

### 4. **전처리 일관성 확보**

#### ❌ ver1, ver2: 전처리 불일치
```python
# 변환 시와 실제 사용 시 전처리가 다름
# 크기나 패딩 방식이 일치하지 않음
```

#### ✅ ver3: 완벽한 전처리 일치
```python
# ONNX 변환 시 크기
imgsz=[PADDED_HEIGHT, PADDED_WIDTH]  # [384, 640]

# 실제 사용 시 전처리 (detecting_ver3.py와 동일)
def preprocess(self, image):
    # 1. 비율 유지 리사이즈
    scale = min(self.input_width / original_w, self.input_height / original_h)
    
    # 2. 패딩 추가 (중앙 정렬)
    padded_image = cv2.copyMakeBorder(
        resized_image, top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # 3. 정규화 및 차원 변환
    input_tensor = padded_image.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
```

---

### 5. **실제 이미지로 검증**

#### ❌ ver1, ver2: 더미 데이터만 테스트
```python
# 랜덤 노이즈로만 테스트
dummy_input = np.random.randn(1, 3, 640, 640)
```

#### ✅ ver3: 실제 이미지로 전체 파이프라인 테스트
```python
# 실제 테스트 이미지 사용
img = cv2.imread(test_image_path)

# 전체 전처리 파이프라인 실행
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (new_w, new_h))
img_padded = cv2.copyMakeBorder(...)  # 실제 패딩
img_tensor = img_padded.astype(np.float32) / 255.0

# ONNX 추론 및 후처리까지 확인
onnx_outputs = session.run([output_info.name], {input_info.name: img_batch})
```

---

### 6. **후처리 로직 최적화**

#### ❌ ver1, ver2: 후처리 문제
```python
# 순환참조나 변환 문제로 제대로 된 후처리 불가
```

#### ✅ ver3: Raw 출력 + 커스텀 후처리
```python
# detecting_ver3.py의 정교한 후처리
def postprocess(self, output, original_width, original_height, scale, padding):
    pred = output.squeeze(0)  # (11, 5040) → 배치 차원 제거
    
    # YOLO 출력 파싱
    boxes_raw = pred[0:4, :].T      # (5040, 4) - cx, cy, w, h
    objectness_raw = pred[4, :]     # (5040,) - 객체성 점수
    class_scores_raw = pred[5:11, :].T  # (5040, 6) - 클래스 점수
    
    # 시그모이드 적용 (중요!)
    objectness = self.sigmoid(objectness_raw)
    class_scores = self.sigmoid(class_scores_raw)
    
    # 최종 신뢰도 = 객체성 × 클래스 점수
    scores = objectness[:, np.newaxis] * class_scores
    
    # NMS 적용
    indices = cv2.dnn.NMSBoxes(...)
```

---

## 📊 성능 비교: ver3 vs 이전 버전들

| 항목 | ver1 | ver2 | ver3 |
|------|------|------|------|
| **변환 성공** | ❌ 실패 | ❌ 실패 | ✅ 성공 |
| **NMS 처리** | 포함 (문제) | 포함 (문제) | 제외 (성공) |
| **해상도 설정** | 부정확 | 부정확 | 정확 (640x384) |
| **전처리 일치** | ❌ 불일치 | ❌ 불일치 | ✅ 완벽 일치 |
| **검증 단계** | 1단계 | 4단계 (실패) | 5단계 (성공) |
| **실제 이미지 테스트** | ❌ 더미만 | ❌ 더미만 | ✅ 실제 이미지 |

---

## 🔑 ver3 성공의 핵심 요소

### 1. **NMS=False 설정**
```python
nms=False,          # 🔥 가장 중요한 설정!
agnostic_nms=False,
```

### 2. **정확한 해상도 매칭**
```python
imgsz=[PADDED_HEIGHT, PADDED_WIDTH],  # [384, 640] - 실제 사용 크기
```

### 3. **전처리-추론-후처리 일관성**
- 변환 시 크기 = 실제 사용 시 크기
- 패딩 방식 동일
- 정규화 방식 동일

### 4. **Raw 출력 + 커스텀 후처리**
- ONNX에서는 Raw 출력만 받음
- Python에서 시그모이드, NMS 등 후처리 수행
- 더 안정적이고 제어 가능

---

## 💡 결론

**ver3가 성공한 이유:**
1. **NMS 제외**로 그래프 복잡도 감소
2. **정확한 해상도** 설정으로 입출력 일치
3. **체계적인 검증**으로 문제 조기 발견
4. **실제 이미지**로 전체 파이프라인 테스트
5. **Raw 출력 + 커스텀 후처리**로 안정성 확보

ver1, ver2는 NMS 포함과 부정확한 해상도 설정으로 실패했지만, ver3는 이러한 문제들을 체계적으로 해결하여 성공했습니다.