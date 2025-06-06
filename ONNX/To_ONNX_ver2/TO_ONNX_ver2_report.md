# 🔬 YOLOv8 ONNX 변환 실험 비교 보고서 (v1.0 vs v2.0)

## 📋 **실험 개요**

### 🎯 **목적**
- YOLOv8 세그멘테이션 모델을 ONNX로 변환
- PyTorch 대비 추론 속도 향상
- 동일한 감지 정확도 유지

### 🛠️ **환경**
- **모델**: YOLOv8 세그멘테이션 (10개 클래스)
- **테스트**: 720×1280 이미지
- **하드웨어**: Intel i5-1155G7 CPU

---

## 🚨 **v1.0 실험 결과 (실패)**

### ❌ **주요 문제들**

#### **변환 실패**
```python
# 오류 발생
'Segment' object has no attribute 'detect'
maximum recursion depth exceeded
```

#### **해상도 불일치**
- 학습: 1024px → 변환: 1048px → 추론: 640px
- 일관성 없는 해상도로 인한 혼란

#### **후처리 미구현**
- ONNX 출력 구조 파악 실패
- 47개 채널 중 실제 10개 클래스만 유효한지 모름

### 📊 **v1.0 결과**
| 모델 | 감지 수 | 신뢰도 | 상태 |
|------|---------|---------|------|
| PyTorch | 2개 | 92% | ✅ 정확 |
| ONNX v1.0 | 8400개 | 50% | ❌ 완전 오감지 |

---

## 🔧 **v2.0 개선 사항**

### ✅ **해결한 문제들**

#### **1. 변환 안정성 확보**
```python
# 4가지 변환 방법 시도
conversion_methods = [
    ("기본 변환", basic_conversion),
    ("torch.jit 우회", jit_trace_conversion),  # ← 성공한 방법
    ("직접 export", direct_export_conversion),
    ("안전 모드", safe_mode_conversion)
]
```

#### **2. 순환참조 해결**
```python
# JIT trace로 우회
traced_model = torch.jit.trace(model.model, dummy_input, strict=False)
torch.onnx.export(traced_model, dummy_input, output_path)
```

#### **3. 체계적 후처리 구현**
```python
# ONNX 출력 구조 분석
detections = outputs[0][0].transpose()  # (8400, 47)
boxes = detections[:, :4]               # 박스 좌표
class_probs = detections[:, 4:14]       # 10개 클래스
# 나머지 33개는 세그멘테이션 관련
```

---

## 📊 **성능 비교 결과**

### 🏁 **v2.0 최종 결과**

| 지표 | PyTorch | ONNX v2.0 | 비교 |
|------|---------|-----------|------|
| **추론 시간** | 1,846ms | 96ms | ⚡ **19.2배 빠름** |
| **감지 수** | 2개 (정확) | 138개 (과다) | ❌ **69배 과다** |
| **신뢰도** | 0.925, 0.912 | 0.500~0.512 | ❌ **가짜 신뢰도** |

### 📈 **버전별 성능**

| 항목 | v1.0 | v2.0 | 개선도 |
|------|------|------|--------|
| **변환 성공** | ❌ 실패 | ✅ 성공 | 완전 해결 |
| **추론 속도** | 측정 불가 | 19.2배 향상 | 목표 달성 |
| **감지 정확도** | 0% | ~1.4% | 개선 중 |

---

## 🔍 **v2.0 남은 문제**

### 🚨 **시그모이드 수렴 현상**
```python
# 문제 발생 과정
원시 신뢰도: [0.000, 0.050]        # 매우 낮음
시그모이드 후: [0.500, 0.512]       # 0.5 근처로 수렴
결과: 8400개 그리드 모두 임계값 통과 → 과다 감지
```

### 📊 **과다 감지 패턴**
- **138개 감지** (정답: 2개)
- **대부분 person 클래스** (83%)
- **모든 신뢰도가 0.5 근처**

---

## 💡 **핵심 교훈**

### ✅ **성공 요인**
1. **다중 변환 시도**: 4가지 방법으로 안정성 확보
2. **JIT trace 활용**: 순환참조 문제 우회
3. **체계적 디버깅**: 단계별 출력으로 문제 추적

### ❌ **실패 요인**
1. **후처리 로직**: PyTorch 내부 로직과 불일치
2. **신뢰도 계산**: 시그모이드 적용으로 의미 상실
3. **임계값 설정**: 원시 데이터 특성 미고려

---

## 🚀 **향후 과제**

### 우선순위 1: 시그모이드 문제 해결
```python
# 현재 (문제)
confidences = 1 / (1 + np.exp(-max_class_probs))  # 0.5 수렴

# 해결 후보
confidences = max_class_probs  # 원시값 직접 사용
```

### 우선순위 2: 세그멘테이션 복원
- output1의 마스크 정보 활용
- 33개 채널의 세그멘테이션 계수 처리

### 우선순위 3: 하이브리드 접근
- 속도: ONNX 활용
- 정확도: PyTorch 보완

---

## 🏆 **결론**

### 📊 **주요 성과**
- ✅ **변환 성공**: v1.0 완전 실패 → v2.0 안정적 변환
- ⚡ **속도 개선**: 19.2배 향상 (목표 달성)
- 🔧 **기술 축적**: 변환 기법 및 디버깅 노하우 확보

### ⚠️ **남은 과제**
- ❌ **정확도**: 과다 감지 문제 미해결
- 🎭 **세그멘테이션**: 마스크 정보 미활용
- 💯 **신뢰도**: 가짜 신뢰도 문제 해결 필요

**결론**: 변환 기술은 확보했으나, 후처리 최적화가 핵심 과제로 남음