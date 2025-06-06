# PT vs ONNX 모델 벤치마크 결과 분석

## 🎯 전체 성능 요약

### ⚡ 속도 성능
- **평균 속도 향상**: **2.66배** - ONNX가 PT보다 월등히 빠름
- **PT 평균 추론 시간**: 127.4ms
- **ONNX 평균 추론 시간**: 41.5ms
- **최대 속도 향상**: 28.58배 (첫 번째 이미지에서 워밍업 효과)

### 🎯 정확도 성능
- **전체 매칭률**: 91.8% (56/61개 매칭)
- **평균 클래스 정확도**: **11.1%** ⚠️ **매우 낮음**
- **평균 IoU**: **0.999** ✅ **매우 높음**

---

## 📊 클래스별 성능 분석 (심각한 문제 발견)

### 🚨 주요 문제점
1. **클래스 라벨 불일치 문제**:
   - PT 모델이 `forklift-vertical`(클래스 4)로 탐지한 35개 객체를 ONNX는 0개만 탐지
   - ONNX 모델이 `person`(클래스 3)으로 탐지한 31개 객체를 PT는 10개만 탐지
   - **클래스 매핑이 완전히 다름**

### 클래스별 상세 분석

| 클래스 | PT 탐지 | ONNX 탐지 | 매칭 | 매칭률 | 문제점 |
|--------|---------|-----------|------|--------|--------|
| **forklift-vertical** | 35개 | 0개 | 0개 | 0% | ❌ ONNX가 이 클래스를 전혀 인식 못함 |
| **person** | 10개 | 31개 | 0개 | 0% | ❌ 완전히 다른 객체를 person으로 분류 |
| **forklift-right** | 11개 | 10개 | 6개 | 55% | ⚠️ 유일하게 어느 정도 일치하는 클래스 |
| **forklift-left** | 4개 | 5개 | 0개 | 0% | ❌ 비슷한 수량이지만 매칭 안됨 |
| **forklift-horizontal** | 1개 | 10개 | 0개 | 0% | ❌ ONNX가 과도하게 탐지 |

### 💡 분석 결과
- **유일한 정상 클래스**: `forklift-right` (매칭률 55%)
- **심각한 클래스 불일치**: PT와 ONNX 모델의 클래스 해석이 완전히 다름

---

## 📈 성능 트렌드 분석

### 🕐 시간 성능
- **첫 번째 이미지**: PT 1,912ms vs ONNX 67ms (28.6배 차이) - 워밍업 효과
- **이후 안정화**: 평균 2-3배 속도 향상으로 안정화
- **ONNX 안정성**: 일관되게 30-40ms 범위 유지

### 🎯 정확도 트렌드
- **대부분 이미지**: 클래스 정확도 0% (클래스 불일치)
- **일부 이미지**: 50-100% 정확도 (우연한 일치)
- **매우 불안정한 패턴**: 예측 불가능한 정확도 변화

---

## 🔍 IoU 분석 (긍정적 결과)

### ✅ 우수한 IoU 성능
- **평균 IoU**: 0.999 (거의 완벽한 위치 정확도)
- **IoU 분포**: 대부분 0.99 이상의 매우 높은 값
- **박스 위치**: PT와 ONNX가 거의 동일한 위치에 박스 생성

### 📌 IoU vs 클래스 정확도 상관관계
- **상관계수**: 0.122 (매우 낮은 상관관계)
- **의미**: 박스 위치는 정확하지만 클래스 분류가 다름

---

## 🚨 핵심 문제점 및 해결 방안

### 1. **클래스 매핑 불일치** (가장 심각)
```
문제: PT와 ONNX 모델의 클래스 해석이 완전히 다름
원인: 
- 클래스 순서 불일치
- 후처리 로직 차이
- 신뢰도 임계값 차이

해결방안:
1. 클래스 매핑 테이블 생성
2. ONNX 후처리 로직 재검토
3. PT 모델과 동일한 클래스 순서로 재학습
```

### 2. **신뢰도 임계값 최적화 필요**
```
현재: PT(0.3) vs ONNX(0.3)
문제: 동일한 임계값이지만 다른 결과
해결: 각 모델별 최적 임계값 찾기
```

### 3. **ONNX 변환 검증 필요**
```
문제: 클래스 4(forklift-vertical)를 전혀 탐지 못함
원인: ONNX 변환 시 일부 클래스 가중치 손실 가능성
해결: 원본 PT 모델과 ONNX 출력을 직접 비교
```

---

## 💯 긍정적 측면

### ✅ 성공한 부분
1. **속도 향상**: 2.66배 빠른 추론 속도
2. **안정성**: ONNX 모델의 일관된 추론 시간
3. **위치 정확도**: IoU 0.999의 완벽한 박스 위치
4. **변환 성공**: 복잡한 세그멘테이션 모델의 성공적인 ONNX 변환

### ⚡ 속도 최적화 효과
- **실시간 처리 가능**: 40ms 추론 시간으로 25fps 가능
- **리소스 효율성**: CPU에서도 빠른 처리 속도
- **배포 용이성**: ONNX 형태로 다양한 플랫폼 지원

---

## 🔧 즉시 해결해야 할 문제

### 우선순위 1: 클래스 매핑 수정
```python
# 클래스 매핑 테이블 추가 필요
class_mapping = {
    'onnx_class_3': 'pt_class_4',  # person -> forklift-vertical
    'onnx_class_4': 'pt_class_3',  # forklift-vertical -> person
    # 기타 매핑...
}
```

### 우선순위 2: 후처리 로직 통일
```python
# 동일한 시그모이드, NMS 파라미터 사용
# 동일한 신뢰도 임계값 적용
# 동일한 클래스 순서 보장
```

---

## 📋 최종 평가

| 항목 | 점수 | 평가 |
|------|------|------|
| **속도 성능** | ⭐⭐⭐⭐⭐ | 탁월함 (2.66배 향상) |
| **위치 정확도** | ⭐⭐⭐⭐⭐ | 완벽함 (IoU 0.999) |
| **클래스 정확도** | ⭐ | 심각한 문제 (11.1%) |
| **안정성** | ⭐⭐⭐⭐ | 우수함 |
| **전체 평가** | ⭐⭐⭐ | 속도는 완벽, 정확도는 수정 필요 |

**결론**: ONNX 변환은 성공했지만 **클래스 매핑 문제로 인한 정확도 이슈**가 있어 즉시 수정이 필요합니다. 속도 향상은 매우 훌륭하므로 클래스 문제만 해결하면 완벽한 최적화가 될 것입니다.