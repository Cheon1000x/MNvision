# YOLOv8 수동 프루닝 트러블슈팅 리포트

## 📋 개요
- **목표**: YOLOv8n 모델의 수동 프루닝을 통한 경량화
- **모델**: best.pt (3,012,213 파라미터, 11.53MB)
- **접근 방법**: torch-pruning 라이브러리 → 수동 프루닝

---

## ❌ 발생한 문제들

### 1차 문제: torch-pruning 라이브러리 오류
**🔧 시도한 방법:**
```python
# torch-pruning 라이브러리 사용
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=example_inputs)
```

**❌ 발생 오류:**
```
RuntimeError: Sizes of tensors must match except in dimension 1. 
Expected size 24 but got size 23 for tensor number 1 in the list.
```

**🔍 원인 분석:**
- torch-pruning이 YOLOv8의 복잡한 구조(C2f, Concat, Skip connection)를 제대로 이해하지 못함
- Dependency graph 생성 과정에서 텐서 차원 불일치 발생

---

### 2차 문제: 개별 레이어 프루닝 시 연결 오류
**🔧 시도한 방법:**
```python
# model.7.conv 레이어의 출력 채널을 256 → 205로 프루닝
target_layer.out_channels = len(keep_indices)
target_layer.weight = nn.Parameter(new_weight)
```

**❌ 발생 오류:**
```
RuntimeError: Given groups=1, weight of size [256, 256, 1, 1], 
expected input[1, 205, 6, 10] to have 256 channels, but got 205 channels instead
```

**🔍 원인 분석:**
- `model.7.conv` (출력: 256→205) → `model.8.cv1.conv` (입력: 여전히 256 기대)
- 연결된 다음 레이어의 입력 채널을 함께 수정하지 않음

---

### 3차 문제: C2f 블록 내부 연결 오류
**🔧 시도한 방법:**
```python
# C2f 블록 내부의 model.8.m.0.cv1.conv 프루닝
# 채널 수: 128 → 109로 감소
```

**❌ 발생 오류:**
```
RuntimeError: Given groups=1, weight of size [128, 128, 3, 3], 
expected input[1, 109, 6, 10] to have 128 channels, but got 109 channels instead
```

**🔍 원인 분석:**
- C2f 블록 내부에서 `m.0.cv1` → `m.0.cv2` 연결
- cv1의 출력 채널을 변경했지만 cv2의 입력 채널은 그대로 유지됨

---

## 🧩 YOLOv8 구조 분석

### C2f 블록의 실제 구조
```
입력 → cv1 → split → [m.0: cv1 → cv2] → concat → cv2 → 출력
                      [m.1: cv1 → cv2]
                      [    ...       ]
```

### 발견된 연결 관계
1. **Backbone 연결**: `model.7.conv` → `model.8.cv1.conv`
2. **C2f 내부 연결**: `m.0.cv1` → `m.0.cv2`
3. **복잡한 Skip/Concat 연결**: 여러 레이어의 출력이 합쳐짐

---

## 🎯 문제의 핵심

### 근본 원인
**YOLOv8의 복잡한 연결 구조**
- 단순한 Sequential 구조가 아님
- Skip connection, Concat, Split 등 복잡한 연결
- 하나의 레이어 변경이 여러 연결된 레이어에 영향

### 간과한 부분
1. **연결 관계 분석 부족**: 단일 레이어만 고려
2. **C2f 블록 구조 이해 부족**: 내부 연결의 복잡성
3. **Tensor 차원 추적 부족**: 변경 후 영향 범위 파악 안됨

---

## 🔍 시도한 해결 접근법

### 1단계: 라이브러리 변경
- torch-pruning → 수동 프루닝으로 전환
- 더 세밀한 제어 가능

### 2단계: 연결 관계 고려
```python
connected_groups = {
    'backbone_1': ['model.1.conv', 'model.2.cv1.conv'],
    'backbone_3': ['model.3.conv', 'model.4.cv1.conv'],
    # ...
}
```

### 3단계: 그룹 프루닝 시도
- 연결된 레이어들을 함께 수정
- 입력/출력 채널 동시 조정

---

## 📊 현재 상태

### 성공한 부분
- ✅ 모델 구조 완전 분석 (64개 Conv2d 레이어 파악)
- ✅ 안전/위험 레이어 분류 (24개 안전, 40개 위험)
- ✅ 프루닝 후보 우선순위 선정

### 여전히 해결되지 않은 부분
- ❌ C2f 블록 내부 연결 구조의 완전한 이해
- ❌ 연결된 모든 레이어의 자동 추적
- ❌ 안전한 프루닝 지점 식별

---

## 🚀 다음 단계 계획

### 필요한 분석
1. **C2f 블록 완전 분석**: 내부 모든 연결 관계 매핑
2. **자동 연결 추적**: Forward pass 중 텐서 흐름 분석
3. **안전 지점 식별**: 연결이 단순한 독립적 레이어 찾기

### 대안적 접근
1. **더 보수적 프루닝**: 매우 작은 비율(5-10%)로 시작
2. **레이어별 영향도 분석**: 각 레이어 변경의 파급 효과 측정
3. **단계별 검증**: 각 프루닝 후 즉시 동작 확인

---

## 💡 학습 포인트

### 기술적 인사이트
- 현대 CNN 구조의 복잡성 체험
- 텐서 차원 관리의 중요성 인식
- 라이브러리 한계와 수동 구현의 필요성

### 프루닝의 현실
- 단순해 보이는 작업의 실제 복잡성
- 모델 구조에 대한 깊은 이해의 필요성
- 실무에서 다른 방법(양자화 등)을 선호하는 이유