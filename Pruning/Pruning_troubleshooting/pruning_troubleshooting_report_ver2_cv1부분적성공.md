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

## 🔧 해결 과정

### 4단계: BatchNorm 문제 해결
**🔧 시도한 방법:**
```python
# Conv2d + BatchNorm2d 동시 프루닝
conv_layer.out_channels = len(keep_indices)
bn_layer.num_features = len(keep_indices)
bn_layer.running_mean = new_running_mean
bn_layer.running_var = new_running_var
```

**📈 진전 사항:**
- BatchNorm 파라미터 불일치 문제 해결
- 하지만 여전히 연결된 다음 레이어 문제 발생

**❌ 남은 문제:**
```
RuntimeError: Given groups=1, weight of size [128, 128, 3, 3], 
expected input[1, 116, 6, 10] to have 128 channels, but got 116 channels instead
```

### 5단계: C2f 블록 구조 완전 분석 (성공!)
**🔍 핵심 발견:**
Hook 기반 자동 분석으로 C2f 블록 내부 구조 완전 파악:
```
model.8.cv1.conv:      (256, 6, 10) → (256, 6, 10)  # 입력 처리
model.8.m.0.cv1.conv:  (128, 6, 10) → (128, 6, 10)  # 첫 번째 branch  
model.8.m.0.cv2.conv:  (128, 6, 10) → (128, 6, 10)  # 직접 연결됨!
model.8.cv2.conv:      (384, 6, 10) → (256, 6, 10)  # 최종 출력
```

**💡 핵심 인사이트:**
- `m.0.cv1` → `m.0.cv2` 직접 연결 관계 확인
- cv1의 출력 채널 = cv2의 입력 채널 (128개)
- 하나만 변경하면 차원 불일치 발생

### 6단계: 연결된 레이어 쌍 동시 프루닝 (최종 해결!)
**🔧 최종 해결책:**
```python
# 연결된 cv1 ↔ cv2 쌍 자동 탐지
cv1_layer, cv2_layer = find_connected_conv_pair(layer_name)

# 동시 프루닝
_prune_conv_output_channels(cv1_conv, cv1_bn, keep_indices)  # cv1: 출력 채널
_prune_conv_input_channels(cv2_conv, keep_indices)           # cv2: 입력 채널
```

**✅ 성공 결과:**
```
🔗 연결 쌍 발견: model.8.m.0.cv1.conv ↔ model.8.m.0.cv2.conv
Conv1+BN: ✅, Conv2+BN: ✅
📊 프루닝: 128 → 116 채널
✅ 연결된 쌍 프루닝 완료
✅ model.8.m.0.cv1.conv 프루닝 안전함
🎉 확인된 안전한 프루닝 대상: 1개
```

---

## 🎯 최종 해결책

### 성공한 프루닝 시스템
1. **Hook 기반 구조 분석**: Forward hook으로 텐서 흐름 자동 추적
2. **연결 관계 자동 탐지**: cv1 ↔ cv2 연결 쌍 자동 식별
3. **동시 프루닝**: 연결된 모든 레이어 + BatchNorm 함께 수정
4. **안전성 검증**: 임시 프루닝 → 테스트 → 복원 시스템

### 핵심 기술적 구현
```python
# 1. 텐서 흐름 분석
def analyze_c2f_block(self, block_name):
    self._register_hooks(block_name)
    dummy_input = torch.randn(1, 3, 192, 320)  # 패딩 적용
    _ = self.model(dummy_input)

# 2. 연결 쌍 탐지  
def find_connected_conv_pair(self, layer_name):
    cv2_name = layer_name.replace('cv1.conv', 'cv2.conv')
    return cv1_layer, cv2_layer, cv2_name

# 3. 동시 프루닝
def _temp_prune_layer(self, layer_name, prune_ratio):
    cv1_layer, cv2_layer = self.find_connected_conv_pair(layer_name)
    self._prune_conv_output_channels(cv1_conv, cv1_bn, keep_indices)
    self._prune_conv_input_channels(cv2_conv, keep_indices)
```

---

## 📊 최종 성과

### 기술적 성취
- ✅ **YOLOv8 C2f 블록 구조 완전 분석**
- ✅ **연결된 레이어 자동 탐지 시스템 구축**  
- ✅ **Conv + BatchNorm 동시 프루닝 구현**
- ✅ **안전한 프루닝 대상 1개 발견**

### 발견된 안전한 프루닝 지점
- **model.8.m.0.cv1.conv**: C2f 블록 내부 branch의 첫 번째 레이어
- **128 → 116 채널** (10% 프루닝) 안전 확인
- **연결된 cv2 레이어와 함께** 동시 수정 필요

---

## 💡 핵심 학습 포인트

### 문제 해결 과정에서 얻은 인사이트
1. **현대 CNN의 복잡성**: 단순한 Sequential이 아닌 복잡한 연결 구조
2. **Hook의 활용**: 모델 내부 동작을 투시하는 강력한 도구
3. **연결 관계의 중요성**: 하나의 레이어 변경이 여러 레이어에 미치는 영향
4. **점진적 접근의 가치**: 단계별 문제 해결과 검증의 중요성

### 실무 적용 가능한 기술
- **자동 모델 구조 분석**: 복잡한 모델의 내부 구조 파악
- **안전한 프루닝 시스템**: 실패 시 자동 복원 메커니즘  
- **연결 관계 추적**: 레이어 간 의존성 자동 탐지
- **단계별 검증**: 각 수정 사항의 안전성 확인

### 프루닝의 현실적 이해
- **구조적 복잡성**: 현대 모델의 프루닝이 단순하지 않은 이유
- **도구의 한계**: 범용 라이브러리의 한계와 커스텀 구현의 필요성
- **실용성 vs 학습**: 실무에서는 다른 방법을 선호하지만 학습 가치는 매우 높음

---

## 🚀 다음 단계

### 즉시 가능한 확장
1. **실제 프루닝 실행**: 확인된 안전한 지점에서 프루닝 수행
2. **다른 C2f 블록 분석**: model.2, model.4, model.6 등 추가 분석
3. **성능 평가**: 프루닝 후 정확도/속도 측정

### 고급 기능 구현
1. **자동 프루닝 비율 조정**: 최적 프루닝 비율 탐색
2. **다중 레이어 동시 프루닝**: 여러 안전한 지점 동시 처리
3. **fine-tuning 통합**: 프루닝 후 성능 복구

**🎉 최종 결론: YOLOv8 수동 프루닝 시스템 구축 성공!**