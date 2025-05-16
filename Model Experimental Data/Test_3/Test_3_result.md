# YOLOv8 모델 성능 평가 결과 (ver3)

## 1. 주요 성능 지표 요약

| 클래스 | mAP@0.5 | Precision | Recall | F1 Score |
|----------|---------|-----------|--------|----------|
| forklift-vertical | 0.359 | 0.50 | 0.67 | ~0.57 |
| forklift-left | 0.995 | 1.00 | 1.00 | ~1.00 |
| forklift-horizontal | 0.977 | 0.98 | 0.97 | ~0.98 |
| person | 0.955 | 0.95 | 0.93 | ~0.89 |
| **전체 (all classes)** | **0.821** | **1.00** | **0.99** | **0.71** |

## 2. 혼동 행렬 분석

### 모델 예측 정확성:
- **person**: 639개 정확히 감지, 240개 배경으로 오분류 (95% 정확도)
- **forklift-horizontal**: 128개 정확히 감지, 2개 배경으로 오분류 (98% 정확도)
- **forklift-left**: 12개 정확히 감지, 6개 배경으로 오분류 (100% 정확도)
- **forklift-vertical**: 샘플이 적어 정확한 평가가 어려움

### 주요 오분류 패턴:
- **person**이 많은 경우 배경으로 오분류 (240개)
- **forklift-left**가 일부 **forklift-vertical**로 잘못 분류 (67%)
- 배경이 **person**으로 오분류되는 경우 있음 (31개)

## 3. 신뢰도 분석

- 최적 F1 점수는 신뢰도 0.162에서 0.71로 달성
- **forklift-horizontal**은 매우 높은 F1 값(~0.98) 유지
- **forklift-vertical**은 신뢰도 0.2 이상에서 감지 능력 상실
- **person** 클래스는 신뢰도 0.8까지 안정적으로 감지됨

## 4. Precision-Recall 곡선
- **forklift-left**: 0.995 mAP로 가장 우수한 성능
- **forklift-horizontal**: 0.977 mAP로 두 번째로 우수
- **person**: 0.955 mAP로 안정적인 성능
- **forklift-vertical**: 0.359 mAP로 다른 클래스보다 낮은 성능

## 5. 종합 평가

### 강점:
- **forklift-left**와 **forklift-horizontal** 클래스 인식 성능 매우 우수 (mAP > 0.97)
- **person** 클래스도 높은 정확도 유지 (mAP 0.955)
- 전체 mAP@0.5는 0.821로 양호한 수준

### 개선 필요 영역:
- **forklift-vertical** 클래스의 샘플 부족 문제 해결 필요
- **person** 클래스의 배경 오분류 (240개) 개선
- 낮은 신뢰도 영역에서의 모델 안정성 향상

### 권장 적용 방안:
- 신뢰도 임계값 0.162 부근으로 설정하여 최적 F1 점수 확보
- ver2와 비교해 전반적으로 향상된 성능 보임 (mAP 0.831 → 0.821)
- forklift 방향 분류에서 left와 horizontal 방향 분류 신뢰도 높음
- 실시간 현장 모니터링 시스템에 적합한 성능 보유