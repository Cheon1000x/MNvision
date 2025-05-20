# YOLOv8 모델 최적화 과정 보고서

## 목차
1. [서론](#서론)
2. [모델 버전별 발전 과정](#모델-버전별-발전-과정)
   - [Version 1: 기본 구현](#version-1-기본-구현)
   - [Version 2: 데이터 증강 강화](#version-2-데이터-증강-강화)
   - [Version 3: 클래스 불균형 해결 및 고급 최적화](#version-3-클래스-불균형-해결-및-고급-최적화)
3. [현재 진행 중인 모델 최적화 탐구](#현재-진행-중인-모델-최적화-탐구)
   - [모델 크기와 성능 간의 균형](#모델-크기와-성능-간의-균형)
   - [추론 속도 최적화](#추론-속도-최적화)
   - [다양한 모델 구조 비교](#다양한-모델-구조-비교)
4. [다음 단계: 전략 및 계획](#다음-단계-전략-및-계획)
5. [결론](#결론)

## 서론

본 보고서는 객체 감지를 위한 YOLOv8 모델의 점진적 최적화 과정을 기록합니다. 초기 기본 구현(Version 1)에서 시작하여 데이터 증강 강화(Version 2), 클래스 불균형 해결 및 고급 최적화(Version 3)를 거쳐 현재는 최적의 모델 구조와 하이퍼파라미터 설정을 탐색하는 단계에 있습니다. 이 과정을 통해 정확도, 처리 속도, 리소스 효율성 간의 최적 균형을 찾는 것이 목표입니다.

## 모델 버전별 발전 과정

### Version 1: 기본 구현

**주요 특징:**
- YOLOv8m 기본 모델 구현
- 기본적인 데이터 전처리 및 학습 파이프라인 구축
- 단순 학습/검증 데이터 분할(8:2 비율)
- 기본 하이퍼파라미터 사용

**코드 구현:**
```python
# 기본 학습 함수
def train_yolov8(yaml_config, epochs=50, batch_size=16, img_size=640):
    model = YOLO('yolov8m.pt')
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_period=5
    )
    return model, results
```

**한계점:**
- 제한된 데이터 증강으로 인한 과적합 위험
- 클래스 불균형 문제 미해결
- 모델 크기와 성능의 균형 미고려
- 단순한 하이퍼파라미터 설정으로 인한 성능 최적화 부족

### Version 2: 데이터 증강 강화

**개선 사항:**
- Albumentations 라이브러리를 활용한 고급 데이터 증강 도입
- 다양한 이미지 변환(회전, 대비 조정, 노이즈 추가 등) 적용
- YOLOv8 내장 데이터 증강 기능 추가 활성화
- 학습 파라미터 최적화(코사인 학습률 스케줄링, 워밍업 에포크 등)

**코드 구현:**
```python
# 데이터 증강을 위한 변환 설정
def get_augmentation_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),                     # 좌우 반전 (50% 확률)
        A.RandomRotate90(p=0.5),                     # 90도 회전 (50% 확률)
        A.RandomBrightnessContrast(p=0.3),           # 밝기와 대비 조정 (30% 확률)
        A.RandomGamma(p=0.3),                        # 감마 조정 (30% 확률)
        A.GaussianBlur(p=0.1),                       # 가우시안 블러 (10% 확률)
        A.CLAHE(p=0.3),                              # 대비 제한 적응형 히스토그램 평활화 (30% 확률)
        A.GaussNoise(p=0.2),                         # 가우시안 노이즈 추가 (20% 확률)
        A.RandomShadow(p=0.1),                       # 무작위 그림자 추가 (10% 확률)
        A.RandomToneCurve(p=0.2),                    # 무작위 톤 커브 조정 (20% 확률)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

**YAML 설정 개선:**
```yaml
# 데이터 증강 및 모델 하이퍼파라미터 추가 (YOLOv8 내장 증강)
augment: True    # YOLOv8 내장 증강 활성화
mosaic: 1.0      # Mosaic 증강 사용 (0.0-1.0)
mixup: 0.3       # Mixup 증강 사용 (0.0-1.0)
copy_paste: 0.3  # Copy-Paste 증강 사용 (0.0-1.0)
```

**학습 함수 개선:**
```python
# 개선된 학습 함수
def train_yolov8(yaml_config, epochs=50, batch_size=16, img_size=640):
    model = YOLO('yolov8m.pt')
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_period=5,  # 5 에포크마다 체크포인트 저장
        augment=True,   # YOLOv8 내장 데이터 증강 적용
        cos_lr=True,    # 코사인 학습률 스케줄링 사용
        warmup_epochs=3 # 워밍업 에포크 수
    )
    return model, results
```

**개선 효과:**
- 데이터 다양성 증가로 과적합 감소
- 데이터 증강을 통한 학습 데이터 증가
- 학습 안정성 향상
- 모델 일반화 능력 향상

**여전한 과제:**
- 클래스 불균형 문제 지속
- 데이터셋 내 희소 클래스에 대한 성능 부족
- 모델 크기와 성능 간의 균형 문제
- 이미지 해상도에 따른 성능 차이 고려 부족

### Version 3: 클래스 불균형 해결 및 고급 최적화

**핵심 개선 사항:**
- 클래스 빈도 분석 및 가중치 계산 기능 추가
- 희소 클래스에 대한 차별화된 데이터 증강 적용
- 클래스별 맞춤형 손실 가중치 설정
- 더 높은 이미지 해상도 지원(640→832)
- 고급 학습 매개변수 최적화(조기 종료 개선, 학습률 미세 조정)

**클래스 빈도 및 가중치 계산:**
```python
# 클래스 빈도 계산 함수
def calculate_class_frequency(json_files, frame_files, classes_file):
    class_counts = Counter()
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    for json_file, frame_file in tqdm(zip(json_files, frame_files), total=len(json_files)):
        # 이미지와 JSON 파일에서 클래스 개수 계산
        # ...생략...
    
    return class_counts

# 클래스 가중치 계산 함수
def calculate_class_weights(class_counts):
    if not class_counts:
        return {}
    
    # 가장 많은 클래스 찾기
    max_count = max(class_counts.values())
    
    # 각 클래스의 가중치 계산 (희소 클래스에 더 큰 가중치)
    weights = {}
    for cls_id, count in class_counts.items():
        # 인스턴스 수에 반비례하는 가중치 (최소 1.0, 최대 30.0)
        weight = min(30.0, max(1.0, max_count / (count + 1)))
        weights[cls_id] = weight
    
    return weights
```

**희소 클래스를 위한 강화된 데이터 증강:**
```python
# 향상된 데이터 증강을 위한 변환 설정
def get_augmentation_transforms(strong=False):
    if strong:  # 희소 클래스를 위한 강화된 증강
        return A.Compose([
            A.HorizontalFlip(p=0.7),                    # 좌우 반전 (70% 확률)
            A.RandomRotate90(p=0.7),                    # 90도 회전 (70% 확률)
            A.VerticalFlip(p=0.3),                      # 상하 반전 (30% 확률)
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            # ... 추가 증강 ...
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:  # 기본 증강
        # ... 기본 증강 설정 ...
```

**클래스 가중치가 적용된 YAML 설정:**
```python
# YAML 설정 파일 생성 함수 (수정: 클래스 가중치 추가)
def create_yaml_config(classes_file, output_yaml, class_weights=None):
    # ... 생략 ...
    
    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
```

**개선된 학습 함수:**
```python
# 학습 함수 (수정: 높은 해상도 및 최적화된 학습 매개변수)
def train_yolov8(yaml_config, epochs=50, batch_size=16, img_size=832):
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=20,        # 조기 종료 인내심 증가
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_period=5,      # 5 에포크마다 체크포인트 저장
        augment=True,       # YOLOv8 내장 데이터 증강 적용
        cos_lr=True,        # 코사인 학습률 스케줄링 사용
        warmup_epochs=5,    # 워밍업 에포크 수 증가
        lr0=0.01,           # 초기 학습률
        lrf=0.001,          # 최종 학습률
        weight_decay=0.0005,# 가중치 감쇠
        overlap_mask=True,  # 마스크 오버랩 허용
        close_mosaic=10     # 마지막 10 에포크에서 mosaic 비활성화
    )
    
    return model, results
```

**클래스 분포 시각화:**
```python
# 클래스 분포 시각화 함수
def visualize_class_distribution(class_counts):
    # 클래스 ID와 개수
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    # 가중치 계산
    class_weights = calculate_class_weights(class_counts)
    weights = [class_weights.get(cls, 1.0) for cls in classes]
    
    # 두 개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 첫 번째 플롯: 클래스 분포
    bars = ax1.bar(classes, counts, color='royalblue')
    ax1.set_xlabel('클래스 ID')
    ax1.set_ylabel('인스턴스 수')
    ax1.set_title('클래스별 인스턴스 분포')
    # ... 생략 ...
```

**개선 효과:**
- 클래스 불균형 문제 완화
- 희소 클래스에 대한 성능 향상
- 더 높은 이미지 해상도로 인한 작은 객체 감지 성능 향상
- 학습 안정성 및 수렴 속도 개선
- 데이터셋 분석 및 시각화를 통한 인사이트 제공

## 현재 진행 중인 모델 최적화 탐구

현재 Version 3 모델을 기반으로 다음과 같은 추가 최적화 탐구를 진행하고 있습니다:

### 모델 크기와 성능 간의 균형

**현재 탐구 영역:**
- YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x 등 다양한 모델 크기 비교
- 각 모델 크기별 정확도와 추론 속도 간의 균형 분석
- 응용 환경에 따른 최적 모델 크기 결정

**주요 고려사항:**
- 실시간 처리가 필요한 경우 더 작은 모델(YOLOv8n/s) 고려
- 높은 정확도가 중요한 경우 더 큰 모델(YOLOv8l/x) 고려
- 모바일/엣지 디바이스 배포 시 모델 크기 제약 고려

**측정 지표:**
- mAP50, mAP50-95 (정확도)
- FPS (초당 프레임 수)
- 모델 크기 (MB)
- 메모리 사용량 (MB)
- 효율성 지표 (mAP/모델크기, mAP/추론시간)

### 추론 속도 최적화

**현재 탐구 영역:**
- 입력 이미지 해상도 최적화 (640 vs 832 vs 1280)
- 모델 양자화(Quantization) 가능성 탐색
- TensorRT, ONNX 변환을 통한 추론 가속화
- 배치 처리를 통한 처리량 최적화

**주요 고려사항:**
- 정확도 손실 대비 속도 향상 균형
- 하드웨어 플랫폼별 최적화 전략 차별화
- 실시간 처리 필요성에 따른 최적화 정도 결정

**목표 성능:**
- 최소 30 FPS (실시간 애플리케이션)
- mAP50 0.5 이상 유지
- 모델 크기 100MB 이하 (가능한 경우)

### 다양한 모델 구조 비교

**현재 탐구 영역:**
- 객체 감지(Detection) vs 세분화(Segmentation) 모델 비교
- YOLOv8 기반 모델과 다른 아키텍처(EfficientDet, SSD 등) 비교
- 앙상블 접근 방식 탐색

**주요 고려사항:**
- 작업 요구사항에 가장 적합한 모델 구조 선택
- 복잡한 시나리오에서의 성능 평가
- 특정 도메인에 최적화된 모델 설계

**비교 방법론:**
- 표준화된 테스트 데이터셋 사용
- 실제 환경 비디오에서의 성능 측정
- 혼합 조건(조명, 거리, 가려짐 등)에서의 강건성 평가

## 다음 단계: 전략 및 계획

현재의 탐구 결과를 바탕으로 다음과 같은 추가 최적화 단계를 계획하고 있습니다:

### 1. 모델 경량화 및 가속화
- **계획:** YOLOv8m 모델을 ONNX로 변환하고 양자화를 적용하여 모델 크기 감소 및 추론 속도 개선
- **목표:** 정확도 손실 최소화(<3%)하면서 추론 속도 50% 향상
- **방법론:**
  ```python
  # ONNX 변환 계획
  model = YOLO('yolov8m_custom.pt')
  model.export(format='onnx', dynamic=True, simplify=True)
  
  # 양자화 계획
  # INT8 양자화 적용 예정
  ```

### 2. 특정 도메인 최적화
- **계획:** 특정 객체 클래스에 대한 성능을 더욱 최적화
- **목표:** 중요 클래스에 대한 F1 점수 15% 향상
- **방법론:**
  - 특정 클래스에 대한 추가 데이터 수집 및 증강
  - 추가 미세 조정 및 클래스별 손실 가중치 최적화

### 3. 배포 파이프라인 구축
- **계획:** 최적화된 모델의 배포를 위한 파이프라인 구축
- **목표:** 다양한 환경(클라우드, 엣지, 모바일)에서의 배포 지원
- **방법론:**
  - Docker 컨테이너화
  - TensorRT, CoreML 등 플랫폼별 변환 자동화
  - 추론 서버 설정 및 API 구축

## 결론

YOLOv8 모델의 최적화 과정(Version 1에서 Version 3까지)을 통해 기본 구현에서 시작하여 데이터 증강, 클래스 불균형 해결, 학습 파라미터 최적화 등 다양한 개선을 이루었습니다. 현재는 모델 크기와 성능 간의 균형, 추론 속도 최적화, 다양한 모델 구조 비교를 중점적으로 탐구하고 있습니다.

이러한 최적화 과정을 통해 얻은 인사이트는 다음과 같습니다:

1. **데이터 품질과 전처리**가 모델 성능에 가장 큰 영향을 미침
2. 클래스 불균형 문제는 **맞춤형 데이터 증강과 손실 가중치**로 효과적으로 해결 가능
3. 응용 환경에 따라 **정확도와 속도 간의 균형**이 중요함
4. 모델 **최적화는 반복적 과정**으로 지속적인 평가와 개선이 필요함

앞으로의 모델 최적화 과정은 모델 경량화, 앙상블 접근법 탐색, 특정 도메인 최적화, 배포 파이프라인 구축 등에 초점을 맞추어 진행할 예정입니다. 이를 통해 실제 환경에서도 높은 성능과 효율성을 제공하는 객체 감지 시스템을 구축하는 것이 목표입니다.
