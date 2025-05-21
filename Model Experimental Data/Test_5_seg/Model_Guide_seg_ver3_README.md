# YOLOv8 통합 학습 프레임워크: 설정 시스템 도입과 모드 통합

[![라이센스: MIT](https://img.shields.io/badge/라이센스-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics: YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

## 📝 개요

이 프레임워크는 객체 감지와 세그멘테이션 모델 학습을 단일 스크립트로 통합하고, 설정 파일 시스템과 명령줄 인터페이스를 도입하여 다양한 환경에서 유연하게 사용할 수 있도록 개발되었습니다.

## ✨ 주요 개선사항

### 1. 설정 파일 시스템 도입
- JSON 형식의 설정 파일로 학습 환경 관리
- 설정 저장/로드 기능으로 재현 가능한 학습 환경 구성

### 2. 객체 감지와 세그멘테이션 모드 통합
- 단일 스크립트로 두 작업 모드를 모두 지원
- 태스크별 최적화된 데이터 처리 및 평가 메트릭 제공

### 3. 강화된 명령줄 인터페이스
- 다양한 매개변수를 통한 유연한 실행 옵션
- 대화형/자동화 모드 전환 가능

### 4. 자동 경로 감지 및 관리
- 데이터셋과 필요 파일의 자동 검색 기능
- 다양한 프로젝트 구조 지원

### 5. 고급 오류 처리 메커니즘
- 포괄적인 예외 처리로 안정적인 실행 보장
- 상세한 디버깅 정보 제공

## 🚀 시작하기

### 필수 라이브러리

```bash
pip install ultralytics torch opencv-python pillow matplotlib tqdm koreanize_matplotlib
```

### 기본 사용법

1. **대화형 모드**:
   ```bash
   python model_seg_ver3.py
   ```

2. **명령줄 파라미터 사용**:
   ```bash
   python model_seg_ver3.py --data_dir "경로/데이터" --task segment --img_size 832 --model_type m
   ```

3. **설정 파일 사용**:
   ```bash
   python model_seg_ver3.py --config config.json --no_input
   ```

## 📋 버전 비교

| 기능 | Seg-Ver1 | Seg-Ver3 |
|---------|-----------|-----------|
| 객체 감지 지원 | ❌ | ✅ |
| 세그멘테이션 지원 | ✅ | ✅ |
| 설정 파일 시스템 | ❌ | ✅ |
| 명령줄 인터페이스 | ❌ | ✅ |
| 다양한 모델 크기 | ✅ (n,s,m만) | ✅ (n,s,m,l,x) |
| 사용자 대화형 인터페이스 | ✅ (단순) | ✅ (고급) |
| 자동 경로 감지 | ❌ | ✅ |
| 다양한 환경 지원 | ❌ | ✅ |
| 설정 저장 기능 | ❌ | ✅ |
| 클래스 불균형 처리 | ✅ | ✅ |
| 오류 처리 메커니즘 | 제한적 | 포괄적 |

## ⚙️ 주요 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|-----------|-----------|
| `--config` | 설정 파일 경로 | - |
| `--data_dir` | 데이터 디렉토리 | - |
| `--work_dir` | 작업 디렉토리 | `yolov8_dataset` |
| `--output_dir` | 출력 디렉토리 | `runs` |
| `--task` | 작업 유형 (detect/segment) | `detect` |
| `--model_type` | 모델 크기 (n/s/m/l/x) | `n` |
| `--img_size` | 이미지 크기 | `640` |
| `--batch_size` | 배치 크기 | `16` |
| `--epochs` | 에폭 수 | `50` |
| `--model_path` | 기존 모델 경로 | - |
| `--no_input` | 사용자 입력 건너뛰기 | `False` |
| `--save_config` | 설정 저장 | `False` |

## 📂 설정 파일 형식

```json
{
  "paths": {
    "work_dir": "yolov8_dataset",
    "output_dir": "runs",
    "data_dir": "path/to/data",
    "classes_file": "classes.txt",
    "model_path": ""
  },
  "training": {
    "img_size": 640,
    "batch_size": 16,
    "epochs": 50,
    "train_ratio": 0.8,
    "task": "detect"
  },
  "model": {
    "model_type": "n",
    "continue_training": false
  }
}
```

## 🧩 실행 모드

### 초보자 모드 (대화형)

대화형 프롬프트를 통해 모든 설정을 단계별로 안내합니다:

```bash
python model_seg_ver3.py
```

### 고급 사용자 모드 (명령줄 매개변수)

명령줄 매개변수를 통해 한 번에 모든 설정을 지정합니다:

```bash
python model_seg_ver3.py --data_dir "data" --task segment --img_size 832 --model_type m --epochs 100 --batch_size 32
```

### 자동화 모드 (설정 파일)

설정 파일을 사용하여 반복 가능한 학습 환경을 구성합니다:

```bash
python model_seg_ver3.py --config my_settings.json --no_input
```

## 📊 성능 최적화 팁

1. **모델 크기 선택**:
   - n: 리소스 제한적인 환경 (빠르지만 정확도 낮음)
   - s/m: 일반적인 용도에 적합 (균형 잡힌 성능)
   - l/x: 높은 정확도가 필요한 경우 (느리지만 정확도 높음)

2. **이미지 크기 최적화**:
   - 640: 표준 학습 (기본값)
   - 832: 세그멘테이션 마스크 정확도 향상
   - 1024: 최대 정확도 (GPU 메모리 많이 필요)

3. **배치 크기 조정**:
   - GPU 메모리에 따라 8-64 사이로 조정
   - 메모리 부족 오류 발생 시 배치 크기 줄이기

4. **에폭 수 조정**:
   - 데이터셋 크기와 복잡도에 따라 30-200 사이 조정
   - 클래스 불균형이 심할수록 더 많은 에폭 필요

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다 - 자세한 내용은 LICENSE 파일을 참조하세요.