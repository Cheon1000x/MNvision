import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torch
import cv2
import shutil
from tqdm import tqdm
import random
import albumentations as A  # 데이터 증강을 위한 라이브러리  
import koreanize_matplotlib
from collections import Counter


'''
1. 새 학습이 시작될 때:

첫 번째 실행 시 runs/detect/train/ 폴더에 모델, 로그, 결과 등이 저장됩니다.


2. 동일한 코드로 새 학습 세션을 또 실행할 경우:

두 번째 실행 시 runs/detect/train2/ 폴더가 생성됩니다.
세 번째 실행 시 runs/detect/train3/ 폴더가 생성됩니다.
이런 식으로 실행할 때마다 숫자가 증가합니다 (train, train2, train3, ...)


3. 각 학습 폴더 내부 구조:

weights/ 폴더: 학습된 모델 저장

best.pt: 검증 세트에서 가장 좋은 성능을 보인 모델
last.pt: 가장 최근 에포크의 모델


results.csv: 각 에포크의 학습/검증 결과 메트릭
confusion_matrix.png: 혼동 행렬 시각화
PR_curve.png: 정밀도-재현율 곡선

4.모델 갱신 조건:

best.pt 모델은 평가 메트릭(주로 mAP)이 이전 최고 성능보다 향상될 때 갱신됩니다.
save_period=5 설정으로 5에포크마다 자동으로 체크포인트도 저장됩니다.


5. 최종 모델 저장:

코드에서는 학습 완료 후 최종 모델을 yolov8m_custom.pt(새 학습) 또는 yolov8m_continued.pt(계속 학습) 파일로 실행 디렉토리에 따로 저장합니다.

'''

# OpenMP 라이브러리 충돌 문제 해결을 위한 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 작업 디렉토리 설정
WORK_DIR = "yolov8_dataset"
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

# 데이터 디렉토리 설정
IMAGES_DIR = os.path.join(WORK_DIR, "images")
LABELS_DIR = os.path.join(WORK_DIR, "labels")
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

# 필요한 디렉토리 생성
for dir_path in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 클래스 빈도 계산 함수 (추가)
def calculate_class_frequency(json_files, frame_files, classes_file):
    """
    모든 데이터에서 클래스별 출현 빈도를 계산합니다.
    """
    class_counts = Counter()
    
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print("클래스별 출현 빈도 계산 중...")
    for json_file, frame_file in tqdm(zip(json_files, frame_files), total=len(json_files)):
        try:
            # 이미지 크기 가져오기
            img = Image.open(frame_file)
            img_width, img_height = img.size
            
            # JSON 파일 파싱
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 라벨 정보 추출
            shapes = data.get("shapes", [])
            
            for shape in shapes:
                label = shape.get("label", "")
                
                if label in classes:
                    class_id = classes.index(label)
                    class_counts[class_id] += 1
        except Exception as e:
            print(f"파일 처리 중 오류: {json_file} - {e}")
            continue
    
    return class_counts

# 클래스 가중치 계산 함수 (추가)
def calculate_class_weights(class_counts):
    """
    클래스별 가중치를 계산합니다 (희소 클래스에 더 큰 가중치).
    """
    if not class_counts:
        return {}
    
    # 가장 많은 클래스 찾기
    max_count = max(class_counts.values())
    
    # 각 클래스의 가중치 계산 (최대 30.0으로 제한)
    weights = {}
    for cls_id, count in class_counts.items():
        # 인스턴스 수에 반비례하는 가중치 (최소 1.0, 최대 30.0)
        weight = min(30.0, max(1.0, max_count / (count + 1)))
        weights[cls_id] = weight
    
    return weights

# 향상된 데이터 증강을 위한 변환 설정 (수정)
def get_augmentation_transforms(strong=False):
    """
    데이터 증강을 위한 변환 파이프라인을 반환합니다.
    strong=True이면 희소 클래스를 위한 강화된 증강을 적용합니다.
    """
    if strong:
        # 희소 클래스를 위한 강화된 증강
        return A.Compose([
            A.HorizontalFlip(p=0.7),                    # 좌우 반전 (70% 확률)
            A.RandomRotate90(p=0.7),                    # 90도 회전 (70% 확률)
            A.VerticalFlip(p=0.3),                      # 상하 반전 (30% 확률)
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),  # 밝기/대비 조정
            A.RandomGamma(p=0.5, gamma_limit=(80, 120)),  # 감마 조정
            A.GaussianBlur(p=0.3, blur_limit=(3, 7)),   # 가우시안 블러
            A.CLAHE(p=0.5, clip_limit=4.0),             # 대비 제한 적응형 히스토그램 평활화
            A.GaussNoise(p=0.4, var_limit=(10.0, 50.0)),  # 가우시안 노이즈
            A.RandomShadow(p=0.3, shadow_roi=(0, 0, 1, 1)),  # 무작위 그림자
            A.RandomToneCurve(p=0.3, scale=0.3),        # 톤 커브 조정
            A.HueSaturationValue(p=0.3, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),  # HSV 조정
            A.RandomSunFlare(p=0.1),                    # 태양 플레어 효과 (10% 확률)
            A.RandomFog(p=0.1),                         # 안개 효과 (10% 확률)
            A.Perspective(p=0.3, scale=(0.05, 0.1)),    # 원근 변환 (30% 확률)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        # 기본 증강
        return A.Compose([
            A.HorizontalFlip(p=0.5),                    # 좌우 반전 (50% 확률)
            A.RandomRotate90(p=0.5),                    # 90도 회전 (50% 확률)
            A.RandomBrightnessContrast(p=0.3),          # 밝기와 대비 조정 (30% 확률)
            A.RandomGamma(p=0.3),                       # 감마 조정 (30% 확률)
            A.GaussianBlur(p=0.1),                      # 가우시안 블러 (10% 확률)
            A.CLAHE(p=0.3),                             # 대비 제한 적응형 히스토그램 평활화 (30% 확률)
            A.GaussNoise(p=0.2),                        # 가우시안 노이즈 추가 (20% 확률)
            A.RandomShadow(p=0.1),                      # 무작위 그림자 추가 (10% 확률)
            A.RandomToneCurve(p=0.2),                   # 무작위 톤 커브 조정 (20% 확률)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# JSON 파일에서 데이터 추출 및 변환 함수
def convert_json_to_yolo_format(json_file, image_file, classes_file):
    """
    JSON 파일에서 라벨 정보를 추출하여 YOLO 형식으로 변환합니다.
    """
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 이미지 크기 가져오기
    img = Image.open(image_file)
    img_width, img_height = img.size
    
    # JSON 파일 파싱
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 라벨 정보 추출
    shapes = data.get("shapes", [])
    yolo_annotations = []
    bboxes = []
    class_ids = []
    
    for shape in shapes:
        label = shape.get("label", "")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")
        
        if label in classes and (shape_type == "polygon" or shape_type == "rectangle"):
            # 클래스 인덱스 찾기
            class_id = classes.index(label)
            
            # YOLO 형식으로 변환 (다각형의 경우 바운딩 박스로 변환)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 바운딩 박스 계산
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # YOLO 형식으로 정규화 (중심 x, 중심 y, 너비, 높이)
            x_center = (x_min + x_max) / (2 * img_width)
            y_center = (y_min + y_max) / (2 * img_height)
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}"
            yolo_annotations.append(yolo_annotation)
            
            # 데이터 증강을 위한 정보 저장
            bboxes.append([x_center, y_center, width, height])
            class_ids.append(class_id)
    
    return yolo_annotations, bboxes, class_ids, np.array(img)

# 데이터 증강 함수 (수정: 클래스 빈도에 따른 증강 강도 조정)
def apply_augmentation(image, bboxes, class_ids, class_counts=None):
    """
    이미지와 바운딩 박스에 데이터 증강을 적용합니다.
    class_counts가 제공되면 희소 클래스에 더 강한 증강을 적용합니다.
    """
    # 강한 증강 적용 여부 결정
    apply_strong_aug = False
    
    # 클래스 빈도 정보가 있으면 희소 클래스 확인
    if class_counts:
        # 이미지에 있는 클래스 중 가장 희소한 클래스 찾기
        min_count = float('inf')
        for cls_id in class_ids:
            if cls_id in class_counts and class_counts[cls_id] < min_count:
                min_count = class_counts[cls_id]
        
        # 특정 임계값보다 적은 클래스가 있으면 강한 증강 적용
        if min_count < 500:  # 500개 미만을 희소 클래스로 간주
            apply_strong_aug = True
    
    # 증강 변환 가져오기
    transforms = get_augmentation_transforms(strong=apply_strong_aug)
    
    # 증강 적용
    transformed = transforms(image=image, bboxes=bboxes, class_labels=class_ids)
    
    # 변환된 결과 반환
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

# 입력 데이터 처리 함수 (수정: 클래스별 증강 횟수 조정)
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir, apply_aug=False, class_counts=None):
    """
    입력 데이터를 처리하고 YOLO 형식으로 변환하여 저장합니다.
    옵션으로 데이터 증강을 적용할 수 있습니다.
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 형식으로 라벨 변환
    yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_format(json_file, frame_file, classes_file)
    
    # 라벨 파일 저장
    label_filename = f"{base_filename}.txt"
    output_label_path = os.path.join(output_label_dir, label_filename)
    
    with open(output_label_path, 'w') as f:
        f.write("\n".join(yolo_annotations))
    
    # 데이터 증강 적용 (학습 데이터만 증강)
    if apply_aug and "train" in output_image_dir and len(bboxes) > 0:
        # 클래스 빈도에 따른 증강 횟수 결정
        aug_count = 3  # 기본 증강 횟수
        
        if class_counts:
            # 이미지에 있는 클래스 중 가장 희소한 클래스 찾기
            min_count = float('inf')
            for cls_id in class_ids:
                if cls_id in class_counts and class_counts[cls_id] < min_count:
                    min_count = class_counts[cls_id]
            
            # 희소 클래스에 따른 증강 횟수 조정
            if min_count < 100:
                aug_count = 10  # 매우 희소한 클래스 (100개 미만)
            elif min_count < 300:
                aug_count = 7   # 희소한 클래스 (300개 미만)
            elif min_count < 500:
                aug_count = 5   # 준희소 클래스 (500개 미만)
        
        # 결정된 횟수만큼 증강 적용
        for aug_idx in range(aug_count):
            # 증강 확률 조정 (희소 클래스는 항상 증강, 흔한 클래스는 50% 확률)
            if aug_count <= 3 and random.random() < 0.5:
                continue
                
            # 이미지와 바운딩 박스에 증강 적용
            try:
                aug_image, aug_bboxes, aug_class_ids = apply_augmentation(image, bboxes, class_ids, class_counts)
                
                # 빈 바운딩 박스 확인
                if not aug_bboxes or len(aug_bboxes) == 0:
                    continue
                
                # 파일명 생성
                aug_image_filename = f"{base_filename}_aug{aug_idx}.jpg"
                aug_label_filename = f"{base_filename}_aug{aug_idx}.txt"
                
                # 이미지 저장
                aug_image_path = os.path.join(output_image_dir, aug_image_filename)
                cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # 라벨 저장
                aug_label_path = os.path.join(output_label_dir, aug_label_filename)
                with open(aug_label_path, 'w') as f:
                    aug_annotations = [f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}" 
                                     for class_id, box in zip(aug_class_ids, aug_bboxes)]
                    f.write("\n".join(aug_annotations))
            except Exception as e:
                print(f"데이터 증강 중 오류 발생: {e}")
                continue
    
    return output_image_path, output_label_path

# 데이터셋 분할 함수
def split_dataset(json_files, frame_files, classes_file, train_ratio=0.8):
    """
    데이터셋을 학습 및 검증 세트로 분할합니다.
    """
    # 클래스 빈도 계산
    class_counts = calculate_class_frequency(json_files, frame_files, classes_file)
    
    # 클래스 분포 출력
    print("\n===== 클래스 분포 =====")
    for cls_id, count in sorted(class_counts.items()):
        print(f"클래스 {cls_id}: {count}개")
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(class_counts)
    print("\n===== 클래스 가중치 =====")
    for cls_id, weight in sorted(class_weights.items()):
        print(f"클래스 {cls_id}: {weight:.2f}")
    
    # 데이터셋 섞기
    indices = np.arange(len(json_files))
    np.random.shuffle(indices)
    
    # 학습 및 검증 세트 분할
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 데이터 처리
    for i in tqdm(train_indices, desc="학습 데이터 처리 중"):
        _, _ = process_data(json_files[i], frame_files[i], classes_file,
                         TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, apply_aug=True, class_counts=class_counts)
    
    for i in tqdm(val_indices, desc="검증 데이터 처리 중"):
        _, _ = process_data(json_files[i], frame_files[i], classes_file,
                         VAL_IMAGES_DIR, VAL_LABELS_DIR, apply_aug=False)
    
    # 데이터셋 개수 계산
    train_images = len([f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"데이터셋 분할 완료: {train_images}개 학습 샘플 (원본 {len(train_indices)}개 + 증강 {train_images - len(train_indices)}개), {val_images}개 검증 샘플")
    
    return class_counts, class_weights

# YAML 설정 파일 생성 함수 (수정: 클래스 가중치 추가)
def create_yaml_config(classes_file, output_yaml, class_weights=None):
    """
    YOLO 학습을 위한 YAML 설정 파일을 생성합니다.
    클래스 가중치가 제공되면 이를 설정에 추가합니다.
    """
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 클래스 가중치 문자열 생성
    if class_weights:
        max_cls_id = max(class_weights.keys())
        weights_list = [class_weights.get(i, 1.0) for i in range(max_cls_id + 1)]
        weights_str = "[" + ", ".join([f"{w:.2f}" for w in weights_list]) + "]"
    
    # YAML 파일 생성
    yaml_content = f"""
path: {os.path.abspath(WORK_DIR)}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}

# 데이터 증강 및 모델 하이퍼파라미터 추가 (YOLOv8 내장 증강)
augment: True    # YOLOv8 내장 증강 활성화
mosaic: 1.0      # Mosaic 증강 사용 (0.0-1.0)
mixup: 0.3       # Mixup 증강 사용 (0.0-1.0)
copy_paste: 0.3  # Copy-Paste 증강 사용 (0.0-1.0)
"""

    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")
    if class_weights:
        print("클래스 가중치가 YAML 설정에 추가되었습니다.")

# 학습 함수 (수정: 높은 해상도 및 최적화된 학습 매개변수)
def train_yolov8(yaml_config, epochs=50, batch_size=16, img_size=640): # 이미지 크기 증가 (640→832)
    """
    YOLOv8 모델을 학습합니다.
    """
    # YOLOv8m 모델 로드
    model = YOLO('yolov8m.pt')
    
    # 모델 학습
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

# 기존 모델에서 추가 학습하는 함수 (수정: 높은 해상도 및 최적화된 학습 매개변수)
def continue_training(pretrained_model_path, yaml_config, epochs=30, batch_size=16, img_size=640):
    """
    기존에 학습된 모델을 로드하고 새 데이터로 추가 학습합니다.
    """
    # 사전 학습된 모델 로드
    model = YOLO(pretrained_model_path)
    
    # 모델 추가 학습
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=20,        # 조기 종료 인내심 증가
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        lr0=0.001,          # 기본값보다 낮은 학습률 설정
        lrf=0.0001,         # 학습률 감소 계수
        resume=False,       # 학습 재개를 위한 설정
        save_period=5,      # 5 에포크마다 체크포인트 저장
        augment=True,       # YOLOv8 내장 데이터 증강 활성화
        cos_lr=True,        # 코사인 학습률 스케줄링
        warmup_epochs=3,    # 워밍업 에포크 수
        weight_decay=0.0005,# 가중치 감쇠
        overlap_mask=True,  # 마스크 오버랩 허용
        close_mosaic=10     # 마지막 10 에포크에서 mosaic 비활성화
    )
    
    return model, results

# 모델 평가 함수
def evaluate_model(model, yaml_config):
    """
    학습된 모델을 평가합니다.
    """
    # 모델 검증
    results = model.val(data=yaml_config)
    
    # 주요 메트릭 출력
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.p,
        'Recall': results.box.r,
        'F1-Score': results.box.f1
    }
    
    print("\n===== 모델 평가 결과 =====")
    for metric_name, metric_value in metrics.items():
        # NumPy 배열 또는 텐서인 경우 첫 번째 값 또는 평균값 사용
        if hasattr(metric_value, 'ndim') and metric_value.ndim > 0:
            # 배열인 경우 첫 번째 값 또는 평균값 사용
            if metric_value.size > 0:
                if metric_value.size == 1:
                    metric_value = float(metric_value[0])
                else:
                    metric_value = float(metric_value.mean())
            else:
                metric_value = 0.0
        
        print(f"{metric_name}: {float(metric_value):.4f}")
    
    return results

# 결과 시각화 함수
def plot_results(results):
    """
    학습 결과를 시각화합니다.
    """
    try:
        # 학습 결과인 경우
        if hasattr(results, 'plot_results'):
            fig = results.plot_results(show=False)
            fig.savefig("training_results.png")
            print("학습 결과 그래프가 'training_results.png'에 저장되었습니다.")
        # 평가 결과인 경우
        elif hasattr(results, 'box'):
            try:
                # 평가 결과 플롯 생성 (다른 방식으로)
               from ultralytics.utils.plotting import plot_pr_curve
               
               fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
               plot_pr_curve(px=results.box.px, py=results.box.py, ap=results.box.ap, 
                             names=results.names, colors=None, show_f1=True, ax=ax)
               fig.savefig("evaluation_results.png")
               print("평가 결과 그래프가 'evaluation_results.png'에 저장되었습니다.")
            except Exception as e:
               print(f"평가 결과 시각화 중 오류 발생: {e}")
               print("결과 시각화를 건너뜁니다.")
        else:
           print("결과 시각화를 위한 적절한 속성이 없습니다.")
    except Exception as e:
       print(f"결과 시각화 중 오류 발생: {e}")
       print("결과 시각화를 건너뜁니다.")

# 클래스 분포 시각화 함수
def visualize_class_distribution(class_counts):
   """
   클래스 분포와 가중치를 시각화합니다.
   """
   if not class_counts:
       print("클래스 분포 데이터가 없습니다.")
       return
   
   try:
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
       ax1.grid(axis='y', linestyle='--', alpha=0.7)
       
       # 각 막대 위에 개수 표시
       for bar, count in zip(bars, counts):
           height = bar.get_height()
           ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{count}', ha='center', va='bottom')
       
       # 두 번째 플롯: 클래스 가중치
       bars = ax2.bar(classes, weights, color='salmon')
       ax2.set_xlabel('클래스 ID')
       ax2.set_ylabel('가중치')
       ax2.set_title('클래스별 손실 함수 가중치')
       ax2.grid(axis='y', linestyle='--', alpha=0.7)
       
       # 각 막대 위에 가중치 표시
       for bar, weight in zip(bars, weights):
           height = bar.get_height()
           ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{weight:.2f}', ha='center', va='bottom')
       
       plt.tight_layout()
       plt.savefig("class_distribution_and_weights.png")
       print("클래스 분포 및 가중치 그래프가 'class_distribution_and_weights.png'에 저장되었습니다.")
   except Exception as e:
       print(f"클래스 분포 시각화 중 오류 발생: {e}")

# 증강된 이미지 샘플 시각화 함수
def visualize_augmented_samples():
   """
   증강된 이미지 샘플을 시각화하여 확인합니다.
   """
   # 학습 디렉토리에서 원본 이미지와 증강된 이미지 쌍 찾기
   orig_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if not '_aug' in f]
   if not orig_files:
       print("시각화할 이미지를 찾을 수 없습니다.")
       return
   
   # 샘플 이미지 선택 (최대 3개)
   sample_count = min(3, len(orig_files))
   sample_files = random.sample(orig_files, sample_count)
   
   fig, axes = plt.subplots(sample_count, 2, figsize=(12, 4 * sample_count))
   if sample_count == 1:
       axes = [axes]
   
   for i, orig_file in enumerate(sample_files):
       base_name = os.path.splitext(orig_file)[0]
       aug_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.startswith(base_name + '_aug')]
       
       # 원본 이미지 표시
       orig_path = os.path.join(TRAIN_IMAGES_DIR, orig_file)
       orig_img = cv2.imread(orig_path)
       orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
       axes[i][0].imshow(orig_img)
       axes[i][0].set_title("원본 이미지")
       axes[i][0].axis('off')
       
       # 증강된 이미지가 있으면 표시
       if aug_files:
           aug_file = random.choice(aug_files)
           aug_path = os.path.join(TRAIN_IMAGES_DIR, aug_file)
           aug_img = cv2.imread(aug_path)
           aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
           axes[i][1].imshow(aug_img)
           axes[i][1].set_title("증강된 이미지")
           axes[i][1].axis('off')
       else:
           axes[i][1].text(0.5, 0.5, "증강 이미지 없음", ha='center', va='center')
           axes[i][1].axis('off')
   
   plt.tight_layout()
   plt.savefig("augmentation_samples.png")
   print("증강 샘플이 'augmentation_samples.png'에 저장되었습니다.")

def main():
   # PyTorch CUDA 사용 가능 여부 확인
   print(f"PyTorch 버전: {torch.__version__}")
   print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA 버전: {torch.version.cuda}")
       print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
   else:
       print("GPU가 감지되지 않았습니다. CPU를 사용하여 학습합니다.")
   
   # 기본 데이터 폴더 경로 설정
   base_data_path = r"C:\Users\KDT34\Desktop\Group 6\MNvision\2.Data\photo"
   classes_file = "classes.txt"  # classes.txt 파일 경로 설정
   
   # 데이터가 포함된 모든 하위 폴더 찾기
   subfolders = [os.path.join(base_data_path, folder) for folder in os.listdir(base_data_path)
                if os.path.isdir(os.path.join(base_data_path, folder))]
   
   # 모든 JSON 및 이미지 파일 경로 저장할 리스트
   all_json_files = []
   all_frame_files = []
   
   # 각 하위 폴더에서 파일 찾기
   for folder in subfolders:
       print(f"폴더 처리 중: {folder}")
       
       # 폴더 내 모든 파일 목록
       files = os.listdir(folder)
       
       # JSON 파일과 이미지 파일 분류
       json_files = [os.path.join(folder, f) for f in files if f.endswith('.json') and f != 'classes.json']
       frame_files = [os.path.join(folder, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
       
       # 파일 경로 일치시키기 (같은 파일명 기반)
       for json_f in json_files:
           base_name = os.path.splitext(os.path.basename(json_f))[0]
           for frame_f in frame_files:
               if base_name == os.path.splitext(os.path.basename(frame_f))[0]:
                   all_json_files.append(json_f)
                   all_frame_files.append(frame_f)
                   break
   
   if not all_json_files:
       print("일치하는 JSON 및 이미지 파일을 찾을 수 없습니다.")
       return
   
   # classes.txt 파일 확인 - 기본 폴더에서 찾거나 하위 폴더에서 찾기
   classes_file_path = os.path.join(base_data_path, classes_file)
   if not os.path.exists(classes_file_path):
       # 첫 번째 하위 폴더에서 찾기
       if subfolders:
           alt_path = os.path.join(subfolders[0], classes_file)
           if os.path.exists(alt_path):
               classes_file_path = alt_path
           else:
               print(f"classes.txt 파일을 찾을 수 없습니다. 기본 폴더나 첫 번째 하위 폴더에 위치시켜주세요.")
               return
   
   print(f"총 {len(all_json_files)}개의 파일 쌍을 찾았습니다.")
   print(f"Classes 파일: {classes_file_path}")
   
   # 파일 경로 저장
   json_files = all_json_files
   frame_files = all_frame_files
   
   # 데이터 증강 설정 및 데이터셋 분할 처리
   print("\n===== 데이터 증강 및 데이터셋 분할 =====")
   class_counts, class_weights = split_dataset(json_files, frame_files, classes_file_path)
   
   # 클래스 분포 및 가중치 시각화
   visualize_class_distribution(class_counts)
   
   # 증강된 이미지 샘플 시각화
   try:
       visualize_augmented_samples()
   except Exception as e:
       print(f"증강 샘플 시각화 중 오류 발생: {e}")
   
   # YAML 설정 파일 생성 (클래스 가중치 포함)
   yaml_config = "dataset.yaml"
   create_yaml_config(classes_file_path, yaml_config, class_weights=class_weights)
   
   # 학습 모드 선택 (새 모델 학습 또는 기존 모델 계속 학습)
   continue_from_checkpoint = input("기존 모델에서 계속 학습하시겠습니까? (y/n): ").lower() == 'y'
   
   try:
       if continue_from_checkpoint:
           # 기존 모델 경로 입력
           model_path = input("기존 모델 경로를 입력하세요 (예: runs/detect/train/weights/best.pt): ")
           print(f"\n===== 기존 모델({model_path})에서 클래스 가중치를 적용하여 학습 계속 =====")
           model, training_results = continue_training(model_path, yaml_config, epochs=1)
           
           # 모델 평가
           evaluation_results = evaluate_model(model, yaml_config)
           
           # 학습 결과 시각화
           if hasattr(training_results, 'plot_results'):
               plot_results(training_results)
       else:
           # 새 모델로 학습 시작
           print("\n===== 클래스 가중치를 적용하여 YOLOv8m 모델 학습 시작 =====")
           model, training_results = train_yolov8(yaml_config, epochs=1)
           
           # 모델 평가
           evaluation_results = evaluate_model(model, yaml_config)
           
           # 학습 결과 시각화
           if hasattr(training_results, 'plot_results'):
               plot_results(training_results)
               
   except Exception as e:
       print(f"학습 중 오류가 발생했습니다: {e}")
       import traceback
       traceback.print_exc()

   # 최종 모델 저장
   output_path = "yolov8m_continued.pt" if continue_from_checkpoint else "yolov8m_custom.pt"
   try:
       # 직접 save 메서드 시도
       model.save(output_path)
   except AttributeError:
       try:
           # 실패하면 export 메서드 시도
           model.export(format="pt", save_dir=os.path.dirname(output_path), 
                       filename=os.path.basename(output_path))
       except Exception as e:
           # 모두 실패하면 torch.save 사용
           print(f"경고: 일반 저장 방법이 실패했습니다. torch.save 사용 중: {e}")
           torch.save(model.model.state_dict(), output_path)
   
   print(f"학습된 모델이 '{output_path}'로 저장되었습니다.")

if __name__ == "__main__":
   main()