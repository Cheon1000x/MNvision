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
from collections import Counter
import koreanize_matplotlib

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

# 클래스 빈도 계산 함수
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

# 클래스 가중치 계산 함수
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

# JSON 파일에서 데이터 추출 및 변환 함수 - 세그멘테이션용으로 수정
def convert_json_to_yolo_format(json_file, image_file, classes_file):
    """
    JSON 파일에서 라벨 정보를 추출하여 YOLO 세그멘테이션 형식으로 변환합니다.
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
    masks = []
    class_ids = []
    
    for shape in shapes:
        label = shape.get("label", "")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")
        
        if label in classes:
            # 클래스 인덱스 찾기
            class_id = classes.index(label)
            
            # 세그멘테이션 포인트 처리
            if shape_type == "polygon":
                # 다각형의 경우 모든 포인트를 정규화 형식으로 변환
                normalized_points = []
                for point in points:
                    x, y = point
                    # YOLO 형식으로 정규화 (x, y 좌표)
                    normalized_x = float(x) / img_width
                    normalized_y = float(y) / img_height
                    normalized_points.extend([normalized_x, normalized_y])
                
                # YOLO 세그멘테이션 형식: class_id x1 y1 x2 y2 ... xn yn
                if len(normalized_points) >= 6:  # 최소 3개의 점이 필요 (3 x 2 좌표)
                    yolo_annotation = f"{class_id} " + " ".join([f"{p}" for p in normalized_points])
                    yolo_annotations.append(yolo_annotation)
                    masks.append(normalized_points)
                    class_ids.append(class_id)
            
            elif shape_type == "rectangle":
                # 사각형의 경우 4개의 점으로 변환 (시계 방향)
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 정규화
                x1_norm = float(x1) / img_width
                y1_norm = float(y1) / img_height
                x2_norm = float(x2) / img_width
                y2_norm = float(y2) / img_height
                
                # 사각형의 4개 점 (시계 방향) - 세그멘테이션용
                rect_points = [
                    x1_norm, y1_norm,  # 좌상단
                    x2_norm, y1_norm,  # 우상단
                    x2_norm, y2_norm,  # 우하단
                    x1_norm, y2_norm   # 좌하단
                ]
                
                yolo_annotation = f"{class_id} " + " ".join([f"{p}" for p in rect_points])
                yolo_annotations.append(yolo_annotation)
                masks.append(rect_points)
                class_ids.append(class_id)
    
    return yolo_annotations, masks, class_ids, np.array(img)

# 입력 데이터 처리 함수 - 세그멘테이션용으로 수정
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir, apply_aug=False, class_counts=None):
    """
    입력 데이터를 처리하고 YOLO 세그멘테이션 형식으로 변환하여 저장합니다.
    옵션으로 클래스 균형을 맞추기 위한 증강을 적용할 수 있습니다.
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 세그멘테이션 형식으로 라벨 변환
    yolo_annotations, masks, class_ids, image = convert_json_to_yolo_format(json_file, frame_file, classes_file)
    
    # 라벨 파일 저장
    label_filename = f"{base_filename}.txt"
    output_label_path = os.path.join(output_label_dir, label_filename)
    
    with open(output_label_path, 'w') as f:
        f.write("\n".join(yolo_annotations))
    
    # 데이터 증강 적용 (학습 데이터만 증강)
    if apply_aug and "train" in output_image_dir and len(masks) > 0:
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
        
        # 결정된 횟수만큼 증강 적용 (클래스 균형 맞추기)
        for aug_idx in range(aug_count):
            # 이미지 저장
            aug_image_filename = f"{base_filename}_aug{aug_idx}.jpg"
            aug_label_filename = f"{base_filename}_aug{aug_idx}.txt"
            aug_image_path = os.path.join(output_image_dir, aug_image_filename)
            
            # 원본 이미지 복사
            shutil.copy(frame_file, aug_image_path)
            
            # 라벨도 그대로 복사 (단순히 수량을 늘리는 방식)
            aug_label_path = os.path.join(output_label_dir, aug_label_filename)
            with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                dst.write(src.read())
    
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

# YAML 설정 파일 생성 함수 (세그멘테이션용으로 수정)
def create_yaml_config(classes_file, output_yaml, class_weights=None):
    """
    YOLO 세그멘테이션 학습을 위한 YAML 설정 파일을 생성합니다.
    클래스 가중치가 제공되면 이를 설정에 추가합니다.
    """
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 클래스 가중치 문자열 생성
    weights_str = ""
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

# 세그멘테이션 및 학습 설정
task: segment
"""

    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")
    if class_weights:
        print("클래스 가중치가 YAML 설정에 추가되었습니다.")

# 세그멘테이션 모델 학습 함수
def train_yolov8_seg(yaml_config, epochs=50, batch_size=16, img_size=640):
    """
    YOLOv8 세그멘테이션 모델을 학습합니다.
    """
    # 선택된 세그멘테이션 모델 로드 (YOLOv8n-seg는 더 가벼운 모델)
    model = YOLO('yolov8n-seg.pt')  # 작은 모델 사용
    
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
        cos_lr=True,        # 코사인 학습률 스케줄링 사용
        warmup_epochs=5,    # 워밍업 에포크 수
        lr0=0.01,           # 초기 학습률
        lrf=0.001,          # 최종 학습률
        weight_decay=0.0005,# 가중치 감쇠
        overlap_mask=True,  # 마스크 오버랩 허용
        close_mosaic=10     # 마지막 10 에포크에서 mosaic 비활성화
    )
    
    return model, results

# 기존 세그멘테이션 모델에서 추가 학습
def continue_training_seg(pretrained_model_path, yaml_config, epochs=30, batch_size=16, img_size=640):
    """
    기존에 학습된 세그멘테이션 모델을 로드하고 새 데이터로 추가 학습합니다.
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
        cos_lr=True,        # 코사인 학습률 스케줄링
        warmup_epochs=3,    # 워밍업 에포크 수
        weight_decay=0.0005,# 가중치 감쇠
        overlap_mask=True,  # 마스크 오버랩 허용
        close_mosaic=10     # 마지막 10 에포크에서 mosaic 비활성화
    )
    
    return model, results

# 모델 평가 함수 (세그멘테이션용으로 수정)
def evaluate_model(model, yaml_config):
    """
    학습된 세그멘테이션 모델을 평가합니다.
    """
    # 모델 검증
    results = model.val(data=yaml_config)
    
    # 주요 메트릭 출력
    metrics = {
        'mAP50(B)': results.box.map50,      # 바운딩 박스 mAP50
        'mAP50-95(B)': results.box.map,      # 바운딩 박스 mAP50-95
        'mAP50(M)': results.seg.map50,       # 세그멘테이션 마스크 mAP50
        'mAP50-95(M)': results.seg.map,      # 세그멘테이션 마스크 mAP50-95
        'Precision(B)': results.box.p,       # 바운딩 박스 정밀도
        'Recall(B)': results.box.r,          # 바운딩 박스 재현율
        'Precision(M)': results.seg.p,       # 마스크 정밀도
        'Recall(M)': results.seg.r           # 마스크 재현율
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

# 학습 결과 시각화 함수
def plot_results(results):
    """
    학습 결과를 시각화합니다.
    """
    try:
        # 결과 시각화 (Ultralytics YOLO는 자동으로 plots 폴더에 그래프를 저장합니다)
        # 추가적인 시각화가 필요한 경우 여기에 구현
        
        print("\n학습 결과가 시각화되었습니다. 'runs/segment/train/results.png'에서 확인하세요.")
        
        # 필요하다면 커스텀 시각화를 위해 메트릭 추출 및 그래프 생성
        if hasattr(results, 'keys') and 'metrics/mAP50(B)' in results.keys():
            epochs = list(range(1, len(results['metrics/mAP50(B)']) + 1))
            
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, results['metrics/mAP50(B)'], label='mAP50 (Box)')
            plt.plot(epochs, results['metrics/mAP50(M)'], label='mAP50 (Mask)')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig("training_progress.png")
            print("추가 학습 진행 그래프가 'training_progress.png'에 저장되었습니다.")
    except Exception as e:
        print(f"학습 결과 시각화 중 오류 발생: {e}")
        print("기본 YOLO 시각화는 'runs/segment/train/' 폴더에서 확인하세요.")

def main():
    # PyTorch CUDA 사용 가능 여부 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU가 감지되지 않았습니다. CPU를 사용하여 학습합니다.")
    
    # 기본 데이터 폴더 경로 설정 - 하드코딩 또는 입력 받기
    use_hardcoded_path = True  # 하드코딩 경로 사용 여부

    if use_hardcoded_path:
        # 하드코딩된 경로 사용 (필요에 따라 수정)
        base_data_path = r"C:\Users\KDT-13\Desktop\Group 6\MNvision\2.Data\photo_cam1"
        print(f"데이터 폴더 경로(고정값): {base_data_path}")
    else:
        # 사용자로부터 데이터 폴더 경로 입력 받기
        print("\n===== 데이터 폴더 경로 입력 =====")
        print("데이터 폴더는 이미지와 JSON 파일이 있는 폴더여야 합니다.")
        print("예: C:\\Users\\KDT-13\\Desktop\\Group 6\\MNvision\\2.Data\\photo_cam1")
        base_data_path = input("데이터 폴더 경로를 입력하세요: ").strip()

    # classes.txt 파일 기본 경로
    classes_file = "classes.txt"
    
    # 경로 존재 확인
    if not os.path.exists(base_data_path):
        print(f"오류: 지정한 경로 '{base_data_path}'가 존재하지 않습니다.")
        return
    
    try:
        # 데이터가 포함된 모든 하위 폴더 찾기
        subfolders = []
        if os.path.isdir(base_data_path):
            for folder in os.listdir(base_data_path):
                folder_path = os.path.join(base_data_path, folder)
                if os.path.isdir(folder_path):
                    subfolders.append(folder_path)
            
            # 하위 폴더가 없으면 기본 폴더를 직접 사용
            if not subfolders:
                print("하위 폴더가 없습니다. 기본 폴더를 직접 사용합니다.")
                subfolders = [base_data_path]
        else:
            print(f"오류: '{base_data_path}'는 폴더가 아닙니다.")
            return
    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {e}")
        return
        
    # 모든 JSON 및 이미지 파일 경로 저장할 리스트
    all_json_files = []
    all_frame_files = []
    
    # 각 하위 폴더에서 파일 찾기
    for folder in subfolders:
        print(f"폴더 처리 중: {folder}")
        
        try:
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
        except Exception as e:
            print(f"폴더 '{folder}' 처리 중 오류 발생: {e}")
            continue
    
    if not all_json_files:
        print("일치하는 JSON 및 이미지 파일을 찾을 수 없습니다.")
        return
    
    # classes.txt 파일 확인 - 기본 폴더에서 찾거나 하위 폴더에서 찾기
    classes_file_path = os.path.join(base_data_path, classes_file)
    if not os.path.exists(classes_file_path):
        # 하위 폴더에서 검색
        found = False
        for folder in subfolders:
            potential_path = os.path.join(folder, classes_file)
            if os.path.exists(potential_path):
                classes_file_path = potential_path
                found = True
                break
        
        if not found:
            print(f"classes.txt 파일을 찾을 수 없습니다. 기본 폴더나 하위 폴더에 위치시켜주세요.")
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
    
    # YAML 설정 파일 생성 (클래스 가중치 포함)
    yaml_config = "dataset.yaml"
    create_yaml_config(classes_file_path, yaml_config, class_weights=class_weights)
    
    # 학습 설정
    use_hardcoded_settings = True  # 하드코딩된 설정 사용 여부
    
    if use_hardcoded_settings: #------------------------------------------------------------------
        # 하드코딩된 설정 사용
        continue_from_checkpoint = False  # 새 모델 학습
        img_size = 640  # 이미지 크기
        epochs = 50     # 에폭 수
        batch_size = 16  # 배치 크기
        model_choice = "n"  # 모델 선택 (n=가벼움, s=중간, m=무거움)
        
        print("\n===== 학습 설정 (고정값) =====")
        print(f"모델 유형: {'기존 모델에서 계속' if continue_from_checkpoint else '새 모델 학습'}")
        print(f"이미지 크기: {img_size}px")
        print(f"에폭 수: {epochs}")
        print(f"배치 크기: {batch_size}")
    else:
        # 사용자로부터 설정 입력 받기
        print("\n===== 학습 설정 입력 =====")
        continue_from_checkpoint = input("기존 모델에서 계속 학습하시겠습니까? (y/n): ").lower() == 'y'
        
        print("\n이미지 크기 입력 (기본: 640, 더 높은 정확도를 위해 832 권장)")
        print("- 작은 크기(320-416): 속도 빠름, 정확도 낮음")
        print("- 중간 크기(640): 균형 잡힌 속도와 정확도")
        print("- 큰 크기(832-1024): 속도 느림, 정확도 높음")
        img_size = int(input("학습 이미지 크기를 입력하세요: ") or "640")
        
        print("\n에폭 수 입력 (기본: 50)")
        print("- 적은 에폭(10-30): 빠른 학습, 낮은 정확도")
        print("- 중간 에폭(50-100): 균형 잡힌 학습 시간과 정확도")
        print("- 많은 에폭(100+): 오랜 학습 시간, 잠재적으로 더 높은 정확도")
        epochs = int(input("학습 에폭 수를 입력하세요: ") or "50")
        
        print("\n배치 크기 입력 (기본: 16, GPU 메모리에 따라 조정)")
        print("- 작은 배치(4-8): 적은 GPU 메모리 필요, 느린 학습")
        print("- 중간 배치(16-32): 균형 잡힌 속도와 메모리 사용량")
        print("- 큰 배치(64+): 빠른 학습, 많은 GPU 메모리 필요")
        batch_size = int(input("배치 크기를 입력하세요: ") or "16")
        
        if not continue_from_checkpoint:
            print("\n모델 선택 (기본: n)")
            print("- n: YOLOv8n-seg (가벼운 모델, 빠르지만 정확도 낮음)")
            print("- s: YOLOv8s-seg (중간 크기 모델, 균형 잡힌 속도와 정확도)")
            print("- m: YOLOv8m-seg (무거운 모델, 정확도 높지만 느림)")
            model_choice = input("어떤 YOLOv8 세그멘테이션 모델을 사용하시겠습니까? (n/s/m): ").lower() or "n"
    
    # 학습 설정 출력
    print(f"\n===== 학습 설정 =====")
    print(f"이미지 크기: {img_size}px")
    print(f"에폭 수: {epochs}")
    print(f"배치 크기: {batch_size}")
    
    try:
        if continue_from_checkpoint:
            # 기존 모델 경로 설정
            if use_hardcoded_settings:
                model_path = "runs/segment/train/weights/best.pt"  # 하드코딩된 경로
                print(f"기존 모델 경로 (고정값): {model_path}")
            else:
                print("\n기존 모델 경로 입력")
                print("- 일반적으로 'runs/segment/train/weights/best.pt' 또는 'runs/segment/train/weights/last.pt'")
                model_path = input("기존 모델 경로를 입력하세요: ")
            
            print(f"\n===== 기존 모델({model_path})에서 클래스 가중치를 적용하여 세그멘테이션 학습 계속 =====")
            model, training_results = continue_training_seg(model_path, yaml_config, epochs=epochs, batch_size=batch_size, img_size=img_size)
            
            # 모델 평가
            evaluation_results = evaluate_model(model, yaml_config)
            
            # 학습 결과 시각화
            plot_results(training_results)
        else:
            # 모델 선택에 따른 기본 모델 설정
            if model_choice == "n":
                base_model = "yolov8n-seg.pt"
                print("YOLOv8n-seg 모델 선택 (가벼운 모델)")
            elif model_choice == "s":
                base_model = "yolov8s-seg.pt"
                print("YOLOv8s-seg 모델 선택 (중간 크기 모델)")
            elif model_choice == "m":
                base_model = "yolov8m-seg.pt"
                print("YOLOv8m-seg 모델 선택 (무거운 모델)")
            else:
                base_model = "yolov8n-seg.pt"
                print("YOLOv8n-seg 모델 선택 (기본 가벼운 모델)")
            
            # 새 모델로 학습 시작
            print(f"\n===== 클래스 가중치를 적용하여 {base_model} 모델 학습 시작 =====")
            
            # YOLOv8 세그멘테이션 모델 로드
            model = YOLO(base_model)
            
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
                cos_lr=True,        # 코사인 학습률 스케줄링 사용
                warmup_epochs=5,    # 워밍업 에포크 수 증가
                lr0=0.01,           # 초기 학습률
                lrf=0.001,          # 최종 학습률
                weight_decay=0.0005,# 가중치 감쇠
                overlap_mask=True,  # 마스크 오버랩 허용
                close_mosaic=10     # 마지막 10 에포크에서 mosaic 비활성화
            )
            
            # 모델 평가
            evaluation_results = evaluate_model(model, yaml_config)
            
            # 학습 결과 시각화
            plot_results(results)
               
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

    # 최종 모델 저장
    output_path = "yolov8_seg_continued.pt" if continue_from_checkpoint else "yolov8_seg_custom.pt"
    try:
        # 직접 save 메서드 시도
        model.save(output_path)
        print(f"학습된 세그멘테이션 모델이 '{output_path}'로 저장되었습니다.")
    except AttributeError:
        try:
            # 실패하면 export 메서드 시도
            model.export(format="pt", save_dir=os.path.dirname(output_path), 
                       filename=os.path.basename(output_path))
            print(f"학습된 세그멘테이션 모델이 '{output_path}'로 저장되었습니다.")
        except Exception as e:
            # 모두 실패하면 torch.save 사용
            print(f"경고: 일반 저장 방법이 실패했습니다. torch.save 사용 중: {e}")
            torch.save(model.model.state_dict(), output_path)
            print(f"학습된 세그멘테이션 모델이 '{output_path}'로 저장되었습니다 (state_dict 형태).")

if __name__ == "__main__":
   main()