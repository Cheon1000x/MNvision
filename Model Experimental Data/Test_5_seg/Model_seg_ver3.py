<<<<<<< HEAD
<<<<<<< HEAD
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
import argparse
from pathlib import Path
import configparser
import sys

'''
이 코드는 YOLOv8 객체 감지 및 세그멘테이션 모델을 학습하는 코드입니다.
경로 설정을 유연하게 하고, 다양한 환경에서도 쉽게 사용할 수 있도록 설계되었습니다.

주요 기능:
1. 사용자 경로 입력 또는 설정 파일을 통한 경로 설정
2. 다양한 하위 폴더 구조 지원
3. 클래스 분포 분석 및 가중치 계산
4. 자동 데이터 증강 및 학습/검증 데이터 분할
5. 학습 매개변수 조정 기능
6. 모델 평가 및 시각화

모델 학습 결과는 지정한 출력 폴더 또는 기본 'runs/' 폴더에 저장됩니다.
'''

# 설정 관리 클래스
class Config:
    def __init__(self, config_file=None):
        # 기본 설정
        self.config = {
            'paths': {
                'work_dir': 'yolov8_dataset',
                'output_dir': 'runs',
                'data_dir': '',
                'classes_file': 'classes.txt',
                'model_path': ''
            },
            'training': {
                'img_size': 640,
                'batch_size': 16,
                'epochs': 50,
                'train_ratio': 0.8,
                'task': 'detect'  # 'detect' 또는 'segment'
            },
            'model': {
                'model_type': 'n',  # 'n', 's', 'm', 'l', 'x'
                'continue_training': False
            }
        }
        
        # 설정 파일이 제공되면 로드
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """설정 파일에서 설정을 로드합니다."""
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # 로드된 설정을 현재 설정에 병합
            for section, values in loaded_config.items():
                if section in self.config:
                    self.config[section].update(values)
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            print("기본 설정을 사용합니다.")
    
    def save_config(self, config_file):
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"설정이 {config_file}에 저장되었습니다.")
        except Exception as e:
            print(f"설정 파일 저장 중 오류 발생: {e}")
    
    def get(self, section, key, default=None):
        """특정 설정 값을 가져옵니다."""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """특정 설정 값을 설정합니다."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_from_args(self, args):
        """명령줄 인수에서 설정을 업데이트합니다."""
        # args에서 받은 값들로 설정 업데이트
        if hasattr(args, 'work_dir') and args.work_dir:
            self.config['paths']['work_dir'] = args.work_dir
        
        if hasattr(args, 'data_dir') and args.data_dir:
            self.config['paths']['data_dir'] = args.data_dir
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config['paths']['output_dir'] = args.output_dir
        
        if hasattr(args, 'classes_file') and args.classes_file:
            self.config['paths']['classes_file'] = args.classes_file
        
        if hasattr(args, 'model_path') and args.model_path:
            self.config['paths']['model_path'] = args.model_path
            self.config['model']['continue_training'] = True
        
        if hasattr(args, 'img_size') and args.img_size:
            self.config['training']['img_size'] = args.img_size
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['training']['batch_size'] = args.batch_size
        
        if hasattr(args, 'epochs') and args.epochs:
            self.config['training']['epochs'] = args.epochs
        
        if hasattr(args, 'model_type') and args.model_type:
            self.config['model']['model_type'] = args.model_type
        
        if hasattr(args, 'task') and args.task:
            self.config['training']['task'] = args.task
    
    def get_dir_structure(self):
        """디렉토리 구조를 반환합니다."""
        work_dir = self.get('paths', 'work_dir')
        
        return {
            'work_dir': work_dir,
            'images_dir': os.path.join(work_dir, "images"),
            'labels_dir': os.path.join(work_dir, "labels"),
            'train_images_dir': os.path.join(work_dir, "images", "train"),
            'train_labels_dir': os.path.join(work_dir, "labels", "train"),
            'val_images_dir': os.path.join(work_dir, "images", "val"),
            'val_labels_dir': os.path.join(work_dir, "labels", "val")
        }

# 명령줄 인수 파서 설정
def setup_argparse():
    parser = argparse.ArgumentParser(description='YOLOv8 모델 학습 프로그램')
    
    # 경로 관련 인수
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--work_dir', type=str, help='작업 디렉토리 경로')
    parser.add_argument('--data_dir', type=str, help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--classes_file', type=str, help='클래스 파일 경로')
    parser.add_argument('--model_path', type=str, help='기존 모델 경로 (계속 학습할 경우)')
    
    # 학습 관련 인수
    parser.add_argument('--img_size', type=int, help='학습 이미지 크기')
    parser.add_argument('--batch_size', type=int, help='배치 크기')
    parser.add_argument('--epochs', type=int, help='에폭 수')
    parser.add_argument('--model_type', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                        help='모델 유형 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--task', type=str, choices=['detect', 'segment'], 
                        help='작업 유형 (detect=객체 탐지, segment=세그멘테이션)')
    
    # 기타 옵션
    parser.add_argument('--no_input', action='store_true', 
                        help='사용자 입력 없이 설정 파일 또는 기본값 사용')
    parser.add_argument('--save_config', action='store_true', 
                        help='현재 설정을 파일로 저장')
    
    return parser.parse_args()

# 디렉토리 설정 함수
def setup_directories(config):
    """작업 디렉토리 구조를 설정합니다."""
    dirs = config.get_dir_structure()
    
    # 각 디렉토리가 존재하지 않으면 생성
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return dirs

# 클래스 빈도 계산 함수
def calculate_class_frequency(json_files, frame_files, classes_file):
    """모든 데이터에서 클래스별 출현 빈도를 계산합니다."""
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
    """클래스별 가중치를 계산합니다 (희소 클래스에 더 큰 가중치)."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (객체 탐지용)
def convert_json_to_yolo_detection(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 객체 탐지 형식으로 변환합니다."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (세그멘테이션용)
def convert_json_to_yolo_segmentation(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 세그멘테이션 형식으로 변환합니다."""
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

# 입력 데이터 처리 함수
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir, 
                 task='detect', apply_aug=False, class_counts=None):
    """
    입력 데이터를 처리하고 YOLO 형식으로 변환하여 저장합니다.
    
    Args:
        task: 'detect' 또는 'segment' (객체 탐지 또는 세그멘테이션)
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 형식으로 라벨 변환 (task에 따라 다른 함수 호출)
    if task == 'detect':
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_detection(
            json_file, frame_file, classes_file)
    else:  # task == 'segment'
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_segmentation(
            json_file, frame_file, classes_file)
    
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
        
        # 결정된 횟수만큼 증강 적용 (클래스 균형 맞추기)
        for aug_idx in range(aug_count):
            # 이미지와 라벨 파일명 설정
            aug_image_filename = f"{base_filename}_aug{aug_idx}.jpg"
            aug_label_filename = f"{base_filename}_aug{aug_idx}.txt"
            
            # 파일 경로 설정
            aug_image_path = os.path.join(output_image_dir, aug_image_filename)
            aug_label_path = os.path.join(output_label_dir, aug_label_filename)
            
            # 세그멘테이션 또는 객체 탐지에 따라 처리 방식 선택
            if task == 'segment':
                # 세그멘테이션의 경우 단순히 원본 복사 (간단한 증강)
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
            else:
                # 객체 탐지의 경우 이미지 자체 변형 적용 (고급 증강)
                # 여기서는 간단한 예시로, 실제로는 albumentations 등의 라이브러리 사용 권장
                # TODO: 고급 증강 기능 구현
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
    
    return output_image_path, output_label_path

# 데이터셋 분할 함수
def split_dataset(json_files, frame_files, classes_file, dirs, config):
    """데이터셋을 학습 및 검증 세트로 분할합니다."""
    task = config.get('training', 'task')
    train_ratio = config.get('training', 'train_ratio')
    
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
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['train_images_dir'], dirs['train_labels_dir'],
            task=task, apply_aug=True, class_counts=class_counts
        )
    
    for i in tqdm(val_indices, desc="검증 데이터 처리 중"):
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['val_images_dir'], dirs['val_labels_dir'],
            task=task, apply_aug=False
        )
    
    # 데이터셋 개수 계산
    train_images = len([f for f in os.listdir(dirs['train_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir(dirs['val_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"데이터셋 분할 완료: {train_images}개 학습 샘플 (원본 {len(train_indices)}개 + 증강 {train_images - len(train_indices)}개), {val_images}개 검증 샘플")
    
    return class_counts, class_weights

# YAML 설정 파일 생성 함수
def create_yaml_config(classes_file, output_yaml, class_weights=None, config=None):
    """YOLO 학습을 위한 YAML 설정 파일을 생성합니다."""
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 클래스 가중치 문자열 생성
    weights_str = ""
    if class_weights:
        max_cls_id = max(class_weights.keys())
        weights_list = [class_weights.get(i, 1.0) for i in range(max_cls_id + 1)]
        weights_str = "[" + ", ".join([f"{w:.2f}" for w in weights_list]) + "]"
    
    # 작업 디렉토리 경로 가져오기
    work_dir = os.path.abspath(config.get('paths', 'work_dir'))
    
    # 작업 유형 가져오기
    task = config.get('training', 'task')
    
    # YAML 파일 생성
    yaml_content = f"""
path: {work_dir}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

    # 세그멘테이션 작업인 경우 task 정보 추가
    if task == 'segment':
        yaml_content += "\n# 세그멘테이션 설정\ntask: segment\n"
    
    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")
    if class_weights:
        print("클래스 가중치가 YAML 설정에 추가되었습니다.")

# 모델 학습 함수
def train_model(yaml_config, config):
    """YOLO 모델을 학습합니다."""
    # 학습 설정 가져오기
    task = config.get('training', 'task')
    epochs = config.get('training', 'epochs')
    batch_size = config.get('training', 'batch_size')
    img_size = config.get('training', 'img_size')
    model_type = config.get('model', 'model_type')
    continue_training = config.get('model', 'continue_training')
    model_path = config.get('paths', 'model_path')
    
    # 모델 유형 결정
    if continue_training and model_path:
        # 기존 모델에서 계속 학습
        model = YOLO(model_path)
        print(f"기존 모델({model_path})에서 학습을 계속합니다.")
    else:
        # 새 모델로 학습 시작
        if task == 'segment':
            # 세그멘테이션 모델
            model_name = f"yolov8{model_type}-seg.pt"
        else:
            # 객체 탐지 모델
            model_name = f"yolov8{model_type}.pt"
        
        model = YOLO(model_name)
        print(f"새 모델({model_name})로 학습을 시작합니다.")

    # 공통 학습 매개변수
    train_args = {
        'data': yaml_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'patience': 20,
        'verbose': True,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'save_period': 5,
        'cos_lr': True,
        'warmup_epochs': 5 if not continue_training else 3,
        'lr0': 0.01 if not continue_training else 0.001,
        'lrf': 0.001 if not continue_training else 0.0001,
        'weight_decay': 0.0005,
        'overlap_mask': True,
        'close_mosaic': 10
    }
    
    # 학습 시작
    results = model.train(**train_args)
    
    return model, results

# 모델 평가 함수
def evaluate_model(model, yaml_config, task='detect'):
    """학습된 모델을 평가합니다."""
    # 모델 검증
    results = model.val(data=yaml_config)
    
    # 주요 메트릭 출력
    print("\n===== 모델 평가 결과 =====")
    
    if task == 'segment':
        # 세그멘테이션 모델 메트릭
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
    else:
        # 객체 탐지 모델 메트릭
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'Precision': results.box.p,
            'Recall': results.box.r,
            'F1-Score': results.box.f1
        }
    
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
def plot_results(results, task='detect'):
    """학습 결과를 시각화합니다."""
    try:
        # 결과 시각화 (Ultralytics YOLO는 자동으로 plots 폴더에 그래프를 저장합니다)
        print(f"\n학습 결과가 시각화되었습니다. 'runs/{task}/train/results.png'에서 확인하세요.")
        
        # 만약 results 객체에 plot_results 메서드가 있다면 사용
        if hasattr(results, 'plot_results'):
            fig = results.plot_results(show=False)
            fig.savefig("training_results.png")
            print("학습 결과 그래프가 'training_results.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"학습 결과 시각화 중 오류 발생: {e}")
        print(f"기본 YOLO 시각화는 'runs/{task}/train/' 폴더에서 확인하세요.")

# 클래스 분포 시각화 함수
def visualize_class_distribution(class_counts):
    """클래스 분포와 가중치를 시각화합니다."""
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

# 경로 찾기 함수
def find_classes_file(base_path, subfolders):
    """클래스 파일을 찾습니다."""
    classes_file = "classes.txt"
    
    # 기본 경로에서 찾기
    classes_file_path = os.path.join(base_path, classes_file)
    if os.path.exists(classes_file_path):
        return classes_file_path
    
    # 하위 폴더에서 찾기
    for folder in subfolders:
        path = os.path.join(folder, classes_file)
        if os.path.exists(path):
            return path
    
    # 현재 스크립트 폴더에서 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, classes_file)
    if os.path.exists(path):
        return path
    
    return None

# 사용자 입력 처리 함수
def get_user_input(config, skip_input=False):
    """사용자로부터 설정 값을 입력받습니다."""
    if skip_input:
        return config
    
    # 데이터 경로 입력
    print("\n===== 데이터 폴더 경로 입력 =====")
    print("데이터 폴더는 이미지와 JSON 파일이 있는 폴더여야 합니다.")
    print("예시: C:\\Users\\Username\\Project\\data")
    data_dir = input(f"데이터 폴더 경로 [{config.get('paths', 'data_dir') or '.'}]: ").strip()
    if data_dir:
        config.set('paths', 'data_dir', data_dir)
    
    # 작업 유형 선택
    print("\n===== 작업 유형 선택 =====")
    print("1: 객체 탐지 (Object Detection)")
    print("2: 세그멘테이션 (Segmentation)")
    task_choice = input(f"작업 유형 선택 (1/2) [{'1' if config.get('training', 'task') == 'detect' else '2'}]: ").strip()
    if task_choice == '1':
        config.set('training', 'task', 'detect')
    elif task_choice == '2':
        config.set('training', 'task', 'segment')
    
    # 학습 방식 선택
    print("\n===== 학습 방식 선택 =====")
    print("1: 새 모델 학습")
    print("2: 기존 모델 계속 학습")
    train_choice = input(f"학습 방식 선택 (1/2) [{'2' if config.get('model', 'continue_training') else '1'}]: ").strip()
    
    if train_choice == '2':
        config.set('model', 'continue_training', True)
        print("\n===== 기존 모델 경로 입력 =====")
        print("예시: runs/detect/train/weights/best.pt")
        model_path = input(f"모델 경로 [{config.get('paths', 'model_path') or 'runs/detect/train/weights/best.pt'}]: ").strip()
        if model_path:
            config.set('paths', 'model_path', model_path)
    else:
        config.set('model', 'continue_training', False)
        
        # 모델 유형 선택 (새 모델 학습 시에만)
        print("\n===== 모델 유형 선택 =====")
        print("n: YOLOv8n (가벼운 모델, 빠르지만 정확도 낮음)")
        print("s: YOLOv8s (소형 모델)")
        print("m: YOLOv8m (중형 모델, 균형 잡힌 속도와 정확도)")
        print("l: YOLOv8l (대형 모델)")
        print("x: YOLOv8x (초대형 모델, 정확도 높지만 느림)")
        model_type = input(f"모델 유형 선택 (n/s/m/l/x) [{config.get('model', 'model_type')}]: ").strip().lower()
        if model_type in ['n', 's', 'm', 'l', 'x']:
            config.set('model', 'model_type', model_type)
    
    # 고급 설정 여부
    advanced_config = input("\n고급 설정을 변경하시겠습니까? (y/n) [n]: ").strip().lower() == 'y'
    
    if advanced_config:
        # 이미지 크기 설정
        print("\n===== 이미지 크기 설정 =====")
        print("- 작은 크기(320-416): 속도 빠름, 정확도 낮음")
        print("- 중간 크기(640): 균형 잡힌 속도와 정확도")
        print("- 큰 크기(832-1024): 속도 느림, 정확도 높음")
        img_size = input(f"이미지 크기 [{config.get('training', 'img_size')}]: ").strip()
        if img_size and img_size.isdigit():
            config.set('training', 'img_size', int(img_size))
        
        # 배치 크기 설정
        print("\n===== 배치 크기 설정 =====")
        print("- 작은 배치(4-8): 적은 GPU 메모리 필요, 느린 학습")
        print("- 중간 배치(16-32): 균형 잡힌 속도와 메모리 사용량")
        print("- 큰 배치(64+): 빠른 학습, 많은 GPU 메모리 필요")
        batch_size = input(f"배치 크기 [{config.get('training', 'batch_size')}]: ").strip()
        if batch_size and batch_size.isdigit():
            config.set('training', 'batch_size', int(batch_size))
        
        # 에폭 수 설정
        print("\n===== 에폭 수 설정 =====")
        print("- 적은 에폭(10-30): 빠른 학습, 낮은 정확도")
        print("- 중간 에폭(50-100): 균형 잡힌 학습 시간과 정확도")
        print("- 많은 에폭(100+): 오랜 학습 시간, 잠재적으로 더 높은 정확도")
        epochs = input(f"에폭 수 [{config.get('training', 'epochs')}]: ").strip()
        if epochs and epochs.isdigit():
            config.set('training', 'epochs', int(epochs))
        
        # 작업 디렉토리 설정
        print("\n===== 작업 디렉토리 설정 =====")
        print("학습 데이터와 라벨이 저장될 디렉토리입니다.")
        work_dir = input(f"작업 디렉토리 [{config.get('paths', 'work_dir')}]: ").strip()
        if work_dir:
            config.set('paths', 'work_dir', work_dir)
    
    return config

# 메인 함수
def main():
    # 시작 메시지 출력
    print("===== YOLO 모델 학습 프로그램 =====")
    print("이 프로그램은 YOLOv8 모델을 학습, 평가, 테스트하는 기능을 제공합니다.")
    
    # 명령줄 인수 파싱
    args = setup_argparse()
    
    # 설정 로드
    config = Config(args.config)
    
    # 명령줄 인수로 설정 업데이트
    config.update_from_args(args)
    
    # 사용자 입력으로 설정 업데이트 (--no_input 옵션이 없을 경우)
    if not args.no_input:
        config = get_user_input(config)
    
    # 설정 저장 (--save_config 옵션이 있을 경우)
    if args.save_config:
        config_file = args.config or "yolo_train_config.json"
        config.save_config(config_file)
    
    # PyTorch CUDA 사용 가능 여부 확인
    print(f"\n===== 시스템 정보 =====")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU가 감지되지 않았습니다. CPU를 사용하여 학습합니다.")
    
    # 작업 디렉토리 설정
    dirs = setup_directories(config)
    
    # 데이터 경로 가져오기
    data_dir = config.get('paths', 'data_dir')
    
    # 경로 존재 확인
    if not os.path.exists(data_dir):
        print(f"오류: 지정한 데이터 경로 '{data_dir}'가 존재하지 않습니다.")
        return
    
    try:
        # 데이터가 포함된 모든 하위 폴더 찾기
        subfolders = []
        if os.path.isdir(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    subfolders.append(folder_path)
            
            # 하위 폴더가 없으면 기본 폴더를 직접 사용
            if not subfolders:
                print("하위 폴더가 없습니다. 기본 폴더를 직접 사용합니다.")
                subfolders = [data_dir]
        else:
            print(f"오류: '{data_dir}'는 폴더가 아닙니다.")
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
    
    # classes.txt 파일 확인 및 찾기
    classes_file_path = config.get('paths', 'classes_file') or find_classes_file(data_dir, subfolders)
    
    if not classes_file_path or not os.path.exists(classes_file_path):
        print("classes.txt 파일을 찾을 수 없습니다.")
        classes_file_path = input("classes.txt 파일의 전체 경로를 입력하세요: ").strip()
        
        if not os.path.exists(classes_file_path):
            print(f"오류: 지정한 classes.txt 파일 '{classes_file_path}'가 존재하지 않습니다.")
            return
    
    # 설정에 classes_file 경로 저장
    config.set('paths', 'classes_file', classes_file_path)
    
    print(f"총 {len(all_json_files)}개의 파일 쌍을 찾았습니다.")
    print(f"Classes 파일: {classes_file_path}")
    
    # 데이터셋 분할 및 처리
    print("\n===== 데이터셋 분할 및 처리 =====")
    class_counts, class_weights = split_dataset(all_json_files, all_frame_files, classes_file_path, dirs, config)
    
    # 클래스 분포 시각화
    visualize_class_distribution(class_counts)
    
    # YAML 설정 파일 생성
    yaml_config = os.path.join(dirs['work_dir'], "dataset.yaml")
    create_yaml_config(classes_file_path, yaml_config, class_weights, config)
    
    # 학습 시작
    print("\n===== 모델 학습 시작 =====")
    model, results = train_model(yaml_config, config)
    
    # 모델 평가
    print("\n===== 모델 평가 =====")
    task = config.get('training', 'task')
    evaluation_results = evaluate_model(model, yaml_config, task)
    
    # 결과 시각화
    plot_results(results, task)
    
    # 모델 저장
    output_name = "yolov8_continued.pt" if config.get('model', 'continue_training') else "yolov8_custom.pt"
    if task == 'segment':
        output_name = output_name.replace('.pt', '_seg.pt')
    
    output_path = os.path.join(os.getcwd(), output_name)
    
    try:
        # 모델 저장
        model.save(output_path)
        print(f"\n학습된 모델이 '{output_path}'로 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
        try:
            # 다른 방법으로 저장 시도
            model.export(format="pt", save_dir=os.path.dirname(output_path), 
                       filename=os.path.basename(output_path))
            print(f"학습된 모델이 '{output_path}'로 저장되었습니다.")
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    print("\n===== 학습 완료 =====")
    print(f"학습 결과는 '{os.path.join('runs', task, 'train')}' 폴더에서 확인할 수 있습니다.")
    print(f"최종 모델 파일: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {e}")
        import traceback
=======
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
import argparse
from pathlib import Path
import configparser
import sys

'''
이 코드는 YOLOv8 객체 감지 및 세그멘테이션 모델을 학습하는 코드입니다.
경로 설정을 유연하게 하고, 다양한 환경에서도 쉽게 사용할 수 있도록 설계되었습니다.

주요 기능:
1. 사용자 경로 입력 또는 설정 파일을 통한 경로 설정
2. 다양한 하위 폴더 구조 지원
3. 클래스 분포 분석 및 가중치 계산
4. 자동 데이터 증강 및 학습/검증 데이터 분할
5. 학습 매개변수 조정 기능
6. 모델 평가 및 시각화

모델 학습 결과는 지정한 출력 폴더 또는 기본 'runs/' 폴더에 저장됩니다.
'''

# 설정 관리 클래스
class Config:
    def __init__(self, config_file=None):
        # 기본 설정
        self.config = {
            'paths': {
                'work_dir': 'yolov8_dataset',
                'output_dir': 'runs',
                'data_dir': '',
                'classes_file': 'classes.txt',
                'model_path': ''
            },
            'training': {
                'img_size': 640,
                'batch_size': 16,
                'epochs': 50,
                'train_ratio': 0.8,
                'task': 'detect'  # 'detect' 또는 'segment'
            },
            'model': {
                'model_type': 'n',  # 'n', 's', 'm', 'l', 'x'
                'continue_training': False
            }
        }
        
        # 설정 파일이 제공되면 로드
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """설정 파일에서 설정을 로드합니다."""
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # 로드된 설정을 현재 설정에 병합
            for section, values in loaded_config.items():
                if section in self.config:
                    self.config[section].update(values)
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            print("기본 설정을 사용합니다.")
    
    def save_config(self, config_file):
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"설정이 {config_file}에 저장되었습니다.")
        except Exception as e:
            print(f"설정 파일 저장 중 오류 발생: {e}")
    
    def get(self, section, key, default=None):
        """특정 설정 값을 가져옵니다."""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """특정 설정 값을 설정합니다."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_from_args(self, args):
        """명령줄 인수에서 설정을 업데이트합니다."""
        # args에서 받은 값들로 설정 업데이트
        if hasattr(args, 'work_dir') and args.work_dir:
            self.config['paths']['work_dir'] = args.work_dir
        
        if hasattr(args, 'data_dir') and args.data_dir:
            self.config['paths']['data_dir'] = args.data_dir
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config['paths']['output_dir'] = args.output_dir
        
        if hasattr(args, 'classes_file') and args.classes_file:
            self.config['paths']['classes_file'] = args.classes_file
        
        if hasattr(args, 'model_path') and args.model_path:
            self.config['paths']['model_path'] = args.model_path
            self.config['model']['continue_training'] = True
        
        if hasattr(args, 'img_size') and args.img_size:
            self.config['training']['img_size'] = args.img_size
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['training']['batch_size'] = args.batch_size
        
        if hasattr(args, 'epochs') and args.epochs:
            self.config['training']['epochs'] = args.epochs
        
        if hasattr(args, 'model_type') and args.model_type:
            self.config['model']['model_type'] = args.model_type
        
        if hasattr(args, 'task') and args.task:
            self.config['training']['task'] = args.task
    
    def get_dir_structure(self):
        """디렉토리 구조를 반환합니다."""
        work_dir = self.get('paths', 'work_dir')
        
        return {
            'work_dir': work_dir,
            'images_dir': os.path.join(work_dir, "images"),
            'labels_dir': os.path.join(work_dir, "labels"),
            'train_images_dir': os.path.join(work_dir, "images", "train"),
            'train_labels_dir': os.path.join(work_dir, "labels", "train"),
            'val_images_dir': os.path.join(work_dir, "images", "val"),
            'val_labels_dir': os.path.join(work_dir, "labels", "val")
        }

# 명령줄 인수 파서 설정
def setup_argparse():
    parser = argparse.ArgumentParser(description='YOLOv8 모델 학습 프로그램')
    
    # 경로 관련 인수
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--work_dir', type=str, help='작업 디렉토리 경로')
    parser.add_argument('--data_dir', type=str, help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--classes_file', type=str, help='클래스 파일 경로')
    parser.add_argument('--model_path', type=str, help='기존 모델 경로 (계속 학습할 경우)')
    
    # 학습 관련 인수
    parser.add_argument('--img_size', type=int, help='학습 이미지 크기')
    parser.add_argument('--batch_size', type=int, help='배치 크기')
    parser.add_argument('--epochs', type=int, help='에폭 수')
    parser.add_argument('--model_type', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                        help='모델 유형 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--task', type=str, choices=['detect', 'segment'], 
                        help='작업 유형 (detect=객체 탐지, segment=세그멘테이션)')
    
    # 기타 옵션
    parser.add_argument('--no_input', action='store_true', 
                        help='사용자 입력 없이 설정 파일 또는 기본값 사용')
    parser.add_argument('--save_config', action='store_true', 
                        help='현재 설정을 파일로 저장')
    
    return parser.parse_args()

# 디렉토리 설정 함수
def setup_directories(config):
    """작업 디렉토리 구조를 설정합니다."""
    dirs = config.get_dir_structure()
    
    # 각 디렉토리가 존재하지 않으면 생성
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return dirs

# 클래스 빈도 계산 함수
def calculate_class_frequency(json_files, frame_files, classes_file):
    """모든 데이터에서 클래스별 출현 빈도를 계산합니다."""
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
    """클래스별 가중치를 계산합니다 (희소 클래스에 더 큰 가중치)."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (객체 탐지용)
def convert_json_to_yolo_detection(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 객체 탐지 형식으로 변환합니다."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (세그멘테이션용)
def convert_json_to_yolo_segmentation(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 세그멘테이션 형식으로 변환합니다."""
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

# 입력 데이터 처리 함수
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir, 
                 task='detect', apply_aug=False, class_counts=None):
    """
    입력 데이터를 처리하고 YOLO 형식으로 변환하여 저장합니다.
    
    Args:
        task: 'detect' 또는 'segment' (객체 탐지 또는 세그멘테이션)
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 형식으로 라벨 변환 (task에 따라 다른 함수 호출)
    if task == 'detect':
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_detection(
            json_file, frame_file, classes_file)
    else:  # task == 'segment'
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_segmentation(
            json_file, frame_file, classes_file)
    
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
        
        # 결정된 횟수만큼 증강 적용 (클래스 균형 맞추기)
        for aug_idx in range(aug_count):
            # 이미지와 라벨 파일명 설정
            aug_image_filename = f"{base_filename}_aug{aug_idx}.jpg"
            aug_label_filename = f"{base_filename}_aug{aug_idx}.txt"
            
            # 파일 경로 설정
            aug_image_path = os.path.join(output_image_dir, aug_image_filename)
            aug_label_path = os.path.join(output_label_dir, aug_label_filename)
            
            # 세그멘테이션 또는 객체 탐지에 따라 처리 방식 선택
            if task == 'segment':
                # 세그멘테이션의 경우 단순히 원본 복사 (간단한 증강)
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
            else:
                # 객체 탐지의 경우 이미지 자체 변형 적용 (고급 증강)
                # 여기서는 간단한 예시로, 실제로는 albumentations 등의 라이브러리 사용 권장
                # TODO: 고급 증강 기능 구현
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
    
    return output_image_path, output_label_path

# 데이터셋 분할 함수
def split_dataset(json_files, frame_files, classes_file, dirs, config):
    """데이터셋을 학습 및 검증 세트로 분할합니다."""
    task = config.get('training', 'task')
    train_ratio = config.get('training', 'train_ratio')
    
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
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['train_images_dir'], dirs['train_labels_dir'],
            task=task, apply_aug=True, class_counts=class_counts
        )
    
    for i in tqdm(val_indices, desc="검증 데이터 처리 중"):
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['val_images_dir'], dirs['val_labels_dir'],
            task=task, apply_aug=False
        )
    
    # 데이터셋 개수 계산
    train_images = len([f for f in os.listdir(dirs['train_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir(dirs['val_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"데이터셋 분할 완료: {train_images}개 학습 샘플 (원본 {len(train_indices)}개 + 증강 {train_images - len(train_indices)}개), {val_images}개 검증 샘플")
    
    return class_counts, class_weights

# YAML 설정 파일 생성 함수
def create_yaml_config(classes_file, output_yaml, class_weights=None, config=None):
    """YOLO 학습을 위한 YAML 설정 파일을 생성합니다."""
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 클래스 가중치 문자열 생성
    weights_str = ""
    if class_weights:
        max_cls_id = max(class_weights.keys())
        weights_list = [class_weights.get(i, 1.0) for i in range(max_cls_id + 1)]
        weights_str = "[" + ", ".join([f"{w:.2f}" for w in weights_list]) + "]"
    
    # 작업 디렉토리 경로 가져오기
    work_dir = os.path.abspath(config.get('paths', 'work_dir'))
    
    # 작업 유형 가져오기
    task = config.get('training', 'task')
    
    # YAML 파일 생성
    yaml_content = f"""
path: {work_dir}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

    # 세그멘테이션 작업인 경우 task 정보 추가
    if task == 'segment':
        yaml_content += "\n# 세그멘테이션 설정\ntask: segment\n"
    
    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")
    if class_weights:
        print("클래스 가중치가 YAML 설정에 추가되었습니다.")

# 모델 학습 함수
def train_model(yaml_config, config):
    """YOLO 모델을 학습합니다."""
    # 학습 설정 가져오기
    task = config.get('training', 'task')
    epochs = config.get('training', 'epochs')
    batch_size = config.get('training', 'batch_size')
    img_size = config.get('training', 'img_size')
    model_type = config.get('model', 'model_type')
    continue_training = config.get('model', 'continue_training')
    model_path = config.get('paths', 'model_path')
    
    # 모델 유형 결정
    if continue_training and model_path:
        # 기존 모델에서 계속 학습
        model = YOLO(model_path)
        print(f"기존 모델({model_path})에서 학습을 계속합니다.")
    else:
        # 새 모델로 학습 시작
        if task == 'segment':
            # 세그멘테이션 모델
            model_name = f"yolov8{model_type}-seg.pt"
        else:
            # 객체 탐지 모델
            model_name = f"yolov8{model_type}.pt"
        
        model = YOLO(model_name)
        print(f"새 모델({model_name})로 학습을 시작합니다.")

    # 공통 학습 매개변수
    train_args = {
        'data': yaml_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'patience': 20,
        'verbose': True,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'save_period': 5,
        'cos_lr': True,
        'warmup_epochs': 5 if not continue_training else 3,
        'lr0': 0.01 if not continue_training else 0.001,
        'lrf': 0.001 if not continue_training else 0.0001,
        'weight_decay': 0.0005,
        'overlap_mask': True,
        'close_mosaic': 10
    }
    
    # 학습 시작
    results = model.train(**train_args)
    
    return model, results

# 모델 평가 함수
def evaluate_model(model, yaml_config, task='detect'):
    """학습된 모델을 평가합니다."""
    # 모델 검증
    results = model.val(data=yaml_config)
    
    # 주요 메트릭 출력
    print("\n===== 모델 평가 결과 =====")
    
    if task == 'segment':
        # 세그멘테이션 모델 메트릭
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
    else:
        # 객체 탐지 모델 메트릭
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'Precision': results.box.p,
            'Recall': results.box.r,
            'F1-Score': results.box.f1
        }
    
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
def plot_results(results, task='detect'):
    """학습 결과를 시각화합니다."""
    try:
        # 결과 시각화 (Ultralytics YOLO는 자동으로 plots 폴더에 그래프를 저장합니다)
        print(f"\n학습 결과가 시각화되었습니다. 'runs/{task}/train/results.png'에서 확인하세요.")
        
        # 만약 results 객체에 plot_results 메서드가 있다면 사용
        if hasattr(results, 'plot_results'):
            fig = results.plot_results(show=False)
            fig.savefig("training_results.png")
            print("학습 결과 그래프가 'training_results.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"학습 결과 시각화 중 오류 발생: {e}")
        print(f"기본 YOLO 시각화는 'runs/{task}/train/' 폴더에서 확인하세요.")

# 클래스 분포 시각화 함수
def visualize_class_distribution(class_counts):
    """클래스 분포와 가중치를 시각화합니다."""
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

# 경로 찾기 함수
def find_classes_file(base_path, subfolders):
    """클래스 파일을 찾습니다."""
    classes_file = "classes.txt"
    
    # 기본 경로에서 찾기
    classes_file_path = os.path.join(base_path, classes_file)
    if os.path.exists(classes_file_path):
        return classes_file_path
    
    # 하위 폴더에서 찾기
    for folder in subfolders:
        path = os.path.join(folder, classes_file)
        if os.path.exists(path):
            return path
    
    # 현재 스크립트 폴더에서 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, classes_file)
    if os.path.exists(path):
        return path
    
    return None

# 사용자 입력 처리 함수
def get_user_input(config, skip_input=False):
    """사용자로부터 설정 값을 입력받습니다."""
    if skip_input:
        return config
    
    # 데이터 경로 입력
    print("\n===== 데이터 폴더 경로 입력 =====")
    print("데이터 폴더는 이미지와 JSON 파일이 있는 폴더여야 합니다.")
    print("예시: C:\\Users\\Username\\Project\\data")
    data_dir = input(f"데이터 폴더 경로 [{config.get('paths', 'data_dir') or '.'}]: ").strip()
    if data_dir:
        config.set('paths', 'data_dir', data_dir)
    
    # 작업 유형 선택
    print("\n===== 작업 유형 선택 =====")
    print("1: 객체 탐지 (Object Detection)")
    print("2: 세그멘테이션 (Segmentation)")
    task_choice = input(f"작업 유형 선택 (1/2) [{'1' if config.get('training', 'task') == 'detect' else '2'}]: ").strip()
    if task_choice == '1':
        config.set('training', 'task', 'detect')
    elif task_choice == '2':
        config.set('training', 'task', 'segment')
    
    # 학습 방식 선택
    print("\n===== 학습 방식 선택 =====")
    print("1: 새 모델 학습")
    print("2: 기존 모델 계속 학습")
    train_choice = input(f"학습 방식 선택 (1/2) [{'2' if config.get('model', 'continue_training') else '1'}]: ").strip()
    
    if train_choice == '2':
        config.set('model', 'continue_training', True)
        print("\n===== 기존 모델 경로 입력 =====")
        print("예시: runs/detect/train/weights/best.pt")
        model_path = input(f"모델 경로 [{config.get('paths', 'model_path') or 'runs/detect/train/weights/best.pt'}]: ").strip()
        if model_path:
            config.set('paths', 'model_path', model_path)
    else:
        config.set('model', 'continue_training', False)
        
        # 모델 유형 선택 (새 모델 학습 시에만)
        print("\n===== 모델 유형 선택 =====")
        print("n: YOLOv8n (가벼운 모델, 빠르지만 정확도 낮음)")
        print("s: YOLOv8s (소형 모델)")
        print("m: YOLOv8m (중형 모델, 균형 잡힌 속도와 정확도)")
        print("l: YOLOv8l (대형 모델)")
        print("x: YOLOv8x (초대형 모델, 정확도 높지만 느림)")
        model_type = input(f"모델 유형 선택 (n/s/m/l/x) [{config.get('model', 'model_type')}]: ").strip().lower()
        if model_type in ['n', 's', 'm', 'l', 'x']:
            config.set('model', 'model_type', model_type)
    
    # 고급 설정 여부
    advanced_config = input("\n고급 설정을 변경하시겠습니까? (y/n) [n]: ").strip().lower() == 'y'
    
    if advanced_config:
        # 이미지 크기 설정
        print("\n===== 이미지 크기 설정 =====")
        print("- 작은 크기(320-416): 속도 빠름, 정확도 낮음")
        print("- 중간 크기(640): 균형 잡힌 속도와 정확도")
        print("- 큰 크기(832-1024): 속도 느림, 정확도 높음")
        img_size = input(f"이미지 크기 [{config.get('training', 'img_size')}]: ").strip()
        if img_size and img_size.isdigit():
            config.set('training', 'img_size', int(img_size))
        
        # 배치 크기 설정
        print("\n===== 배치 크기 설정 =====")
        print("- 작은 배치(4-8): 적은 GPU 메모리 필요, 느린 학습")
        print("- 중간 배치(16-32): 균형 잡힌 속도와 메모리 사용량")
        print("- 큰 배치(64+): 빠른 학습, 많은 GPU 메모리 필요")
        batch_size = input(f"배치 크기 [{config.get('training', 'batch_size')}]: ").strip()
        if batch_size and batch_size.isdigit():
            config.set('training', 'batch_size', int(batch_size))
        
        # 에폭 수 설정
        print("\n===== 에폭 수 설정 =====")
        print("- 적은 에폭(10-30): 빠른 학습, 낮은 정확도")
        print("- 중간 에폭(50-100): 균형 잡힌 학습 시간과 정확도")
        print("- 많은 에폭(100+): 오랜 학습 시간, 잠재적으로 더 높은 정확도")
        epochs = input(f"에폭 수 [{config.get('training', 'epochs')}]: ").strip()
        if epochs and epochs.isdigit():
            config.set('training', 'epochs', int(epochs))
        
        # 작업 디렉토리 설정
        print("\n===== 작업 디렉토리 설정 =====")
        print("학습 데이터와 라벨이 저장될 디렉토리입니다.")
        work_dir = input(f"작업 디렉토리 [{config.get('paths', 'work_dir')}]: ").strip()
        if work_dir:
            config.set('paths', 'work_dir', work_dir)
    
    return config

# 메인 함수
def main():
    # 시작 메시지 출력
    print("===== YOLO 모델 학습 프로그램 =====")
    print("이 프로그램은 YOLOv8 모델을 학습, 평가, 테스트하는 기능을 제공합니다.")
    
    # 명령줄 인수 파싱
    args = setup_argparse()
    
    # 설정 로드
    config = Config(args.config)
    
    # 명령줄 인수로 설정 업데이트
    config.update_from_args(args)
    
    # 사용자 입력으로 설정 업데이트 (--no_input 옵션이 없을 경우)
    if not args.no_input:
        config = get_user_input(config)
    
    # 설정 저장 (--save_config 옵션이 있을 경우)
    if args.save_config:
        config_file = args.config or "yolo_train_config.json"
        config.save_config(config_file)
    
    # PyTorch CUDA 사용 가능 여부 확인
    print(f"\n===== 시스템 정보 =====")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU가 감지되지 않았습니다. CPU를 사용하여 학습합니다.")
    
    # 작업 디렉토리 설정
    dirs = setup_directories(config)
    
    # 데이터 경로 가져오기
    data_dir = config.get('paths', 'data_dir')
    
    # 경로 존재 확인
    if not os.path.exists(data_dir):
        print(f"오류: 지정한 데이터 경로 '{data_dir}'가 존재하지 않습니다.")
        return
    
    try:
        # 데이터가 포함된 모든 하위 폴더 찾기
        subfolders = []
        if os.path.isdir(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    subfolders.append(folder_path)
            
            # 하위 폴더가 없으면 기본 폴더를 직접 사용
            if not subfolders:
                print("하위 폴더가 없습니다. 기본 폴더를 직접 사용합니다.")
                subfolders = [data_dir]
        else:
            print(f"오류: '{data_dir}'는 폴더가 아닙니다.")
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
    
    # classes.txt 파일 확인 및 찾기
    classes_file_path = config.get('paths', 'classes_file') or find_classes_file(data_dir, subfolders)
    
    if not classes_file_path or not os.path.exists(classes_file_path):
        print("classes.txt 파일을 찾을 수 없습니다.")
        classes_file_path = input("classes.txt 파일의 전체 경로를 입력하세요: ").strip()
        
        if not os.path.exists(classes_file_path):
            print(f"오류: 지정한 classes.txt 파일 '{classes_file_path}'가 존재하지 않습니다.")
            return
    
    # 설정에 classes_file 경로 저장
    config.set('paths', 'classes_file', classes_file_path)
    
    print(f"총 {len(all_json_files)}개의 파일 쌍을 찾았습니다.")
    print(f"Classes 파일: {classes_file_path}")
    
    # 데이터셋 분할 및 처리
    print("\n===== 데이터셋 분할 및 처리 =====")
    class_counts, class_weights = split_dataset(all_json_files, all_frame_files, classes_file_path, dirs, config)
    
    # 클래스 분포 시각화
    visualize_class_distribution(class_counts)
    
    # YAML 설정 파일 생성
    yaml_config = os.path.join(dirs['work_dir'], "dataset.yaml")
    create_yaml_config(classes_file_path, yaml_config, class_weights, config)
    
    # 학습 시작
    print("\n===== 모델 학습 시작 =====")
    model, results = train_model(yaml_config, config)
    
    # 모델 평가
    print("\n===== 모델 평가 =====")
    task = config.get('training', 'task')
    evaluation_results = evaluate_model(model, yaml_config, task)
    
    # 결과 시각화
    plot_results(results, task)
    
    # 모델 저장
    output_name = "yolov8_continued.pt" if config.get('model', 'continue_training') else "yolov8_custom.pt"
    if task == 'segment':
        output_name = output_name.replace('.pt', '_seg.pt')
    
    output_path = os.path.join(os.getcwd(), output_name)
    
    try:
        # 모델 저장
        model.save(output_path)
        print(f"\n학습된 모델이 '{output_path}'로 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
        try:
            # 다른 방법으로 저장 시도
            model.export(format="pt", save_dir=os.path.dirname(output_path), 
                       filename=os.path.basename(output_path))
            print(f"학습된 모델이 '{output_path}'로 저장되었습니다.")
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    print("\n===== 학습 완료 =====")
    print(f"학습 결과는 '{os.path.join('runs', task, 'train')}' 폴더에서 확인할 수 있습니다.")
    print(f"최종 모델 파일: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {e}")
        import traceback
>>>>>>> 93bbb41db9c4711833322ee6859a7d8ee4f26442
=======
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
import argparse
from pathlib import Path
import configparser
import sys

'''
이 코드는 YOLOv8 객체 감지 및 세그멘테이션 모델을 학습하는 코드입니다.
경로 설정을 유연하게 하고, 다양한 환경에서도 쉽게 사용할 수 있도록 설계되었습니다.

주요 기능:
1. 사용자 경로 입력 또는 설정 파일을 통한 경로 설정
2. 다양한 하위 폴더 구조 지원
3. 클래스 분포 분석 및 가중치 계산
4. 자동 데이터 증강 및 학습/검증 데이터 분할
5. 학습 매개변수 조정 기능
6. 모델 평가 및 시각화

모델 학습 결과는 지정한 출력 폴더 또는 기본 'runs/' 폴더에 저장됩니다.
'''

# 설정 관리 클래스
class Config:
    def __init__(self, config_file=None):
        # 기본 설정
        self.config = {
            'paths': {
                'work_dir': 'yolov8_dataset',
                'output_dir': 'runs',
                'data_dir': '',
                'classes_file': 'classes.txt',
                'model_path': ''
            },
            'training': {
                'img_size': 640,
                'batch_size': 16,
                'epochs': 50,
                'train_ratio': 0.8,
                'task': 'detect'  # 'detect' 또는 'segment'
            },
            'model': {
                'model_type': 'n',  # 'n', 's', 'm', 'l', 'x'
                'continue_training': False
            }
        }
        
        # 설정 파일이 제공되면 로드
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """설정 파일에서 설정을 로드합니다."""
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # 로드된 설정을 현재 설정에 병합
            for section, values in loaded_config.items():
                if section in self.config:
                    self.config[section].update(values)
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            print("기본 설정을 사용합니다.")
    
    def save_config(self, config_file):
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"설정이 {config_file}에 저장되었습니다.")
        except Exception as e:
            print(f"설정 파일 저장 중 오류 발생: {e}")
    
    def get(self, section, key, default=None):
        """특정 설정 값을 가져옵니다."""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """특정 설정 값을 설정합니다."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_from_args(self, args):
        """명령줄 인수에서 설정을 업데이트합니다."""
        # args에서 받은 값들로 설정 업데이트
        if hasattr(args, 'work_dir') and args.work_dir:
            self.config['paths']['work_dir'] = args.work_dir
        
        if hasattr(args, 'data_dir') and args.data_dir:
            self.config['paths']['data_dir'] = args.data_dir
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config['paths']['output_dir'] = args.output_dir
        
        if hasattr(args, 'classes_file') and args.classes_file:
            self.config['paths']['classes_file'] = args.classes_file
        
        if hasattr(args, 'model_path') and args.model_path:
            self.config['paths']['model_path'] = args.model_path
            self.config['model']['continue_training'] = True
        
        if hasattr(args, 'img_size') and args.img_size:
            self.config['training']['img_size'] = args.img_size
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['training']['batch_size'] = args.batch_size
        
        if hasattr(args, 'epochs') and args.epochs:
            self.config['training']['epochs'] = args.epochs
        
        if hasattr(args, 'model_type') and args.model_type:
            self.config['model']['model_type'] = args.model_type
        
        if hasattr(args, 'task') and args.task:
            self.config['training']['task'] = args.task
    
    def get_dir_structure(self):
        """디렉토리 구조를 반환합니다."""
        work_dir = self.get('paths', 'work_dir')
        
        return {
            'work_dir': work_dir,
            'images_dir': os.path.join(work_dir, "images"),
            'labels_dir': os.path.join(work_dir, "labels"),
            'train_images_dir': os.path.join(work_dir, "images", "train"),
            'train_labels_dir': os.path.join(work_dir, "labels", "train"),
            'val_images_dir': os.path.join(work_dir, "images", "val"),
            'val_labels_dir': os.path.join(work_dir, "labels", "val")
        }

# 명령줄 인수 파서 설정
def setup_argparse():
    parser = argparse.ArgumentParser(description='YOLOv8 모델 학습 프로그램')
    
    # 경로 관련 인수
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--work_dir', type=str, help='작업 디렉토리 경로')
    parser.add_argument('--data_dir', type=str, help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--classes_file', type=str, help='클래스 파일 경로')
    parser.add_argument('--model_path', type=str, help='기존 모델 경로 (계속 학습할 경우)')
    
    # 학습 관련 인수
    parser.add_argument('--img_size', type=int, help='학습 이미지 크기')
    parser.add_argument('--batch_size', type=int, help='배치 크기')
    parser.add_argument('--epochs', type=int, help='에폭 수')
    parser.add_argument('--model_type', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                        help='모델 유형 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--task', type=str, choices=['detect', 'segment'], 
                        help='작업 유형 (detect=객체 탐지, segment=세그멘테이션)')
    
    # 기타 옵션
    parser.add_argument('--no_input', action='store_true', 
                        help='사용자 입력 없이 설정 파일 또는 기본값 사용')
    parser.add_argument('--save_config', action='store_true', 
                        help='현재 설정을 파일로 저장')
    
    return parser.parse_args()

# 디렉토리 설정 함수
def setup_directories(config):
    """작업 디렉토리 구조를 설정합니다."""
    dirs = config.get_dir_structure()
    
    # 각 디렉토리가 존재하지 않으면 생성
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return dirs

# 클래스 빈도 계산 함수
def calculate_class_frequency(json_files, frame_files, classes_file):
    """모든 데이터에서 클래스별 출현 빈도를 계산합니다."""
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
    """클래스별 가중치를 계산합니다 (희소 클래스에 더 큰 가중치)."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (객체 탐지용)
def convert_json_to_yolo_detection(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 객체 탐지 형식으로 변환합니다."""
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

# JSON 파일에서 데이터 추출 및 변환 함수 (세그멘테이션용)
def convert_json_to_yolo_segmentation(json_file, image_file, classes_file):
    """JSON 파일에서 라벨 정보를 추출하여 YOLO 세그멘테이션 형식으로 변환합니다."""
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

# 입력 데이터 처리 함수
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir, 
                 task='detect', apply_aug=False, class_counts=None):
    """
    입력 데이터를 처리하고 YOLO 형식으로 변환하여 저장합니다.
    
    Args:
        task: 'detect' 또는 'segment' (객체 탐지 또는 세그멘테이션)
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 형식으로 라벨 변환 (task에 따라 다른 함수 호출)
    if task == 'detect':
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_detection(
            json_file, frame_file, classes_file)
    else:  # task == 'segment'
        yolo_annotations, bboxes, class_ids, image = convert_json_to_yolo_segmentation(
            json_file, frame_file, classes_file)
    
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
        
        # 결정된 횟수만큼 증강 적용 (클래스 균형 맞추기)
        for aug_idx in range(aug_count):
            # 이미지와 라벨 파일명 설정
            aug_image_filename = f"{base_filename}_aug{aug_idx}.jpg"
            aug_label_filename = f"{base_filename}_aug{aug_idx}.txt"
            
            # 파일 경로 설정
            aug_image_path = os.path.join(output_image_dir, aug_image_filename)
            aug_label_path = os.path.join(output_label_dir, aug_label_filename)
            
            # 세그멘테이션 또는 객체 탐지에 따라 처리 방식 선택
            if task == 'segment':
                # 세그멘테이션의 경우 단순히 원본 복사 (간단한 증강)
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
            else:
                # 객체 탐지의 경우 이미지 자체 변형 적용 (고급 증강)
                # 여기서는 간단한 예시로, 실제로는 albumentations 등의 라이브러리 사용 권장
                # TODO: 고급 증강 기능 구현
                shutil.copy(frame_file, aug_image_path)
                with open(output_label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
    
    return output_image_path, output_label_path

# 데이터셋 분할 함수
def split_dataset(json_files, frame_files, classes_file, dirs, config):
    """데이터셋을 학습 및 검증 세트로 분할합니다."""
    task = config.get('training', 'task')
    train_ratio = config.get('training', 'train_ratio')
    
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
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['train_images_dir'], dirs['train_labels_dir'],
            task=task, apply_aug=True, class_counts=class_counts
        )
    
    for i in tqdm(val_indices, desc="검증 데이터 처리 중"):
        _, _ = process_data(
            json_files[i], frame_files[i], classes_file,
            dirs['val_images_dir'], dirs['val_labels_dir'],
            task=task, apply_aug=False
        )
    
    # 데이터셋 개수 계산
    train_images = len([f for f in os.listdir(dirs['train_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir(dirs['val_images_dir']) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"데이터셋 분할 완료: {train_images}개 학습 샘플 (원본 {len(train_indices)}개 + 증강 {train_images - len(train_indices)}개), {val_images}개 검증 샘플")
    
    return class_counts, class_weights

# YAML 설정 파일 생성 함수
def create_yaml_config(classes_file, output_yaml, class_weights=None, config=None):
    """YOLO 학습을 위한 YAML 설정 파일을 생성합니다."""
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 클래스 가중치 문자열 생성
    weights_str = ""
    if class_weights:
        max_cls_id = max(class_weights.keys())
        weights_list = [class_weights.get(i, 1.0) for i in range(max_cls_id + 1)]
        weights_str = "[" + ", ".join([f"{w:.2f}" for w in weights_list]) + "]"
    
    # 작업 디렉토리 경로 가져오기
    work_dir = os.path.abspath(config.get('paths', 'work_dir'))
    
    # 작업 유형 가져오기
    task = config.get('training', 'task')
    
    # YAML 파일 생성
    yaml_content = f"""
path: {work_dir}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

    # 세그멘테이션 작업인 경우 task 정보 추가
    if task == 'segment':
        yaml_content += "\n# 세그멘테이션 설정\ntask: segment\n"
    
    # 클래스 가중치 추가
    if class_weights:
        yaml_content += f"\n# 클래스 가중치 (희소 클래스에 더 높은 가중치)\ncls_weights: {weights_str}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")
    if class_weights:
        print("클래스 가중치가 YAML 설정에 추가되었습니다.")

# 모델 학습 함수
def train_model(yaml_config, config):
    """YOLO 모델을 학습합니다."""
    # 학습 설정 가져오기
    task = config.get('training', 'task')
    epochs = config.get('training', 'epochs')
    batch_size = config.get('training', 'batch_size')
    img_size = config.get('training', 'img_size')
    model_type = config.get('model', 'model_type')
    continue_training = config.get('model', 'continue_training')
    model_path = config.get('paths', 'model_path')
    
    # 모델 유형 결정
    if continue_training and model_path:
        # 기존 모델에서 계속 학습
        model = YOLO(model_path)
        print(f"기존 모델({model_path})에서 학습을 계속합니다.")
    else:
        # 새 모델로 학습 시작
        if task == 'segment':
            # 세그멘테이션 모델
            model_name = f"yolov8{model_type}-seg.pt"
        else:
            # 객체 탐지 모델
            model_name = f"yolov8{model_type}.pt"
        
        model = YOLO(model_name)
        print(f"새 모델({model_name})로 학습을 시작합니다.")

    # 공통 학습 매개변수
    train_args = {
        'data': yaml_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'patience': 20,
        'verbose': True,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'save_period': 5,
        'cos_lr': True,
        'warmup_epochs': 5 if not continue_training else 3,
        'lr0': 0.01 if not continue_training else 0.001,
        'lrf': 0.001 if not continue_training else 0.0001,
        'weight_decay': 0.0005,
        'overlap_mask': True,
        'close_mosaic': 10
    }
    
    # 학습 시작
    results = model.train(**train_args)
    
    return model, results

# 모델 평가 함수
def evaluate_model(model, yaml_config, task='detect'):
    """학습된 모델을 평가합니다."""
    # 모델 검증
    results = model.val(data=yaml_config)
    
    # 주요 메트릭 출력
    print("\n===== 모델 평가 결과 =====")
    
    if task == 'segment':
        # 세그멘테이션 모델 메트릭
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
    else:
        # 객체 탐지 모델 메트릭
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'Precision': results.box.p,
            'Recall': results.box.r,
            'F1-Score': results.box.f1
        }
    
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
def plot_results(results, task='detect'):
    """학습 결과를 시각화합니다."""
    try:
        # 결과 시각화 (Ultralytics YOLO는 자동으로 plots 폴더에 그래프를 저장합니다)
        print(f"\n학습 결과가 시각화되었습니다. 'runs/{task}/train/results.png'에서 확인하세요.")
        
        # 만약 results 객체에 plot_results 메서드가 있다면 사용
        if hasattr(results, 'plot_results'):
            fig = results.plot_results(show=False)
            fig.savefig("training_results.png")
            print("학습 결과 그래프가 'training_results.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"학습 결과 시각화 중 오류 발생: {e}")
        print(f"기본 YOLO 시각화는 'runs/{task}/train/' 폴더에서 확인하세요.")

# 클래스 분포 시각화 함수
def visualize_class_distribution(class_counts):
    """클래스 분포와 가중치를 시각화합니다."""
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

# 경로 찾기 함수
def find_classes_file(base_path, subfolders):
    """클래스 파일을 찾습니다."""
    classes_file = "classes.txt"
    
    # 기본 경로에서 찾기
    classes_file_path = os.path.join(base_path, classes_file)
    if os.path.exists(classes_file_path):
        return classes_file_path
    
    # 하위 폴더에서 찾기
    for folder in subfolders:
        path = os.path.join(folder, classes_file)
        if os.path.exists(path):
            return path
    
    # 현재 스크립트 폴더에서 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, classes_file)
    if os.path.exists(path):
        return path
    
    return None

# 사용자 입력 처리 함수
def get_user_input(config, skip_input=False):
    """사용자로부터 설정 값을 입력받습니다."""
    if skip_input:
        return config
    
    # 데이터 경로 입력
    print("\n===== 데이터 폴더 경로 입력 =====")
    print("데이터 폴더는 이미지와 JSON 파일이 있는 폴더여야 합니다.")
    print("예시: C:\\Users\\Username\\Project\\data")
    data_dir = input(f"데이터 폴더 경로 [{config.get('paths', 'data_dir') or '.'}]: ").strip()
    if data_dir:
        config.set('paths', 'data_dir', data_dir)
    
    # 작업 유형 선택
    print("\n===== 작업 유형 선택 =====")
    print("1: 객체 탐지 (Object Detection)")
    print("2: 세그멘테이션 (Segmentation)")
    task_choice = input(f"작업 유형 선택 (1/2) [{'1' if config.get('training', 'task') == 'detect' else '2'}]: ").strip()
    if task_choice == '1':
        config.set('training', 'task', 'detect')
    elif task_choice == '2':
        config.set('training', 'task', 'segment')
    
    # 학습 방식 선택
    print("\n===== 학습 방식 선택 =====")
    print("1: 새 모델 학습")
    print("2: 기존 모델 계속 학습")
    train_choice = input(f"학습 방식 선택 (1/2) [{'2' if config.get('model', 'continue_training') else '1'}]: ").strip()
    
    if train_choice == '2':
        config.set('model', 'continue_training', True)
        print("\n===== 기존 모델 경로 입력 =====")
        print("예시: runs/detect/train/weights/best.pt")
        model_path = input(f"모델 경로 [{config.get('paths', 'model_path') or 'runs/detect/train/weights/best.pt'}]: ").strip()
        if model_path:
            config.set('paths', 'model_path', model_path)
    else:
        config.set('model', 'continue_training', False)
        
        # 모델 유형 선택 (새 모델 학습 시에만)
        print("\n===== 모델 유형 선택 =====")
        print("n: YOLOv8n (가벼운 모델, 빠르지만 정확도 낮음)")
        print("s: YOLOv8s (소형 모델)")
        print("m: YOLOv8m (중형 모델, 균형 잡힌 속도와 정확도)")
        print("l: YOLOv8l (대형 모델)")
        print("x: YOLOv8x (초대형 모델, 정확도 높지만 느림)")
        model_type = input(f"모델 유형 선택 (n/s/m/l/x) [{config.get('model', 'model_type')}]: ").strip().lower()
        if model_type in ['n', 's', 'm', 'l', 'x']:
            config.set('model', 'model_type', model_type)
    
    # 고급 설정 여부
    advanced_config = input("\n고급 설정을 변경하시겠습니까? (y/n) [n]: ").strip().lower() == 'y'
    
    if advanced_config:
        # 이미지 크기 설정
        print("\n===== 이미지 크기 설정 =====")
        print("- 작은 크기(320-416): 속도 빠름, 정확도 낮음")
        print("- 중간 크기(640): 균형 잡힌 속도와 정확도")
        print("- 큰 크기(832-1024): 속도 느림, 정확도 높음")
        img_size = input(f"이미지 크기 [{config.get('training', 'img_size')}]: ").strip()
        if img_size and img_size.isdigit():
            config.set('training', 'img_size', int(img_size))
        
        # 배치 크기 설정
        print("\n===== 배치 크기 설정 =====")
        print("- 작은 배치(4-8): 적은 GPU 메모리 필요, 느린 학습")
        print("- 중간 배치(16-32): 균형 잡힌 속도와 메모리 사용량")
        print("- 큰 배치(64+): 빠른 학습, 많은 GPU 메모리 필요")
        batch_size = input(f"배치 크기 [{config.get('training', 'batch_size')}]: ").strip()
        if batch_size and batch_size.isdigit():
            config.set('training', 'batch_size', int(batch_size))
        
        # 에폭 수 설정
        print("\n===== 에폭 수 설정 =====")
        print("- 적은 에폭(10-30): 빠른 학습, 낮은 정확도")
        print("- 중간 에폭(50-100): 균형 잡힌 학습 시간과 정확도")
        print("- 많은 에폭(100+): 오랜 학습 시간, 잠재적으로 더 높은 정확도")
        epochs = input(f"에폭 수 [{config.get('training', 'epochs')}]: ").strip()
        if epochs and epochs.isdigit():
            config.set('training', 'epochs', int(epochs))
        
        # 작업 디렉토리 설정
        print("\n===== 작업 디렉토리 설정 =====")
        print("학습 데이터와 라벨이 저장될 디렉토리입니다.")
        work_dir = input(f"작업 디렉토리 [{config.get('paths', 'work_dir')}]: ").strip()
        if work_dir:
            config.set('paths', 'work_dir', work_dir)
    
    return config

# 메인 함수
def main():
    # 시작 메시지 출력
    print("===== YOLO 모델 학습 프로그램 =====")
    print("이 프로그램은 YOLOv8 모델을 학습, 평가, 테스트하는 기능을 제공합니다.")
    
    # 명령줄 인수 파싱
    args = setup_argparse()
    
    # 설정 로드
    config = Config(args.config)
    
    # 명령줄 인수로 설정 업데이트
    config.update_from_args(args)
    
    # 사용자 입력으로 설정 업데이트 (--no_input 옵션이 없을 경우)
    if not args.no_input:
        config = get_user_input(config)
    
    # 설정 저장 (--save_config 옵션이 있을 경우)
    if args.save_config:
        config_file = args.config or "yolo_train_config.json"
        config.save_config(config_file)
    
    # PyTorch CUDA 사용 가능 여부 확인
    print(f"\n===== 시스템 정보 =====")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU가 감지되지 않았습니다. CPU를 사용하여 학습합니다.")
    
    # 작업 디렉토리 설정
    dirs = setup_directories(config)
    
    # 데이터 경로 가져오기
    data_dir = config.get('paths', 'data_dir')
    
    # 경로 존재 확인
    if not os.path.exists(data_dir):
        print(f"오류: 지정한 데이터 경로 '{data_dir}'가 존재하지 않습니다.")
        return
    
    try:
        # 데이터가 포함된 모든 하위 폴더 찾기
        subfolders = []
        if os.path.isdir(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    subfolders.append(folder_path)
            
            # 하위 폴더가 없으면 기본 폴더를 직접 사용
            if not subfolders:
                print("하위 폴더가 없습니다. 기본 폴더를 직접 사용합니다.")
                subfolders = [data_dir]
        else:
            print(f"오류: '{data_dir}'는 폴더가 아닙니다.")
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
    
    # classes.txt 파일 확인 및 찾기
    classes_file_path = config.get('paths', 'classes_file') or find_classes_file(data_dir, subfolders)
    
    if not classes_file_path or not os.path.exists(classes_file_path):
        print("classes.txt 파일을 찾을 수 없습니다.")
        classes_file_path = input("classes.txt 파일의 전체 경로를 입력하세요: ").strip()
        
        if not os.path.exists(classes_file_path):
            print(f"오류: 지정한 classes.txt 파일 '{classes_file_path}'가 존재하지 않습니다.")
            return
    
    # 설정에 classes_file 경로 저장
    config.set('paths', 'classes_file', classes_file_path)
    
    print(f"총 {len(all_json_files)}개의 파일 쌍을 찾았습니다.")
    print(f"Classes 파일: {classes_file_path}")
    
    # 데이터셋 분할 및 처리
    print("\n===== 데이터셋 분할 및 처리 =====")
    class_counts, class_weights = split_dataset(all_json_files, all_frame_files, classes_file_path, dirs, config)
    
    # 클래스 분포 시각화
    visualize_class_distribution(class_counts)
    
    # YAML 설정 파일 생성
    yaml_config = os.path.join(dirs['work_dir'], "dataset.yaml")
    create_yaml_config(classes_file_path, yaml_config, class_weights, config)
    
    # 학습 시작
    print("\n===== 모델 학습 시작 =====")
    model, results = train_model(yaml_config, config)
    
    # 모델 평가
    print("\n===== 모델 평가 =====")
    task = config.get('training', 'task')
    evaluation_results = evaluate_model(model, yaml_config, task)
    
    # 결과 시각화
    plot_results(results, task)
    
    # 모델 저장
    output_name = "yolov8_continued.pt" if config.get('model', 'continue_training') else "yolov8_custom.pt"
    if task == 'segment':
        output_name = output_name.replace('.pt', '_seg.pt')
    
    output_path = os.path.join(os.getcwd(), output_name)
    
    try:
        # 모델 저장
        model.save(output_path)
        print(f"\n학습된 모델이 '{output_path}'로 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
        try:
            # 다른 방법으로 저장 시도
            model.export(format="pt", save_dir=os.path.dirname(output_path), 
                       filename=os.path.basename(output_path))
            print(f"학습된 모델이 '{output_path}'로 저장되었습니다.")
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    print("\n===== 학습 완료 =====")
    print(f"학습 결과는 '{os.path.join('runs', task, 'train')}' 폴더에서 확인할 수 있습니다.")
    print(f"최종 모델 파일: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {e}")
        import traceback
>>>>>>> 93bbb41db9c4711833322ee6859a7d8ee4f26442
        traceback.print_exc()