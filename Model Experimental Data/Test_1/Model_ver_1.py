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


'''
* 성능 부족시 *

객체 감지로 시작하고 필요시 세그멘테이션으로 발전:

우선 현재처럼 객체 감지 모델로 시작하여 성능을 평가합니다.
정확도가 충분하지 않으면 세그멘테이션으로 전환을 고려합니다.


더 가벼운 세그멘테이션 모델 선택:

YOLOv8n-seg와 같은 더 작은 모델 사용 (YOLOv8m-seg 대신)
입력 이미지 크기 줄이기 (1280 → 640)


모델 최적화 기술 적용:

양자화(Quantization)
모델 프루닝(Pruning)
지식 증류(Knowledge Distillation)


하이브리드 접근법:

빠른 객체 감지로 객체를 찾은 다음, 필요한 객체에만 세그멘테이션 적용

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
            
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

# 입력 데이터 처리 함수
def process_data(json_file, frame_file, classes_file, output_image_dir, output_label_dir):
    """
    입력 데이터를 처리하고 YOLO 형식으로 변환하여 저장합니다.
    """
    # 이미지 파일명 추출
    image_filename = os.path.basename(frame_file)
    base_filename = os.path.splitext(image_filename)[0]
    
    # 이미지 복사
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(frame_file, output_image_path)
    
    # YOLO 형식으로 라벨 변환
    yolo_annotations = convert_json_to_yolo_format(json_file, frame_file, classes_file)
    
    # 라벨 파일 저장
    label_filename = f"{base_filename}.txt"
    output_label_path = os.path.join(output_label_dir, label_filename)
    
    with open(output_label_path, 'w') as f:
        f.write("\n".join(yolo_annotations))
    
    return output_image_path, output_label_path

# 데이터셋 분할 함수
def split_dataset(json_files, frame_files, classes_file, train_ratio=0.8):
    """
    데이터셋을 학습 및 검증 세트로 분할합니다.
    """
    # 데이터셋 섞기
    indices = np.arange(len(json_files))
    np.random.shuffle(indices)
    
    # 학습 및 검증 세트 분할
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 데이터 처리
    for i in tqdm(train_indices, desc="Processing Training Data"):
        process_data(json_files[i], frame_files[i], classes_file,
                    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    
    for i in tqdm(val_indices, desc="Processing Validation Data"):
        process_data(json_files[i], frame_files[i], classes_file,
                    VAL_IMAGES_DIR, VAL_LABELS_DIR)
    
    print(f"데이터셋 분할 완료: {len(train_indices)}개 학습 샘플, {len(val_indices)}개 검증 샘플")

# YAML 설정 파일 생성 함수
def create_yaml_config(classes_file, output_yaml):
    """
    YOLO 학습을 위한 YAML 설정 파일을 생성합니다.
    """
    # 클래스 목록 로드
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # YAML 파일 생성
    yaml_content = f"""
path: {os.path.abspath(WORK_DIR)}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 설정 파일 생성 완료: {output_yaml}")

# 학습 함수
def train_yolov8(yaml_config, epochs=50, batch_size=16, img_size=640): # 원본 너비와 비슷 하게 바꾸려면 img_size 변경 (1280)
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
        patience=10,
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_period=5  # 5 에포크마다 체크포인트 저장
    )
    
    return model, results

# 기존 모델에서 추가 학습하는 함수
def continue_training(pretrained_model_path, yaml_config, epochs=30, batch_size=16, img_size=640):
    """
    기존에 학습된 모델을 로드하고 새 데이터로 추가 학습합니다.
    
    Args:
        pretrained_model_path: 이전에 학습된 모델 경로 (예: 'runs/detect/train/weights/best.pt')
        yaml_config: 데이터셋 YAML 설정 파일 경로
        epochs: 추가 학습할 에포크 수
        batch_size: 배치 크기
        img_size: 입력 이미지 크기
    
    Returns:
        model: 학습된 모델
        results: 학습 결과
    """
    # 사전 학습된 모델 로드
    model = YOLO(pretrained_model_path)
    
    # 모델 추가 학습
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        lr0=0.001,  # 기본값보다 낮은 학습률 설정
        lrf=0.01,   # 학습률 감소 계수
        resume=False,  # 학습 재개를 위한 설정
        save_period=5  # 5 에포크마다 체크포인트 저장
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
    base_data_path = r"C:\Users\KDT34\Desktop\MNvision\data\photo"
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
    
    # 데이터셋 분할 및 처리
    split_dataset(json_files, frame_files, classes_file_path)
    
    # YAML 설정 파일 생성
    yaml_config = "dataset.yaml"
    create_yaml_config(classes_file_path, yaml_config)
    
    # 학습 모드 선택 (새 모델 학습 또는 기존 모델 계속 학습)
    continue_from_checkpoint = input("기존 모델에서 계속 학습하시겠습니까? (y/n): ").lower() == 'y'
    
    try:
        if continue_from_checkpoint:
            # 기존 모델 경로 입력
            model_path = input("기존 모델 경로를 입력하세요 (예: runs/detect/train/weights/best.pt): ")
            print(f"\n===== 기존 모델({model_path})에서 학습 계속 =====")
            model, training_results = continue_training(model_path, yaml_config, epochs=30)
            
            # 모델 평가
            evaluation_results = evaluate_model(model, yaml_config)
            
            # 학습 결과 시각화
            if hasattr(training_results, 'plot_results'):
                plot_results(training_results)
        else:
            # 새 모델로 학습 시작
            print("\n===== YOLOv8m 모델 학습 시작 =====")
            model, training_results = train_yolov8(yaml_config, epochs=1) # 새 학습 epoch
            
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