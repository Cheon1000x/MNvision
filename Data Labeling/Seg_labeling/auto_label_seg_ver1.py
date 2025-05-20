import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
from pathlib import Path
import argparse

def create_labelme_json_segmentation(image_path, results, class_names, output_dir, camera_type):
    """
    YOLOv8 세그멘테이션 예측 결과를 LabelMe JSON 형식으로 변환합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        results: YOLO 예측 결과
        class_names (dict): 클래스 ID와 이름 매핑 딕셔너리
        output_dir (str): 출력 JSON 파일 저장 디렉토리
        camera_type (str): 카메라 타입 ('cam1' 또는 'cam2')
    """
    # 이미지 로드하여 크기 얻기
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    
    # 이미지 파일명 가져오기
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    
    # JSON 데이터 구조 생성
    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "auto_labeled": True  # 자동 라벨링으로 생성된 파일임을 표시
    }
    
    # YOLO 결과에서 세그멘테이션 마스크 및 클래스 정보 추출
    if hasattr(results, 'masks') and results.masks is not None:
        # 마스크 원래 크기 가져오기
        orig_shape = results.masks.orig_shape  # 원본 이미지 크기
        
        for i, mask in enumerate(results.masks.data):
            # 클래스 ID 및 신뢰도 점수 가져오기
            cls_id = int(results.boxes.cls[i].item())
            conf = float(results.boxes.conf[i].item())
            
            # 기본 클래스 이름 가져오기
            class_name = class_names[cls_id]
            
            # cam2인 경우 forklift 관련 라벨에만 "-cam2" 접미사 추가
            if camera_type == "cam2" and "forklift" in class_name:
                class_name = f"{class_name}(cam2)"
            
            # 마스크 크기 조정 (원본 이미지 크기에 맞게)
            mask_np = mask.cpu().numpy().astype(np.uint8)
            # YOLO에서 반환된 마스크 크기를 원본 이미지 크기로 변환
            mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 가장 큰 윤곽선 선택
            if contours:
                contour = max(contours, key=cv2.contourArea)
                
                # 윤곽선 단순화 (Douglas-Peucker 알고리즘)
                epsilon = 0.003 * cv2.arcLength(contour, True)  # 더 정밀한 윤곽선을 위해 조정
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 포인트가 너무 많으면 더 줄이기
                if len(approx) > 100:
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 포인트 목록 생성
                points = []
                for point in approx:
                    x, y = point[0]
                    points.append([float(x), float(y)])
                
                # 최소 3개의 점이 있는지 확인
                if len(points) >= 3:
                    # shape 항목 생성
                    shape = {
                        "label": class_name,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                        "confidence": float(conf)  # 신뢰도 정보 추가
                    }
                    
                    json_data["shapes"].append(shape)
    
    # 현재 시간 정보 추가
    now = datetime.datetime.now()
    json_data["imageData"] = None
    json_data["time_updated"] = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 출력 파일 경로 생성
    output_path = os.path.join(output_dir, f"{base_name}.json")
    
    # JSON 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return output_path

def yolo_seg_to_labelme(model_path, image_dir, output_base_dir=None, conf_threshold=0.3, camera_type="cam1"):
    """
    YOLOv8 세그멘테이션 모델을 사용하여 이미지를 자동 라벨링하고 LabelMe 형식으로 저장합니다.
    하위 폴더를 포함한 모든 이미지를 재귀적으로 처리합니다.
    
    Args:
        model_path (str): YOLOv8 세그멘테이션 모델 경로
        image_dir (str): 이미지 디렉토리 경로 (하위 폴더 포함)
        output_base_dir (str, optional): 출력 JSON 파일 저장 디렉토리. 기본값은 None으로, 
                                       이 경우 이미지와 동일한 위치에 JSON 저장
        conf_threshold (float): 신뢰도 임계값 (이 값 이상의 예측만 유지)
        camera_type (str): 카메라 타입 ('cam1' 또는 'cam2')
    """
    # 모델 로드
    model = YOLO(model_path)
    
    # 모델이 세그멘테이션 모델인지 확인
    if not model.task == 'segment':
        print(f"경고: 제공된 모델이 세그멘테이션 모델이 아닙니다. 현재 태스크: {model.task}")
        if input("계속 진행하시겠습니까? (y/n): ").lower() != 'y':
            print("작업을 중단합니다.")
            return
    
    # 클래스 이름 가져오기
    class_names = model.names
    
    # 모든 이미지 파일 찾기 (재귀적으로)
    all_image_files = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 전체 경로 저장
                img_path = os.path.join(root, file)
                all_image_files.append(img_path)
    
    print(f"총 {len(all_image_files)}개의 이미지 파일을 세그멘테이션 처리합니다. (카메라 타입: {camera_type})")
    
    processed_count = 0
    for image_path in all_image_files:
        # 이미지 경로의 상대 경로 계산
        rel_path = os.path.relpath(os.path.dirname(image_path), image_dir)
        
        # 출력 디렉토리 설정 (기본: 이미지와 같은 위치)
        if output_base_dir:
            # 출력 베이스 디렉토리가 제공된 경우, 상대 경로를 유지하여 해당 위치에 저장
            output_dir = os.path.join(output_base_dir, rel_path)
        else:
            # 출력 베이스 디렉토리가 없는 경우, 이미지와 같은 위치에 저장
            output_dir = os.path.dirname(image_path)
            
        # 출력 디렉토리가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 이미 자동 라벨링 JSON 파일이 있는지 확인
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}.json")
        
        if os.path.exists(json_path):
            print(f"건너뜁니다: {image_path} (이미 자동 라벨링된 파일)")
            continue
        
        # 이미지에 대한 예측 수행
        try:
            results = model.predict(image_path, conf=conf_threshold, verbose=False)[0]
            
            # 세그멘테이션 마스크가 있는지 확인
            if hasattr(results, 'masks') and results.masks is not None and len(results.masks) > 0:
                # LabelMe JSON 생성
                json_path = create_labelme_json_segmentation(image_path, results, class_names, output_dir, camera_type)
                print(f"생성됨: {json_path}")
            else:
                print(f"감지된 세그멘테이션 마스크 없음: {image_path}")
                
        except Exception as e:
            print(f"처리 중 오류 발생: {image_path} - {str(e)}")
            continue
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"진행 상황: {processed_count}/{len(all_image_files)}")
    
    print(f"모든 처리 완료: {processed_count}개 이미지 세그멘테이션 처리됨")

def visualize_segmentation(model_path, input_image, output_image=None, conf_threshold=0.3):
    """
    YOLOv8 세그멘테이션 모델을 사용하여 이미지에 세그멘테이션 마스크를 시각화합니다.
    
    Args:
        model_path (str): YOLOv8 세그멘테이션 모델 경로
        input_image (str): 입력 이미지 경로
        output_image (str, optional): 출력 이미지 경로. 기본값은 None으로, 이 경우 input_image_seg.jpg로 저장
        conf_threshold (float): 신뢰도 임계값
    """
    # 모델 로드
    model = YOLO(model_path)
    
    # 세그멘테이션 모델인지 확인
    if not (model.task == 'segment' or hasattr(model, 'masks')):
        print(f"경고: 제공된 모델이 세그멘테이션 모델이 아닙니다.")
        return
    
    # 클래스 이름
    class_names = model.names
    
    # 기본 출력 이미지 경로 설정
    if output_image is None:
        base_name = os.path.splitext(input_image)[0]
        output_image = f"{base_name}_seg.jpg"
    
    # 세그멘테이션 예측 수행 (plot=True로 설정하여 시각화)
    results = model.predict(input_image, conf=conf_threshold, save=True, save_crop=True)
    
    # 결과 표시
    image_result = results[0].plot()
    cv2.imwrite(output_image, image_result)
    
    print(f"세그멘테이션 결과가 {output_image}에 저장되었습니다.")

if __name__ == "__main__":
    model_path = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\4.Model Experimental Data\Test_4_Seg\yolov8_seg_custom.pt"
    image_dir = r"C:\Users\KDT-13\Desktop\Group 6_\MNvision\2.Data\photo_cam2"
    output_dir = None  # 이미지와 같은 폴더에 저장
    conf_threshold = 0.3
    
    # 카메라 타입 입력 받기
    camera_type = None
    while camera_type not in [1, 2]:
        try:
            camera_type = int(input("카메라 타입을 입력하세요 (1 또는 2): "))
            if camera_type not in [1, 2]:
                print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")
    
    # 카메라 타입에 따른 문자열 변환
    camera_str = f"cam{camera_type}"
    
    print(f"선택된 카메라 타입: {camera_str}")
    print(f"forklift 관련 라벨만 {camera_str} 접미사가 추가됩니다.")
    
    # 사용 모드 선택 (라벨링 또는 시각화)
    mode = input("모드를 선택하세요 (1: 자동 라벨링, 2: 시각화): ")
    if mode == "1":
        yolo_seg_to_labelme(model_path, image_dir, output_dir, conf_threshold, camera_str)
        print(f"세그멘테이션 자동 라벨링이 완료되었습니다. ({camera_str} 적용) LabelMe에서 결과를 검토하고 수정하세요.")
    elif mode == "2":
        # 시각화할 이미지 경로 입력
        image_path = input("시각화할 이미지 경로를 입력하세요: ")
        visualize_segmentation(model_path, image_path, None, conf_threshold)
    else:
        print("잘못된 모드 선택입니다. 프로그램을 종료합니다.")