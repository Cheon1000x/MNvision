import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import datetime
from pathlib import Path
import argparse

def create_labelme_json(image_path, boxes, class_names, confidence_scores, output_dir):
    """
    YOLOv8 예측 결과를 LabelMe JSON 형식으로 변환합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        boxes (list): 예측된 바운딩 박스 (x1, y1, x2, y2, class_id, conf)
        class_names (list): 클래스 이름 목록
        confidence_scores (list): 신뢰도 점수 목록
        output_dir (str): 출력 JSON 파일 저장 디렉토리
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
    
    # 각 바운딩 박스를 JSON 형식으로 변환
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        class_id = int(box[4])
        conf = confidence_scores[i] if i < len(confidence_scores) else 0.0
        
        class_name = class_names[class_id]
        
        # LabelMe에서 사용하는 형식으로 포인트 변환 (바운딩 박스 좌표)
        points = [[float(x1), float(y1)], [float(x2), float(y2)]]
        
        # shape 항목 생성
        shape = {
            "label": class_name,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "confidence": float(conf)  # 신뢰도 정보 추가
        }
        
        json_data["shapes"].append(shape)
    
    # 현재 시간 정보 추가
    now = datetime.datetime.now()
    json_data["imageData"] = None
    json_data["time_updated"] = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 자동 라벨링 접두어 추가하여 JSON 파일 저장
    output_path = os.path.join(output_dir, f"auto_{base_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return output_path

def yolo_to_labelme(model_path, image_dir, output_base_dir=None, conf_threshold=0.3):
    """
    YOLOv8 모델을 사용하여 이미지를 자동 라벨링하고 LabelMe 형식으로 저장합니다.
    하위 폴더를 포함한 모든 이미지를 재귀적으로 처리합니다.
    
    Args:
        model_path (str): YOLOv8 모델 경로
        image_dir (str): 이미지 디렉토리 경로 (하위 폴더 포함)
        output_base_dir (str, optional): 출력 JSON 파일 저장 디렉토리. 기본값은 None으로, 
                                       이 경우 이미지와 동일한 위치에 JSON 저장
        conf_threshold (float): 신뢰도 임계값 (이 값 이상의 예측만 유지)
    """
    # 모델 로드
    model = YOLO(model_path)
    
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
    
    print(f"총 {len(all_image_files)}개의 이미지 파일을 처리합니다.")
    
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
        auto_json_path = os.path.join(output_dir, f"auto_{base_name}.json")
        
        if os.path.exists(auto_json_path):
            print(f"건너뜁니다: {image_path} (이미 자동 라벨링된 파일)")
            continue
        
        # 이미지에 대한 예측 수행
        try:
            results = model.predict(image_path, conf=conf_threshold, verbose=False)[0]
            
            # 예측 결과 가져오기
            boxes = []
            confidence_scores = []
            
            if len(results.boxes) > 0:
                for box in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = box
                    if conf >= conf_threshold:
                        boxes.append([x1, y1, x2, y2, cls_id, conf])
                        confidence_scores.append(conf)
            
            # LabelMe JSON 생성
            if boxes:
                json_path = create_labelme_json(image_path, boxes, class_names, confidence_scores, output_dir)
                print(f"생성됨: {json_path}")
            else:
                print(f"감지된 객체 없음: {image_path}")
                
        except Exception as e:
            print(f"처리 중 오류 발생: {image_path} - {str(e)}")
            continue
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"진행 상황: {processed_count}/{len(all_image_files)}")
    
    print(f"모든 처리 완료: {processed_count}개 이미지 처리됨")

if __name__ == "__main__":
    model_path = r"C:\Users\KDT-13\Desktop\Group 6\MNvision\4.Model Experimental Data\Test_3\yolo_ver3.pt"
    image_dir = r"C:\Users\KDT-13\Desktop\Group 6\MNvision\2.Data\photo\sample1"
    output_dir = None  # 이미지와 같은 폴더에 저장
    conf_threshold = 0.3
    
    yolo_to_labelme(model_path, image_dir, output_dir, conf_threshold)
    print("자동 라벨링이 완료되었습니다. LabelMe에서 결과를 검토하고 수정하세요.")