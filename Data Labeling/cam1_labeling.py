import cv2
import os
import pandas as pd

# 🎥 처리할 영상 3개 목록 (파일명 직접 입력)
video_files = [
    "../PROJECT/data/cam1/20231214095219.avi",
    "../PROJECT/data/cam1/20231231143549.avi",
    "../PROJECT/data/cam1/20231231145055.avi"
]

# 📁 프레임 저장 폴더
output_dir = "../PROJECT/data/extracted_frames"
os.makedirs(output_dir, exist_ok=True)

all_frames = []

# 각 영상마다 반복
for video_path in video_files:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️ 영상 열기 실패: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 1초 간격
    frame_idx = 0
    saved_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"🎞 전처리 시작: {video_name} (fps={fps})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_name = f"{video_name}_{saved_count:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            all_frames.append(frame_name)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"✅ 완료: {video_name} → {saved_count}개 프레임 추출")

# 결과 프레임 목록 출력
df = pd.DataFrame({"파일명": sorted(all_frames)})
print("📸 전체 추출된 프레임 목록:")
print(df)


