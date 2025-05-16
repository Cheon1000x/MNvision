import cv2
import os

def extract_frames_per_second(video_path, output_folder):
    # 비디오 파일 이름 추출 (확장자 제외)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 비디오별 출력 폴더 생성
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    # 비디오가 제대로 열렸는지 확인
    if not cap.isOpened():
        print(f"Error: 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # 비디오 길이(초)
    
    print(f"처리 중: {video_path}")
    print(f"FPS: {fps}")
    print(f"총 프레임 수: {frame_count}")
    print(f"비디오 길이: {duration:.2f}초")
    
    # 초당 1프레임을 위한 프레임 간격 계산
    frame_interval = int(fps)
    if frame_interval < 1:
        frame_interval = 1  # FPS가 1보다 작은 경우 모든 프레임 사용
    
    print(f"프레임 간격: {frame_interval} (매 {frame_interval}번째 프레임만 저장)")
    
    # 프레임 추출
    frame_idx = 0
    saved_count = 0
    
    while True:
        # 다음 프레임 읽기
        ret, frame = cap.read()
        
        # 더 이상 읽을 프레임이 없으면 종료
        if not ret:
            break
        
        # 초당 1프레임만 저장 (frame_interval 간격으로)
        if frame_idx % frame_interval == 0:
            # 프레임 저장
            frame_path = os.path.join(video_output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            # 저장 진행 상황 출력 (10프레임마다)
            if saved_count % 10 == 0:
                print(f"저장됨: {saved_count}개 프레임 (원본 프레임 {frame_idx}/{frame_count})")
        
        frame_idx += 1
    
    # 자원 해제
    cap.release()
    print(f"완료: {video_name} - 총 {saved_count}개의 프레임이 저장되었습니다 (예상: 약 {int(duration)}개)")

def process_video_folder(video_folder, output_folder):
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 비디오 파일 확장자 목록
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 폴더 내 모든 파일 검사
    video_files = []
    for filename in os.listdir(video_folder):
        file_path = os.path.join(video_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file_path)
    
    if not video_files:
        print(f"Error: {video_folder} 폴더에 비디오 파일이 없습니다.")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
    
    # 각 비디오 파일 처리
    for video_path in video_files:
        extract_frames_per_second(video_path, output_folder)
    
    print("모든 비디오 처리 완료!")

# 사용 예시
video_folder = r"C:\Users\KDT34\Desktop\MNvision\data\video"  # 비디오 파일들이 있는 폴더
output_folder = r"C:\Users\KDT34\Desktop\MNvision\data\photo"  # 프레임이 저장될 폴더
process_video_folder(video_folder, output_folder)