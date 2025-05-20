# recorder/saver.py

import cv2
import os
from datetime import datetime

class VideoSaver:
    """ 
    이벤트 발생 시 영상과 기록을 저장하는 클래스
    """
    def __init__(self, cam_num, save_dir="resources/logs/", fps=30, log_viewer=None):
        self.cam_num = cam_num
        self.save_dir = save_dir
        self.fps = fps
        self.log_viewer = log_viewer  # ← LogViewer 연결
        os.makedirs(self.save_dir, exist_ok=True)

    def save_clip(self, frames, event_time, label="event", iou=0.0):
        # if not frames:
        #     print("저장할 프레임이 없습니다.")
        #     return None
        # print(frames.shape, frames.dtype)
        timestamp_str = datetime.fromtimestamp(event_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{self.cam_num}_{label}.mp4"
        save_video_path = os.path.join(self.save_dir, filename)

        print(f"[DEBUG] 저장할 영상 파일 경로: {save_video_path}")
        
        height, width, _  = frames[0].shape
        print(f"[DEBUG] 프레임 크기: {height}x{width}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video_path, fourcc, self.fps, (width, height))

        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"[INFO] 영상 저장 완료: {save_video_path}")
        return save_video_path
    
    def save_logs(self, event_time, label="event", cam_num=1, iou=0.0):
        timestamp_str = datetime.fromtimestamp(event_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{self.cam_num}_{label}.txt"
        save_log_path = os.path.join(self.save_dir, filename)
        log_text = f"{timestamp_str} | {self.cam_num} | {label} | {iou} in ROI"
        
        with open(save_log_path, 'w', encoding='utf-8') as file:
            file.write(log_text)
        
        if self.log_viewer:
            self.log_viewer.append_log_text(log_text)  # ← 실시간 로그 뷰 업데이트

        
        return save_log_path

