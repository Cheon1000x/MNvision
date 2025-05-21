# recorder/saver.py

import cv2
import os
import threading
from datetime import datetime

class VideoSaver:
    """
    이벤트 발생 시 영상과 로그를 저장하는 클래스 (스레드 지원)
    """
    def __init__(self, cam_num, save_dir="resources/logs/", fps=30, log_viewer=None):
        self.cam_num = cam_num
        self.fps = fps
        self.log_viewer = log_viewer
        self.save_dir = os.path.join(save_dir, str(cam_num))
        os.makedirs(self.save_dir, exist_ok=True)

    def save_clip(self, frames, event_time, label="event", iou=0.0):
        if not frames:
            print("[WARN] 저장할 프레임이 없습니다.")
            return None

        timestamp_str = datetime.fromtimestamp(event_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{self.cam_num}_{label}.mp4"
        save_path = os.path.join(self.save_dir, filename)

        try:
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, self.fps, (width, height))

            for frame in frames:
                out.write(frame)
            out.release()

            print(f"[INFO] 🎞️ 영상 저장 완료: {save_path}")
            return save_path
        except Exception as e:
            print(f"[ERROR] 영상 저장 중 오류: {e}")
            return None

    def save_logs(self, event_time, label="event", iou=0.0):
        timestamp_str = datetime.fromtimestamp(event_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{self.cam_num}_{label}.txt"
        save_path = os.path.join(self.save_dir, filename)
        log_text = f"{timestamp_str},{self.cam_num},{label},{iou:.3f}"

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(log_text)

            if self.log_viewer:
                self.log_viewer.append_log_text(log_text)

            print(f"[INFO] 📝 로그 저장 완료: {save_path}")
            return save_path
        except Exception as e:
            print(f"[ERROR] 로그 저장 실패: {e}")
            return None

    def save_event(self, frames, event_time, label="event", iou=0.0):
        """
        영상 + 로그를 동기 방식으로 저장
        """
        self.save_clip(frames, event_time, label, iou)
        self.save_logs(event_time, label, iou)

    def save_event_async(self, frames, event_time, label="event", iou=0.0):
        """
        영상 + 로그 저장을 비동기로 처리 (스레드 기반)
        """
        thread = threading.Thread(
            target=self.save_event,
            args=(frames, event_time, label, iou),
            daemon=True
        )
        thread.start()
