# recorder/saver.py

import cv2
import os
import threading
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime

class VideoSaver(QObject):
    clip_saved_signal = pyqtSignal(int)
    log_appended_signal = pyqtSignal(str) # ⭐ 새로운 시그널 추가 ⭐

    def __init__(self, cam_num, save_dir="resources/logs/", fps=30, log_viewer=None):
        super().__init__()
        self.cam_num = cam_num
        self.fps = fps
        # self.log_viewer = log_viewer # ⭐ 직접 참조하지 않도록 제거 (옵션) ⭐
        self.save_dir = os.path.join(save_dir, str(cam_num))
        os.makedirs(self.save_dir, exist_ok=True)
        # ...

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
            self.clip_saved_signal.emit(self.cam_num)
            
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

             # if self.log_viewer: # ⭐ 이 부분 제거 ⭐
            #     self.log_viewer.append_log_text(log_text) # ⭐ 직접 호출 대신 시그널 사용 ⭐

            self.log_appended_signal.emit(log_text) # ⭐ 시그널 emit ⭐
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
