# recorder/video_buffer.py

import cv2
import time
import threading
from collections import deque

class VideoBuffer:
    def __init__(self, max_seconds=5, fps=30):
        self.fps = fps
        self.max_frames = max_seconds * fps
        self.buffer = deque(maxlen=self.max_frames * 2)  # ±5초니까 10초치 저장
        self.lock = threading.Lock()

    def add_frame(self, frame):
        timestamp = time.time()
        with self.lock:
            self.buffer.append((timestamp, frame.copy()))

    def get_clip(self, event_time, seconds_before=5, seconds_after=5):
        start_time = event_time - seconds_before
        end_time = event_time + seconds_after
        clip_frames = []

        with self.lock:
            for timestamp, frame in list(self.buffer):
                if start_time <= timestamp <= end_time:
                    clip_frames.append(frame)

        return clip_frames
