from collections import deque

class VideoRecorder:
    def __init__(self, buffer_size=300):  # 예: 10초 @30fps
        self.buffer = deque(maxlen=buffer_size)

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_clip(self, output_path):
        # TODO: 현재 버퍼 내용 저장
        pass