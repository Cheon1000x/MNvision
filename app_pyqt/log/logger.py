class EventLogger:
    def __init__(self):
        self.logs = []

    def log_event(self, timestamp, camera_id, video_path):
        self.logs.append((timestamp, camera_id, video_path))

    def get_logs(self):
        return self.logs
