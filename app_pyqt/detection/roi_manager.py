class ROIManager:
    def __init__(self):
        self.roi_rects = []

    def set_roi(self, camera_id, rect):
        self.roi_rects.append((camera_id, rect))

    def is_within_roi(self, camera_id, bbox):
        # TODO: bbox가 ROI 내부에 있는지 확인
        return True
