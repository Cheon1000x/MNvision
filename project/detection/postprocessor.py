import numpy as np

class PostProcessor:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

    def filter_results(self, detections):
        """
        감지 결과 중 confidence가 일정 기준 이상인 것만 남김
        """
        filtered = []
        for (bbox, conf, cls) in detections:
            if conf >= self.conf_threshold:
                filtered.append((bbox, conf, cls))
        return filtered


    # def is_in_roi(self, xyxy):
    #     # ROI 내에 객체가 있는지 확인
    #     x1, y1, x2, y2 = xyxy
    #     if self.roi:
    #         roi_x1, roi_y1 = self.roi[0]
    #         roi_x2, roi_y2 = self.roi[1]
    #         return not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)
    #     return True
