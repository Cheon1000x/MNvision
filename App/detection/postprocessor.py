import numpy as np

class PostProcessor:
    """ 
    예측값 처리.
    thresholds 설정.
    """
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

    def filter_results(self, detections):
        """
        감지 결과 중 confidence가 thresholds 이상인 것만 남김
        """
        filtered = []
        for (bbox, conf, cls) in detections:
            if conf >= self.conf_threshold:
                filtered.append((bbox, conf, cls))
        return filtered

