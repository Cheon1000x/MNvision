from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

class ROIEditor(QWidget):
    """ 
    ROI 영역을 그리고 확정하는 기능
    """
    roi_defined = pyqtSignal(list, int)  # 추가된 부분: 카메라 ID를 포함한 시그널

    def __init__(self, video_widget, cam_id):
        super().__init__(video_widget)
        self.cam_id = cam_id  # 카메라 ID 저장
        self.points = []
        self.finished = False
        self.temp_point = None
        self.saved_polygon = None  # 확정된 ROI 폴리곤 저장용
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 좌클릭으로 점을 추가
            self.points.append(QPointF(event.pos()))
            self.update()  # 화면 업데이트

        elif event.button() == Qt.RightButton and len(self.points) >= 3:
            # 우클릭으로 ROI 확정
            self.finished = True
            self.temp_point = None
            self.saved_polygon = self.points.copy()  # ROI 유지용 저장
            polygon_coords = [(pt.x(), pt.y()) for pt in self.points]
            self.roi_defined.emit(polygon_coords, self.cam_id)
            self.update()

    def mouseMoveEvent(self, event):
        if not self.finished:
            # 마우스를 이동하면서 마지막 점과 현재 점을 선으로 그리기
            self.temp_point = event.pos()
            self.update()

    def paintEvent(self, event):
        if not self.points:
            return

        painter = QPainter(self)
        polygon = QPolygonF(self.saved_polygon if self.finished else self.points)

        if not self.finished and self.temp_point:
            # ROI 설정 중일 때 (마우스로 그릴 때)
            painter.setPen(QPen(Qt.red, 2))
            painter.drawPolyline(polygon)
            painter.drawLine(self.points[-1], self.temp_point)
        else:
            # 확정된 ROI가 있으면 파란색 다각형 (닫힌 형태)
            closed_polygon = polygon + QPolygonF([polygon[0]])
            painter.setPen(QPen(Qt.blue, 2))
            painter.drawPolygon(closed_polygon)

        # 점도 항상 파란색으로 그림
        dot_pen = QPen(Qt.blue, 5)
        painter.setPen(dot_pen)
        for pt in polygon:
            painter.drawPoint(pt)

    def reset(self):
        # ROI 설정 초기화
        self.points.clear()
        self.finished = False
        self.temp_point = None
        self.saved_polygon = None  # 저장된 ROI도 초기화
        self.update()

    def get_polygon(self):
        """[(x1, y1), (x2, y2), ...] 형식으로 반환"""
        return [(pt.x(), pt.y()) for pt in self.points] if self.finished else None

    def load_polygon(self, polygon_coords):
        """
        외부에서 ROI를 설정할 때 사용. [(x1, y1), (x2, y2), ...] 형식의 좌표 리스트
        """
        print('roi_e',[QPointF(x, y) for x, y in polygon_coords] )
        self.points = [QPointF(x, y) for x, y in polygon_coords]
        self.saved_polygon = self.points.copy()
        self.finished = True
        self.temp_point = None
        self.update()
