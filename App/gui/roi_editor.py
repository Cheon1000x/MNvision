from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
import datetime

class ROIEditor(QWidget):
    roi_defined = pyqtSignal(list, int)  # 추가된 부분: 카메라 ID를 포함한 시그널

    def __init__(self, video_widget, cam_id):
        super().__init__(video_widget)
        self.video_widget = video_widget
        self.cam_id = cam_id  # 카메라 ID 저장
        self.points = []
        self.finished = False
        self.temp_point = None
        self.now = datetime.datetime.now()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 좌클릭으로 점을 추가
            pos = event.pos()
            self.points.append(QPointF(pos))
            self.update()  # 화면 업데이트

        elif event.button() == Qt.RightButton and len(self.points) >= 3:
            # 우클릭 시 폴리곤을 확정
            self.finished = True
            polygon_coords = [(pt.x(), pt.y()) for pt in self.points]
            self.roi_defined.emit(polygon_coords, self.cam_id)  # 카메라 ID 포함하여 시그널 전송
            self.update()


    def mouseMoveEvent(self, event):
        if not self.finished:
            # 마우스를 이동하면서 마지막 점과 현재 점을 선으로 그리기
            self.temp_point = event.pos()
            self.update()

    def paintEvent(self, event):
        print(__class__.__name__)
        print(f"{self.now.second}[DEBUG] paintEvent called for cam_id: {self.cam_id}, points: {self.points}" )
        painter = QPainter(self)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
                    
        if self.points:
            # 점을 클릭해서 다각형을 만듦
            polygon = QPolygonF(self.points)
            if self.finished:
                # 다각형이 완성되었으면 첫 점과 마지막 점을 연결하여 그린다.
                polygon.append(self.points[0])  
                painter.drawPolygon(polygon)
            else:
                # 다각형을 그리기 전에 폴리라인으로 점을 연결하고, 마우스를 이동하면서 그려지는 선
                painter.drawPolyline(polygon)
                if self.temp_point:
                    painter.drawLine(self.points[-1], self.temp_point)

        # 점 찍기
        dot_pen = QPen(Qt.blue, 5)
        painter.setPen(dot_pen)
        for pt in self.points:
            painter.drawPoint(pt)

    def reset(self):
        # ROI 설정 초기화
        self.points.clear()
        self.finished = False
        self.temp_point = None
        self.update()

    def get_polygon(self):
        """[(x1, y1), (x2, y2), ...] 형식으로 반환"""
        return [(pt.x(), pt.y()) for pt in self.points] if self.finished else None
