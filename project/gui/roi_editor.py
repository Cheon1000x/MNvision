from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF, pyqtSignal


class ROIEditor(QWidget):
    roi_defined = pyqtSignal(list)  # 최종 ROI 좌표를 외부에 전달하는 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 마우스 추적
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 부모 위젯의 이벤트를 받게 하기 위해 설정

        self.points = []  # 사용자 클릭 좌표들
        self.finished = False
        self.temp_point = None  # 마우스 현재 위치

    def mousePressEvent(self, event):
        if self.finished:
            return

        if event.button() == Qt.LeftButton:
            # 좌클릭으로 점을 추가
            pos = event.pos()
            self.points.append(QPointF(pos))
            self.update()  # 화면 업데이트

        elif event.button() == Qt.RightButton and len(self.points) >= 3:
            # 우클릭 시 폴리곤을 확정
            self.finished = True
            polygon_coords = [(pt.x(), pt.y()) for pt in self.points]
            self.roi_defined.emit(polygon_coords)  # ROI가 완성되면 시그널 전송
            self.update()

    def mouseMoveEvent(self, event):
        if not self.finished:
            # 마우스를 이동하면서 마지막 점과 현재 점을 선으로 그리기
            self.temp_point = event.pos()
            self.update()

    def paintEvent(self, event):
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
