from PyQt5.QtWidgets import QMessageBox, QPushButton
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QBrush
from PyQt5.QtCore import Qt
import sys

def remove_custom_messagebox(parent):
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle("확인")
    msg_box.setText("\n정말 삭제하시겠습니까?\n")
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)

    # 먼저 show()를 호출한 후 버튼을 얻어 스타일 적용
    msg_box.show()
    for button in msg_box.findChildren(QPushButton):
        button.setStyleSheet("""
            QPushButton {
                background-color: #161616;
                color: #00D2B5;
                width:100px;
                height:50px;
                padding: 6px 12px;
                font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
                font-weight: bold;
                font-size: 25px;
            }
            QPushButton:hover {
                background-color: #00D2B5;
                color: black;
            }   
        """)
        
    
    msg_box.setStyleSheet("""
        QLabel {
        color: white;
        font-size: 35px;
        margin: 10px 30px;
        }
        
        QMessageBox {
            width:300px;
            height:200px;
            background-color: #161616;
            color: white;
            font-family: 'Pretendard', 'Helvetica Neue', Arial, sans-serif;
            font-weight: bold;
        }
    """)

    return msg_box.exec_()



class CircleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.circle_color = Qt.blue  # 기본 색상

    def setCircleColor(self, color):
        self.circle_color = color
        self.update()  # 다시 그리도록 요청

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(self.circle_color, Qt.SolidPattern))
        painter.drawEllipse(50, 0, 75, 75)  # x, y, width, height
