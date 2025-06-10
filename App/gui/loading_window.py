# splash_screen.py
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

class SplashScreen(QSplashScreen):
    # 로딩 완료 신호를 보낼 수 있습니다.    
    finished = pyqtSignal() 

    def __init__(self, parent=None):
        # QPixmap을 사용하여 이미지를 배경으로 설정할 수 있습니다.
        # super().__init__(QPixmap("path/to/your/splash_image.png")) 
        super().__init__() # 이미지 없이 생성

        self.setWindowFlags(Qt.FramelessWindowHint) # 타이틀바 제거
        self.setStyleSheet("background-color: #2c3e50; color: white;") # 배경색 설정

        layout = QVBoxLayout(self)
        label = QLabel("앱 로딩 중...", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        layout.addWidget(self.progressBar)
        
        self.showMessage("초기화 중...", Qt.AlignBottom | Qt.AlignRight, Qt.white)

        # 3초 후에 시작 화면을 닫는 타이머 (예시)
        QTimer.singleShot(3000, self.close_splash) 
        
        # (실제 로딩 로직을 여기에 넣거나 별도 메서드 호출)
        # self.start_loading_process() 

    def close_splash(self):
        self.close()
        self.finished.emit() # 로딩 완료 신호 전송

    # 실제 로딩 작업을 시뮬레이션하는 메서드 (비동기로 실행될 수 있음)
    def start_loading_process(self):
        # 실제로는 여기서 파일 로드, 네트워크 연결, 데이터베이스 초기화 등을 수행
        # 진행률을 업데이트할 때 progressBar.setValue() 호출
        for i in range(101):
            self.progressBar.setValue(i)
            # time.sleep(0.03) # 실제 작업이 있을 때만
        self.close_splash()