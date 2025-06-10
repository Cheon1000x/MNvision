import subprocess
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,  QSplashScreen, QProgressBar, QVBoxLayout, QLabel,
    QPushButton
)
import os, json
from PyQt5.QtGui import QFont, QGuiApplication, QCursor, QIcon, QMouseEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from gui.main_window import MainWindow
from gui.config_window import ConfigWindow

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class StartWindow(QWidget):
    start_main_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        ## 컨트롤할 변수들 생성 - 통합 초기화
        self.current_config = None
        
        self.old_pos = None 
        self.normal_geometry = self.geometry() 
        
        self.setWindowFlags(Qt.FramelessWindowHint) 
        screen = QGuiApplication.primaryScreen()
        self.size = screen.availableGeometry()
        self.setStyleSheet(""" background-image: url(resources/icons/bg_start.png);
                background-repeat: no-repeat;
                background-position: center; """)
        # self.setWindowTitle("Forklift Detection")
        
        self.setMinimumSize(600, 400)
        self.setContentsMargins(0,0,0,0)
        
        ## 전체 영역 분할
        self.main_v_layout = QVBoxLayout(self)
        self.main_v_layout.setContentsMargins(0,0,0,0)
        self.main_v_layout.setSpacing(0) # 레이아웃 아이템 간 간격 제거 (필요시)
        
        
        ## 윈도우 버튼 영역 설정
        self.ui_area = QWidget(self) # 부모 위젯을 self로 명시
        self.ui_area.setFixedHeight(40) # 고정 높이
        # self.ui_area.setStyleSheet("background-color: #06E9E1;")
        
        ui_layout = QHBoxLayout(self.ui_area)
        ui_layout.setContentsMargins(0,0,0,0)
        ui_layout.setSpacing(0)
        
        # 타이틀바 왼쪽에 여백 추가 (버튼들을 오른쪽으로 밀기 위함)
        ui_layout.addStretch(1) 
        
        minimize_btn = QPushButton("")
        self.maximize_restore_btn = QPushButton("") # self.maximize_restore_btn을 인스턴스 변수로 변경 (토글 기능 위함)
        exit_btn = QPushButton("")

        minimize_btn.clicked.connect(self.showMinimized)
        # self.maximize_restore_btn에 대한 연결을 toggle_maximize_restore로 통일
        self.maximize_restore_btn.clicked.connect(self.toggle_maximize_restore)
        exit_btn.clicked.connect(self.close)
        
        # 각 버튼에 맞는 아이콘 스타일 적용
        minimize_btn.setStyleSheet(
            """
            QPushButton {
                background-image: url(resources/icons/mini_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:hover {
                background-image: url(resources/icons/mini_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:pressed {
                background-image: url(resources/icons/mini_sb.png);
                background-repeat: no-repeat;
                background-position: center;
                
                background-color: #2FDFD9;
                border: 3px solid #00D2B5;
                border-radius: 5px;
            }
            """
        )
        self.maximize_restore_btn.setStyleSheet(
            """
            QPushButton {
                background-image: url(resources/icons/max_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:hover {
                background-image: url(resources/icons/max_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:pressed {
                background-image: url(resources/icons/max_sb.png);
                background-repeat: no-repeat;
                background-position: center;
                
                background-color: #2FDFD9;
                border: 3px solid #00D2B5;
                border-radius: 5px;
            }
            """
        )
        exit_btn.setStyleSheet(
            """
            QPushButton {
                background-image: url(resources/icons/close_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:hover {
                background-image: url(resources/icons/close_s.png);
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton:pressed {
                background-image: url(resources/icons/close_sb.png);
                background-repeat: no-repeat;
                background-position: center;
                
                background-color: #2FDFD9;
                border: 3px solid #00D2B5;
                border-radius: 5px;
                
            }
            """
        )
        
        minimize_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.maximize_restore_btn.setCursor(QCursor(Qt.PointingHandCursor))
        exit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        minimize_btn.setFixedSize(40,40)
        self.maximize_restore_btn.setFixedSize(40,40)
        exit_btn.setFixedSize(40,40)
        
        minimize_btn.setFlat(True)
        self.maximize_restore_btn.setFlat(True)
        exit_btn.setFlat(True)
        
        ui_layout.addWidget(minimize_btn)
        # ui_layout.addWidget(self.maximize_restore_btn)
        ui_layout.addWidget(exit_btn)
        
        self.main_v_layout.addWidget(self.ui_area) 
        
        ## 하단 영역 생성
        self.sm_area = QWidget(self)
        sm_layout = QHBoxLayout(self.sm_area)
        sm_layout.setContentsMargins(0,0,0,40)
        
        
        ## startmain logo area
        self.logo_area = QWidget(self) # 부모 위젯을 self로 명시
        logo_layout = QVBoxLayout(self.logo_area) # 로고 영역 내부 레이아웃
        logo_layout.setContentsMargins(50,0,0,0)
        logo_layout.addStretch(1) # 로고를 중앙으로
        
        logo_btn = QPushButton()
        logo_btn.setEnabled(False) # 클릭 불가능하게
        logo_btn.setFixedSize(255, 255) # 로고 버튼 크기 고정 (예시)
        # logo_icon = QIcon("resources/icons/logo_main.png")
        # logo_btn.setIcon(logo_icon)
        # logo_btn.setIconSize(QSize(255, 255)) # QSize 임포트 필요: from PyQt5.QtCore import QSize
        # logo_btn.setFixedSize(255, 255) 
        logo_btn.setStyleSheet("""
            QPushButton {
                background-image: url(resources/icons/logo_main.png);
                background-repeat: no-repeat;
                background-position: center;
                border: none; /* 로고 버튼 테두리 제거 */
            }
        """)
        logo_layout.addWidget(logo_btn, alignment=Qt.AlignCenter) # 로고 중앙 정렬
        logo_layout.addStretch(1)
        
        sm_layout.addWidget(self.logo_area) 
        
     
         ## btn area
        self.btn_area = QWidget(self) # 부모 위젯을 self로 명시
        btn_layout = QVBoxLayout(self.btn_area)
        btn_layout.setContentsMargins(0,20,0,20)
        btn_layout.setSpacing(10) # 버튼 간 간격
        
        # 버튼을 중앙에 모으기 위해 위아래 스트레치 추가
        btn_layout.addStretch(1) 
        
        start_btn = QPushButton("") # 텍스트 추가
        config_btn = QPushButton("")
        logs_btn = QPushButton("")
        
        start_btn.setCursor(QCursor(Qt.PointingHandCursor))
        config_btn.setCursor(QCursor(Qt.PointingHandCursor))
        logs_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        # 버튼의 높이 고정 (필요시)
        start_btn.setFixedSize(160, 160) 
        config_btn.setFixedSize(160, 40)
        logs_btn.setFixedSize(160, 40)
        
        start_btn.setFlat(True)
        config_btn.setFlat(True)
        logs_btn.setFlat(True)
        
        start_btn.clicked.connect(self.on_start)
        config_btn.clicked.connect(self.openConfig)
        logs_btn.clicked.connect(self.openFolder)
        
        
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background-image: url(resources/icons/on.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/on_c.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:pressed {{
                background-image: url(resources/icons/on_b.png);
                background-repeat: no-repeat;
                background-position: center;
                border: none;
            }}
        """)

        # Config 버튼
        config_btn.setStyleSheet(f"""
            QPushButton {{
                
                background-image: url(resources/icons/config.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/config_c.png);
                background-repeat: no-repeat;
                background-position: center;
                
            }} 
            QPushButton:pressed {{
                background-image: url(resources/icons/config_b.png);
                background-repeat: no-repeat;
                background-position: center;
                border: none;
            }}
        """)

        # logs 버튼
        logs_btn.setStyleSheet(f"""
            QPushButton {{
                
                background-image: url(resources/icons/logs.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/logs_c.png);
                background-repeat: no-repeat;
                background-position: center;
                
            }} 
            QPushButton:pressed {{
                background-image: url(resources/icons/logs_b.png);
                background-repeat: no-repeat;
                background-position: center;
                border: none;
            }}
        """)
        
        # 버튼들을 레이아웃에 추가 (중앙 정렬)
        btn_layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(config_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(logs_btn, alignment=Qt.AlignCenter)
        btn_layout.addStretch(1) # 아래쪽 스트레치
        
        sm_layout.addWidget(self.btn_area) 
        self.main_v_layout.addWidget(self.sm_area) 
         
        
        # --- 창 드래그 기능 구현 ---
    def mousePressEvent(self, event: QMouseEvent):
        # 마우스 왼쪽 버튼이 눌렸을 때만 처리
        if event.button() == Qt.LeftButton:
            # 현재 마우스 위치(전역 좌표)에서 창의 왼쪽 상단 위치를 뺀 값 저장
            self.old_pos = event.globalPos() - self.pos()
            event.accept() # 이벤트 처리 완료

    def mouseMoveEvent(self, event: QMouseEvent):
        # 마우스 왼쪽 버튼이 눌린 상태에서 이동 중일 때만 처리
        if event.buttons() == Qt.LeftButton and self.old_pos is not None:
            # 새로운 창 위치 = 현재 마우스 전역 위치 - 저장된 오프셋
            self.move(event.globalPos() - self.old_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        # 마우스 버튼이 놓였을 때 오프셋 초기화
        self.old_pos = None
        event.accept()
        
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # 마우스 왼쪽 버튼이 더블클릭되었고, 타이틀바 영역 내에서 발생했을 때
        if event.button() == Qt.LeftButton:
            self.toggle_maximize_restore() # 최대화/복원 토글 함수 호출
            event.accept()
        super().mouseDoubleClickEvent(event) # 부모 클래스의 이벤트도 호출 (필요시)

    # 최대화/복원 토글 기능
    def toggle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
            # self.maximize_button.setText("□") # 최대화 버튼 텍스트 변경
        else:
            # self.showMaximized()
            self.showFullScreen()
            # self.maximize_button.setText("❐") # 복원 버튼 텍스트 변경

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    # --------------------------
    
    def on_start(self):
        if not self.current_config:
            self.current_config = self.get_default_config()
        
        # config와 함께 시그널 전송
        print('current_config',self.current_config)
        self.start_main_signal.emit(self.current_config)
        self.close() # 현재 StartWindow 닫기    
                        
    def openFolder(self):
        """ 
        폴더 열기 메서드
        """
        folder_path = os.path.abspath('./resources/logs/')
        # print(folder_path)
        if os.path.exists(folder_path):
            subprocess.Popen(["explorer", folder_path])
            print(f"{folder_path}")
        else:
            print(f"경로가 존재하지 않습니다. {folder_path}")

    def openConfig(self):
        try:
            # 완전히 독립적인 창으로 생성
            config_window = ConfigWindow()
            
            # 창 설정을 더 명확하게
            config_window.setParent(None)
            config_window.setAttribute(Qt.WA_DeleteOnClose, True)
            
            # 초기 설정값 로드
            if hasattr(self, 'current_config') and self.current_config:
                config_window.load_initial_config(self.current_config)
            
            # 시그널 연결
            config_window.config_changed.connect(self.on_config_changed)
            
            # 창 표시
            config_window.show()  # exec_() 대신 show() 사용
            
        except Exception as e:
            print(f"ConfigWindow 열기 오류: {e}")
    
    def get_default_config(self):
        """기본 설정값 반환"""
        import json
        default_config = {
            "confidence": 0.65,
            "cam1_mute": False,
            "cam2_mute": False,
            "show_labels": True,
            "default_confidence": 0.6,
            "default_cam1_mute": False,
            "default_cam2_mute": False,
            "default_show_labels": True
        }
        return json.dumps(default_config, ensure_ascii=False)
    
    def on_config_changed(self, config_json):
        """JSON 형태로 모든 설정을 한번에 받기"""
        print("받은 설정 JSON:", config_json)
        self.current_config = config_json
        
        # JSON을 딕셔너리로 파싱해서 사용
        import json
        config = json.loads(config_json)
        print(f"Confidence: {config['confidence']}")
        print(f"Cam1 Mute: {config['cam1_mute']}")
        print(f"Cam2 Mute: {config['cam2_mute']}")
        print(f"Show Labels: {config['show_labels']}")
        
        # MainWindow에 전달
        self.send_config_to_main_window(config_json)
    
    def send_config_to_main_window(self, config_json):
        """MainWindow에 설정 전달"""
        # 방법 1: 직접 메서드 호출
        if hasattr(self, 'main_window'):
            self.main_window.update_config(config_json)
            
        # 방법 2: 시그널로 전달
        # self.config_signal.emit(config_json)
    
    def on_confidence_changed(self, confidence):
        """개별 설정 변경 (기존 방식)"""
        print(f"Confidence 변경: {confidence}")
    
    def on_sound_changed(self, cam1_mute, cam2_mute):
        print(f"Sound 설정 변경: Cam1={cam1_mute}, Cam2={cam2_mute}")
    
    def on_label_changed(self, show_labels):
        print(f"Label 설정 변경: {show_labels}")