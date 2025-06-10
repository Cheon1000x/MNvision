KMP_DUPLICATE_LIB_OK='TRUE'

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTabWidget, QSizePolicy, QLabel, QGraphicsDropShadowEffect,
    QApplication
)
import os, json
from PyQt5.QtGui import QFontDatabase, QFont, QGuiApplication, QCursor, QIcon, QMouseEvent 
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from gui.video_widget import VideoWidget
from gui.roi_editor import ROIEditor
from gui.log_viewer import LogViewer
import time
from collections import Counter
from utils import design

# from utils.alert_manager import AlertManager 

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = {
                "confidence": 0.6,
                "cam1_mute": True,
                "cam2_mute": True,
                "show_labels": False
            }
            print('default')
        else:
            self.config = json.loads(config)
            
        # print(self.config)
        
        ## 컨트롤할 변수들 생성 - 통합 초기화
        self.video_widgets = {}
        self.roi_editors = {}
        self.info_widgets = {}
        self.log_viewers = {}  
        self.reset_timers = {}
         
        self.spotlights = {}
        self.onoff_labels = {}
        self.event_labels = {}  
        self.info_labels = {}
        
        self.info_sss = None
        self.infor_sss = None
        
        self.old_pos = None 
        # QMainWindow는 frameGeometry()를 사용하면 창 테두리까지 포함한 정확한 크기를 얻습니다.
        self.normal_geometry = self.geometry() 
        
        self.setWindowFlags(Qt.FramelessWindowHint) 
        screen = QGuiApplication.primaryScreen()
        self.size = screen.availableGeometry()
        self.setWindowTitle("Forklift Detection")
        
        self.setMinimumSize(400, 400)
        self.setContentsMargins(0,0,0,0)
        
        ## 전체 영역 분할
        central_widget = QWidget()
        central_widget.setObjectName("central_widget")  # ✔ 스타일링 타겟 명확히

        central_widget.setStyleSheet("""
            #central_widget {
                background-image: url(resources/icons/bg.png);
                background-repeat: no-repeat;
                background-position: center;
            }
        """)
        self.setCentralWidget(central_widget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0,10,0,10)
        
        ## UI 영역 설정
        self.ui_area = QWidget()
        ui_layout = QHBoxLayout(self.ui_area)
        ui_layout.setContentsMargins(50,0,0,0)
        self.ui_area.setFixedHeight(50) 
        # self.ui_area.setStyleSheet("background-color: #161616;")
        main_layout.addWidget(self.ui_area, alignment=Qt.AlignCenter) 
        
        # 스타일시트 통합
        self.btn_design = """
            background-color: transparent;
            border-radius:  5px;
        """
        
        # Start 버튼
        # logo
        logo_btn = QPushButton("")
        logo_btn.setFlat(True)
        logo_btn.setEnabled(False) # 클릭 불가능하게
        # Config 버튼
        
        logo_btn.setStyleSheet(f"""
            QPushButton {{
                background-image: url(resources/icons/logo_wide.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)
        logo_btn.setFixedSize(210,50)
        ui_layout.addWidget(logo_btn, alignment=Qt.AlignLeft)
        
        
        config_btn = QPushButton("")
        minimize_btn = QPushButton("")
        minimize_btn.clicked.connect(self.showMinimized)
        maximize_btn = QPushButton("")
        maximize_btn.clicked.connect(lambda: self.move(0, 0))
        maximize_btn.clicked.connect(self.showFullScreen)
        exit_btn = QPushButton("")
        exit_btn.clicked.connect(self.close)
        
        config_btn.setCursor(QCursor(Qt.PointingHandCursor))
        minimize_btn.setCursor(QCursor(Qt.PointingHandCursor))
        maximize_btn.setCursor(QCursor(Qt.PointingHandCursor))
        exit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Config 버튼
        config_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/config_m.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/config_mc.png);
                background-repeat: no-repeat;
                background-position: center;
            }} 
            QPushButton:pressed {{
                background-color: #00D2B5;
                
                border-radius:5px;
                background-image: url(resources/icons/config_mb.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)

        # minimize 버튼
        minimize_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/mini_m.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/mini_mc.png);
                background-repeat: no-repeat;
                background-position: center;
            }} 
            QPushButton:pressed {{
                background-color: #00D2B5;
                
                border-radius:5px;
                background-image: url(resources/icons/mini_mb.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)
        
        # maximize 버튼
        maximize_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/max_m.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/max_mc.png);
                background-repeat: no-repeat;
                background-position: center;
                
            }} 
            QPushButton:pressed {{
                background-color: #00D2B5;
                
                border-radius:5px;
                background-image: url(resources/icons/max_mb.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)
        
        # Exit 버튼
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                {self.btn_design}
                background-image: url(resources/icons/close_m.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:hover {{
                background-image: url(resources/icons/close_mc.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
            QPushButton:pressed {{
                background-color: #00D2B5;
                
                border-radius:5px;
                background-image: url(resources/icons/close_mb.png);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)
        
        config_btn.setFixedSize(60,50)
        minimize_btn.setFixedSize(60,50)
        maximize_btn.setFixedSize(60,50)
        exit_btn.setFixedSize(60,50)
        
        # ui_layout.addWidget(config_btn)
        ui_layout.addWidget(minimize_btn)
        ui_layout.addWidget(maximize_btn)
        ui_layout.addWidget(exit_btn)
        
        ## 비디오 영역
        self.video_area = QWidget()
        self.video_layout = QHBoxLayout()
        self.video_layout.setContentsMargins(40,0,40,0)
        self.video_layout.setSpacing(20)
        self.video_area.setLayout(self.video_layout)
        
        main_layout.addWidget(self.video_area)
        self.main_layout = main_layout
        
        # Config가 전달된 경우에만 초기화 진행
        if self.config:
            # QTimer를 사용하여 UI가 완전히 그려진 후 초기화 작업 실행
            QTimer.singleShot(100, self.start_initialization)
    
    def start_initialization(self):
        """초기화 작업을 순차적으로 실행"""
        try:
            # 1단계: on_start 실행 (config 사용)
            self.on_start(self.config)
            
            # 2단계: on_start 완료 후 추가 작업들
            QTimer.singleShot(500, self.post_start_tasks)
            
        except Exception as e:
            print(f"초기화 오류: {e}")
    
    def post_start_tasks(self):
        """on_start 완료 후 실행할 추가 작업들"""
        try:
            # 여기에 on_start 이후에 실행할 작업들을 추가
            self.setup_additional_features()
            self.finalize_ui()
            print("모든 초기화 작업 완료")
            
        except Exception as e:
            print(f"후속 작업 오류: {e}")

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
        
    def update_config(self, config_json):
        """StartWindow에서 받은 설정 적용"""
        import json
        try:
            self.config = json.loads(config_json)
            print("MainWindow에서 설정 업데이트:", self.config)
            
            # 실제 설정 적용
            self.apply_confidence(self.config['confidence'])
            self.apply_sound_settings(self.config['cam1_mute'], self.config['cam2_mute'])
            self.apply_label_settings(self.config['show_labels'])
            
        except json.JSONDecodeError:
            print("JSON 파싱 오류")
    
    def apply_confidence(self, confidence):
        print(f"Confidence 적용: {confidence}")
        # 실제 confidence 로직 적용
    
    def apply_sound_settings(self, cam1_mute, cam2_mute):
        print(f"소리 설정 적용: Cam1={cam1_mute}, Cam2={cam2_mute}")
        # 실제 소리 설정 로직 적용
    
    def apply_label_settings(self, show_labels):
        print(f"라벨 설정 적용: {show_labels}")
        # 실제 라벨 표시 로직 적용
    
    def get_current_config_json(self):
        """현재 설정을 JSON으로 반환"""
        import json
        return json.dumps(self.config, ensure_ascii=False)    
        
        
        
    def on_start(self, config):

        """ 
        시작시 cam의 vw, roi, log 생성
        """
        print('-'*50)
        print(config)
        print('-'*50)
        self.main_layout.removeWidget(self.ui_area)
        self.main_layout.insertWidget(0, self.ui_area)
        self.setMinimumSize(self.size.width(), self.size.height())
        self.showFullScreen()
        
        if self.video_widgets:
            return
        
        for cam_id in range(1, 3):  # CAM 1과 CAM 2
            cam_widget = QWidget()
            cam_widget.setObjectName("cam_widget")  # ✔ 스타일링 타겟 명확히
            
            cam_widget.setStyleSheet(f"""
                QWidget#cam_widget {{
                background-image: url(resources/icons/bg_cam_test.png);
                background-repeat: no-repeat;
                background-position: center;
                border-radius:  10px;
                }}
                           """)
            cam_layout = QVBoxLayout(cam_widget)
            cam_layout.setContentsMargins(20, 20, 20, 20)
            cam_layout.setSpacing(20)
            cam_widget.setLayout(cam_layout)

            # 비디오 위젯
            vw = VideoWidget(cam_num=cam_id, video_path=f"resources/videos/sample{cam_id}.avi" , conf_threshold = config['confidence'])
            if config['show_labels'] is False:
                vw.vthread.label_visible = False
            else:
                vw.vthread.label_visible = True
            
            vw_size = [int((self.size.width()-200) * 0.5), int((self.size.width()-200) * 9/32)]
            vw.setFixedSize(vw_size[0], vw_size[1])
            vw.setStyleSheet(""" 
                            border: 2px solid #000000;
                            border-radius: 10px;
                             """)
            self.video_widgets[cam_id] = vw

            self.create_roi_editor(cam_id, vw)
            polygon = self.load_roi_from_file(cam_id)
            if polygon:
                print(""" 
                      11111111111111111111
                      vthread
                      """)
                vw.set_roi_editor(self.roi_editors[cam_id])
                vw.set_roi(polygon)
                vw.vthread.set_roi(polygon)
                print(f'main_vw{cam_id}', polygon)

            # 정보 영역
            # 인포에어리아 00, 03은 크기 고정 및 비율고정
            ## 01, 02 는 최소 설정하되 비디오크기에 따라 변하도록.
            ## vw.setFixedSize(int((self.size.width()-180) * 0.5) 이므로 
            ## 신호등사이즈 120, 버튼 사이즈 100과 spacing 10씩 고려 총 240을 추가로 빼고 /2로 나누어 설정
            # background-image: url(resources/icons/bg_info.png);
            #     background-repeat: no-repeat;
            #     background-position: center;
            #     background-color: #ffffff;
                
            #     border: 10px solid white;
            #     border-radius: 20px; 
                
            
            
            info_sss =(f"""
                QWidget#info_widget {{
                background-image: url(resources/icons/bg_info.png);
                background-repeat: no-repeat;
                background-position: center;
                
                }}
            """)
            infor_sss =(f"""
                QWidget#info_widget {{
                background-image: url(resources/icons/bg_info_r.png);
                background-repeat: no-repeat;
                background-position: center;
                
                }}
            """)
            self.info_sss = info_sss
            self.infor_sss = infor_sss
            
            info_widget = QWidget()
            info_widget.setObjectName("info_widget")  # ✔ 스타일링 타겟 명확히
            info_widget.setStyleSheet(info_sss)
            info_width = vw_size[0]
            info_widget.setFixedSize(info_width, 130)
            
            info_layout = QHBoxLayout()
            info_widget.setLayout(info_layout)
            
            info_layout.setContentsMargins(20,0,20,0)
            info_layout.setSpacing(50)
            self.info_widgets[cam_id] = info_widget

            info00w = QWidget()
            info00 = QVBoxLayout(info00w)
            info00w.setStyleSheet(""" 
                                  border:none;
                                  """)
            # 신호등
            cam_name_btn = QPushButton()
            cam_name_btn.setEnabled(False)
            cam_name_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/cam{cam_id}.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    border-radius:  5px;
                }}
            """)
            cam_name_btn.setFixedSize(120,45)
            info00.addWidget(cam_name_btn)
            
            spotlight =  QPushButton()
            spotlight.setEnabled(False)
            spotlight.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/safe.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    border-radius:  5px;
                }}
            """)
            self.spotlights[cam_id] = spotlight
            spotlight.setFixedSize(120,45)
            info00.addWidget(spotlight)
            
            ##                  cam_w -spot.w-reset.b-margin-spacing*3 /2
            # info12_width = int(info_width-120-100-60-60)*0.5
            info12_width = 200
            # 왼쪽 정보
            info01w = QWidget()
            info01 = QVBoxLayout(info01w)
            info01.setContentsMargins(0,10,0,10)
            info01w.setFixedSize(200, 130)
            info01w.setObjectName("info01w")  # ✔ 스타일링 타겟 명확히
            info01w.setStyleSheet(f"""
                QWidget#info01w {{
                background-image: url(resources/icons/bg_info_test.png);
                background-repeat: no-repeat;
                background-position: center;
                border-radius:  10px;
                }}
                """)
            
            info_label_sss = (""" 
                color: #E6E6E6; 
                font-size: 25px;
                font-family: 'Koulen-Regular', 'pretendard-bold', Arial;
                font-weight: bold;     
             """)
            
            info1 = QLabel(f"{'mute/on'}")
            info1.setStyleSheet(info_label_sss)
            
            info2 = QLabel(f"event_type")
            info2.setStyleSheet(info_label_sss)
            
            for lbl in (info1, info2):
                info01.addWidget(lbl, alignment=Qt.AlignCenter)
            self.onoff_labels[cam_id] = info1
            self.event_labels[cam_id] = info2

            # 가운데 정보
            info02w = QWidget()
            info02 = QVBoxLayout(info02w)
            info02.setContentsMargins(0,10,0,10)
            info02w.setFixedSize(200, 130)
            info02w.setObjectName("info02w")  # ✔ 스타일링 타겟 명확히
            info02w.setStyleSheet(f"""
                QWidget#info02w {{
                background-image: url(resources/icons/bg_info_test.png);
                background-repeat: no-repeat;
                background-position: center;
                border-radius:  10px;
                }}
                """)
            
            
            info3 = QLabel("")
            info3.setStyleSheet(info_label_sss)
            info4 = QLabel("")
            info4.setStyleSheet(info_label_sss)
            
            for lbl in (info3, info4):
                info02.addWidget(lbl)
            self.info_labels[cam_id] = (info3, info4)
            
            # 버튼
            reset_btn = QPushButton("")
            reset_btn.setStyleSheet(f"""
                QPushButton {{
                    background-image: url(resources/icons/reset.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    border-radius:  5px;
                }}
                QPushButton:hover {{
                    background-image: url(resources/icons/reset_c.png);
                    background-repeat: no-repeat;
                    background-position: center;
                }}
                QPushButton:pressed {{
                    background-image: url(resources/icons/reset_b.png);
                    background-repeat: no-repeat;
                    background-position: center;
                }}
            """)
            reset_btn.setFixedSize(100, 100)
            reset_btn.clicked.connect(lambda _, cid=cam_id: self.reset_roi(cid))

            info_layout.addWidget(info00w, alignment=Qt.AlignLeft)
            info_layout.addWidget(info01w, alignment=Qt.AlignLeft)
            info_layout.addWidget(info02w, alignment=Qt.AlignCenter)
            info_layout.addWidget(reset_btn, alignment=Qt.AlignRight)

            # LogViewer
            lv = LogViewer(cam_id)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setFixedWidth(vw_size[0])
            
            
            print('-1')
            print("config[f'cam cam_id_mute']", config[f'cam{cam_id}_mute'])
            print('-1')
            print('lv.mute_optl', lv.mute_opt)
            print('vw.vthread.alert_manager.mute_opt', vw.vthread.alert_manager.mute_opt)
            print('-1')
            
            if config[f'cam{cam_id}_mute'] != lv.mute_opt:
                vw.vthread.alert_manager.mute_opt = config[f'cam{cam_id}_mute']
                lv.mute_opt = config[f'cam{cam_id}_mute']
                print('lv.load_mute_btn ')
                lv.load_mute_btn()
                # print("config[f'cam cam_id_mute']", config[f'cam{cam_id}_mute'], '==', lv.mute_opt)
            else:
                pass
            
            lv.mute_opt_TF.connect(vw.vthread.alert_manager.mute_control)
            lv.mute_opt_TF.connect(lv.mute_control)
            
            print('-')
            print("config[f'cam cam_id_mute']", config[f'cam{cam_id}_mute'])
            print('-')
            print('lv.mute_optl', lv.mute_opt)
            print('vw.vthread.alert_manager.mute_opt', vw.vthread.alert_manager.mute_opt)
            print('-')
            
            
            self.log_viewers[cam_id] = lv  # ✅ 딕셔너리에 저장 추가
            
            
            # ROI 에디터 설정
            roi_editor = self.roi_editors.get(cam_id)
            if roi_editor and vw:
                roi_editor.setParent(vw)
                roi_editor.setGeometry(vw.rect())
                roi_editor.show()
                roi_editor.raise_()

            # 시그널 연결
            vw.video_saver.clip_saved_signal.connect(lv.loadLogs)
            vw.vthread.on_triggered.connect(self.onoff_info)
            vw.vthread.mute_triggered.connect(self.onoff_info)
            vw.vthread.event_triggered.connect(self.event_info)
            vw.vthread.alert_manager.on_alert_signal.connect(self.lightControl, type=Qt.QueuedConnection)
            # vw.vthread.event_triggered.connect(lambda: self.make_delayed_loader(lv)())
            vw.vthread.info_triggered.connect(self.handle_result)
           
            if vw.video_saver: # video_saver 인스턴스가 존재한다면
                vw.video_saver.log_appended_signal.connect(lv.append_log_text, type=Qt.QueuedConnection)

            # 레이아웃 추가
            cam_layout.addWidget(vw, alignment=Qt.AlignCenter)
            cam_layout.addWidget(info_widget, alignment=Qt.AlignCenter)
            cam_layout.addWidget(lv, alignment=Qt.AlignCenter)
            self.video_layout.addWidget(cam_widget)

     
    def lightControl(self, str, cam_num):
        # print(f"\n[lightControl 호출 시작] cam_num: {cam_num}") 

        target = self.spotlights.get(cam_num, None)
        if not target:
            # print(f"   cam_num {cam_num}에 Spotlight 없음.")
            return

        # print(f"   Spotlight 객체 유효: {target}")

        target.setStyleSheet("""
            QPushButton {
                background-image: url(resources/icons/danger.png);
                background-repeat: no-repeat;
                background-position: center;
                border-radius: 5px;
            }
        """)
        ## 빨강으로 바꿔야함
        self.info_widgets[cam_num].setObjectName("info_widget")  # ✔ 스타일링 타겟 명확히
        self.info_widgets[cam_num].setStyleSheet(self.infor_sss)

        def reset_to_green():
            # print(f"[타이머 만료] cam_num: {cam_num} - 초록불로 리셋 및 no event 설정.")
            target.setStyleSheet("""
                QPushButton {
                    background-image: url(resources/icons/safe.png);
                    background-repeat: no-repeat;
                    background-position: center;
                    border-radius: 5px;
                }
            """)
            self.info_widgets[cam_num].setObjectName("info_widget")  # ✔ 스타일링 타겟 명확히
            self.info_widgets[cam_num].setStyleSheet(self.info_sss)
            event_label = self.event_labels.get(cam_num)
            if event_label:
                event_label.setText("no event")
                print(f"  cam_num {cam_num} 이벤트 라벨 설정 완료.")
            else:
                print(f"  경고: event_label 없음 (cam_num: {cam_num})")

        timer = self.reset_timers.get(cam_num)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(reset_to_green)
            self.reset_timers[cam_num] = timer
            # print(f"  타이머 생성됨 (cam_num: {cam_num})")
        else:
            if timer.isActive():
                timer.stop()
                # print(f"  기존 타이머 중지됨 (cam_num: {cam_num})")

        timer.start(5000)
        # print(f"  타이머 시작 (5초 후 리셋) (cam_num: {cam_num})")

        # print("[lightControl 호출 종료]")
    
    
    def onoff_info(self, type, label, cam_num):
        text1 = f"{type.upper()}"
        info1 = self.onoff_labels.get(cam_num, None)
        if info1:
            info1.setText(f"{text1}")
            
    def event_info(self, event_time, cam_num, label):
        text2 = f"{label}"
        info2 = self.event_labels.get(cam_num, None)
        if info2:
            info2.setText(text2)
            
    def handle_result(self, data, cam_num):
        
        class_names = [det['class_name'] for det in data]
        counts = Counter(class_names)
        class_count = sorted([[count, class_name] for class_name, count in counts.items()], reverse=True)

        print(class_count)

        if len(class_count) < 2:
            print("감지된 클래스가 2개 미만입니다.")
            return

        text10 = f"   {('FORKLIFT' if class_count[0][1].startswith('fork') else 'PERSON'):6s}"
        text11 = f"{class_count[0][0]}   "
        text20 = f"   {('FORKLIFT' if class_count[1][1].startswith('fork') else 'PERSON'):6s}"
        text21 = f"{class_count[1][0]}   "

        info3, info4 = self.info_labels.get(cam_num, (None, None))
        info3.setText(f"{text10}  {text11:>8}")
        info4.setText(f"{text20}  {text21:>6}")
        # print('_'*50)
        # print(text20, text21)
        

    
    def make_delayed_loader(self, log_viewer):
        print('delayed_loader debug')
        def loader(*args, **kwargs):
            QTimer.singleShot(5000, lambda: log_viewer.loadLogs())
        return loader

    
    def reset_roi(self, cam_id):
        """ 
        roi 리셋 버튼을 눌렀을 시 작동하는 함수
        video_widget의 clear_roi()
        같은 cam_id의 roi_editors를 제거함
        """
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.clear_roi()

        old_editor = self.roi_editors.pop(cam_id, None)
        if old_editor:
            old_editor.hide()
            old_editor.setParent(None)
            ## 참조 제거
            if hasattr(vw, 'roi_editor'):
                vw.roi_editor = None
            old_editor.deleteLater()

        self.create_roi_editor(cam_id, vw)
        
        # 설정 파일 안전하게 정리
        config_dir = 'resources/config/'
        if os.path.exists(config_dir):
            for filename in os.listdir(config_dir):
                filepath = os.path.join(config_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        print(f"카메라 {cam_id} ROI 초기화됨 및 새 ROIEditor 생성됨")

    def create_roi_editor(self, cam_id, vw):
        """ 
        ROI 에디터 객체 생성
        """
        editor = ROIEditor(vw, cam_id)
        editor.setParent(vw)
        editor.setGeometry(vw.rect())
        editor.roi_defined.connect(self.on_roi_defined)
        editor.show()
        editor.raise_()
        self.roi_editors[cam_id] = editor
        
    def adjust_video_size(self, vw, parent_height, num_videos):
        """ 
        영상 출력 사이즈 조절
        최소크기를 현재 창크기로부터 받아서 설정하고
        Policy로 남은 공간을 다 채우도록 함.
        """
        aspect_ratio = 1280 / 720
        if parent_height > 0 and num_videos > 0:
            target_height = parent_height / num_videos
            target_width = int(target_height * aspect_ratio)

            vw.setMinimumSize(target_width, int(target_height))
            vw.setMaximumSize(16777215, 16777215)  # 최대 크기 제한 없음 (Qt.WA_Unlimited)
            vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        else:
            vw.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


    def on_roi_defined(self, polygon, cam_id):
        """ 
        roi 확정시 이벤트 발생.
        """
        if len(polygon) < 3:
            print(f"카메라 {cam_id}에서 ROI는 최소 3개의 점이 필요합니다.")
            return
        print(f"카메라 {cam_id} ROI 확정:", polygon)
        # 1. VideoWidget에 ROI 설정
        vw = self.video_widgets.get(cam_id)
        if vw:
            vw.set_roi(polygon)
            vw.vthread.set_roi(polygon)

        # 2. ROI 설정 저장
        self.save_roi_to_file(polygon, cam_id)

    
    def save_roi_to_file(self, polygon, cam_id, base_dir="resources/config/"):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"roi_cam_{cam_id}.json")
        with open(filepath, "w") as f:
            json.dump({"cam_id": cam_id, "roi": polygon}, f)
    
    
    def load_roi_from_file(self, cam_id, base_dir="resources/config/"):
        filepath = os.path.join(base_dir, f"roi_cam_{cam_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            return data["roi"]
        return None
    
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        print('get resized')

        screen = QGuiApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        width = screen_size.width()

        if self.video_widgets:
            new_width = int((width - 200) * 0.5)
            new_height = int((width - 200) * 9 / 32)

            for cam_id, vw in self.video_widgets.items():
                vw.setFixedSize(new_width, new_height)
                roi_editor = self.roi_editors.get(cam_id)
                info_widget = self.info_widgets.get(cam_id)
                log_viewer = self.log_viewers.get(cam_id)
                if roi_editor:
                    roi_editor.setGeometry(vw.rect())  # 위치 및 크기 재조정
                if vw.vthread:
                    vw.vthread.set_ui_size(new_width, new_height)
                if log_viewer:
                    log_viewer.setFixedWidth(new_width)
                    
    def loadConfig(self):
        with open('config.json', encoding='utf-8') as config:
            if config['label'][0] == 1:
                self.video_w1idgets[1].vthread.label_visible == config['label'][1]
                
                
                
                
    def closeEvent(self, event):
        """종료 시 안전한 정리 - 수정된 부분"""
        try:
            # 타이머들 정리
            for timer in self.reset_timers.values():
                if timer and timer.isActive():
                    timer.stop()
                timer.deleteLater()
            
            # ROI 저장 (안전하게)
            for cam_id in [1, 2]:
                if cam_id in self.video_widgets:
                    vw = self.video_widgets[cam_id]
                    if hasattr(vw, 'roi') and vw.roi is not None:
                        self.save_roi_to_file(vw.roi.tolist(), cam_id)
                        
        except Exception as e:
            print(f"종료 중 오류 발생: {e}")
        finally:
            super().closeEvent(event)