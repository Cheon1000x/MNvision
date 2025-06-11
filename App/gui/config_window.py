from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, 
    QLabel, QSlider, QCheckBox, QSpinBox, QGroupBox, QApplication, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QSettings
from PyQt5.QtGui import QGuiApplication
import json

class ConfigWindow(QDialog):
    # JSON 형태로 모든 설정을 한번에 전달하는 시그널
    config_changed = pyqtSignal(str)  # JSON string으로 전달
    
    # 기존 개별 시그널들도 유지 (호환성을 위해)
    conf_signal = pyqtSignal(float)
    label_signal = pyqtSignal(bool)
    sound_signal = pyqtSignal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Config")
        self.settings = QSettings("MyOrganization", "MyApplication")
        
        # 기본값으로 설정값 로드
        self._load_from_settings()
        
        self.setStyleSheet(f""" 
                           background-color:#161616;
                           color:#e6e6e6;
                           font-size: 20px;

                           """)
        self.btn_ss = f""" 
                           QPushButton {{
                               border: 1px solid white;
                               padding: 1px 6px;
                           }}
                           
                           QPushButton:hover {{
                               color: #2FDFD9;
                               border: 1px solid #2FDFD9;
                           }}
                           
                        """
            
        # 창 플래그 설정 - 독립적인 창으로 만들기
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setModal(True)  # 모달 다이얼로그로 설정
        self._setup_ui()

    def load_initial_config(self, config_json):
        """외부에서 초기 설정값을 로드하는 메서드"""
        if config_json:
            self._load_from_json(config_json)
            self._update_ui_from_current_values()

    def _update_ui_from_current_values(self):
        """현재 값으로 UI 업데이트"""
        self.confidence_slider.setValue(int(self.current_confidence * 100))
        self.confidence_spinbox.setValue(self.current_confidence)
        self.sound_checkbox1.setChecked(self.cam1_mute)
        self.sound_checkbox2.setChecked(self.cam2_mute)
        self.label_checkbox.setChecked(self.show_labels)

    def _load_from_json(self, config_json):
        """JSON 문자열에서 설정값 로드"""
        try:
            if isinstance(config_json, str):
                config = json.loads(config_json)
            else:
                config = config_json
                
            self.current_confidence = config.get("confidence", 0.5)
            self.cam1_mute = config.get("cam1_mute", False)
            self.cam2_mute = config.get("cam2_mute", False)
            self.show_labels = config.get("show_labels", False)
            
            # 기본값들
            self.default_confidence = config.get("default_confidence", 0.6)
            self.default_cam1_mute = config.get("default_cam1_mute", False)
            self.default_cam2_mute = config.get("default_cam2_mute", False)
            self.default_show_labels = config.get("default_show_labels", False)
            
        except (json.JSONDecodeError, AttributeError):
            self._load_from_settings()

    def _load_from_settings(self):
        """QSettings에서 설정값 로드"""
        self.current_confidence = self.settings.value("confidence", 0.5, type=float)
        self.cam1_mute = self.settings.value("cam1_mute", False, type=bool)
        self.cam2_mute = self.settings.value("cam2_mute", False, type=bool)
        self.show_labels = self.settings.value("show_labels", False, type=bool)
        self.default_confidence = self.settings.value("default_confidence", 0.5, type=float)
        self.default_cam1_mute = self.settings.value("default_cam1_mute", False, type=bool)
        self.default_cam2_mute = self.settings.value("default_cam2_mute", False, type=bool)
        self.default_show_labels = self.settings.value("default_show_labels", False, type=bool)

    def get_config_json(self):
        """현재 설정을 JSON 문자열로 반환"""
        config = {
            "confidence": self.current_confidence,
            "cam1_mute": self.cam1_mute,
            "cam2_mute": self.cam2_mute,
            "show_labels": self.show_labels,
            "default_confidence": self.default_confidence,
            "default_cam1_mute": self.default_cam1_mute,
            "default_cam2_mute": self.default_cam2_mute,
            "default_show_labels": self.default_show_labels
        }
        return json.dumps(config, ensure_ascii=False, indent=2)

    def get_config_dict(self):
        """현재 설정을 딕셔너리로 반환"""
        return {
            "confidence": self.current_confidence,
            "cam1_mute": self.cam1_mute,
            "cam2_mute": self.cam2_mute,
            "show_labels": self.show_labels,
            "default_confidence": self.default_confidence,
            "default_cam1_mute": self.default_cam1_mute,
            "default_cam2_mute": self.default_cam2_mute,
            "default_show_labels": self.default_show_labels
        }

    
    def _setup_ui(self):
        """UI 구성"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        

        # --- Confidence 설정 ---
        confidence_group = QGroupBox("Confidence Config")
        conf_layout = QVBoxLayout()
        confidence_input_layout = QHBoxLayout()
        self.conf_label_prefix = QLabel("current:")
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.00, 1.00)
        self.confidence_spinbox.setSingleStep(0.01)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self.current_confidence)
        self.confidence_spinbox.valueChanged.connect(self._update_confidence_from_spinbox)
        
        confidence_input_layout.addWidget(self.conf_label_prefix)
        confidence_input_layout.addWidget(self.confidence_spinbox)
        confidence_input_layout.addStretch()
        conf_layout.addLayout(confidence_input_layout)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(self.current_confidence * 100))
        self.confidence_slider.setSingleStep(1)
        self.confidence_slider.valueChanged.connect(self._update_confidence_from_slider)
        conf_layout.addWidget(self.confidence_slider)

        conf_button_layout = QHBoxLayout()
        self.reset_conf_btn = QPushButton("Load Default")
        self.reset_conf_btn.clicked.connect(self._reset_confidence_to_default)
        self.set_conf_as_default_btn = QPushButton("Update Default")
        self.set_conf_as_default_btn.clicked.connect(self._set_current_confidence_as_default)
        conf_button_layout.addWidget(self.reset_conf_btn)
        conf_button_layout.addWidget(self.set_conf_as_default_btn)
        conf_layout.addLayout(conf_button_layout)
        confidence_group.setLayout(conf_layout)
        confidence_group.setStyleSheet(self.btn_ss)
        main_layout.addWidget(confidence_group)

        # --- 소리 설정 ---
        sound_group = QGroupBox("Sound Config")
        sound_layout = QVBoxLayout()

        self.sound_checkbox1 = QCheckBox("cam1 mute")
        self.sound_checkbox1.setChecked(self.cam1_mute)
        self.sound_checkbox1.stateChanged.connect(lambda state: self._update_sound_setting(state, 1))
        sound_layout.addWidget(self.sound_checkbox1)
        
        self.sound_checkbox2 = QCheckBox("cam2 mute")
        self.sound_checkbox2.setChecked(self.cam2_mute)
        self.sound_checkbox2.stateChanged.connect(lambda state: self._update_sound_setting(state, 2))
        sound_layout.addWidget(self.sound_checkbox2)

        sound_button_layout = QHBoxLayout()
        self.reset_sound_btn = QPushButton("Load Default")
        self.reset_sound_btn.clicked.connect(self._reset_sound_to_default)
        self.set_sound_as_default_btn = QPushButton("Update Default")
        self.set_sound_as_default_btn.clicked.connect(self._set_current_sound_as_default)
        sound_button_layout.addWidget(self.reset_sound_btn)
        sound_button_layout.addWidget(self.set_sound_as_default_btn)
        sound_layout.addLayout(sound_button_layout)

        sound_group.setLayout(sound_layout)
        sound_group.setStyleSheet(self.btn_ss)
        main_layout.addWidget(sound_group)

        # --- 라벨 출력 여부 설정 ---
        label_group = QGroupBox("Print Label Config")
        label_layout = QVBoxLayout()

        self.label_checkbox = QCheckBox("Print Label")
        self.label_checkbox.setChecked(self.show_labels)
        self.label_checkbox.stateChanged.connect(self._update_label_setting)
        label_layout.addWidget(self.label_checkbox)

        label_button_layout = QHBoxLayout()
        self.reset_label_btn = QPushButton("Load Default")
        self.reset_label_btn.clicked.connect(self._reset_label_to_default)
        self.set_label_as_default_btn = QPushButton("Update Default")
        self.set_label_as_default_btn.clicked.connect(self._set_current_label_as_default)
        label_button_layout.addWidget(self.reset_label_btn)
        label_button_layout.addWidget(self.set_label_as_default_btn)
        label_layout.addLayout(label_button_layout)
        
        label_group.setLayout(label_layout)
        label_group.setStyleSheet(self.btn_ss)
        main_layout.addWidget(label_group)
        
        # --- 확인/취소 버튼 ---
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        confirm_btn = QPushButton("Confirm")
        
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(confirm_btn)
        main_layout.addLayout(button_layout)

        cancel_btn.clicked.connect(self.reject)  # reject()로 변경
        confirm_btn.clicked.connect(self._apply_settings)
        
        cancel_btn.setStyleSheet(self.btn_ss)
        confirm_btn.setStyleSheet(self.btn_ss)
        

    def _update_confidence_from_slider(self, value):
        self.current_confidence = value / 100.0
        self.confidence_spinbox.setValue(self.current_confidence)

    def _update_confidence_from_spinbox(self, value):
        self.current_confidence = value
        self.confidence_slider.setValue(int(self.current_confidence * 100))

    def _update_sound_setting(self, state, cam_num):
        if cam_num == 1:
            self.cam1_mute = (state == Qt.Checked)
        elif cam_num == 2:
            self.cam2_mute = (state == Qt.Checked)

    def _update_label_setting(self, state):
        self.show_labels = (state == Qt.Checked)

    def _apply_settings(self):
        # QSettings에 저장
        self.settings.setValue("confidence", self.current_confidence)
        self.settings.setValue("cam1_mute", self.cam1_mute)
        self.settings.setValue("cam2_mute", self.cam2_mute)
        self.settings.setValue("show_labels", self.show_labels)

        # JSON 형태로 시그널 전송
        self.config_changed.emit(self.get_config_json())
        
        # 기존 개별 시그널들도 전송 (호환성)
        self.conf_signal.emit(self.current_confidence)
        self.sound_signal.emit(self.cam1_mute, self.cam2_mute)
        self.label_signal.emit(self.show_labels)
        
        self.close()

    def _reset_confidence_to_default(self):
        self.current_confidence = self.default_confidence
        self.confidence_slider.setValue(int(self.current_confidence * 100))
        self.confidence_spinbox.setValue(self.current_confidence)

    def _set_current_confidence_as_default(self):
        self.default_confidence = self.current_confidence
        self.settings.setValue("default_confidence", self.default_confidence)

    def _reset_sound_to_default(self):
        self.cam1_mute = self.default_cam1_mute
        self.cam2_mute = self.default_cam2_mute
        self.sound_checkbox1.setChecked(self.cam1_mute)
        self.sound_checkbox2.setChecked(self.cam2_mute)

    def _set_current_sound_as_default(self):
        self.default_cam1_mute = self.cam1_mute
        self.default_cam2_mute = self.cam2_mute
        self.settings.setValue("default_cam1_mute", self.default_cam1_mute)
        self.settings.setValue("default_cam2_mute", self.default_cam2_mute)

    def _reset_label_to_default(self):
        self.show_labels = self.default_show_labels
        self.label_checkbox.setChecked(self.show_labels)

    def _set_current_label_as_default(self):
        self.default_show_labels = self.show_labels
        self.settings.setValue("default_show_labels", self.default_show_labels)

    def closeEvent(self, event):
        """창 닫기 이벤트 처리"""
        # 부모 창에 영향을 주지 않도록 이벤트 처리
        event.accept()
        self.deleteLater()  # 객체 정리