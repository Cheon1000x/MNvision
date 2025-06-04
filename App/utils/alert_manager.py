import pygame
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import time
import threading
import os

class AlertManager(QObject):
    on_alert_signal = pyqtSignal(str, int)  # 클래스 변수로 시그널 정의
    
    def __init__(self):
        super().__init__()
        self.last_alert_time = 0
        self.cooldown_seconds = 5
        self.is_playing = False
        self.mute_opt = True
        self.on_alert_signal.connect(self.handle_alert_signal)
        
        

    @pyqtSlot(str)
    def handle_alert_signal(self, message):
        now = time.time()
        if now - self.last_alert_time > self.cooldown_seconds:
            self.last_alert_time = now
            if not self.is_playing:
                self.play_alert_sound(message)
                
    def mute_contorl(self):
        print('am.mute_opt',self.mute_opt)
        if self.mute_opt:
            self.mute_opt = False
        else: 
            self.mute_opt = True
           

    def play_alert_sound(self, message, sound_path=os.path.abspath('./resources/etc/')):
        if self.mute_opt:
            return
        else:
            def play():
                self.is_playing = True
                pygame.mixer.init()
                pygame.mixer.music.load(sound_path+'/'+message+'.wav')
                print(sound_path+'/'+message+'.wav')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                self.is_playing = False

            threading.Thread(target=play, daemon=True).start()

    def trigger_alert(self, message, sound_path=os.path.abspath('./resources/etc/forklift_away.wav')):
        if self.mute_opt:
            return
        else:
            now = time.time()
            if now - self.last_alert_time > self.cooldown_seconds:
                self.last_alert_time = now
                if not self.is_playing:
                    print(f"[🔊 재생 시도] {message}")
                    self.play_alert_sound(sound_path)

# 사용 예
alert_manager = AlertManager()
# sound_path = os.path.abspath('./resources/etc/forklift_away.wav')
# alert_manager.trigger_alert("person-forklift overlap", sound_path)