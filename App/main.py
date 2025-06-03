import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.start_window import StartWindow

def main():
    print("메인 애플리케이션을 시작합니다!")
    # 여기에 실제 메인 애플리케이션의 인스턴스를 생성하고 표시합니다.
    global main_window
    main_window = MainWindow()
    main_window.show()
    # 주의: sys.exit(app.exec_())는 QApplicatoin.exec_() 호출이 필요할 때만 사용합니다.
    # 일반적으로 최상위 앱 진입점에서만 한 번 호출합니다.


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # StartWindow 인스턴스를 생성하고 표시합니다.
    start_window = StartWindow()
    start_window.show()

    # ⭐ StartWindow의 start_main_signal이 발생하면 run_main_app 함수를 호출하도록 연결합니다. ⭐
    start_window.start_main_signal.connect(main)

    # QApplication의 이벤트 루프를 시작합니다.
    # start_window가 닫히거나, run_main_app에서 main_window가 닫히면 앱이 종료됩니다.
    sys.exit(app.exec_())
    
# if __name__ == "__main__":
#     main()
