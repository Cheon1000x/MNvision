import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.start_window import StartWindow

def main():
    print("메인 애플리케이션을 시작합니다!")
    # 여기에 실제 메인 애플리케이션의 인스턴스를 생성하고 표시합니다.
    global main_window
    main_window = MainWindow()
    main_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font_path = os.path.abspath("./resources/fonts/Koulen-Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print("❌ Koulen 폰트 로드 실패")
        raise RuntimeError(f"❌ 폰트 로드 실패: {font_path}")
    else:
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            Koulen_Regular = QFont(font_families[0])
            QApplication.setFont(Koulen_Regular)
            print(f"✅ Koulen-Regular 폰트 전체 적용 완료: {font_families[0]}")        
    

    # StartWindow 인스턴스를 생성하고 표시합니다.
    start_window = StartWindow()
    start_window.show()

    # main() 호출
    start_window.start_main_signal.connect(main)

    # QApplication의 이벤트 루프를 시작
    sys.exit(app.exec_())
