# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow

if __name__ == '__main__':
    # 解决部分系统下高DPI屏幕的显示问题
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont("微软雅黑", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())