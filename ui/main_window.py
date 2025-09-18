'''
主窗口 (ui/main_window.py)
包含两个标签页：实时对话和文件分析
统一管理模型加载，避免重复初始化
'''

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QMessageBox
from .camera_tab import CameraTab
from .file_tab import FileTab

# -------------------- 主窗口类 --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("心语情感助手")
        self.setGeometry(100, 100, 1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.camera_tab = CameraTab()
        self.file_tab = FileTab()

        self.tabs.addTab(self.camera_tab, "实时对话")
        self.tabs.addTab(self.file_tab, "文件分析")

        # 关键：在主窗口中统一加载模型
        # 并通过信号将加载好的模型实例传递给另一个tab
        self.camera_tab.models_loaded_signal.connect(self.file_tab.on_models_ready)
        self.camera_tab.load_models()

    def closeEvent(self, event):
        # 确保关闭窗口时能安全释放硬件
        reply = QMessageBox.question(self, '退出', "您确定要退出程序吗?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.camera_tab.close_hardware()
            event.accept()
        else:
            event.ignore()