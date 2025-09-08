from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QVBoxLayout, 
                             QPushButton, QHBoxLayout, QGroupBox, QSizePolicy,
                             QSplitter, QFrame, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor
from datetime import datetime
import os
import config
from ui.workers import ModelLoaderWorker, HardwareSetupWorker, AnalysisWorker

class CameraTab(QWidget):
    # 定义信号，用于传递加载好的模型给其他标签页
    models_loaded_signal = pyqtSignal(object, object, object)
    
    def __init__(self):
        super().__init__()
        
        # 设置整体布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ===== 上部显示区域（左右分栏） =====
        display_area = QSplitter(Qt.Horizontal)
        display_area.setChildrenCollapsible(False)
        
        # ----- 左侧区域（视频和状态） -----
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        # 视频显示
        self.video_label = QLabel("正在初始化系统，请稍候...", left_panel)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: #000000; 
            color: #ffffff; 
            border-radius: 8px;
            padding: 10px;
        """)
        left_layout.addWidget(self.video_label)
        
        # 状态信息组
        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("系统正在初始化...", status_group)
        self.status_label.setStyleSheet("font-size: 15px; color: #333333;")
        
        self.model_info_label = QLabel("情感分析模型: Erlangshen-Roberta | 大语言模型: Qwen1.5-1.8B", status_group)
        self.model_info_label.setStyleSheet("font-size: 13px; color: #666666;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.model_info_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # ----- 右侧区域（对话框） -----
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        # 对话历史组
        chat_group = QGroupBox("对话历史")
        chat_layout = QVBoxLayout()
        
        self.dialogue_box = QTextEdit()
        self.dialogue_box.setReadOnly(True)
        self.dialogue_box.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                line-height: 1.5;
                background-color: #ffffff;
                padding: 15px;
            }
        """)
        
        chat_layout.addWidget(self.dialogue_box)
        chat_group.setLayout(chat_layout)
        right_layout.addWidget(chat_group)
        
        # 将左右面板添加到分割器
        display_area.addWidget(left_panel)
        display_area.addWidget(right_panel)
        display_area.setSizes([600, 400])  # 设置初始分割比例
        
        # ===== 下部控制区域 =====
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout()  # 使用网格布局以便将来扩展按钮
        
        self.start_button = QPushButton("开始录制")
        self.start_button.setIcon(QIcon.fromTheme("media-record"))
        self.start_button.setMinimumHeight(50)
        
        self.clear_button = QPushButton("清空对话")
        self.clear_button.setIcon(QIcon.fromTheme("edit-clear"))
        self.clear_button.setMinimumHeight(50)
        
        control_layout.addWidget(self.start_button, 0, 0, 1, 1)
        control_layout.addWidget(self.clear_button, 0, 1, 1, 1)
        
        control_group.setLayout(control_layout)
        
        # 将显示区域和控制区域添加到主布局
        main_layout.addWidget(display_area, 8)  # 显示区域占80%
        main_layout.addWidget(control_group, 2)  # 控制区域占20%
        
        # 连接信号和槽
        self.start_button.clicked.connect(self.toggle_recording)
        self.clear_button.clicked.connect(self.dialogue_box.clear)
        
        # 初始化其他变量
        self.recorder = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_recording = False
        self.is_processing = False
        
        self.transcriber = None
        self.analyzer = None
        self.responder = None
        self.models_loaded = False
        self.current_basename = None
        
        # 禁用按钮，直到系统完全初始化
        self.start_button.setDisabled(True)
        self.clear_button.setDisabled(True)
        
        # 欢迎消息
        self.dialogue_box.append('<div style="color: #0078d7; font-weight: bold; margin-bottom: 10px;">系统启动</div>')
        self.dialogue_box.append('<div style="color: #666666; margin-bottom: 20px;">欢迎使用心语情感助手，我可以聆听您的想法并给予回应。请等待系统初始化完成...</div>')

    def load_models(self):
        """加载所有AI模型"""
        self.status_label.setText("正在初始化AI模型，请稍候...")
        self.model_loader = ModelLoaderWorker()
        self.model_loader.finished.connect(self.on_models_loaded)
        self.model_loader.status_update.connect(self.update_status)
        self.model_loader.start()

    def on_models_loaded(self, transcriber, analyzer, responder):
        """AI模型加载完成后，开始初始化硬件"""
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.responder = responder
        self.models_loaded = True
        self.status_label.setText("AI模型加载完成，正在初始化摄像头和麦克风...")
        
        # 发送信号，将模型传递给其他标签页
        self.models_loaded_signal.emit(transcriber, analyzer, responder)
        
        # AI模型加载完成后自动开始初始化硬件
        self.hw_worker = HardwareSetupWorker()
        self.hw_worker.finished.connect(self.on_hardware_ready)
        self.hw_worker.error.connect(self.on_hardware_error)
        self.hw_worker.start()

    def on_hardware_ready(self, recorder_instance):
        """硬件初始化成功后的槽函数"""
        self.recorder = recorder_instance
        self.timer.start(50)  # 启动定时器以更新视频画面
        self.clear_button.setDisabled(False)
        self.start_button.setDisabled(False)
        self.status_label.setText("系统初始化完成，可以开始录制。")
        
        # 更新对话框，通知用户系统已就绪
        self.dialogue_box.append('<div style="color: #28a745; font-weight: bold; margin-top: 10px;">系统就绪</div>')
        self.dialogue_box.append('<div style="color: #333333; margin-bottom: 20px;">摄像头和麦克风已准备就绪，点击"开始录制"按钮开始与我交流。</div>')

    def on_hardware_error(self, error_message):
        """硬件初始化失败后的槽函数"""
        self.status_label.setText(f"错误: 无法启动硬件 - {error_message}")
        # 即使硬件初始化失败，仍然启用清空按钮
        self.clear_button.setDisabled(False)
        
        # 在对话框中显示错误
        self.dialogue_box.append(f'<div style="color: #dc3545; font-weight: bold;">硬件初始化失败</div>')
        self.dialogue_box.append(f'<div style="color: #dc3545;">{error_message}</div>')

    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            # 开始录制
            self.start_recording()
        else:
            # 停止录制
            self.stop_recording()

    def start_recording(self):
        """开始录制"""
        if not self.recorder:
            self.status_label.setText("错误：硬件尚未初始化")
            return
            
        self.is_recording = True
        self.start_button.setText("停止录制")
        self.start_button.setStyleSheet("background-color: #dc3545;")  # 红色按钮表示正在录制
        self.status_label.setText("正在录制...")
        
        # 生成本次录制的基础文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_basename = f"output_{timestamp}"
        
        # 清空之前的录制数据
        self.recorder.audio_frames = []
        self.recorder.video_frames = []
        self.recorder.is_recording = True

    def stop_recording(self):
        """停止录制并开始处理"""
        if not self.recorder or not self.is_recording:
            return
            
        self.is_recording = False
        self.is_processing = True
        self.start_button.setText("处理中...")
        self.start_button.setStyleSheet("")  # 恢复默认样式
        self.start_button.setDisabled(True)
        self.status_label.setText("正在保存录制文件...")
        
        # 停止录制
        self.recorder.is_recording = False
        
        # 保存录制的文件
        basename = self.recorder.manual_save_recording(self.current_basename)
        if basename:
            self.start_analysis(basename)
        else:
            self.status_label.setText("错误: 保存录制文件失败")
            self.is_processing = False
            self.start_button.setText("开始录制")
            self.start_button.setDisabled(False)

    def update_frame(self):
        """更新视频画面"""
        if not self.recorder: 
            return
            
        frame = self.recorder.get_current_frame()
        if frame is not None:
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def start_analysis(self, basename):
        """开始分析录制内容"""
        if not self.models_loaded:
            self.status_label.setText("错误：模型仍在加载中，请稍候！")
            self.is_processing = False
            self.start_button.setText("开始录制")
            self.start_button.setDisabled(False)
            return
            
        self.dialogue_box.append('<div style="color: #333333; font-weight: bold; margin-top: 15px;">你:</div>')
        self.dialogue_box.append('<div style="color: #666666; margin-left: 20px; margin-bottom: 10px;">(正在识别语音...)</div>')
        
        self.worker = AnalysisWorker(basename, self.transcriber, self.analyzer, self.responder)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.status_update.connect(self.update_status)
        self.worker.start()

    def on_analysis_complete(self, user_text, ai_response, text_sentiment, video_emotion):
        """分析完成后更新UI"""
        # 删除"正在识别语音..."文本
        self.dialogue_box.undo()
        
        # 直接添加用户文本，不再添加重复的"你:"
        # self.dialogue_box.append(f'<div style="color: #333333; font-weight: bold; margin-top: 15px;">你:</div>') 
        # 删除这一行，因为之前已经添加过"你:"标签
        
        # 直接添加实际文本内容
        self.dialogue_box.append(f'<div style="color: #333333; margin-left: 20px; margin-bottom: 15px;">{user_text}</div>')
        
        # 添加情感分析结果
        emotion_color = "#28a745" if text_sentiment == "Positive" else "#dc3545" if text_sentiment == "Negative" else "#6c757d"
        self.dialogue_box.append(f'<div style="color: #666666; margin-left: 20px; font-style: italic; font-size: 14px;">文本情感: <span style="color:{emotion_color}">{self._translate_sentiment(text_sentiment)}</span> | 视频情绪: <span style="color:{emotion_color}">{self._translate_emotion(video_emotion)}</span></div>')
        
        # 添加AI回复
        self.dialogue_box.append('<div style="color: #0078d7; font-weight: bold; margin-top: 10px;">心语 (AI):</div>')
        self.dialogue_box.append(f'<div style="color: #333333; margin-left: 20px; margin-bottom: 20px; line-height: 1.5;">{ai_response}</div>')
        
        # 添加分隔线
        self.dialogue_box.append('<hr style="border: 0; height: 1px; background-color: #e0e0e0; margin: 15px 0;">')

        self.status_label.setText("AI已回复。请点击\"开始录制\"继续。")
        self.start_button.setText("开始录制")
        self.start_button.setDisabled(False)
        
        self.is_processing = False
        
        # 自动滚动到底部
        self.dialogue_box.verticalScrollBar().setValue(self.dialogue_box.verticalScrollBar().maximum())

    def _translate_sentiment(self, sentiment):
        sentiment_map = {
            "Positive": "积极",
            "Negative": "消极",
            "Neutral": "中性",
            "Unknown": "未知"
        }
        return sentiment_map.get(sentiment, sentiment)

    def _translate_emotion(self, emotion):
        emotion_map = {
            "Happy": "开心",
            "Sad": "悲伤",
            "Angry": "愤怒",
            "Surprise": "惊讶",
            "Fear": "恐惧",
            "Disgust": "厌恶",
            "Neutral": "中性",
            "NoFace": "未检测到面部",
            "Unknown": "未知"
        }
        return emotion_map.get(emotion, emotion)

    def update_status(self, message):
        """更新状态栏信息"""
        self.status_label.setText(message)

    def close_hardware(self):
        """关闭摄像头和麦克风"""
        self.timer.stop()
        if self.recorder: 
            self.recorder.close()