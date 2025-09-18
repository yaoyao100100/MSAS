'''
文件分析标签页 (ui/file_tab.py)
支持上传视频文件进行分析
显示分析结果和AI回复
'''

from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QVBoxLayout, 
                            QPushButton, QHBoxLayout, QGroupBox, QFileDialog,
                            QSplitter, QFrame, QMessageBox, QApplication)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import os
import cv2
from ui.workers import VideoPreprocessor, AnalysisWorker

# -------------------- 文件分析标签页类 --------------------
class FileTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 创建主布局
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
        self.video_label = QLabel("上传视频后将在此处显示预览", left_panel)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: #000000; 
            color: #ffffff; 
            border-radius: 8px;
            padding: 10px;
        """)
        left_layout.addWidget(self.video_label)
        
        # 移除进度条组件
        
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
        
        # 分析结果组
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                line-height: 1.5;
                background-color: #ffffff;
                padding: 15px;
            }
        """)
        
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # 将左右面板添加到分割器
        display_area.addWidget(left_panel)
        display_area.addWidget(right_panel)
        display_area.setSizes([600, 400])  # 设置初始分割比例
        
        # ===== 下部控制区域 =====
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("上传视频文件")
        self.upload_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogOpenButton))
        
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_MediaPlay))
        self.analyze_btn.setDisabled(True)
        
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogResetButton))
        
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.clear_btn)
        
        control_group.setLayout(control_layout)
        
        # 将显示区域和控制区域添加到主布局
        main_layout.addWidget(display_area, 8)
        main_layout.addWidget(control_group, 2)
        
        # 绑定按钮事件
        self.upload_btn.clicked.connect(self.upload_video)
        self.analyze_btn.clicked.connect(self.analyze_video)
        self.clear_btn.clicked.connect(self.clear_results)
        
        # 初始化变量
        self.video_path = None
        self.processed_video = None
        self.processed_audio = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.current_frame = 0
        self.total_frames = 0
        
        # AI模型
        self.transcriber = None
        self.analyzer = None
        self.responder = None
        self.models_loaded = False
        
        # 欢迎消息
        self.result_text.append('<div style="color: #0078d7; font-weight: bold; margin-bottom: 10px;">视频文件情感分析系统</div>')
        self.result_text.append('<div style="color: #666666; margin-bottom: 20px;">通过上传视频文件，系统将进行语音转文本、情感分析和AI回复。请等待系统初始化完成...</div>')
        
        # 禁用上传按钮，直到模型加载完成
        self.upload_btn.setDisabled(True)
        self.status_label.setText("正在等待AI模型初始化...")
    
    def on_models_ready(self, transcriber, analyzer, responder):
        """当主窗口中的模型加载完成后，接收模型并启用功能"""
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.responder = responder
        self.models_loaded = True
        self.status_label.setText("系统初始化完成，请上传视频文件。")
        self.upload_btn.setDisabled(False)
        
        # 更新结果文本
        self.result_text.append('<div style="color: #28a745; font-weight: bold; margin-top: 10px;">系统就绪</div>')
        self.result_text.append('<div style="color: #333333; margin-bottom: 20px;">AI模型已加载完成，点击"上传视频文件"按钮开始分析。</div>')
    
    def upload_video(self):
        """上传视频文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.wmv *.mkv);;所有文件 (*)", 
            options=options
        )
        
        if file_path:
            # 关闭之前的视频预览
            if self.cap is not None:
                self.cap.release()
                self.timer.stop()
            
            # 保存视频路径
            self.video_path = file_path
            self.status_label.setText(f"已选择视频: {os.path.basename(file_path)}")
            
            # 打开视频以预览
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.status_label.setText(f"错误: 无法打开视频文件")
                return
                
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            
            # 显示第一帧作为预览
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                
                # 开始预览循环
                self.timer.start(100)  # 每100ms更新一次预览
                
                # 启用分析按钮
                self.analyze_btn.setDisabled(False)
                
                # 更新状态
                self.result_text.append('<div style="color: #0078d7; font-weight: bold; margin-top: 10px;">视频已上传</div>')
                self.result_text.append(f'<div style="color: #333333; margin-bottom: 20px;">文件: {os.path.basename(file_path)}<br>点击"开始分析"按钮进行处理。</div>')
            else:
                self.status_label.setText("错误: 无法读取视频帧")
    
    def update_preview(self):
        """更新视频预览"""
        if self.cap is not None:
            # 循环播放预览
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
            else:
                # 视频结束，重新开始
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame(self, frame):
        """在UI上显示视频帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 调整尺寸以适应标签
        label_size = self.video_label.size()
        pixmap_scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(pixmap_scaled)
    
    def analyze_video(self):
        """开始处理和分析视频"""
        if not self.video_path:
            return
            
        # 禁用按钮
        self.upload_btn.setDisabled(True)
        self.analyze_btn.setDisabled(True)
        
        # 更新状态
        self.status_label.setText("正在处理视频文件...")
        self.result_text.append('<div style="color: #0078d7; font-weight: bold; margin-top: 10px;">开始处理</div>')
        self.result_text.append('<div style="color: #333333; margin-bottom: 20px;">正在预处理视频和提取音频...</div>')
        
        # 启动预处理线程
        self.preprocessor = VideoPreprocessor(self.video_path)
        # 不再连接progress信号
        self.preprocessor.finished.connect(self.on_preprocessing_complete)
        self.preprocessor.error.connect(self.on_preprocessing_error)
        self.preprocessor.start()
    
    def on_preprocessing_complete(self, video_path, audio_path):
        """视频预处理完成后的回调"""
        self.processed_video = video_path
        self.processed_audio = audio_path
        
        self.status_label.setText("预处理完成，开始分析...")
        self.result_text.append('<div style="color: #28a745; margin-top: 10px;">预处理完成</div>')
        self.result_text.append('<div style="color: #333333; margin-bottom: 20px;">视频和音频提取成功，开始分析内容...</div>')
        
        # 获取基础文件名
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 开始分析
        self.analysis_worker = AnalysisWorker(
            basename,
            self.transcriber,
            self.analyzer,
            self.responder
        )
        # 不再连接progress信号
        self.analysis_worker.status_update.connect(self.update_status)
        self.analysis_worker.finished.connect(self.on_analysis_complete)
        self.analysis_worker.start()
    
    def on_preprocessing_error(self, error_message):
        """视频预处理错误的回调"""
        self.status_label.setText(f"错误: {error_message}")
        self.result_text.append(f'<div style="color: #dc3545; font-weight: bold; margin-top: 10px;">处理错误</div>')
        self.result_text.append(f'<div style="color: #dc3545; margin-bottom: 20px;">{error_message}</div>')
        
        # 重新启用上传按钮
        self.upload_btn.setDisabled(False)
        self.analyze_btn.setDisabled(False)
    
    def update_status(self, message):
        """更新状态标签"""
        self.status_label.setText(message)
    
    def on_analysis_complete(self, user_text, ai_response, text_sentiment, video_emotion):
        """分析完成后的回调"""
        # 更新状态
        self.status_label.setText("分析完成")
        
        # 清空之前的结果
        self.result_text.clear()
        
        # 添加分析结果
        self.result_text.append('<div style="color: #28a745; font-weight: bold; margin-bottom: 15px;">分析结果</div>')
        
        # 添加用户文本
        self.result_text.append('<div style="color: #333333; font-weight: bold; margin-top: 15px;">语音内容:</div>')
        self.result_text.append(f'<div style="color: #333333; margin-left: 20px; margin-bottom: 15px;">{user_text}</div>')
        
        # 添加情感分析结果
        emotion_color = "#28a745" if text_sentiment == "Positive" else "#dc3545" if text_sentiment == "Negative" else "#6c757d"
        self.result_text.append('<div style="color: #333333; font-weight: bold; margin-top: 15px;">情感分析:</div>')
        self.result_text.append(f'<div style="color: #666666; margin-left: 20px; font-size: 15px;">文本情感: <span style="color:{emotion_color}">{self._translate_sentiment(text_sentiment)}</span></div>')
        self.result_text.append(f'<div style="color: #666666; margin-left: 20px; margin-bottom: 15px; font-size: 15px;">视频情绪: <span style="color:{emotion_color}">{self._translate_emotion(video_emotion)}</span></div>')
        
        # 添加AI回复
        self.result_text.append('<div style="color: #333333; font-weight: bold; margin-top: 15px;">AI回复:</div>')
        self.result_text.append(f'<div style="color: #0078d7; margin-left: 20px; margin-bottom: 20px; line-height: 1.5;">{ai_response}</div>')
        
        # 重新启用按钮
        self.upload_btn.setDisabled(False)
        self.analyze_btn.setDisabled(False)
    
    def _translate_sentiment(self, sentiment):
        """将英文情感标签翻译为中文"""
        sentiment_map = {
            "Positive": "积极",
            "Negative": "消极",
            "Neutral": "中性",
            "Unknown": "未知"
        }
        return sentiment_map.get(sentiment, sentiment)
    
    def _translate_emotion(self, emotion):
        """将英文情绪标签翻译为中文"""
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
    
    def clear_results(self):
        """清空结果"""
        self.result_text.clear()
        self.result_text.append('<div style="color: #0078d7; font-weight: bold; margin-bottom: 10px;">视频文件情感分析系统</div>')
        self.result_text.append('<div style="color: #666666; margin-bottom: 20px;">上传视频文件进行分析，或者选择之前上传的视频继续分析。</div>')