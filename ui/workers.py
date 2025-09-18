'''
后台工作线程 (ui/workers.py)
ModelLoaderWorker: 加载AI模型
HardwareSetupWorker: 初始化摄像头和麦克风
AnalysisWorker: 分析处理视频和音频
VideoPreprocessor: 处理上传的视频，提取音频
'''

from PyQt5.QtCore import QThread, pyqtSignal
import os
import subprocess
import cv2
from datetime import datetime
import config
from core.transcriber import AudioTranscriber
from core.sentiment import SentimentAnalyzer
from core.responder import LLMResponder
from core.analysis_pipeline import AnalysisPipeline
from utils.hardware import VADRecorderUI

# -------------------- 模型加载线程 --------------------
class ModelLoaderWorker(QThread):
    """线程：加载所有AI模型"""
    finished = pyqtSignal(object, object, object) 
    status_update = pyqtSignal(str)

    def run(self):
        """在后台加载所有AI模型。"""
        try:
            self.status_update.emit("正在加载语音识别模型...")
            transcriber = AudioTranscriber()
            
            self.status_update.emit("正在加载情感分析模型...")
            analyzer = SentimentAnalyzer()

            self.status_update.emit("正在加载大语言模型(首次加载约需3-5分钟)...")
            responder = LLMResponder()
            
            self.finished.emit(transcriber, analyzer, responder)
        except Exception as e:
            self.status_update.emit(f"模型加载失败: {e}")

# -------------------- 硬件初始化线程 --------------------
class HardwareSetupWorker(QThread):
    """线程：初始化摄像头和麦克风"""
    finished = pyqtSignal(object) # 信号返回一个已初始化的 VADRecorderUI 实例
    error = pyqtSignal(str)       # 信号返回错误信息

    def run(self):
        """在后台初始化摄像头和麦克风，避免阻塞UI。"""
        try:
            recorder = VADRecorderUI()
            self.finished.emit(recorder)
        except Exception as e:
            self.error.emit(str(e))

# -------------------- 分析工作线程 --------------------
class AnalysisWorker(QThread):
    """线程：使用AnalysisPipeline分析处理后的视频和音频"""
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    # 信号：文本、AI回复、文本情感、视频情绪
    finished = pyqtSignal(str, str, str, str)  
    
    def __init__(self, basename, transcriber, analyzer, responder, parent=None):
        super().__init__(parent)
        self.basename = basename
        # 创建分析管道
        self.pipeline = AnalysisPipeline(transcriber, analyzer, responder)
        
    def run(self):
        try:
            # 构建文件路径
            audio_path = os.path.join(config.RESULTS_DIR, self.basename + ".wav")
            video_path = os.path.join(config.RESULTS_DIR, self.basename + ".avi")
            
            # 进度更新
            self.status_update.emit("正在分析数据...")
            self.progress.emit(25)  # 开始分析
            
            # 使用pipeline执行全部分析
            user_text, ai_response, text_sentiment, video_emotion = self.pipeline.run(audio_path, video_path)
            
            self.progress.emit(100)  # 分析完成
            
            # 发送结果
            self.finished.emit(user_text, ai_response, text_sentiment, video_emotion)
            
        except Exception as e:
            self.status_update.emit(f"处理时发生错误: {e}")
            self.finished.emit("处理出错", "抱歉，我遇到了一点问题。", "未知", "未知")

# -------------------- 视频预处理线程 --------------------
class VideoPreprocessor(QThread):
    """线程：处理上传的视频，提取音频"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)  # 输出视频和音频路径
    error = pyqtSignal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        try:
            # 创建results目录（如果不存在）
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = f"uploaded_{timestamp}"
            
            # 输出文件路径
            video_output = os.path.join(config.RESULTS_DIR, f"{basename}.avi")
            audio_output = os.path.join(config.RESULTS_DIR, f"{basename}.wav")
            
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频文件: {self.video_path}")
                return
                
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
            
            # 处理每一帧
            processed_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                processed_frames += 1
                
                # 更新进度
                progress = int((processed_frames / frame_count) * 50)  # 视频处理占总进度的50%
                self.progress.emit(progress)
            
            # 释放资源
            cap.release()
            out.release()
            
            # 使用FFmpeg提取音频
            self.progress.emit(50)  # 视频处理完成，开始提取音频
            import subprocess
            
            try:
                command = [
                    "ffmpeg", "-y", "-i", self.video_path, 
                    "-vn", "-acodec", "pcm_s16le", 
                    "-ar", "16000", "-ac", "1", audio_output
                ]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.progress.emit(100)  # 音频提取完成
                self.finished.emit(video_output, audio_output)
                
            except subprocess.CalledProcessError as e:
                self.error.emit(f"音频提取失败: {e}")
            except FileNotFoundError:
                self.error.emit("找不到FFmpeg，请确保已正确安装并添加到系统路径")
                
        except Exception as e:
            self.error.emit(f"处理视频时发生错误: {str(e)}")