'''
硬件接口 (utils/hardware.py)
VADRecorderUI类：管理摄像头和麦克风捕获
支持音视频数据的获取和保存
'''

import cv2
import pyaudio
import numpy as np
import wave
import time
from datetime import datetime
import os
from PIL import Image, ImageDraw, ImageFont  # 新增导入PIL库

# -------------------- VADRecorderUI类 --------------------
class VADRecorderUI:
    """
    一个专门为PyQt5 UI设计的、非阻塞的音视频采集器。
    UI的主循环通过反复调用 get_current_frame() 来获取最新的摄像头帧。
    """
    def __init__(self, rms_threshold=500, silence_limit=2, video_fps=20.0):
        # --- 可调参数 ---
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RMS_THRESHOLD = rms_threshold
        self.SILENCE_LIMIT = silence_limit
        self.VIDEO_FPS = video_fps
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头。")
            
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        self.is_recording = False
        self.silence_start_time = None
        self.audio_frames = []
        self.video_frames = []
        
        # 确保 results 文件夹存在
        os.makedirs("results", exist_ok=True)
        
        # 查找系统中可用的中文字体
        self.font = self._find_chinese_font()

    def _find_chinese_font(self):
        """查找系统中可用的中文字体"""
        # 常见的中文字体路径列表
        font_paths = [
            "C:/Windows/Fonts/SimHei.ttf",       # 黑体
            "C:/Windows/Fonts/SimSun.ttf",       # 宋体
            "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
            "C:/Windows/Fonts/Microsoft YaHei.ttf",
            "/System/Library/Fonts/PingFang.ttc", # macOS
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf" # Linux
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, 30)
                except Exception:
                    continue
                
        print("警告: 未找到中文字体，将使用默认字体")
        return ImageFont.load_default()

    def _calculate_rms(self, data):
        """计算音频数据的RMS值，增加健壮性检查"""
        if not data:
            return 0
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            if audio_data.size == 0:
                return 0
            rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
            return int(rms)
        except (ValueError, TypeError):
            return 0

    def manual_save_recording(self, basename=None):
        """手动保存录制的音视频数据"""
        if not self.audio_frames or not self.video_frames:
            return None

        if basename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = f"output_{timestamp}"
            
        audio_filename = os.path.join("results", f"{basename}.wav")
        video_filename = os.path.join("results", f"{basename}.avi")

        try:
            # 保存音频
            wf = wave.open(audio_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()

            # 保存视频
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, self.VIDEO_FPS, (self.frame_width, self.frame_height))
            for frame in self.video_frames:
                out.write(frame)
            out.release()
            
            print(f"音视频已保存: {basename}")
            return basename
        except Exception as e:
            print(f"保存录制文件失败: {e}")
            return None

    def get_current_frame(self):
        """获取当前的摄像头帧"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # 读取音频数据计算音量
        audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        rms = self._calculate_rms(audio_data)
        
        # 如果正在录制，保存数据
        if self.is_recording:
            self.audio_frames.append(audio_data)
            self.video_frames.append(frame.copy())
        
        # 使用PIL绘制中文文本
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制状态信息
        status_text = "状态: 录制中..." if self.is_recording else "状态: 准备就绪"
        status_color = (255, 0, 0) if self.is_recording else (0, 255, 0)  # PIL中是RGB顺序
        draw.text((10, 30), status_text, font=self.font, fill=status_color)
        
        # 绘制音量信息
        draw.text((10, 60), f"音量 RMS: {rms}", font=self.font, fill=(0, 100, 255))
        
        # 将PIL图像转回OpenCV格式
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return frame

    def close(self):
        """释放所有硬件资源"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.cap.release()
