'''
语音转文本 (core/transcriber.py)
使用OpenAI的Whisper模型将用户语音转换为文本
支持音频预处理以提高识别质量
'''

import whisper
import time
import os
import torch
import shutil
import numpy as np
from scipy import signal
import config  # 导入配置模块

def check_ffmpeg():
    """检查系统中是否能找到ffmpeg"""
    if shutil.which("ffmpeg"):
        print("FFmpeg 已找到，环境正常。")
        return True
    else:
        print("="*50)
        print("!! 警告：在系统的PATH环境变量中找不到 FFmpeg !!")
        print("Whisper 需要 FFmpeg 来处理音频文件。")
        print("请按照以下步骤操作：")
        print("1. 从 https://www.gyan.dev/ffmpeg/builds/ 下载 FFmpeg。")
        print("2. 解压文件 (例如解压到 C:\\ffmpeg)。")
        print("3. 将解压后的 'bin' 目录 (例如 C:\\ffmpeg\\bin) 添加到 Windows 的系统环境变量 'Path' 中。")
        print("4. 重要：重启你的终端 (PowerShell/CMD) 甚至重启电脑，以使环境变量生效。")
        print("="*50)
        return False

# -------------------- AudioTranscriber类 --------------------
class AudioTranscriber:
    """
    一个使用OpenAI Whisper模型进行语音转文本的类。
    """
    def __init__(self):
        """
        初始化并加载Whisper模型。
        """
        print("--- 模块二：语音转文本 ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_size = config.MODEL_TRANSCRIPTION  # 从配置中读取模型大小
        print(f"正在加载 Whisper 模型 ('{model_size}')...")
        print(f"使用的计算设备: {self.device.upper()}")
        
        start_time = time.time()
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            end_time = time.time()
            print(f"模型加载完毕，耗时 {end_time - start_time:.2f} 秒。")
        except Exception as e:
            print(f"加载Whisper模型失败: {e}")
            raise

    def _preprocess_audio(self, audio):
        """预处理音频以提高识别质量"""
        try:
            import librosa
            import soundfile as sf
            
            # 加载音频
            y, sr = librosa.load(audio, sr=None)
            
            # 1. 降噪处理
            y_denoised = librosa.effects.preemphasis(y)
            
            # 2. 音量归一化
            y_normalized = librosa.util.normalize(y_denoised)
            
            # 3. 静音修剪
            y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=20)
            
            # 临时保存处理后的音频
            temp_audio = os.path.join(os.path.dirname(audio), "temp_processed.wav")
            sf.write(temp_audio, y_trimmed, sr)
            
            return temp_audio
        except ImportError:
            print("警告: librosa或soundfile库未安装，跳过音频预处理")
            return audio
        except Exception as e:
            print(f"音频预处理失败: {e}，使用原始音频")
            return audio

    def transcribe_audio(self, audio_path):
        """
        接收一个音频文件路径，返回识别出的文本。
        """
        absolute_audio_path = os.path.abspath(audio_path)

        if not os.path.exists(absolute_audio_path):
            print(f"错误：找不到音频文件 -> {absolute_audio_path}")
            return ""
            
        print(f"\n正在处理音频文件: {absolute_audio_path}")
        start_time = time.time()
        
        try:
            # 预处理音频
            try:
                processed_audio = self._preprocess_audio(absolute_audio_path)
                print("音频预处理完成")
            except Exception as e:
                print(f"音频预处理失败: {e}，使用原始音频")
                processed_audio = absolute_audio_path
            
            use_fp16 = self.device == "cuda"
            
            # 高级转录设置，提高准确率
            options = {
                "language": 'zh',       # 设置为中文
                "fp16": use_fp16,       # 根据设备使用FP16
                "temperature": 0.0,     # 确定性输出
                "beam_size": 5,         # 使用集束搜索
                "best_of": 5,           # 返回最佳结果
                "patience": 1.0,        # 提高搜索耐心
                "without_timestamps": True,  # 我们只需要文本
                "initial_prompt": "这是一段中文对话。"  # 提供初始提示
            }
            
            # 进行转录
            result = self.model.transcribe(processed_audio, **options)
            transcribed_text = result["text"].strip()
            
            # 如果使用了临时处理文件，删除它
            if processed_audio != absolute_audio_path and os.path.exists(processed_audio):
                try:
                    os.remove(processed_audio)
                except:
                    pass
            
            end_time = time.time()
            print(f"语音识别完成，耗时 {end_time - start_time:.2f} 秒。")
            print(f"识别结果: {transcribed_text}")
            
            return transcribed_text
        except Exception as e:
            print(f"语音识别过程中发生错误: {e}")
            return ""

# --- 主程序入口，用于独立测试此模块 ---
if __name__ == '__main__':
    if not check_ffmpeg():
        exit()

    try:
        # 安装必要的音频处理库
        try:
            import librosa
            import soundfile
        except ImportError:
            print("提示: 要获得更好的音频处理效果，请安装以下库:")
            print("pip install librosa soundfile")
        
        transcriber = AudioTranscriber()
        result_dir = "results"
        test_audio_file = None

        if not os.path.exists(result_dir):
            print(f"错误：找不到结果文件夹 '{result_dir}'。")
        else:
            wav_files = [f for f in os.listdir(result_dir) if f.endswith('.wav')]
            if not wav_files:
                print(f"错误：在 '{result_dir}' 文件夹中没有找到任何 .wav 音频文件。")
            else:
                latest_file = max(wav_files, key=lambda f: os.path.getmtime(os.path.join(result_dir, f)))
                test_audio_file = os.path.join(result_dir, latest_file)
                print(f"\n自动选择最新的音频文件进行测试: {test_audio_file}")

        if test_audio_file:
            text_result = transcriber.transcribe_audio(test_audio_file)
            if text_result:
                print("\n--- 测试成功 ---")
            else:
                print("\n--- 测试失败：未能识别出文本 ---")
        else:
            print("\n--- 测试终止：未找到可用的音频文件 ---")

    except Exception as e:
        print(f"\n程序运行出现意外错误: {e}")

