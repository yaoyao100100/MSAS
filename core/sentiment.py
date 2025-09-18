'''
情感分析 (core/sentiment.py)
文本情感分析：使用Erlangshen-Roberta模型
视频情绪识别：使用FER(Facial Emotion Recognition)分析面部表情
多模态融合：综合文本和视频的情感结果
'''

import os
# 在导入TensorFlow之前，设置环境变量来抑制其提示信息 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导入warnings库并过滤掉特定的Keras UserWarning ---
import warnings
# 我们只忽略这条特定的、来自Keras的、关于输入结构不匹配的警告
warnings.filterwarnings('ignore', message='The structure of `inputs` doesn\'t match the expected structure.*')


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fer import FER
import cv2 # OpenCV 用于视频处理
import pandas as pd
import time
import shutil # 用于文件操作
import config  # 导入配置模块

# -------------------- SentimentAnalyzer类 --------------------
class SentimentAnalyzer:
    """
    一个结合文本和视频进行多模态情感分析的类。
    """
    def __init__(self):
        print("--- 模块三：多模态情感分析 ---")
        # --- 1. 初始化文本情感分析模型 ---
        print("正在加载中文文本情感分析模型...")
        start_time = time.time()
        try:
            self.text_model_name = config.MODEL_SENTIMENT  # 从配置中读取模型名称
            self.tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
            self.text_model = BertForSequenceClassification.from_pretrained(self.text_model_name)
            print(f"文本模型加载完毕，耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            print(f"错误：加载文本模型失败。请检查网络连接。错误信息: {e}")
            raise

        # --- 2. 初始化视频面部情绪识别模型 ---
        print("\n正在加载面部情绪识别(FER)模型...")
        start_time = time.time()
        try:
            self.video_analyzer = FER()
            print(f"FER模型加载完毕，耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            print(f"错误：加载FER模型失败。请检查网络和tensorflow安装。错误信息: {e}")
            raise

    def analyze_text_sentiment(self, text):
        """分析单个文本的情感（积极/消极）"""
        if not text:
            return "Neutral", 0.0

        print(f"\n正在分析文本: '{text}'")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
        positive_prob = probabilities[1].item()
        
        if positive_prob > 0.6:
            sentiment = "Positive"
        elif positive_prob < 0.4:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        print(f"文本情感分析结果: {sentiment} (积极概率: {positive_prob:.2f})")
        return sentiment, positive_prob

    def analyze_video_emotion(self, video_path):
        """
        [核心逻辑重构] 通过逐帧分析视频中的主要面部情绪，提高稳定性。
        """
        if not os.path.exists(video_path):
            print(f"错误: 找不到视频文件 {video_path}")
            return "Unknown"
        
        print(f"\n正在分析视频 (逐帧处理): {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误: OpenCV无法打开视频文件 {video_path}")
                return "Error"

            all_emotions = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调用fer的核心功能，对单张图片（帧）进行分析
                # result 是一个列表，包含视频中每个脸的数据
                result = self.video_analyzer.detect_emotions(frame)
                
                # result 结构: [{'box': [x, y, w, h], 'emotions': {'angry': 0.0, ...}}]
                if result:
                    # 我们只关心第一个检测到的人脸的情绪数据
                    all_emotions.append(result[0]['emotions'])

            cap.release()

            if not all_emotions:
                print("警告: 在视频所有帧中都未能检测到有效的情绪数据。")
                return "NoFace"

            # 将收集到的所有情绪数据转换为DataFrame，方便分析
            emotion_df = pd.DataFrame(all_emotions)
            
            # 手动保存详细的帧数据
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            target_csv_path = os.path.join(config.RESULTS_DIR, f"{base_filename}_details.csv")
            emotion_df.to_csv(target_csv_path, index=False)
            print(f"详细情绪数据已保存至: {target_csv_path}")

            # 计算在所有帧中，哪种情绪作为主要情绪出现的次数最多
            dominant_emotion = emotion_df.idxmax(axis=1).mode()
            
            if dominant_emotion.empty:
                return "Unknown"

            final_emotion = dominant_emotion[0]
            print(f"视频主要面部情绪分析结果: {final_emotion.capitalize()}")
            return final_emotion.capitalize()

        except Exception as e:
            print(f"错误：处理视频时发生意外: {e}")
            return "Error"
    
    def get_multimodal_sentiment(self, video_path, text):
        """
        执行多模态情感分析并返回结果
        """
        text_sentiment, _ = self.analyze_text_sentiment(text)
        video_emotion = self.analyze_video_emotion(video_path)

        print(f"\n--- 融合分析 ---")
        print(f"文本情感: {text_sentiment}, 视频情绪: {video_emotion}")

        #修改计算权重
        text_weights = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0}
        video_weights = {
            "Happy": 1, "Surprise": 0.5, "Disgust": -1.0,
            "Sad": -0.7, "Angry": -1.0, "Fear": -0.6,
            "Neutral": 0.1
        }
        # 计算加权得分
        text_score = text_weights.get(text_sentiment, 0)
        video_score = video_weights.get(video_emotion, 0)
        total_score = text_score * 0.6 + video_score * 0.4  # 文本权重60%，视频40%
        # 判定最终情感
        if total_score >= 0.3:
            final_sentiment = "Positive"
        elif total_score <= -0.3:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"
        
        print(f"综合情感判断结果: {final_sentiment}")
        
        # 返回包含详细信息的字典，而不是直接写入文件
        result = {
            "text_sentiment": text_sentiment,
            "video_emotion": video_emotion,
            "final_sentiment": final_sentiment
        }
        
        return result

    def _cleanup(self):
        """清理分析过程中产生的临时文件/文件夹"""
        # 由于我们不再依赖fer.Video，它不会再自动创建output和data.csv
        # 但保留这个函数以备不时之需
        output_dir = "output"
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            if not os.listdir(output_dir): # 检查文件夹是否为空
                try:
                    os.rmdir(output_dir)
                    print(f"\n已自动清理空的 '{output_dir}' 文件夹。")
                except OSError as e:
                    print(f"清理 '{output_dir}' 文件夹时出错: {e}")

# --- 主程序入口，用于独立测试此模块 ---
if __name__ == '__main__':
    try:
        analyzer = SentimentAnalyzer()
        result_dir = config.RESULTS_DIR  # 从配置中读取结果目录
        latest_video = None
        latest_text_content = ""

        if not os.path.exists(result_dir):
             print(f"错误：找不到结果文件夹 '{result_dir}'。")
        else:
            avi_files = [f for f in os.listdir(result_dir) if f.endswith('.avi')]
            if not avi_files:
                print(f"错误：在 '{result_dir}' 中未找到 .avi 视频文件。")
            else:
                latest_video_name = max(avi_files, key=lambda f: os.path.getmtime(os.path.join(result_dir, f)))
                latest_video = os.path.join(result_dir, latest_video_name)
                
                latest_text_path = os.path.splitext(latest_video)[0] + ".txt"
                if os.path.exists(latest_text_path):
                    with open(latest_text_path, 'r', encoding='utf-8') as f:
                        latest_text_content = f.read()
                else:
                    print(f"警告：找到了视频文件 '{latest_video_name}'，但没有找到对应的文本文件。")

        if latest_video and latest_text_content:
            print("\n" + "="*50)
            print(f"自动选择最新文件进行多模态情感分析:")
            print(f"视频文件: {latest_video}")
            print(f"文本内容: '{latest_text_content}'")
            print("="*50)
            
            # 分析情感
            result = analyzer.get_multimodal_sentiment(latest_video, latest_text_content)
            
            # 将分析结果保存到文件
            base_filename = os.path.splitext(latest_video)[0]
            result_filename = base_filename + "_sentiment.txt"
            with open(result_filename, 'w', encoding='utf-8') as f:
                f.write(f"Text Sentiment: {result['text_sentiment']}\n")
                f.write(f"Video Emotion: {result['video_emotion']}\n")
                f.write(f"Final Sentiment: {result['final_sentiment']}\n")
            print(f"分析结果已保存至: {result_filename}")
            
            print("\n--- 测试成功 ---")
        else:
            print("\n--- 测试终止：未能找到成对的视频和文本文件 ---")
            
    except Exception as e:
        print(f"\n程序运行出现意外错误: {e}")

