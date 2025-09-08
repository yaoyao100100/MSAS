import os
from .transcriber import AudioTranscriber
from .sentiment import SentimentAnalyzer
from .responder import LLMResponder
from config import RESULTS_DIR

class AnalysisPipeline:
    def __init__(self, transcriber: AudioTranscriber, analyzer: SentimentAnalyzer, responder: LLMResponder):
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.responder = responder

    def run(self, audio_path, video_path):
        """
        执行完整的分析流程。
        返回: (用户文本, AI回复, 文本情感, 视频情绪)
        """
        print("\n--- 开始分析流程 ---")
        # 1. 语音转文本
        user_text = self.transcriber.transcribe_audio(audio_path)
        if not user_text:
            user_text = "(未能识别语音)"
        # 将文本保存到文件 (这是 pipeline 的职责)
        self._save_text(audio_path, user_text)

        # 2. 多模态情感分析
        final_sentiment = self.analyzer.get_multimodal_sentiment(video_path, user_text)

        # 从结果文件中获取详细情感信息
        basename = os.path.splitext(os.path.basename(video_path))[0]
        sentiment_file = os.path.join(RESULTS_DIR, f"{basename}_sentiment.txt")
        text_sentiment, video_emotion = self._read_sentiment_details(sentiment_file)

        # 3. 生成AI回复
        ai_response = self.responder.generate_response(user_text, final_sentiment)
        print("--- 分析流程结束 ---")
        
        return user_text, ai_response, text_sentiment, video_emotion

    def _save_text(self, audio_path, text):
        base_filename = os.path.splitext(audio_path)[0]
        text_filename = base_filename + ".txt"
        try:
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"语音识别文本已保存至: {text_filename}")
        except Exception as e:
            print(f"错误：保存文本文件失败: {e}")

    def _read_sentiment_details(self, file_path):
        text_sentiment = "未知"
        video_emotion = "未知"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Text Sentiment:"):
                        text_sentiment = line.strip().split(": ")[1]
                    elif line.startswith("Video Emotion:"):
                        video_emotion = line.strip().split(": ")[1]
        return text_sentiment, video_emotion