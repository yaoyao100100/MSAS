'''
大语言模型回复 (core/responder.py)
使用Qwen1.5-1.8B-Chat模型生成回复
根据检测到的情感调整回复的语气和内容
'''

import os
import warnings

# 抑制 TensorFlow 和 transformers 库的非关键信息 ---
# 1. 抑制 TensorFlow 的 INFO 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. 忽略来自 transformers 的 FutureWarning (例如 'torch_dtype' is deprecated)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import config  # 导入配置模块

# -------------------- LLMResponder类 --------------------
class LLMResponder:
    """
    一个用于与大语言模型交互，根据情感生成回复的类。
    """
    def __init__(self, model_name=None):
        print("--- 模块四：大语言模型交互 ---")
        
        # 如果没有提供模型名称，使用配置中的模型
        if model_name is None:
            model_name = config.MODEL_LLM
            
        print(f"正在加载大语言模型: {model_name}...")
        start_time = time.time()

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用的计算设备: {self.device.upper()}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            print(f"大语言模型加载完毕，耗时 {time.time() - start_time:.2f} 秒。")
        except Exception as e:
            print(f"错误：加载大语言模型失败。请检查网络连接或模型名称。错误信息: {e}")
            raise

    def _build_prompt(self, user_text, sentiment):
        """
        [核心] 根据用户文本和情感，构建一个高质量的Prompt。
        """
        system_prompt = "你是一个富有同情心和洞察力的AI情感伙伴，你的名字叫'心语'。你的任务是倾听用户的话语，并根据他们的情感状态，给出简短、温暖且有帮助的回应。请不要在回复中暴露你是一个AI模型。"

        if sentiment == "Positive":
            emotion_instruction = "用户现在的情绪是积极的、快乐的。请用同样阳光、鼓励的语气来回应他，分享他的喜悦。"
        elif sentiment == "Negative":
            emotion_instruction = "用户现在的情绪是消极的、悲伤的。请用非常温柔、有耐心、能给予安慰的语气来回应他，让他感受到被理解和支持。"
        else: # Neutral, Unknown, etc.
            emotion_instruction = "用户现在的情绪是平静的、中性的。请用友好、自然的语气与他交谈，可以提出一个开放性的问题来鼓励他多分享一些。"

        messages = [
            {"role": "system", "content": f"{system_prompt}\n{emotion_instruction}"},
            {"role": "user", "content": user_text}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_text

    def generate_response(self, user_text, sentiment):
        """
        接收用户文本和情感，生成模型的回复。
        """
        if not user_text:
            return "我在这里，准备好倾听你的心声。"

        prompt = self._build_prompt(user_text, sentiment)
        print("\n--- 构建的Prompt ---")
        print(prompt)
        print("--------------------")

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        print("正在生成AI回复...")
        start_time = time.time()
        
        # --- 核心修改：加入 attention_mask 和 pad_token_id 来消除警告 ---
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # --- 修改结束 ---

        response_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        
        # --- 核心修改：增加 .strip() 来清理回复文本 ---
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        print(f"AI回复生成完毕，耗时 {time.time() - start_time:.2f} 秒。")
        return response

# --- 主程序入口，自动读取最新结果 ---
if __name__ == '__main__':
    try:
        responder = LLMResponder()
        result_dir = config.RESULTS_DIR  # 从配置中读取结果目录
        latest_user_text = ""
        latest_sentiment = ""

        # 1. 检查results文件夹是否存在
        if not os.path.exists(result_dir):
             print(f"错误：找不到结果文件夹 '{result_dir}'。")
        else:
            # 2. 找到最新的情感分析总结文件
            sentiment_files = [f for f in os.listdir(result_dir) if f.endswith('_sentiment.txt')]
            if not sentiment_files:
                print(f"错误：在 '{result_dir}' 中未找到 _sentiment.txt 分析文件。")
            else:
                # 找到最新的文件
                latest_sentiment_filename = max(sentiment_files, key=lambda f: os.path.getmtime(os.path.join(result_dir, f)))
                
                # 3. 从文件名推导出对应的文本文件名
                base_name = latest_sentiment_filename.replace('_sentiment.txt', '')
                text_filename = base_name + ".txt"
                
                # 4. 读取文本内容和情感标签
                text_filepath = os.path.join(result_dir, text_filename)
                sentiment_filepath = os.path.join(result_dir, latest_sentiment_filename)

                if os.path.exists(text_filepath) and os.path.exists(sentiment_filepath):
                    with open(text_filepath, 'r', encoding='utf-8') as f:
                        latest_user_text = f.read().strip()
                    
                    with open(sentiment_filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith("Final Sentiment:"):
                                latest_sentiment = line.strip().split(": ")[1]
                                break
                else:
                    print("错误：找到了分析文件但对应的文本文件不存在。")

        # 5. 如果成功获取到数据，则调用大模型
        if latest_user_text and latest_sentiment:
            print("\n" + "="*50)
            print("      LLM 交互模块 - 自动读取最新结果")
            print("="*50)
            print(f"\n--- 读取到的用户输入 (情感: {latest_sentiment}) ---\n'{latest_user_text}'")
            
            response = responder.generate_response(latest_user_text, latest_sentiment)
            
            print(f"\nAI (心语) 回复:\n{response}")
            print("\n--- 测试成功 ---")
        else:
            print("\n--- 测试终止：未能从 'results' 文件夹获取到有效的用户输入和情感标签 ---")
            
    except Exception as e:
        print(f"\n程序运行出现意外错误: {e}")

