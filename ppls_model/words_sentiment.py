import os
import math
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        # 缓存已分析过的词语情感，避免重复计算
        self.word_sentiment_cache = {}
        # 缓存已分析过的上下文情感
        self.context_sentiment_cache = {}
        
        # 初始化模型为None
        self.tokenizer = None
        self.model = None
        
        # 内置一些常见词的情感分数作为初始值
        self._init_common_words_cache()
        
        # 尝试加载预训练的情感分析模型，使用本地缓存并禁用自动转换
        try:
            # 设置环境变量禁用自动转换
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_DISABLE_AUTO_CONVERSION"] = "1"
            
            # 使用local_files_only参数强制使用本地缓存
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"尝试从本地缓存目录加载: {cache_dir}")
            
            self.tokenizer = BertTokenizer.from_pretrained(
                "uer/roberta-base-finetuned-jd-binary-chinese",
                local_files_only=True,
                cache_dir=cache_dir
            )
            self.model = BertForSequenceClassification.from_pretrained(
                "uer/roberta-base-finetuned-jd-binary-chinese",
                local_files_only=True,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=True
            )
            self.model.eval()  # 设置为评估模式
            print("情感分析模型加载成功（仅使用本地缓存）")
        except Exception as e:
            print(f"无法加载情感分析模型: {e}")
            print("将使用内置的情感词典进行分析")
            # 禁用模型使用
            self.tokenizer = None
            self.model = None
    
    def _init_common_words_cache(self):
        """初始化一些常见词的情感缓存"""
        common_words = {
            "开心": 0.8,
            "高兴": 0.7,
            "快乐": 0.85,
            "伤心": -0.7,
            "难过": -0.6,
            "唉": -0.8,
            "爱你": 0.9,
            "不错": 0.6,
            "挺好": 0.65,
            "哈哈哈哈": 0.9,
            "好吧": 0.0  # 中性词
        }
        
        for word, score in common_words.items():
            self.word_sentiment_cache[word] = score
    
    def calculate_enhanced_sentiment_score(self, logits):
        """
        计算增强的情感得分，结合logits差值和熵
        :param logits: 模型原始输出的logits
        :return: -1到1之间的情感得分
        """
        # 计算概率
        probabilities = torch.softmax(logits, dim=-1)
        pos_prob = probabilities[0][1].item()
        neg_prob = probabilities[0][0].item()
        
        # 计算基础得分（概率差值）
        base_score = pos_prob - neg_prob
        
        # 计算熵（反映不确定性）
        if pos_prob < 1e-10:
            pos_term = 0
        else:
            pos_term = pos_prob * math.log(pos_prob)
        if neg_prob < 1e-10:
            neg_term = 0
        else:
            neg_term = neg_prob * math.log(neg_prob)
        entropy = - (pos_term + neg_term)
        max_entropy = math.log(2)  # 二分类的最大熵
        uncertainty_factor = 1 - (entropy / max_entropy)
        
        # 结合基础得分和不确定性因子
        enhanced_score = base_score * uncertainty_factor
        
        # 确保得分在-1到1之间
        enhanced_score = max(-1.0, min(1.0, enhanced_score))
        
        return enhanced_score
    
    def get_word_sentiment_score(self, word):
        """
        获取词语的情感分数
        :param word: 要分析的词语
        :return: -1到1之间的情感分数，正值表示积极，负值表示消极
        """
        # 优先从缓存获取
        if word in self.word_sentiment_cache:
            return self.word_sentiment_cache[word]
        
        # 如果有预训练模型，使用模型分析
        if self.tokenizer is not None and self.model is not None:
            try:
                inputs = self.tokenizer(f"这个{word}的体验", return_tensors="pt")
                with torch.no_grad():  # 不计算梯度，节省内存
                    outputs = self.model(**inputs)
                # 使用增强的情感得分计算方法
                normalized_score = self.calculate_enhanced_sentiment_score(outputs.logits)
                
                # 缓存结果
                self.word_sentiment_cache[word] = normalized_score
                return normalized_score
            except Exception as e:
                print(f"分析词语 '{word}' 情感时出错: {e}")
        
        # 如果没有模型或分析失败，使用基于字符的简单情感判断
        positive_chars = set("好棒赞喜欢爱乐欢笑甜美")
        negative_chars = set("坏差讨厌恨悲痛苦哭愁")
        
        pos_count = sum(1 for char in word if char in positive_chars)
        neg_count = sum(1 for char in word if char in negative_chars)
        
        # 计算简单的情感分数
        if pos_count > neg_count:
            score = 0.5 + (pos_count - neg_count) * 0.1
        elif neg_count > pos_count:
            score = -0.5 - (neg_count - pos_count) * 0.1
        else:
            score = 0.0
        
        # 限制在-1到1之间
        score = max(-1.0, min(1.0, score))
        
        # 缓存结果
        self.word_sentiment_cache[word] = score
        return score
    
    def get_context_sentiment_score(self, context_text):
        """
        获取上下文文本的情感分数
        :param context_text: 要分析的上下文文本
        :return: -1到1之间的情感分数，正值表示积极，负值表示消极
        """
        # 优先从缓存获取（对于较短的上下文）
        if len(context_text) < 100 and context_text in self.context_sentiment_cache:
            return self.context_sentiment_cache[context_text]
        
        # 对于很长的文本，取最后一部分进行分析
        # 这样更关注最近的情感倾向
        analysis_text = context_text[-200:] if len(context_text) > 200 else context_text
        
        # 如果有预训练模型，使用模型分析
        if self.tokenizer is not None and self.model is not None:
            try:
                # 对整个文本进行情感分析
                inputs = self.tokenizer(analysis_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 使用增强的情感得分计算方法
                normalized_score = self.calculate_enhanced_sentiment_score(outputs.logits)
                
                # 只缓存较短的上下文
                if len(context_text) < 100:
                    self.context_sentiment_cache[context_text] = normalized_score
                
                return normalized_score
            except Exception as e:
                print(f"分析上下文情感时出错: {e}")
        
        # 如果没有模型或分析失败，使用基于字符的简单情感判断
        positive_chars = set("好棒赞喜欢爱乐欢笑甜美!")
        negative_chars = set("坏差讨厌恨悲痛苦哭愁!")
        
        pos_count = sum(1 for char in analysis_text if char in positive_chars)
        neg_count = sum(1 for char in analysis_text if char in negative_chars)
        
        # 计算简单的情感分数
        if pos_count > neg_count:
            score = 0.5 + (pos_count - neg_count) * 0.05
        elif neg_count > pos_count:
            score = -0.5 - (neg_count - pos_count) * 0.05
        else:
            score = 0.0
        
        # 限制在-1到1之间
        score = max(-1.0, min(1.0, score))
        
        # 只缓存较短的上下文
        if len(context_text) < 100:
            self.context_sentiment_cache[context_text] = score
        
        return score
    
    def calculate_sentiment_factor(self, context_score, word_score, beta=1.0):
        """
        计算情感匹配因子
        :param context_score: 上下文情感分数 (-1到1)
        :param word_score: 词语情感分数 (-1到1)
        :param beta: 情感扰动系数，控制情感影响的强度
        :return: 情感匹配因子
        """
        # 基本公式：beta * (context_score * word_score)
        # 这会产生一个-1到1之间的值
        raw_factor = beta * (context_score * word_score)
        
        # 将结果映射到一个乘数因子，而不是直接加减
        # 当情感匹配时（同正或同负），增强权重
        # 当情感不匹配时，降低权重
        # 使用指数函数平滑过渡
        if raw_factor >= 0:
            # 情感匹配时，权重范围：1.0到e^beta
            factor = 1.0 + (raw_factor * 0.5)  # 更温和的增长
        else:
            # 情感不匹配时，权重范围：1/e^beta到1.0
            factor = 1.0 + (raw_factor * 0.3)  # 稍微降低惩罚力度
        
        # 确保因子在合理范围内
        factor = max(0.2, min(3.0, factor))
        return factor
    
    def analyze_bag_of_words(self, words_list, context_text, beta=1.0):
        """
        分析词袋中所有词的情感，并计算它们相对于上下文的情感匹配因子
        :param words_list: 词袋列表
        :param context_text: 当前上下文
        :param beta: 情感扰动系数
        :return: 包含每个词的情感信息和匹配因子的字典
        """
        # 获取上下文情感分数
        context_score = self.get_context_sentiment_score(context_text)
        
        # 分析每个词的情感
        results = {}
        for word in words_list:
            word_score = self.get_word_sentiment_score(word)
            sentiment_factor = self.calculate_sentiment_factor(context_score, word_score, beta)
            
            results[word] = {
                "word_score": word_score,
                "sentiment_direction": "positive" if word_score > 0.2 else "negative" if word_score < -0.2 else "neutral",
                "intensity": abs(word_score),
                "sentiment_factor": sentiment_factor
            }
        
        return results

# 创建一个全局实例，方便直接使用
sentiment_analyzer = SentimentAnalyzer()

# 简化的API函数
def get_word_sentiment(word):
    return sentiment_analyzer.get_word_sentiment_score(word)

def get_context_sentiment(context):
    return sentiment_analyzer.get_context_sentiment_score(context)

def get_sentiment_factor(context, word, beta=1.0):
    context_score = get_context_sentiment(context)
    word_score = get_word_sentiment(word)
    return sentiment_analyzer.calculate_sentiment_factor(context_score, word_score, beta)

# 测试示例
if __name__ == "__main__":
    # 测试词语情感分析
    print(f"开心的情感分数: {get_word_sentiment('开心')}")
    print(f"难过的情感分数: {get_word_sentiment('难过')}")
    print(f"爱你的情感分数: {get_word_sentiment('爱你')}")
    
    # 测试上下文情感分析
    positive_context = "今天天气真好，心情非常愉快！"
    negative_context = "我很不开心，遇到了很多麻烦。"
    neutral_context = "今天是星期三，温度适宜。"
    
    print(f"积极上下文情感分数: {get_context_sentiment(positive_context)}")
    print(f"消极上下文情感分数: {get_context_sentiment(negative_context)}")
    print(f"中性上下文情感分数: {get_context_sentiment(neutral_context)}")
    
    # 测试情感因子
    print(f"积极上下文中'开心'的情感因子: {get_sentiment_factor(positive_context, '开心', beta=1.0)}")
    print(f"消极上下文中'开心'的情感因子: {get_sentiment_factor(negative_context, '开心', beta=1.0)}")
    print(f"消极上下文中'难过'的情感因子: {get_sentiment_factor(negative_context, '难过', beta=1.0)}")