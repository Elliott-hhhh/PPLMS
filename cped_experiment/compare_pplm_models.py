#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPLM模型对比实验
比较纯模型生成（不带干预）和修改后的PPLM模型在对话生成中的表现
使用CPED数据集进行评估
评估指标：情感匹配度、词袋命中概率等
"""

import os
import sys
import argparse
import torch
import json
import jieba
from collections import defaultdict, Counter
import numpy as np
from ..words_sentiment import sentiment_analyzer, get_context_sentiment

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修改后的PPLM
from ..ppls_model.generate_ppls import run_pplm_example as run_modified_pplm

# 设置中文字符
torch.manual_seed(42)
np.random.seed(42)

# 情感标签映射
SENTIMENT_MAPPING = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

class CharacterWordbagManager:
    """
    角色词袋管理器
    支持从CPED数据集直接提取说话人发言并生成词袋
    """
    def __init__(self, cped_dataset_path=None):
        self.cped_dataset_path = cped_dataset_path
        self.character_wordbags = {}
        self.sentiment_wordbags = {}
        self.dataset_speakers = []
        self.dataset_dialogues = []
        
        # 如果提供了数据集路径，加载数据集
        if self.cped_dataset_path:
            self._load_dataset()
    
    def _load_dataset(self):
        """
        加载CPED数据集并提取说话人和对话
        """
        try:
            print(f"从 {self.cped_dataset_path} 加载数据集以提取说话人信息...")
            
            if not os.path.exists(self.cped_dataset_path):
                print(f"警告: 数据集文件不存在: {self.cped_dataset_path}")
                return
            
            # 假设数据格式为CSV
            import csv
            dialogue_data = []
            
            with open(self.cped_dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 提取说话人和对话内容
                    if 'Speaker' in row and 'Utterance' in row:
                        speaker = row['Speaker'].strip()
                        text = row['Utterance'].strip()
                        if speaker and text:
                            dialogue_data.append((speaker, text))
                            self.dataset_dialogues.append({
                                'speaker': speaker,
                                'text': text
                            })
                            # 收集唯一的说话人
                            if speaker not in self.dataset_speakers:
                                self.dataset_speakers.append(speaker)
            
            print(f"成功加载 {len(dialogue_data)} 条对话记录")
            print(f"发现 {len(self.dataset_speakers)} 个不同的说话人")
            print(f"前5个说话人: {self.dataset_speakers[:5]}")
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
    
    def generate_wordbag_from_speaker(self, speaker_name, min_word_freq=1, top_n_words=20):
        """
        从指定说话人的所有发言中生成词袋
        
        Args:
            speaker_name: 说话人名称
            min_word_freq: 词频阈值，只包含出现次数>=此值的词
            top_n_words: 返回频率最高的N个词
            
        Returns:
            词袋集合
        """
        if speaker_name in self.character_wordbags:
            print(f"使用已生成的 {speaker_name} 词袋")
            return self.character_wordbags[speaker_name]
        
        # 收集该说话人的所有文本
        speaker_texts = []
        for dialogue in self.dataset_dialogues:
            if dialogue['speaker'] == speaker_name:
                speaker_texts.append(dialogue['text'])
        
        if not speaker_texts:
            print(f"警告: 未找到说话人 {speaker_name} 的发言记录")
            return set()
        
        print(f"为说话人 {speaker_name} 从 {len(speaker_texts)} 条发言中生成词袋...")
        
        # 分词并统计词频
        word_counter = Counter()
        for text in speaker_texts:
            words = jieba.lcut(text)
            # 过滤掉太短的词和停用词
            filtered_words = [
                word for word in words 
                if len(word) > 1 and 
                word.strip() and 
                word not in ['，', '。', '！', '？', '：', '；', '的', '了', '是', '在']
            ]
            word_counter.update(filtered_words)
        
        # 过滤低频词
        filtered_words = [(word, count) for word, count in word_counter.items() if count >= min_word_freq]
        
        # 按频率排序并取前N个词
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top_n_words]
        
        # 创建词袋集合
        wordbag = set(word[0] for word in sorted_words)
        
        # 保存词袋
        self.character_wordbags[speaker_name] = wordbag
        
        print(f"生成的词袋大小: {len(wordbag)}")
        print(f"最常用的10个词: {[word[0] for word in sorted_words[:10]]}")
        
        return wordbag
    
    def get_available_speakers(self):
        """
        获取数据集中所有可用的说话人列表
        """
        return self.dataset_speakers
    
    def get_character_wordbag(self, character):
        """
        获取指定角色的词袋
        """
        return self.character_wordbags.get(character, set())
    
    def get_sentiment_wordbag(self, sentiment):
        """
        获取指定情感的词袋（这里使用简单的情感词典）
        """
        if sentiment not in self.sentiment_wordbags:
            # 简单的情感词典
            if sentiment == 'positive':
                self.sentiment_wordbags[sentiment] = {'好', '喜欢', '开心', '快乐', '满意', '高兴', '棒', '优秀', '精彩', '成功'}
            elif sentiment == 'negative':
                self.sentiment_wordbags[sentiment] = {'不好', '讨厌', '伤心', '难过', '不满意', '生气', '糟糕', '失败', '失望', '遗憾'}
            else:  # neutral
                self.sentiment_wordbags[sentiment] = {'一般', '还行', '普通', '正常', '平常', '可能', '也许', '大概', '据说', '听说'}
        return self.sentiment_wordbags[sentiment]

# 修改后的加载模型和分词器函数
def load_model_and_tokenizer(model_name=r"C:\Users\23921\.cache\modelscope\Qwen3-4B", device=None):
    """
    加载模型和分词器
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"正在从本地路径加载模型: {model_name}")
    # 直接使用用户指定的本地路径加载模型，强制使用本地文件
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件，避免联网下载
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件
        )
        print("模型和分词器加载成功！")
    except Exception as e:
        print(f"加载本地模型失败: {e}")
        raise
    
    print(f"模型已使用device_map自动分配到设备")
    return model, tokenizer, device

# 从CPED数据集提取提示词的函数
def extract_prompts_from_cped(dataset_path=None, num_prompts=10):
    """
    从CPED数据集提取对话提示词
    
    Args:
        dataset_path: CPED数据集路径
        num_prompts: 要提取的提示词数量
        
    Returns:
        prompts: 用于生成的提示词列表
    """
    # 默认使用train_split.csv作为CPED数据集路径
    if dataset_path is None:
        dataset_path = r"../data/cped/train_split.csv"
    
    prompts = []
    
    try:
        print(f"从CPED数据集提取提示词: {dataset_path}")
        
        # 检查文件是否存在
        if not os.path.exists(dataset_path):
            print(f"警告: CPED数据集文件不存在: {dataset_path}")
            return None
        
        import csv
        dialogues = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 尝试不同的字段名获取对话内容
                for field in ['context', 'dialogue', 'text', 'content']:
                    if field in row and row[field].strip():
                        dialogues.append(row[field].strip())
                        break
        
        # 去重并限制数量
        dialogues = list(set(dialogues))
        if len(dialogues) > num_prompts:
            import random
            random.seed(42)
            prompts = random.sample(dialogues, num_prompts)
        else:
            prompts = dialogues
            
        print(f"成功提取 {len(prompts)} 条提示词")
        return prompts
        
    except Exception as e:
        print(f"CPED数据集加载失败: {e}")
        return None

# 纯模型生成函数保持不变
def run_vanilla_model_generation(model, tokenizer, prompt, device,
                               temperature=0.7, top_k=10, top_p=1.0,
                               sample=True, num_tokens=40):
    """
    纯模型生成（不带干预）
    使用原始模型直接进行文本生成，不应用任何PPLM干预
    使用tqdm内置速度计算功能
    """
    import time
    from tqdm import trange
    
    try:
        print("运行纯模型生成...")
        
        # 注意：使用device_map="auto"加载的模型不能手动调用to()方法移动设备
        # accelerate已自动将模型分配到合适的设备上
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            # 确保输入张量在指定设备上
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # 使用tqdm来跟踪生成进度和速度
            with trange(num_tokens, desc="纯模型生成", unit="token") as pbar:
                start_time = time.time()
                generated_tokens = 0
                
                # 逐步生成tokens，以便使用tqdm跟踪
                current_inputs = inputs
                for i in pbar:
                    outputs = model.generate(
                        **current_inputs,
                        max_new_tokens=1,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=sample,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # 获取新生成的token
                    new_token = outputs[0][-1:]
                    generated_tokens += 1
                    
                    # 更新进度条，tqdm会自动计算和显示速度
                    pbar.set_postfix({"tokens": generated_tokens})
                    
                    # 为下一次生成准备输入
                    current_inputs = {
                        'input_ids': new_token.unsqueeze(0),
                        'attention_mask': torch.ones_like(new_token.unsqueeze(0))
                    }
                    for key in current_inputs:
                        current_inputs[key] = current_inputs[key].to(device)
                
                # 重新生成完整输出
                full_outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=sample
                )
                output = tokenizer.decode(full_outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        # 从tqdm进度条获取速度信息
        tokens_per_second = pbar.format_dict["rate"] if "rate" in pbar.format_dict else generated_tokens / generation_time
        
        # 计算生成的文本部分
        generated_text = output[len(prompt):].strip() if output.startswith(prompt) else output
        actual_generated_tokens = len(generated_text) if generated_text else 0
        
        return output, generation_time, actual_generated_tokens, tokens_per_second
    except Exception as e:
        print(f"纯模型生成失败: {e}")
        return prompt, 0.0, 0, 0.0

def run_modified_pplm_adapter(prompt, bow_words, sentiment, pretrained_model_path, num_tokens=50):
    """修改后的PPLM适配器，直接调用generate_ppls.py的完整功能，返回生成文本和时间信息，使用tqdm内置速度计算"""
    import subprocess
    import sys
    import os
    import time
    from tqdm import trange
    
    # 创建临时词袋文件
    temp_bow_path = "temp_bow.txt"
    with open(temp_bow_path, "w", encoding="utf-8") as f:
        for word in bow_words:
            f.write(word + "\n")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 使用tqdm来跟踪生成进度和速度
        with trange(num_tokens, desc="PPLM生成", unit="token") as pbar:
            # 直接调用generate_ppls.py脚本
            # 使用subprocess运行，捕获输出
            process = subprocess.Popen(
                [
                    sys.executable,
                    "generate_ppls.py",
                    "--cond_text", prompt,
                    "--bag_of_words", temp_bow_path,
                    "--length", str(num_tokens),
                    "--num_samples", "1",
                    "--sample"
                ],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 读取输出，尝试多种编码
            stdout, stderr = process.communicate()
            
            # 记录结束时间并计算生成时间
            generation_time = time.time() - start_time
            
            # 更新进度条到完成状态
            pbar.update(num_tokens)
        
        # 尝试解码输出，使用utf-8，如果失败则尝试gbk
        try:
            output_text = stdout.decode('utf-8')
        except UnicodeDecodeError:
            try:
                output_text = stdout.decode('gbk')
            except UnicodeDecodeError:
                # 如果都失败，使用replace策略
                output_text = stdout.decode('utf-8', errors='replace')
        
        # 解析输出，提取生成的文本
        # 过滤掉模型加载日志，只保留实际生成的文本
        output_lines = output_text.splitlines()
        generated_text = ""
        
        # 遍历所有输出行
        for line in output_lines:
            stripped_line = line.strip()
            if stripped_line:
                # 过滤掉明显的日志行
                if not ("加载" in stripped_line and "模型" in stripped_line or 
                        "尝试从本地缓存目录加载" in stripped_line or 
                        "modelscope" in stripped_line.lower() or
                        "huggingface" in stripped_line.lower() or
                        "cache" in stripped_line.lower()):
                    # 检查是否是日志格式（包含日期时间、日志级别等）
                    if not (len(stripped_line) > 20 and ":" in stripped_line[:20]):
                        generated_text = stripped_line
                        break
        
        # 如果没有找到合适的生成文本，尝试用更宽松的方式
        if not generated_text:
            for line in output_lines:
                stripped_line = line.strip()
                if stripped_line and len(stripped_line) > 10:  # 至少10个字符，排除短日志
                    generated_text = stripped_line
                    break
        
        # 计算生成速度（tokens/秒）
        final_text = generated_text.strip() if generated_text else prompt
        if final_text and final_text != prompt:
            # 去除原始提示词，计算生成的部分
            generated_part = final_text[len(prompt):].strip() if final_text.startswith(prompt) else final_text
            # 粗略估算token数量（中文字符数）
            generated_tokens = len(generated_part)
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        else:
            generated_tokens = 0
            tokens_per_second = 0
        
        print(f"调试: 从generate_ppls.py获取的原始输出:\n{output_text}")
        print(f"调试: 解析出的生成文本: {final_text}")
        print(f"调试: 生成时间: {generation_time:.4f}秒, 生成tokens: {generated_tokens}, 速度: {tokens_per_second:.2f} tokens/秒")
        
        return final_text, generation_time, generated_tokens, tokens_per_second
    except Exception as e:
        print(f"调用generate_ppls.py时出错: {e}")
        return prompt, 0.0, 0, 0.0
    finally:
        # 清理临时文件
        if os.path.exists(temp_bow_path):
            try:
                os.remove(temp_bow_path)
            except:
                pass

# 评估函数保持不变
def calculate_bag_hit_rate(generated_text, bag_of_words):
    """
    计算词袋命中概率
    :param generated_text: 生成的文本
    :param bag_of_words: 词袋集合
    :return: 命中概率和命中的词列表
    """
    if not bag_of_words:
        return 0.0, []
    
    # 使用jieba分词
    words = jieba.lcut(generated_text)
    # 统计命中的词
    hit_words = set()
    for word in words:
        if word in bag_of_words:
            hit_words.add(word)
    
    # 计算命中概率
    hit_rate = len(hit_words) / len(bag_of_words) if bag_of_words else 0.0
    # 归一化到合理范围
    hit_rate = min(hit_rate, 1.0)
    
    return hit_rate, list(hit_words)

def calculate_sentiment_similarity(generated_text, target_sentiment, sentiment_wordbags):
    """
    计算情感匹配度（使用预训练情感模型）
    :param generated_text: 生成的文本
    :param target_sentiment: 目标情感标签
    :param sentiment_wordbags: 情感词袋字典（保留参数以保持兼容性）
    :return: 情感匹配度分数
    """
    # 定义目标情感对应的理想分数
    target_sentiment_scores = {
        'positive': 0.8,    # 积极情感
        'negative': -0.8,   # 消极情感
        'neutral': 0.0      # 中性情感
    }
    
    # 获取生成文本的实际情感分数
    actual_score = get_context_sentiment(generated_text)
    
    # 获取目标情感的理想分数
    target_score = target_sentiment_scores.get(target_sentiment.lower(), 0.0)
    
    # 计算情感匹配度：1.0减去实际分数与目标分数的绝对差的归一化值
    # 匹配度范围：0.0（完全不匹配）到1.0（完全匹配）
    sentiment_match = 1.0 - (abs(actual_score - target_score) / 2.0)
    
    # 确保匹配度在0.0到1.0之间
    sentiment_match = max(0.0, min(1.0, sentiment_match))
    
    return sentiment_match

def evaluate_generated_text(generated_text, character, target_sentiment, wordbag_manager, generation_time=0.0, generated_tokens=0, tokens_per_second=0.0):
    """
    评估生成文本的质量
    """
    # 获取角色词袋和情感词袋
    character_bag = wordbag_manager.get_character_wordbag(character)
    sentiment_bags = wordbag_manager.sentiment_wordbags
    
    # 计算词袋命中概率
    char_hit_rate, char_hit_words = calculate_bag_hit_rate(generated_text, character_bag)
    
    # 计算情感匹配度
    sentiment_match = calculate_sentiment_similarity(generated_text, target_sentiment, sentiment_bags)
    
    # 计算文本长度
    text_length = len(generated_text)
    
    # 计算多样性指标（去重后词数/总词数）
    words = jieba.lcut(generated_text)
    unique_words = set(words)
    diversity = len(unique_words) / len(words) if words else 0.0
    
    return {
        'text': generated_text,
        'character_hit_rate': char_hit_rate,
        'character_hit_words': char_hit_words,
        'sentiment_match': sentiment_match,
        'text_length': text_length,
        'diversity': diversity,
        'generation_time': generation_time,
        'generated_tokens': generated_tokens,
        'tokens_per_second': tokens_per_second
    }

# 运行对比实验函数
def run_comparison_experiment(model, tokenizer, device, wordbag_manager, character, target_sentiment,
                              prompts, num_samples=5, num_tokens=40):
    """
    运行对比实验 - 比较纯模型生成（不带干预）和修改后的PPLM模型
    """
    results = defaultdict(list)
    
    # 获取角色词袋（从数据集中生成）
    character_bag = list(wordbag_manager.get_character_wordbag(character))
    
    # 获取模型路径
    pretrained_model_path = r"C:\Users\23921\.cache\modelscope\Qwen3-4B"
    
    print(f"\n开始对比实验：角色={character}, 目标情感={target_sentiment}")
    print(f"角色词袋大小: {len(character_bag)}")
    
    for i in range(num_samples):
        for prompt in prompts:
            print(f"\n样本 {i+1}/{num_samples}")
            print(f"提示词: {prompt}")
            
            # 1. 运行纯模型生成（不带干预）
            print("运行纯模型生成...")
            vanilla_output, vanilla_time, vanilla_tokens, vanilla_speed = run_vanilla_model_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                num_tokens=num_tokens,
                top_k=10,
                sample=True
            )
            
            # 评估纯模型输出
            vanilla_eval = evaluate_generated_text(
                vanilla_output,
                character,
                target_sentiment,
                wordbag_manager,
                vanilla_time,
                vanilla_tokens,
                vanilla_speed
            )
            results['vanilla'].append(vanilla_eval)
            
            print(f"纯模型输出: {vanilla_output}")
            print(f"纯模型评估: 词袋命中率={vanilla_eval['character_hit_rate']:.4f}, 情感匹配度={vanilla_eval['sentiment_match']:.4f}")
            
            # 2. 运行修改后的PPLM
            print("运行修改后的PPLM...")
            modified_output, modified_time, modified_tokens, modified_speed = run_modified_pplm_adapter(
                prompt=prompt,
                bow_words=character_bag,
                sentiment=target_sentiment,
                pretrained_model_path=pretrained_model_path,
                num_tokens=num_tokens
            )
            
            # 评估修改后的PPLM输出
            modified_eval = evaluate_generated_text(
                modified_output,
                character,
                target_sentiment,
                wordbag_manager,
                modified_time,
                modified_tokens,
                modified_speed
            )
            results['modified'].append(modified_eval)
            
            print(f"修改后的PPLM输出: {modified_output}")
            print(f"修改后的PPLM评估: 词袋命中率={modified_eval['character_hit_rate']:.4f}, 情感匹配度={modified_eval['sentiment_match']:.4f}")
    
    return results

# 生成对比报告函数
def generate_comparison_report(results, character, target_sentiment, output_file):
    """
    生成对比报告
    """
    # 计算平均指标
    report = {
        'character': character,
        'target_sentiment': target_sentiment,
        'metrics': {}
    }
    
    for model_type, model_results in results.items():
        char_hit_rates = [r['character_hit_rate'] for r in model_results]
        sentiment_matches = [r['sentiment_match'] for r in model_results]
        text_lengths = [r['text_length'] for r in model_results]
        diversities = [r['diversity'] for r in model_results]
        generation_times = [r['generation_time'] for r in model_results]
        generated_tokens = [r['generated_tokens'] for r in model_results]
        tokens_per_second = [r['tokens_per_second'] for r in model_results]
        
        report['metrics'][model_type] = {
            'avg_character_hit_rate': np.mean(char_hit_rates),
            'avg_sentiment_match': np.mean(sentiment_matches),
            'avg_text_length': np.mean(text_lengths),
            'avg_diversity': np.mean(diversities),
            'avg_generation_time': np.mean(generation_times),
            'avg_generated_tokens': np.mean(generated_tokens),
            'avg_tokens_per_second': np.mean(tokens_per_second) if any(t > 0 for t in tokens_per_second) else 0,
            'character_hit_rate_std': np.std(char_hit_rates),
            'sentiment_match_std': np.std(sentiment_matches),
            'samples': len(model_results)
        }
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比报告已保存到: {output_file}")
    print("\n===== 对比结果摘要 =====")
    print(f"角色: {character}")
    print(f"目标情感: {target_sentiment}")
    
    # 打印详细对比表格
    print(f"\n{'指标':<20} {'纯模型':<15} {'修改后PPLM':<15} {'提升':<15}")
    print("-" * 65)
    
    metrics = [
        ("词袋命中率", "avg_character_hit_rate", True),
        ("情感匹配度", "avg_sentiment_match", True),
        ("文本长度", "avg_text_length", False),
        ("多样性", "avg_diversity", True),
        ("生成时间(s)", "avg_generation_time", False),
        ("生成tokens", "avg_generated_tokens", False),
        ("速度(tok/s)", "avg_tokens_per_second", True)
    ]
    
    for metric_name, key, higher_is_better in metrics:
        vanilla_val = report['metrics']['vanilla'][key]
        pplm_val = report['metrics']['modified'][key]
        
        if key in ["avg_character_hit_rate", "avg_sentiment_match", "avg_diversity", "avg_tokens_per_second"]:
            vanilla_str = f"{vanilla_val:.4f}"
            pplm_str = f"{pplm_val:.4f}"
        else:
            vanilla_str = f"{vanilla_val:.2f}"
            pplm_str = f"{pplm_val:.2f}"
        
        if higher_is_better:
            improvement = ((pplm_val - vanilla_val) / vanilla_val * 100) if vanilla_val != 0 else 0
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        else:
            improvement = ((vanilla_val - pplm_val) / vanilla_val * 100) if vanilla_val != 0 else 0
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        
        print(f"{metric_name:<20} {vanilla_str:<15} {pplm_str:<15} {improvement_str:<15}")
    
    # 详细分析
    print(f"\n详细分析:")
    print(f"1. 词袋命中率: PPLM {'显著提升' if report['metrics']['modified']['avg_character_hit_rate'] > report['metrics']['vanilla']['avg_character_hit_rate'] + 0.1 else '略有提升' if report['metrics']['modified']['avg_character_hit_rate'] > report['metrics']['vanilla']['avg_character_hit_rate'] else '无明显提升'}")
    print(f"2. 情感匹配度: PPLM {'显著提升' if report['metrics']['modified']['avg_sentiment_match'] > report['metrics']['vanilla']['avg_sentiment_match'] + 0.1 else '略有提升' if report['metrics']['modified']['avg_sentiment_match'] > report['metrics']['vanilla']['avg_sentiment_match'] else '无明显提升'}")
    print(f"3. 生成速度: PPLM {'显著变慢' if report['metrics']['vanilla']['avg_tokens_per_second'] > report['metrics']['modified']['avg_tokens_per_second'] * 2 else '略有变慢' if report['metrics']['vanilla']['avg_tokens_per_second'] > report['metrics']['modified']['avg_tokens_per_second'] * 1.2 else '速度相当'}")
    print(f"4. 文本质量: PPLM {'质量更高' if report['metrics']['modified']['avg_diversity'] > report['metrics']['vanilla']['avg_diversity'] else '质量相当'}")

# 解析命令行参数
def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='PPLM模型对比实验 - 纯模型生成与修改后PPLM对比')
    parser.add_argument('--model', type=str, default=r'C:\Users\23921\.cache\modelscope\Qwen3-4B', help='模型名称或路径，默认为本地Qwen3-4B模型')
    parser.add_argument('--cped_dataset',
                        default=r'c:\Users\23921\PycharmProjects\pythonProject\PPLM\PPLM\train_split.csv',
                        type=str,
                        help='Path to CPED dataset file')
    parser.add_argument('--speakers', type=str, nargs='+', default=['江天昊'], help='要分析的说话人列表')
    parser.add_argument('--sentiment', type=str, default='positive', 
                       choices=['positive', 'negative', 'neutral'], help='目标情感')
    parser.add_argument('--num_samples', type=int, default=3, help='每个模型生成的样本数量')
    parser.add_argument('--num_tokens', type=int, default=40, help='生成的token数量')
    parser.add_argument('--output_dir', type=str, 
                       default=r'c:\Users\23921\PycharmProjects\pythonProject\PPLM\PPLM\comparison_results',
                       help='输出结果目录')
    parser.add_argument('--min_word_freq', type=int, default=1, help='词袋中词的最小频率阈值')
    parser.add_argument('--top_n_words', type=int, default=20, help='词袋中包含的高频词数量')
    return parser.parse_args()

# 主函数
def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建词袋管理器并加载数据集
    print("初始化词袋管理器并加载CPED数据集...")
    wordbag_manager = CharacterWordbagManager(args.cped_dataset)
    
    # 如果没有找到说话人，使用默认值
    if not wordbag_manager.dataset_speakers:
        print("警告: 无法从数据集中识别说话人，将使用默认角色")
        # 手动设置一些可能的角色
        wordbag_manager.dataset_speakers = ['林妙妙', '钱三一', '邓小棋', '江天昊', '小琪']
    
    # 显示可用的说话人
    print(f"\n数据集中可用的说话人: {wordbag_manager.get_available_speakers()[:10]}")
    
    # 对于每个指定的说话人，生成词袋并运行对比实验
    for speaker in args.speakers:
        print(f"\n===================================================")
        print(f"处理说话人: {speaker}")
        
        # 生成该说话人的词袋
        wordbag_manager.generate_wordbag_from_speaker(
            speaker, 
            min_word_freq=args.min_word_freq, 
            top_n_words=args.top_n_words
        )
        
        # 加载模型和分词器
        print("加载模型和分词器...")
        model, tokenizer, device = load_model_and_tokenizer(args.model)
        print(f"模型加载成功，运行设备: {device}")
        
        # 从CPED数据集提取提示词
        print("从CPED数据集提取提示词...")
        prompts = extract_prompts_from_cped(args.cped_dataset)
        
        # 如果提示词提取失败，使用默认提示词
        if prompts is None or len(prompts) == 0:
            print(f"使用默认提示词进行评估，角色: {speaker}")
            prompts = [
                f"{speaker}说：",
                f"{speaker}开心地说：",
                f"{speaker}思考了一下，然后说：",
                f"面对这个问题，{speaker}回答：",
                f"{speaker}笑着说："
            ]
        
        # 运行对比实验
        results = run_comparison_experiment(
            model=model,
            tokenizer=tokenizer,
            device=device,
            wordbag_manager=wordbag_manager,
            character=speaker,
            target_sentiment=args.sentiment,
            prompts=prompts,
            num_samples=args.num_samples,
            num_tokens=args.num_tokens
        )
        
        # 生成报告
        output_file = os.path.join(
            args.output_dir,
            f"comparison_{speaker}_{args.sentiment}.json"
        )
        generate_comparison_report(results, speaker, args.sentiment, output_file)
        
        # 保存详细结果
        detailed_file = os.path.join(
            args.output_dir,
            f"detailed_results_vanilla_vs_modified_{speaker}_{args.sentiment}.json"
        )
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {detailed_file}")
        print(f"===================================================")

if __name__ == "__main__":
    main()