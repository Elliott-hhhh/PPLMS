#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集人物常用词提取脚本
用于提取每个角色在对话中的常用词，生成人物特定的词袋文件
"""

import os
import csv
import jieba
import jieba.analyse
from collections import defaultdict, Counter
import json
import re

# 设置中文字符
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 情感标签映射
SENTIMENT_MAPPING = {
    'positive': 1,      # 积极
    'neutral': 0,       # 中性
    'negative': -1,     # 消极
    'negative-other': -1, # 其他消极情感
    'positive-other': 1  # 其他积极情感
}

# 情感词汇分类
EMOTION_WORDS_CATEGORIES = {
    'positive': ['高兴', '快乐', '喜欢', '感谢', '好', '棒', '赞', '精彩', '优秀', '不错', '满意', '成功'],
    'negative': ['难过', '生气', '失望', '糟糕', '差', '讨厌', '失败', '错误', '不行', '拒绝', '反对'],
    'neutral': ['今天', '明天', '时间', '地点', '问题', '建议', '情况', '需要', '可以', '应该', '可能']
}

# 停用词列表
STOPWORDS = {
    '的', '了', '和', '与', '或', '在', '是', '有', '我', '你', '他', '她', '它', '我们', '你们', '他们', 
    '这', '那', '这些', '那些', '个', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '也', '就', '都', '还', '但', '而', '却', '并', '又', '之', '以', '于', '为', '被', '把', '将',
    '从', '对', '对于', '关于', '在', '向', '朝', '往', '当', '到', '使', '让', '叫', '被', '由',
    '因为', '所以', '如果', '虽然', '但是', '然而', '而且', '并且', '或者', '要么', '只有', '只要',
    '只是', '不过', '其实', '实际上', '事实上', '的确', '确实', '真的', '假的', '可能', '也许',
    '大概', '大约', '应该', '应当', '必须', '需要', '可以', '能够', '会', '要', '想', '希望',
    '啊', '呀', '呢', '吗', '吧', '啦', '哦', '嗯', '哼', '哈', '嘿', '喂', '嗨',
    '，', '。', '！', '？', '；', '：', '、', '（', '）', '【', '】', '{', '}', '"', '\'', '“', '”',
    '‘', '’', '<', '>', '《', '》', '〈', '〉', '……', '—', '～', '——', ' ', '\t', '\n', '\r'
}

def is_valid_word(word):
    """判断是否为有效词"""
    # 过滤掉太短的词（通常是标点或单字无意义词）
    if len(word) <= 1:
        return False
    # 过滤掉停用词
    if word in STOPWORDS:
        return False
    # 过滤掉纯数字或特殊字符
    if re.match(r'^[\d\W]+$', word):
        return False
    return True

def clean_text(text):
    """清理文本"""
    # 移除多余的空格和特殊字符
    text = re.sub(r'\s+', ' ', text)
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、（）【】"\'“”‘’]', ' ', text)
    return text.strip()

def extract_character_words(csv_file_path):
    """
    从CSV文件中提取每个角色的常用词
    :param csv_file_path: CSV文件路径
    :return: 包含每个人物词汇统计的字典
    """
    # 初始化数据结构
    character_data = defaultdict(lambda: {
        'words': [],         # 所有词的列表
        'emotions': [],      # 对应的情感标签
        'word_counts': Counter(),  # 词频统计
        'sentiment_counts': Counter()  # 情感标签统计
    })
    
    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 获取必要字段
            speaker = row.get('Speaker', '').strip()
            utterance = row.get('Utterance', '').strip()
            sentiment = row.get('Sentiment', '').strip()
            emotion = row.get('Emotion', '').strip()
            
            # 跳过无效数据
            if not speaker or not utterance:
                continue
            
            # 清理文本
            clean_utterance = clean_text(utterance)
            if not clean_utterance:
                continue
            
            # 使用jieba分词
            words = jieba.lcut(clean_utterance)
            
            # 过滤有效词
            valid_words = [word for word in words if is_valid_word(word)]
            
            # 更新数据
            if valid_words:
                character_data[speaker]['words'].extend(valid_words)
                character_data[speaker]['word_counts'].update(valid_words)
                
                # 更新情感信息
                sentiment_value = SENTIMENT_MAPPING.get(sentiment.lower(), 0)
                character_data[speaker]['emotions'].append(sentiment_value)
                character_data[speaker]['sentiment_counts'][sentiment_value] += 1
    
    return character_data

def generate_character_wordbags(character_data, output_dir, min_freq=3):
    """
    为每个角色生成词袋文件
    :param character_data: 包含角色词频的字典
    :param output_dir: 输出目录
    :param min_freq: 最小词频阈值
    :return: 生成的词袋文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    # 为每个角色生成词袋文件
    for speaker, data in character_data.items():
        # 过滤低频词
        filtered_words = [(word, count) for word, count in data['word_counts'].items() if count >= min_freq]
        
        # 按词频降序排序
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        if filtered_words:
            # 生成词袋文件
            wordbag_filename = os.path.join(output_dir, f'character_{speaker}.txt')
            with open(wordbag_filename, 'w', encoding='utf-8') as f:
                for word, count in filtered_words:
                    f.write(f"{word}\n")
            
            output_files.append(wordbag_filename)
            print(f"生成角色 {speaker} 的词袋文件：{wordbag_filename}，包含 {len(filtered_words)} 个词")
    
    return output_files

def generate_emotion_wordbags(character_data, output_dir, min_freq=3):
    """
    生成情感相关的词袋
    :param character_data: 包含角色词频的字典
    :param output_dir: 输出目录
    :param min_freq: 最小词频阈值
    """
    # 初始化情感词汇统计
    emotion_words = defaultdict(Counter)
    
    # 为每个角色收集情感相关词汇
    for speaker, data in character_data.items():
        words = data['words']
        emotions = data['emotions']
        
        # 确保词和情感标签数量一致
        for i, word in enumerate(words):
            if i < len(emotions):
                sentiment = emotions[i]
                emotion_words[sentiment][word] += 1
    
    # 生成不同情感的词袋文件
    for sentiment_value, word_counts in emotion_words.items():
        # 过滤低频词
        filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        if filtered_words:
            # 确定情感标签
            if sentiment_value == 1:
                sentiment_label = 'positive'
            elif sentiment_value == -1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # 生成词袋文件
            wordbag_filename = os.path.join(output_dir, f'sentiment_{sentiment_label}.txt')
            with open(wordbag_filename, 'w', encoding='utf-8') as f:
                for word, count in filtered_words:
                    f.write(f"{word}\n")
            
            print(f"生成情感词袋 {sentiment_label}：{wordbag_filename}，包含 {len(filtered_words)} 个词")

def generate_character_info(character_data, output_file):
    """
    生成角色信息文件
    :param character_data: 包含角色数据的字典
    :param output_file: 输出文件路径
    """
    character_info = {}
    
    for speaker, data in character_data.items():
        # 计算平均情感倾向
        total_emotion = sum(data['emotions'])
        avg_emotion = total_emotion / len(data['emotions']) if data['emotions'] else 0
        
        # 获取最常使用的50个词
        top_words = [(word, count) for word, count in data['word_counts'].most_common(50)]
        
        character_info[speaker] = {
            'total_utterances': len(data['emotions']),
            'avg_sentiment': avg_emotion,
            'sentiment_distribution': dict(data['sentiment_counts']),
            'top_words': top_words
        }
    
    # 保存角色信息
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(character_info, f, ensure_ascii=False, indent=2)
    
    print(f"生成角色信息文件：{output_file}")

def main():
    # 数据集路径
    csv_file_path = 'c:\\Users\\23921\\PycharmProjects\\pythonProject\\PPLM\\PPLM\\train_split.csv'
    
    # 输出目录
    output_dir = 'c:\\Users\\23921\\PycharmProjects\\pythonProject\\PPLM\\PPLM\\character_wordbags'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理数据集：{csv_file_path}")
    
    # 提取角色常用词
    print("提取每个角色的常用词...")
    character_data = extract_character_words(csv_file_path)
    
    print(f"共提取到 {len(character_data)} 个角色的数据")
    
    # 生成角色词袋
    print("生成角色特定的词袋文件...")
    character_wordbags = generate_character_wordbags(character_data, output_dir, min_freq=3)
    
    # 生成情感词袋
    print("生成情感相关的词袋文件...")
    generate_emotion_wordbags(character_data, output_dir, min_freq=3)
    
    # 生成角色信息文件
    character_info_file = os.path.join(output_dir, 'character_info.json')
    generate_character_info(character_data, character_info_file)
    
    print("所有词袋文件生成完成！")

if __name__ == "__main__":
    main()