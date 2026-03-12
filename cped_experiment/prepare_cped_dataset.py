#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集预处理脚本
用于处理train_split.csv格式的情感对话数据集
"""

import os
import json
import csv
import argparse
from collections import Counter
import jieba
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CPED对话数据集预处理脚本")
    parser.add_argument("--input_file", type=str, default="./train_split.csv",
                        help="CPED数据集输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./processed_cped",
                        help="处理后数据输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    return parser.parse_args()

def load_cped_dataset(input_file):
    """
    加载CPED对话数据集（CSV格式）
    提取Utterance文本和Sentiment/Emotion标签
    """
    data = []
    print(f"正在加载CPED数据集: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 提取关键信息：文本和情感标签
            if 'Utterance' in row and 'Sentiment' in row and 'Emotion' in row:
                data.append({
                    'text': row['Utterance'],
                    'sentiment': row['Sentiment'],  # 情感极性（positive/neutral/negative）
                    'emotion': row['Emotion'],      # 具体情感类别
                    'speaker': row.get('Speaker', ''),
                    'gender': row.get('Gender', ''),
                    'age': row.get('Age', ''),
                    'da': row.get('DA', '')  # 对话行为
                })
    
    print(f"加载了 {len(data)} 个样本")
    return data

def preprocess_text(text):
    """
    文本预处理
    1. 去除多余空白字符
    """
    # 去除多余空白字符
    text = ' '.join(text.strip().split())
    return text

def extract_emotion_labels(data):
    """
    提取情感标签统计信息
    返回sentiment和emotion两个维度的统计
    """
    sentiment_counter = Counter()
    emotion_counter = Counter()
    
    for item in data:
        if 'sentiment' in item and item['sentiment']:
            sentiment_counter[item['sentiment']] += 1
        if 'emotion' in item and item['emotion']:
            emotion_counter[item['emotion']] += 1
    
    print("情感极性分布:")
    for sentiment, count in sentiment_counter.most_common():
        print(f"  {sentiment}: {count}")
    
    print("\n具体情感类别分布:")
    for emotion, count in emotion_counter.most_common():
        print(f"  {emotion}: {count}")
    
    return sentiment_counter, emotion_counter

def convert_to_pplm_format(data, output_dir, sentiment_counter, emotion_counter):
    """
    转换数据为PPLM可用的格式
    1. 创建词袋文件（基于sentiment和emotion）
    2. 创建用于评估的数据集文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整数据集信息
    dataset_info = {
        'total_samples': len(data),
        'sentiment_distribution': dict(sentiment_counter),
        'emotion_distribution': dict(emotion_counter)
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 为sentiment创建词袋文件（positive/negative/neutral）
    sentiment_groups = {sentiment: [] for sentiment in sentiment_counter.keys()}
    for item in data:
        if 'sentiment' in item and item['sentiment'] in sentiment_groups and 'text' in item:
            # 使用jieba分词提取关键词
            words = jieba.cut(preprocess_text(item['text']))
            sentiment_groups[item['sentiment']].extend(words)
    
    # 为emotion创建词袋文件
    emotion_groups = {emotion: [] for emotion in emotion_counter.keys()}
    for item in data:
        if 'emotion' in item and item['emotion'] in emotion_groups and 'text' in item:
            words = jieba.cut(preprocess_text(item['text']))
            emotion_groups[item['emotion']].extend(words)
    
    # 创建词袋目录
    bow_dir = os.path.join(output_dir, "bow_files")
    os.makedirs(bow_dir, exist_ok=True)
    
    # 保存sentiment词袋
    print("\n生成情感极性词袋文件:")
    for sentiment, words in sentiment_groups.items():
        # 过滤空词袋
        if not words:
            continue
            
        # 统计词频
        word_counter = Counter(words)
        # 过滤掉停用词和低频词
        filtered_words = [word for word, count in word_counter.most_common(500) 
                         if len(word) > 1 and count > 1]
        
        # 保存词袋文件
        bow_file = os.path.join(bow_dir, f"sentiment_{sentiment}.txt")
        with open(bow_file, 'w', encoding='utf-8') as f:
            for word in filtered_words:
                f.write(f"{word}\n")
        
        print(f"  情感极性 '{sentiment}' 词袋已保存，包含 {len(filtered_words)} 个词: {bow_file}")
    
    # 保存emotion词袋
    print("\n生成具体情感类别词袋文件:")
    for emotion, words in emotion_groups.items():
        # 过滤空词袋
        if not words:
            continue
            
        # 统计词频
        word_counter = Counter(words)
        # 过滤掉停用词和低频词
        filtered_words = [word for word, count in word_counter.most_common(300) 
                         if len(word) > 1 and count > 1]
        
        # 保存词袋文件
        bow_file = os.path.join(bow_dir, f"emotion_{emotion}.txt")
        with open(bow_file, 'w', encoding='utf-8') as f:
            for word in filtered_words:
                f.write(f"{word}\n")
        
        print(f"  情感类别 '{emotion}' 词袋已保存，包含 {len(filtered_words)} 个词: {bow_file}")

def split_train_test(data, output_dir, train_ratio=0.8):
    """划分训练集和测试集"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    import random
    random.shuffle(data)
    
    split_point = int(len(data) * train_ratio)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    # 保存训练集和测试集
    train_path = os.path.join(output_dir, "cped_train.json")
    test_path = os.path.join(output_dir, "cped_test.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练集已保存: {train_path} ({len(train_data)} 样本)")
    print(f"测试集已保存: {test_path} ({len(test_data)} 样本)")
    
    return train_data, test_data

def create_pplm_eval_file(data, output_dir):
    """创建PPLM评估用的文件格式"""
    eval_file_path = os.path.join(output_dir, "pplm_eval_data.tsv")
    
    with open(eval_file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['text', 'sentiment', 'emotion'])
        
        # 写入数据
        for item in data:
            if 'text' in item and 'sentiment' in item and 'emotion' in item:
                writer.writerow([
                    preprocess_text(item['text']),
                    item['sentiment'],
                    item['emotion']
                ])
    
    print(f"\nPPLM评估数据已保存: {eval_file_path} ({len(data)} 样本)")

def main():
    """主函数"""
    args = parse_args()
    
    # 加载数据集
    data = load_cped_dataset(args.input_file)
    
    # 限制样本数量（如果指定）
    if args.max_samples and len(data) > args.max_samples:
        data = data[:args.max_samples]
        print(f"限制样本数量为: {args.max_samples}")
    
    # 提取情感标签统计
    sentiment_counter, emotion_counter = extract_emotion_labels(data)
    
    # 划分训练集和测试集
    train_data, test_data = split_train_test(data, args.output_dir, args.train_ratio)
    
    # 创建PPLM评估文件
    create_pplm_eval_file(data, args.output_dir)
    
    # 转换为PPLM词袋格式
    convert_to_pplm_format(data, args.output_dir, sentiment_counter, emotion_counter)
    
    print("\n预处理完成！")
    print(f"处理后数据目录: {args.output_dir}")
    print("包含以下文件:")
    print("  1. dataset_info.json - 数据集统计信息")
    print("  2. cped_train.json - 训练集")
    print("  3. cped_test.json - 测试集")
    print("  4. pplm_eval_data.tsv - PPLM评估数据")
    print("  5. bow_files/ - 情感词袋文件（sentiment_*.txt 和 emotion_*.txt）")

if __name__ == "__main__":
    main()