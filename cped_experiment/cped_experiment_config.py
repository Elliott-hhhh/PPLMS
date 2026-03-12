#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集对比实验配置文件
定义实验参数、模型配置和评估指标
"""

import os
import json
import numpy as np
import torch

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_cped")
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_results")
BOW_DIR = os.path.join(DATA_DIR, "bow_files")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集配置
DATASET_CONFIG = {
    "name": "CPED情感对话数据集",
    "description": "中文情感对话数据集，包含日常对话内容和对应的情感极性与类别标签",
    "eval_file": os.path.join(DATA_DIR, "pplm_eval_data.tsv"),
    "dataset_info": os.path.join(DATA_DIR, "dataset_info.json")
}

# 模型配置 - 原始PPLM
ORIGINAL_PPLM_CONFIG = {
    "base_model": "gpt2-xl",  # 使用GPT2-XL模型
    "tokenizer": "gpt2-xl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "num_samples": 5,  # 每个提示生成的样本数
    "length": 50,  # 生成文本长度
    "stop_token": "",
    
    # 词袋控制参数
    "bow_scale": 0.9,  # 词袋控制强度，原始值
    "kl_scale": 0.05,  # KL散度权重，原始值
    "stepsize": 0.02,  # 每步PPLM梯度步长，原始值
    "num_iterations": 3,  # 每步PPLM迭代次数，原始值
    "gamma": 1.5,  # 控制词袋影响的参数，原始值
    "gm_scale": 0.9,  # 生成模型权重，原始值
    "window_length": 5,  # 上下文窗口长度，原始值
    "weight_decay": 0.01  # 权重衰减，原始值
}

# 模型配置 - 修改版PPLM (增强的中文情感控制)
MODIFIED_PPLM_CONFIG = {
    "base_model": "gpt2-xl",
    "tokenizer": "gpt2-xl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "num_samples": 5,
    "length": 50,
    "stop_token": "",
    
    # 修改后的词袋控制参数 (针对中文优化)
    "bow_scale": 1.5,  # 增加词袋控制强度，更好地引导情感
    "kl_scale": 0.1,  # 增加KL散度权重，平衡生成质量
    "stepsize": 0.03,  # 增加步长，加速情感引导
    "num_iterations": 4,  # 增加迭代次数，增强控制效果
    "gamma": 2.0,  # 增加gamma，强化词袋影响
    "gm_scale": 0.8,  # 略微降低生成模型权重
    "window_length": 7,  # 增加上下文窗口，更好地理解长句子
    "weight_decay": 0.02  # 略微增加权重衰减
}

# 生成参数 - 通用配置
GENERATION_CONFIG = {
    "prompt_templates": [
        "今天的天气很",  # 中性开头，测试情感控制
        "我觉得这个问题",  # 中性开头，测试情感控制
        "你好，",  # 问候开头
        "谢谢你",  # 致谢开头
        "这个想法",  # 评价开头
        "我感到",  # 情感表达开头
        "我想",  # 思考开头
        "虽然",  # 转折开头
        "因为",  # 原因开头
        "但是"   # 转折开头
    ],
    "max_prompts": 50,  # 限制使用的提示数量，避免实验时间过长
    "batch_size": 5  # 批处理大小
}

# 评估指标配置
EVALUATION_CONFIG = {
    "metrics": [
        "perplexity",      # 困惑度
        "bleu",           # BLEU分数
        "rouge",          # ROUGE分数
        "diversity",      # 多样性指标
        "sentiment_accuracy"  # 情感准确率（使用数据集自带标签）
    ],
    "diversity_window": 5,  # 多样性计算窗口大小
    "diversity_topk": 20    # 多样性计算中的topk
}

# 对比实验配置
EXPERIMENTS = [
    # 1. 基线模型 - 无情感控制
    {
        "name": "vanilla_gpt2",
        "description": "基线模型，无情感控制",
        "model_type": "vanilla",
        "config": {
            "base_model": MODIFIED_PPLM_CONFIG["base_model"],
            "device": MODIFIED_PPLM_CONFIG["device"],
            "temperature": MODIFIED_PPLM_CONFIG["temperature"],
            "top_k": MODIFIED_PPLM_CONFIG["top_k"],
            "top_p": MODIFIED_PPLM_CONFIG["top_p"],
            "length": MODIFIED_PPLM_CONFIG["length"],
            "num_samples": MODIFIED_PPLM_CONFIG["num_samples"]
        },
        "priority": "high"
    },
    
    # 2. 原始PPLM - 积极情感词袋控制
    {
        "name": "original_pplm_positive",
        "description": "原始PPLM配置，使用积极情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_positive.txt"),
        "config": {
            **ORIGINAL_PPLM_CONFIG,
            "class_label": "positive"
        },
        "priority": "high"
    },
    
    # 3. 修改版PPLM - 积极情感词袋控制
    {
        "name": "modified_pplm_positive",
        "description": "修改后的PPLM配置，使用积极情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_positive.txt"),
        "config": {
            **MODIFIED_PPLM_CONFIG,
            "class_label": "positive"
        },
        "priority": "high"
    },
    
    # 4. 原始PPLM - 消极情感词袋控制
    {
        "name": "original_pplm_negative",
        "description": "原始PPLM配置，使用消极情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_negative.txt"),
        "config": {
            **ORIGINAL_PPLM_CONFIG,
            "class_label": "negative"
        },
        "priority": "high"
    },
    
    # 5. 修改版PPLM - 消极情感词袋控制
    {
        "name": "modified_pplm_negative",
        "description": "修改后的PPLM配置，使用消极情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_negative.txt"),
        "config": {
            **MODIFIED_PPLM_CONFIG,
            "class_label": "negative"
        },
        "priority": "high"
    },
    
    # 6. 原始PPLM - 中性情感词袋控制
    {
        "name": "original_pplm_neutral",
        "description": "原始PPLM配置，使用中性情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_neutral.txt"),
        "config": {
            **ORIGINAL_PPLM_CONFIG,
            "class_label": "neutral"
        },
        "priority": "medium"
    },
    
    # 7. 修改版PPLM - 中性情感词袋控制
    {
        "name": "modified_pplm_neutral",
        "description": "修改后的PPLM配置，使用中性情感词袋控制",
        "model_type": "pplm_bow",
        "bow_file": os.path.join(BOW_DIR, "sentiment_neutral.txt"),
        "config": {
            **MODIFIED_PPLM_CONFIG,
            "class_label": "neutral"
        },
        "priority": "medium"
    }
]

# 实验条件配置
EXPERIMENT_CONDITIONS = {
    "run_in_parallel": False,  # 是否并行运行实验
    "max_workers": 4,  # 并行工作线程数
    "timeout": 3600,  # 单个实验超时时间（秒）
    "max_experiment_duration": 86400,  # 总实验超时时间（秒）
    "early_stopping": True,  # 是否启用早停
    "save_checkpoints": True,  # 是否保存检查点
    "log_interval": 10,  # 日志记录间隔
    "use_cached_results": False  # 是否使用缓存结果
}

# 加载数据集信息

def load_dataset_info():
    """加载数据集统计信息"""
    if os.path.exists(DATASET_CONFIG["dataset_info"]):
        with open(DATASET_CONFIG["dataset_info"], 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 评估指标计算辅助函数

def calculate_bleu(references, candidates, max_n=4):
    """计算BLEU分数"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method4
        
        bleu_scores = []
        for ref, cand in zip(references, candidates):
            # 中文分词
            ref_tokens = list(jieba.cut(ref))
            cand_tokens = list(jieba.cut(cand))
            
            if len(cand_tokens) < max_n:
                # 对于短句子，适当降低max_n
                current_max_n = len(cand_tokens)
                weights = tuple((1.0/current_max_n for _ in range(current_max_n)))
            else:
                weights = tuple((1.0/max_n for _ in range(max_n)))
            
            score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothie)
            bleu_scores.append(score)
        
        return {
            f"bleu-{i+1}": np.mean([sentence_bleu([list(jieba.cut(r))], list(jieba.cut(c)), 
                                                 weights=tuple(1.0 if j==i else 0.0 for j in range(4))) 
                                   for r, c in zip(references, candidates)])
            for i in range(max_n)
        }
    except Exception as e:
        print(f"计算BLEU分数时出错: {str(e)}")
        return {}

def calculate_rouge(references, candidates):
    """计算ROUGE分数"""
    try:
        from rouge import Rouge
        # 中文分词并连接
        ref_tokens = [' '.join(list(jieba.cut(ref))) for ref in references]
        cand_tokens = [' '.join(list(jieba.cut(cand))) for cand in candidates]
        
        rouge = Rouge()
        scores = rouge.get_scores(cand_tokens, ref_tokens, avg=True)
        return scores
    except Exception as e:
        print(f"计算ROUGE分数时出错: {str(e)}")
        return {}

def calculate_diversity(texts, window_size=5, topk=20):
    """计算文本多样性指标"""
    try:
        from collections import Counter
        import jieba
        
        all_tokens = []
        for text in texts:
            tokens = list(jieba.cut(text))
            all_tokens.extend(tokens)
        
        # 计算词汇多样性 (Type-Token Ratio)
        if len(all_tokens) == 0:
            return {"ttr": 0.0, "topk_ratio": 0.0}
        
        ttr = len(set(all_tokens)) / len(all_tokens)
        
        # 计算topk词比例
        token_counts = Counter(all_tokens)
        topk_tokens_count = sum(count for _, count in token_counts.most_common(topk))
        topk_ratio = topk_tokens_count / len(all_tokens)
        
        return {
            "ttr": ttr,
            "topk_ratio": topk_ratio
        }
    except Exception as e:
        print(f"计算多样性指标时出错: {str(e)}")
        return {}

# 检查配置有效性
def validate_config():
    """验证配置有效性"""
    # 检查必要目录是否存在
    assert os.path.exists(DATA_DIR), f"数据集目录不存在: {DATA_DIR}"
    assert os.path.exists(BOW_DIR), f"词袋文件目录不存在: {BOW_DIR}"
    
    # 检查情感词袋文件是否存在
    sentiment_files = ["sentiment_positive.txt", "sentiment_negative.txt", "sentiment_neutral.txt"]
    for file in sentiment_files:
        file_path = os.path.join(BOW_DIR, file)
        if not os.path.exists(file_path):
            print(f"警告: 情感词袋文件不存在: {file_path}")
    
    print("配置验证完成")

# 确保导入
import torch
import jieba

# 如果是主程序运行，验证配置
if __name__ == "__main__":
    validate_config()
    print(f"\nCPED实验配置信息:")
    print(f"数据集名称: {DATASET_CONFIG['name']}")
    print(f"实验数量: {len(EXPERIMENTS)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"可用模型配置: 原始PPLM和修改版PPLM")
    print(f"评估指标: {', '.join(EVALUATION_CONFIG['metrics'])}")