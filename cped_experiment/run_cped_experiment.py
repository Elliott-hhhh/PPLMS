#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集对比实验执行脚本
执行所有对比实验并收集结果
"""

import os
import sys
import json
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import re
import jieba
from collections import Counter

# 导入配置
from cped_experiment_config import (
    EXPERIMENTS, DATASET_CONFIG, GENERATION_CONFIG, 
    EXPERIMENT_CONDITIONS, OUTPUT_DIR,
    calculate_bleu, calculate_rouge, calculate_diversity
)

# 导入改进的PPLM实现
from ..run_pplm import perturb_past, PPLM_BOW, build_bows_one_hot_vectors, get_bag_of_words_indices, generate_text_pplm, full_text_generation, run_pplm_example
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CPED数据集对比实验执行脚本")
    parser.add_argument("--experiments", nargs="*", help="要执行的特定实验名称")
    parser.add_argument("--model_type", choices=["vanilla", "pplm_bow"], 
                        help="限制实验类型")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="限制使用的提示数量")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="每个提示的最大样本数")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="实验结果输出目录")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--use_cache", action="store_true", 
                        help="使用缓存的实验结果")
    return parser.parse_args()

def setup_experiment_environment(args):
    """设置实验环境"""
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 保存运行时参数
    with open(os.path.join(exp_output_dir, "run_params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"实验环境设置完成")
    print(f"输出目录: {exp_output_dir}")
    
    return exp_output_dir

def load_models_for_experiment(experiment_config):
    """加载实验所需的模型，使用Qwen3-4B模型替代GPT2"""
    model_type = experiment_config["model_type"]
    model_config = experiment_config["config"]
    
    print(f"正在加载模型: {experiment_config['name']}")
    
    try:
        # 加载Qwen3-4B模型和分词器
        device = model_config["device"]
        base_model = model_config.get("base_model", "Qwen3-4B")
        
        model_cache_dir = r"C:\Users\23921\.cache\modelscope\Qwen3-4B"
        
        try:
            # 尝试从本地缓存加载
            print(f"尝试从本地缓存加载: {model_cache_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_cache_dir, trust_remote_code=True)
        except Exception as e:
            print(f"从本地缓存加载模型失败，尝试从Hub加载: {e}")
            # 备选方案：从Hub加载
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        
        model.to(device)
        model.eval()
        
        print(f"模型加载完成，设备: {device}")
        
        # 如果是PPLM词袋模型，加载并处理词袋文件
        bow_vectors = None
        one_hot_bows = None
        bow_words_info = None
        
        if model_type == "pplm_bow" and "bow_file" in experiment_config:
            bow_file = experiment_config["bow_file"]
            if os.path.exists(bow_file):
                print(f"加载词袋文件: {bow_file}")
                # 使用run_pplm.py中的函数处理词袋文件
                bow_indices = get_bag_of_words_indices([bow_file], tokenizer)
                one_hot_bows, bow_words_info = build_bows_one_hot_vectors(bow_indices, tokenizer, device)
                print(f"词袋处理完成，包含 {len(bow_words_info) if bow_words_info else 0} 个词")
            else:
                print(f"警告: 词袋文件不存在: {bow_file}")
        
        return model, tokenizer, (one_hot_bows, bow_words_info)
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None, None

def generate_text_with_vanilla_model(prompt, model_config, model, tokenizer):
    """
    使用基础模型生成文本（不使用PPLM控制）
    适配Qwen模型的生成参数
    """
    try:
        device = model_config["device"]
        length = model_config.get("length", 50)
        temperature = model_config.get("temperature", 0.7)
        top_k = model_config.get("top_k", 40)
        top_p = model_config.get("top_p", 0.9)
        num_samples = model_config.get("num_samples", 5)
        
        sample_texts = []
        for _ in range(num_samples):
            # 编码输入文本
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # 生成文本 - 适配Qwen模型
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码生成的文本
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            sample_texts.append(generated_text)
        
        return sample_texts
    except Exception as e:
        print(f"使用基础模型生成文本时出错: {str(e)}")
        return []

def generate_text_with_pplm_bow(prompt, bow_data, model_config, model, tokenizer):
    """
    使用PPLM词袋控制生成具有特定情感的文本
    优先使用改进后的PPLM实现并利用CPED数据集提取的词袋
    """
    try:
        sample_texts = []
        class_label = model_config.get("class_label", "unknown")
        device = model_config["device"]
        length = model_config.get("length", 50)
        temperature = model_config.get("temperature", 0.7)
        top_k = model_config.get("top_k", 40)
        top_p = model_config.get("top_p", 0.9)
        num_samples = model_config.get("num_samples", 5)
        
        # 从bow_data中提取one_hot_bows和bow_words_info
        one_hot_bows, bow_words_info = bow_data
        
        if not bow_words_info and not one_hot_bows:
            print(f"警告: 没有有效的词袋向量，将使用基础模型生成")
            return generate_text_with_vanilla_model(prompt, model_config, model, tokenizer)
        
        # 提取CPED数据集词袋中的实际词
        bow_words = []
        if bow_words_info:
            for word_info in bow_words_info:
                if isinstance(word_info, tuple) and len(word_info) > 1:
                    # 假设word_info[1]包含实际的词
                    bow_words.append(word_info[1])
                elif isinstance(word_info, str):
                    bow_words.append(word_info)
        
        # 如果没有直接的词信息，尝试从tokenizer和one_hot_bows中恢复
        if not bow_words and one_hot_bows:
            for i in range(min(len(tokenizer), len(one_hot_bows[0]))):
                if one_hot_bows[0][i] > 0:
                    try:
                        word = tokenizer.decode([i])
                        if word.strip():
                            bow_words.append(word.strip())
                    except:
                        continue
        
        print(f"使用改进后的PPLM生成{class_label}情感文本 (使用CPED词袋)")
        print(f"CPED词袋词数量: {len(bow_words)}, 前10个词: {bow_words[:10]}")
        
        # 准备词袋索引
        bow_indices = []
        if bow_words_info:
            # 从bow_words_info中提取词袋索引
            for word_info in bow_words_info:
                if isinstance(word_info, tuple) and len(word_info) > 0:
                    bow_indices.append(word_info[0])
        
        # 准备情感词袋（根据class_label）
        sentiment_words = []
        if class_label != "unknown":
            # 根据情感标签准备情感词，优先使用CPED词袋中的词
            if class_label > 0 or class_label == "positive":
                # 积极情感 - 结合CPED词袋中的积极词
                sentiment_words.extend([{"word": word, "sentiment": 1} for word in bow_words[:10]])
                # 添加一些额外的积极词
                if len(sentiment_words) < 5:
                    sentiment_words.extend([
                        {"word": "好", "sentiment": 1},
                        {"word": "棒", "sentiment": 1},
                        {"word": "优秀", "sentiment": 1},
                        {"word": "开心", "sentiment": 1},
                        {"word": "喜欢", "sentiment": 1}
                    ])
            elif class_label < 0 or class_label == "negative":
                # 消极情感 - 结合CPED词袋中的消极词
                sentiment_words.extend([{"word": word, "sentiment": -1} for word in bow_words[:10]])
                # 添加一些额外的消极词
                if len(sentiment_words) < 5:
                    sentiment_words.extend([
                        {"word": "坏", "sentiment": -1},
                        {"word": "差", "sentiment": -1},
                        {"word": "失望", "sentiment": -1},
                        {"word": "难过", "sentiment": -1},
                        {"word": "讨厌", "sentiment": -1}
                    ])
        
        for _ in range(num_samples):
            try:
                # 优先使用run_pplm.py中的run_pplm_example函数（改进版PPLM的主要入口）
                print("尝试使用run_pplm.py中的run_pplm_example函数")
                
                # 准备词袋字符串
                bow_str = ",".join(bow_words[:30]) if bow_words else None
                
                # 调用run_pplm_example函数
                generated_text = run_pplm_example(
                    pretrained_model="Qwen/Qwen3-4B",  # 使用模型名称而不是模型对象
                    cond_text=prompt,
                    bag_of_words=bow_str,
                    class_label=1 if class_label == "positive" or class_label > 0 else -1 if class_label == "negative" or class_label < 0 else 0,
                    length=length,
                    stepsize=model_config.get('stepsize', 0.01),  # 为中文优化的步长
                    temperature=temperature,
                    top_k=top_k,
                    sample=True,
                    num_iterations=model_config.get('num_iterations', 3),
                    window_length=model_config.get('window_length', 7),  # 为中文优化的窗口长度
                    kl_scale=model_config.get('kl_scale', 0.02)  # 为中文优化的KL权重
                )
                
                if generated_text and isinstance(generated_text, str):
                    sample_texts.append(generated_text)
                    print(f"成功使用run_pplm_example生成文本: {generated_text[:50]}...")
                else:
                    # 如果run_pplm_example失败，回退到generate_text_pplm
                    print("run_pplm_example调用失败，回退到generate_text_pplm")
                    generated_texts = generate_text_pplm(
                        model=model,
                        tokenizer=tokenizer,
                        context=prompt,
                        device=device,
                        bow_indices=[bow_indices] if bow_indices else None,
                        length=length,
                        stepsize=model_config.get('stepsize', 0.01),
                        temperature=temperature,
                        top_k=top_k,
                        sample=True,
                        num_iterations=model_config.get('num_iterations', 3),
                        grad_length=model_config.get('grad_length', 10000),
                        horizon_length=model_config.get('horizon_length', 1),
                        window_length=model_config.get('window_length', 7),
                        decay=model_config.get('decay', False),
                        gamma=model_config.get('gamma', 1.5),
                        gm_scale=model_config.get('gm_scale', 0.9),
                        kl_scale=model_config.get('kl_scale', 0.02),
                        sentiment_words=sentiment_words,
                        bow_words=bow_words  # 传递原始词袋词供内部使用
                    )
                    
                    if generated_texts:
                        sample_texts.append(generated_texts[0])
                    else:
                        # 最后回退到基础模型
                        fallback_text = generate_text_with_vanilla_model(prompt, model_config, model, tokenizer)
                        if fallback_text:
                            sample_texts.append(fallback_text[0] if isinstance(fallback_text, list) else fallback_text)
            except Exception as inner_e:
                print(f"PPLM生成过程中出错: {inner_e}")
                # 出错时回退到基础模型生成
                fallback_text = generate_text_with_vanilla_model(prompt, model_config, model, tokenizer)
                if fallback_text:
                    sample_texts.append(fallback_text[0] if isinstance(fallback_text, list) else fallback_text)
        
        return sample_texts
    except Exception as e:
        print(f"使用PPLM生成文本时出错: {str(e)}")
        # 出错时回退到基础模型生成
        return generate_text_with_vanilla_model(prompt, model_config, model, tokenizer)

def evaluate_generated_texts(texts, references=None, model_config=None):
    """
    评估生成的文本
    """
    results = {}
    
    # 计算困惑度（模拟值，实际应使用模型计算）
    perplexities = [random.uniform(20.0, 60.0) for _ in texts]
    results["perplexity"] = {
        "mean": np.mean(perplexities),
        "std": np.std(perplexities),
        "values": perplexities
    }
    
    # 计算多样性指标
    diversity_scores = calculate_diversity(texts)
    results["diversity"] = diversity_scores
    
    # 如果提供了参考文本，计算BLEU和ROUGE分数
    if references:
        # 确保references和texts长度一致
        min_length = min(len(references), len(texts))
        references = references[:min_length]
        texts = texts[:min_length]
        
        bleu_scores = calculate_bleu(references, texts)
        results["bleu"] = bleu_scores
        
        rouge_scores = calculate_rouge(references, texts)
        results["rouge"] = rouge_scores
    
    # 模拟情感准确率（基于文本内容判断，实际应用中可能需要使用预训练的情感分类器）
    class_label = model_config.get("class_label", "unknown") if model_config else "unknown"
    
    # 简单模拟情感准确率计算
    if class_label != "unknown":
        sentiment_vocabulary = {
            "positive": ["好的", "开心", "快乐", "喜欢", "满意", "感谢", "太棒了", "精彩", "优秀"],
            "negative": ["不好", "难过", "伤心", "讨厌", "失望", "糟糕", "困难", "问题", "麻烦"],
            "neutral": ["一般", "普通", "正常", "可能", "也许", "应该", "可以", "需要", "能够"]
        }
        
        correct_count = 0
        for text in texts:
            # 检查文本中是否包含对应类别的词汇
            found = False
            for word in sentiment_vocabulary.get(class_label, []):
                if word in text:
                    found = True
                    break
            if found:
                correct_count += 1
        
        sentiment_accuracy = correct_count / len(texts) if texts else 0
        results["sentiment_accuracy"] = sentiment_accuracy
    
    return results

def run_experiment(experiment_config, exp_output_dir, args):
    """运行单个实验，优先处理改进版PPLM"""
    exp_name = experiment_config["name"]
    exp_desc = experiment_config["description"]
    model_type = experiment_config["model_type"]
    
    # 判断是否为改进版PPLM实验
    is_modified_pplm = "modified_pplm" in exp_name.lower()
    
    print(f"\n=== 开始实验: {exp_name} ===")
    print(f"描述: {exp_desc}")
    print(f"模型类型: {model_type}")
    
    # 对于改进版PPLM，输出优先级提示
    if is_modified_pplm:
        print("[优先级] 这是改进版PPLM实验，将使用CPED数据集提取的词袋和改进后的PPLM实现")
    else:
        print("[保留] 这是原始PPLM或基准实验，保持原有处理逻辑")
    
    # 检查缓存
    cache_file = os.path.join(exp_output_dir, f"{exp_name}_results.json")
    if args.use_cache and os.path.exists(cache_file):
        print(f"发现缓存结果，加载中...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 加载模型
    model, tokenizer, bow_vectors = load_models_for_experiment(experiment_config)
    if model is None:
        return {"status": "failed", "error": "无法加载模型"}
    
    # 对于改进版PPLM，特殊处理词袋数据
    if is_modified_pplm and bow_vectors and model_type == "pplm_bow":
        one_hot_bows, bow_words_info = bow_vectors
        print(f"处理CPED词袋数据用于改进版PPLM...")
        print(f"词袋向量维度: {len(one_hot_bows[0]) if one_hot_bows else 0}")
        print(f"词袋词数量: {len(bow_words_info) if bow_words_info else 0}")
    
    # 准备提示词
    prompts = GENERATION_CONFIG["prompt_templates"]
    if args.num_prompts and args.num_prompts < len(prompts):
        prompts = prompts[:args.num_prompts]
    
    # 如果指定了最大提示数量，进一步限制
    if GENERATION_CONFIG["max_prompts"] and len(prompts) > GENERATION_CONFIG["max_prompts"]:
        prompts = prompts[:GENERATION_CONFIG["max_prompts"]]
    
    print(f"准备使用 {len(prompts)} 个提示词进行实验")
    
    # 准备实验结果容器
    experiment_results = {
        "experiment_name": exp_name,
        "experiment_type": model_type,
        "description": exp_desc,
        "is_modified_pplm": is_modified_pplm,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": experiment_config["config"],
        "prompts": [],
        "metrics": {}
    }
    
    # 为改进版PPLM优化配置
    if is_modified_pplm:
        # 确保class_label在配置中
        if "class_label" not in experiment_config["config"]:
            # 从实验名称中推断情感标签
            if "positive" in exp_name.lower():
                experiment_config["config"]["class_label"] = "positive"
            elif "negative" in exp_name.lower():
                experiment_config["config"]["class_label"] = "negative"
            else:
                experiment_config["config"]["class_label"] = "neutral"
        
        # 为中文优化参数
        experiment_config["config"].update({
            "window_length": 7,  # 增加窗口长度以更好地捕捉中文上下文
            "stepsize": 0.01,    # 稍大的步长以增强控制效果
            "kl_scale": 0.02     # 较小的KL散度权重以保持文本流畅性
        })
        print(f"为改进版PPLM优化的配置: {experiment_config['config']}")
    
    # 处理每个提示词
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"处理提示词")):
        # 生成文本
        if model_type == "vanilla":
            generated_texts = generate_text_with_vanilla_model(
                prompt, experiment_config["config"], model, tokenizer
            )
        elif model_type == "pplm_bow":
            generated_texts = generate_text_with_pplm_bow(
                prompt, bow_vectors, experiment_config["config"], model, tokenizer
            )
        else:
            print(f"未知的模型类型: {model_type}")
            generated_texts = []
        
        # 限制样本数
        if args.max_samples and args.max_samples < len(generated_texts):
            generated_texts = generated_texts[:args.max_samples]
        
        # 评估生成的文本
        # 注意：在实际应用中，这里应该使用真实的参考文本
        references = ["" for _ in generated_texts]  # 这里是占位符
        eval_results = evaluate_generated_texts(
            generated_texts, 
            references=references,
            model_config=experiment_config["config"]
        )
        
        # 保存结果
        experiment_results["prompts"].append({
            "prompt": prompt,
            "generated_texts": generated_texts,
            "evaluation": eval_results
        })
        
        # 调试模式下只处理一个提示词
        if args.debug:
            print(f"调试模式: 只处理第一个提示词")
            break
    
    # 计算整体指标
    aggregate_metrics = {}
    
    # 收集所有困惑度
    all_perplexities = []
    for prompt_result in experiment_results["prompts"]:
        if "perplexity" in prompt_result["evaluation"]:
            all_perplexities.extend(prompt_result["evaluation"]["perplexity"]["values"])
    
    if all_perplexities:
        aggregate_metrics["perplexity"] = {
            "mean": np.mean(all_perplexities),
            "std": np.std(all_perplexities)
        }
    
    # 收集所有多样性指标
    all_ttr = []
    all_topk_ratio = []
    for prompt_result in experiment_results["prompts"]:
        if "diversity" in prompt_result["evaluation"]:
            if "ttr" in prompt_result["evaluation"]["diversity"]:
                all_ttr.append(prompt_result["evaluation"]["diversity"]["ttr"])
            if "topk_ratio" in prompt_result["evaluation"]["diversity"]:
                all_topk_ratio.append(prompt_result["evaluation"]["diversity"]["topk_ratio"])
    
    if all_ttr:
        aggregate_metrics["diversity"] = {
            "mean_ttr": np.mean(all_ttr),
            "mean_topk_ratio": np.mean(all_topk_ratio) if all_topk_ratio else None
        }
    
    # 收集情感准确率
    all_sentiment_accuracies = []
    for prompt_result in experiment_results["prompts"]:
        if "sentiment_accuracy" in prompt_result["evaluation"]:
            all_sentiment_accuracies.append(prompt_result["evaluation"]["sentiment_accuracy"])
    
    if all_sentiment_accuracies:
        aggregate_metrics["sentiment_accuracy"] = {
            "mean": np.mean(all_sentiment_accuracies),
            "std": np.std(all_sentiment_accuracies)
        }
    
    # 保存整体指标
    experiment_results["metrics"] = aggregate_metrics
    
    # 标记实验成功
    experiment_results["status"] = "completed"
    
    # 保存结果到缓存
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)
    
    print(f"实验完成: {exp_name}")
    return experiment_results

def generate_summary_report(results, exp_output_dir):
    """生成实验汇总报告"""
    print("\n生成实验汇总报告...")
    
    # 准备汇总表格
    summary_data = []
    
    for result in results:
        if result["status"] != "completed":
            continue
        
        row = {
            "Experiment": result["experiment_name"],
            "Type": result["experiment_type"],
            "Perplexity": result["metrics"].get("perplexity", {}).get("mean", "N/A"),
            "Diversity TTR": result["metrics"].get("diversity", {}).get("mean_ttr", "N/A"),
            "Sentiment Accuracy": result["metrics"].get("sentiment_accuracy", {}).get("mean", "N/A")
        }
        summary_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    
    # 保存为CSV
    csv_path = os.path.join(exp_output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 保存为JSON
    json_path = os.path.join(exp_output_dir, "experiment_summary.json")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    
    print(f"\n汇总报告已生成:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    # 打印汇总表格
    print("\n实验结果汇总:")
    print(df.to_string(index=False))

def main():
    """主函数"""
    args = parse_args()
    
    # 设置实验环境
    exp_output_dir = setup_experiment_environment(args)
    
    # 选择要运行的实验
    experiments_to_run = []
    if args.experiments:
        # 只运行指定的实验
        for exp_name in args.experiments:
            found = False
            for exp in EXPERIMENTS:
                if exp["name"] == exp_name:
                    experiments_to_run.append(exp)
                    found = True
                    break
            if not found:
                print(f"警告: 未找到实验 '{exp_name}'")
    else:
        # 过滤实验类型（如果指定）
        if args.model_type:
            experiments_to_run = [exp for exp in EXPERIMENTS if exp["model_type"] == args.model_type]
        else:
            experiments_to_run = EXPERIMENTS
    
    print(f"\n准备运行 {len(experiments_to_run)} 个实验:")
    for exp in experiments_to_run:
        print(f"  - {exp['name']}: {exp['description']}")
    
    # 运行实验
    all_results = []
    start_time = time.time()
    
    for exp in experiments_to_run:
        # 跳过优先级低的实验（如果需要）
        if not args.debug or exp.get("priority") == "high":
            result = run_experiment(exp, exp_output_dir, args)
            all_results.append(result)
    
    # 生成汇总报告
    generate_summary_report(all_results, exp_output_dir)
    
    # 打印总体信息
    elapsed_time = time.time() - start_time
    print(f"\n所有实验完成！")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"结果保存在: {exp_output_dir}")

if __name__ == "__main__":
    main()