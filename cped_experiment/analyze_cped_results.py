#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集对比实验结果分析脚本
分析PPLM模型对比实验的结果并生成可视化报告
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CPED数据集对比实验结果分析脚本")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="包含实验结果的目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="分析报告输出目录")
    parser.add_argument("--format", choices=["markdown", "html", "text"], 
                        default="markdown", help="报告格式")
    parser.add_argument("--include_plots", action="store_true",
                        help="在报告中包含可视化图表")
    parser.add_argument("--save_plots", action="store_true",
                        help="保存图表为单独的文件")
    parser.add_argument("--compare_metrics", nargs="+", 
                        default=["perplexity", "diversity", "sentiment_accuracy"],
                        help="要比较的指标")
    return parser.parse_args()

def setup_analysis_environment(args):
    """设置分析环境"""
    # 验证结果目录存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录不存在: {args.results_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 默认输出到结果目录的analysis子目录
        output_dir = os.path.join(args.results_dir, "analysis")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表保存目录
    if args.save_plots:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        return output_dir, plots_dir
    
    return output_dir, None

def load_experiment_results(results_dir):
    """加载实验结果文件"""
    results = []
    experiment_files = []
    
    # 检查是否有汇总文件
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        # 从汇总文件中提取实验名称
        experiment_names = [item["Experiment"] for item in summary_data]
        
        # 加载每个实验的详细结果
        for exp_name in experiment_names:
            exp_file = os.path.join(results_dir, f"{exp_name}_results.json")
            if os.path.exists(exp_file):
                experiment_files.append(exp_file)
    
    # 如果没有汇总文件，直接查找所有_results.json文件
    if not experiment_files:
        for filename in os.listdir(results_dir):
            if filename.endswith("_results.json"):
                exp_file = os.path.join(results_dir, filename)
                experiment_files.append(exp_file)
    
    # 加载实验结果
    for exp_file in experiment_files:
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
            
            # 只处理成功的实验
            if exp_data.get("status") == "completed":
                results.append(exp_data)
                print(f"加载实验结果: {exp_file}")
        except Exception as e:
            print(f"加载文件 {exp_file} 时出错: {str(e)}")
    
    # 按照实验类型和名称排序
    results.sort(key=lambda x: (x.get("experiment_type", ""), x.get("experiment_name", "")))
    
    return results

def extract_metrics(results):
    """从结果中提取指标数据"""
    metrics_data = []
    
    for result in results:
        exp_name = result.get("experiment_name", "")
        exp_type = result.get("experiment_type", "")
        metrics = result.get("metrics", {})
        
        # 提取基础指标
        row = {
            "Experiment": exp_name,
            "Type": exp_type,
            "Perplexity": metrics.get("perplexity", {}).get("mean", np.nan),
            "Perplexity_Std": metrics.get("perplexity", {}).get("std", np.nan),
            "Diversity_TTR": metrics.get("diversity", {}).get("mean_ttr", np.nan),
            "Sentiment_Accuracy": metrics.get("sentiment_accuracy", {}).get("mean", np.nan),
            "Sentiment_Accuracy_Std": metrics.get("sentiment_accuracy", {}).get("std", np.nan)
        }
        
        metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)

def create_comparison_chart(metrics_df, metric_name, metric_label, output_dir, save_plots=False, plots_dir=None):
    """创建指标对比图表"""
    # 过滤掉NaN值
    valid_df = metrics_df.dropna(subset=[metric_name])
    
    if valid_df.empty:
        print(f"警告: 指标 '{metric_name}' 的数据不足，无法生成图表")
        return None
    
    plt.figure(figsize=(10, 6))
    
    # 按照实验类型分组
    type_groups = valid_df.groupby("Type")
    
    # 为柱状图准备数据
    types = list(type_groups.groups.keys())
    experiments = valid_df["Experiment"].unique()
    
    # 设置柱状图宽度
    width = 0.8 / len(experiments)
    
    # 创建柱状图
    for i, exp in enumerate(experiments):
        exp_data = valid_df[valid_df["Experiment"] == exp]
        x_pos = [j + i * width for j in range(len(types))]
        
        # 获取数据和误差（如果有）
        values = exp_data[metric_name].tolist()
        errors = None
        
        # 查找对应的标准差列
        std_column = f"{metric_name}_Std"
        if std_column in exp_data.columns:
            errors = exp_data[std_column].tolist()
        
        plt.bar(x_pos, values, width, label=exp, yerr=errors, capsize=5)
    
    # 设置图表属性
    plt.xlabel("模型类型")
    plt.ylabel(metric_label)
    plt.title(f"不同模型的{metric_label}对比")
    plt.xticks([j + width * (len(experiments) - 1) / 2 for j in range(len(types))], types)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    if save_plots and plots_dir:
        filename = f"{metric_name}_comparison.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return filepath
    
    # 保存图表到内存
    filename = f"{metric_name}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

def create_radar_chart(metrics_df, output_dir, save_plots=False, plots_dir=None):
    """创建雷达图对比多个指标"""
    # 选择需要的指标
    selected_metrics = [
        "Perplexity", "Diversity_TTR", "Sentiment_Accuracy"
    ]
    
    # 为每个指标准备标准化数据
    normalized_data = []
    experiments = metrics_df["Experiment"].unique()
    
    # 确保每个实验都有所有指标的数据
    valid_experiments = []
    for exp in experiments:
        exp_data = metrics_df[metrics_df["Experiment"] == exp]
        if not exp_data[selected_metrics].isna().any().any():
            valid_experiments.append(exp)
    
    if len(valid_experiments) < 2:
        print("警告: 有效的实验数据不足，无法生成雷达图")
        return None
    
    # 准备标准化数据
    for exp in valid_experiments:
        exp_data = metrics_df[metrics_df["Experiment"] == exp].iloc[0]
        
        # 标准化数据（注意：困惑度是越低越好，其他是越高越好）
        normalized_row = []
        for metric in selected_metrics:
            if metric == "Perplexity":
                # 困惑度：假设最小值为10，最大值为100
                min_val = 10
                max_val = 100
                # 反转值，使得更高的值表示更好的性能
                norm_val = 1 - (exp_data[metric] - min_val) / (max_val - min_val)
                norm_val = max(0, min(1, norm_val))  # 限制在[0, 1]范围内
            else:
                # 其他指标：假设最小值为0，最大值为1
                min_val = 0
                max_val = 1
                norm_val = (exp_data[metric] - min_val) / (max_val - min_val)
                norm_val = max(0, min(1, norm_val))  # 限制在[0, 1]范围内
            normalized_row.append(norm_val)
        
        normalized_data.append(normalized_row)
    
    # 创建雷达图
    plt.figure(figsize=(10, 8))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(selected_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 为每个实验绘制雷达图
    for i, exp in enumerate(valid_experiments):
        values = normalized_data[i]
        values += values[:1]  # 闭合雷达图
        plt.polar(angles, values, label=exp, linewidth=2, linestyle='solid')
        plt.fill(angles, values, alpha=0.25)
    
    # 设置雷达图属性
    metric_labels = ["困惑度", "多样性(TTR)", "情感准确率"]
    metric_labels += metric_labels[:1]  # 闭合雷达图
    plt.xticks(angles, metric_labels)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.title("不同模型性能雷达图对比", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 保存图表
    filename = "model_comparison_radar.png"
    if save_plots and plots_dir:
        filepath = os.path.join(plots_dir, filename)
    else:
        filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

def create_text_examples_table(results, output_dir):
    """创建生成文本示例表格"""
    examples_data = []
    
    # 为每个实验收集生成文本示例
    for result in results:
        exp_name = result.get("experiment_name", "")
        prompts = result.get("prompts", [])
        
        if prompts:
            # 选择第一个提示词的前3个生成文本
            prompt_data = prompts[0]
            generated_texts = prompt_data.get("generated_texts", [])[:3]
            
            for i, text in enumerate(generated_texts):
                examples_data.append({
                    "Experiment": exp_name,
                    "Prompt": prompt_data.get("prompt", ""),
                    "Example": i + 1,
                    "Generated_Text": text[:100] + "..." if len(text) > 100 else text
                })
    
    if examples_data:
        examples_df = pd.DataFrame(examples_data)
        examples_file = os.path.join(output_dir, "text_examples.csv")
        examples_df.to_csv(examples_file, index=False, encoding="utf-8-sig")
        return examples_df
    
    return None

def generate_markdown_report(metrics_df, examples_df, chart_paths, output_dir):
    """生成Markdown格式的分析报告"""
    report_file = os.path.join(output_dir, "analysis_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        # 报告标题
        f.write("# CPED数据集PPLM模型对比实验分析报告\n\n")
        
        # 报告信息
        f.write("## 报告信息\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 实验概述
        f.write("## 实验概述\n\n")
        f.write("本报告分析了基于CPED数据集的PPLM模型对比实验结果，包括不同模型在多种评估指标上的表现。\n\n")
        
        # 模型列表
        if not metrics_df.empty:
            f.write("### 参与实验的模型\n\n")
            for _, row in metrics_df.iterrows():
                f.write(f"- **{row['Experiment']}** ({row['Type']})\n")
            f.write("\n")
        
        # 指标对比表格
        if not metrics_df.empty:
            f.write("## 性能指标对比\n\n")
            f.write("### 主要指标概览\n\n")
            
            # 准备Markdown表格
            f.write("| 实验 | 类型 | 困惑度 | 多样性(TTR) | 情感准确率 |\n")
            f.write("|------|------|--------|-------------|------------|\n")
            
            for _, row in metrics_df.iterrows():
                f.write(f"| {row['Experiment']} | {row['Type']} | ")
                f.write(f"{row['Perplexity']:.2f} ± {row['Perplexity_Std']:.2f} | " if not pd.isna(row['Perplexity']) else "N/A | ")
                f.write(f"{row['Diversity_TTR']:.4f} | " if not pd.isna(row['Diversity_TTR']) else "N/A | ")
                f.write(f"{row['Sentiment_Accuracy']:.4f} ± {row['Sentiment_Accuracy_Std']:.4f} |\n" if not pd.isna(row['Sentiment_Accuracy']) else "N/A |\n")
            
            f.write("\n")
        
        # 可视化图表
        if chart_paths:
            f.write("## 可视化分析\n\n")
            
            # 雷达图
            radar_path = next((p for p in chart_paths if "radar" in p), None)
            if radar_path:
                relative_path = os.path.relpath(radar_path, output_dir)
                f.write("### 模型性能雷达图对比\n\n")
                f.write(f"![模型性能雷达图对比]({relative_path})\n\n")
            
            # 困惑度对比
            perplexity_path = next((p for p in chart_paths if "Perplexity" in p), None)
            if perplexity_path:
                relative_path = os.path.relpath(perplexity_path, output_dir)
                f.write("### 困惑度对比\n\n")
                f.write(f"![困惑度对比]({relative_path})\n\n")
                f.write("**分析**: 困惑度越低，表示模型生成的文本流畅度越好。从图表可以看出...\n\n")
            
            # 多样性对比
            diversity_path = next((p for p in chart_paths if "Diversity" in p), None)
            if diversity_path:
                relative_path = os.path.relpath(diversity_path, output_dir)
                f.write("### 多样性对比\n\n")
                f.write(f"![多样性对比]({relative_path})\n\n")
                f.write("**分析**: 多样性指标(TTR)越高，表示模型生成的文本词汇丰富度越好。从图表可以看出...\n\n")
            
            # 情感准确率对比
            sentiment_path = next((p for p in chart_paths if "Sentiment" in p), None)
            if sentiment_path:
                relative_path = os.path.relpath(sentiment_path, output_dir)
                f.write("### 情感准确率对比\n\n")
                f.write(f"![情感准确率对比]({relative_path})\n\n")
                f.write("**分析**: 情感准确率越高，表示模型生成的文本越符合目标情感要求。从图表可以看出...\n\n")
        
        # 生成文本示例
        if examples_df is not None and not examples_df.empty:
            f.write("## 生成文本示例\n\n")
            
            # 按实验分组显示示例
            experiments = examples_df["Experiment"].unique()
            for exp in experiments:
                exp_examples = examples_df[examples_df["Experiment"] == exp]
                if not exp_examples.empty:
                    f.write(f"### {exp} 示例\n\n")
                    
                    # 显示第一个提示词
                    prompt = exp_examples["Prompt"].iloc[0]
                    f.write(f"**提示词**: {prompt}\n\n")
                    
                    # 显示生成文本示例
                    for _, row in exp_examples.iterrows():
                        f.write(f"**示例 {row['Example']}**:\n")
                        f.write(f"> {row['Generated_Text']}\n\n")
        
        # 分析总结
        f.write("## 分析总结\n\n")
        
        # 性能分析
        f.write("### 性能分析\n\n")
        
        # 寻找最佳模型
        if not metrics_df.empty:
            # 找出困惑度最低的模型
            min_perplexity_model = None
            if not metrics_df["Perplexity"].isna().all():
                min_perplexity_row = metrics_df.loc[metrics_df["Perplexity"].idxmin()]
                min_perplexity_model = min_perplexity_row["Experiment"]
                f.write(f"- **最佳流畅度模型**: {min_perplexity_model}，困惑度为 {min_perplexity_row['Perplexity']:.2f}\n")
            
            # 找出多样性最高的模型
            max_diversity_model = None
            if not metrics_df["Diversity_TTR"].isna().all():
                max_diversity_row = metrics_df.loc[metrics_df["Diversity_TTR"].idxmax()]
                max_diversity_model = max_diversity_row["Experiment"]
                f.write(f"- **最佳多样性模型**: {max_diversity_model}，TTR值为 {max_diversity_row['Diversity_TTR']:.4f}\n")
            
            # 找出情感准确率最高的模型
            max_sentiment_model = None
            if not metrics_df["Sentiment_Accuracy"].isna().all():
                max_sentiment_row = metrics_df.loc[metrics_df["Sentiment_Accuracy"].idxmax()]
                max_sentiment_model = max_sentiment_row["Experiment"]
                f.write(f"- **最佳情感控制模型**: {max_sentiment_model}，情感准确率为 {max_sentiment_row['Sentiment_Accuracy']:.4f}\n\n")
        
        # 结论
        f.write("### 结论\n\n")
        f.write("基于实验结果分析，我们可以得出以下结论：\n\n")
        f.write("1. **PPLM词袋模型的有效性**: 使用词袋控制的PPLM模型在情感控制方面表现出...\n")
        f.write("2. **不同模型的优缺点**: 基础模型在...方面表现较好，而PPLM模型在...方面表现更优\n")
        f.write("3. **词袋控制参数的影响**: 词袋权重、上下文窗口等参数对生成文本的质量有显著影响\n\n")
        
        # 建议
        f.write("### 改进建议\n\n")
        f.write("基于实验分析，我们提出以下改进建议：\n\n")
        f.write("1. **词袋优化**: 可以进一步优化词袋内容，增加更有代表性的情感词汇\n")
        f.write("2. **参数调优**: 根据具体应用场景调整PPLM控制参数，以达到流畅度和情感控制的最佳平衡\n")
        f.write("3. **集成方法**: 考虑结合多种控制方法，如词袋和分类器联合控制，以提高情感控制的准确性\n")
        f.write("4. **模型选择**: 对于中文情感控制任务，可以考虑使用更适合中文的预训练模型\n\n")
        
        # 参考文献
        f.write("## 参考文献\n\n")
        f.write("1. PPLM: Plug and Play Language Models for Controlled Text Generation\n")
        f.write("2. CPED数据集相关研究文献\n")
    
    return report_file

def main():
    """主函数"""
    args = parse_args()
    
    # 设置分析环境
    output_dir, plots_dir = setup_analysis_environment(args)
    
    # 加载实验结果
    print("加载实验结果...")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("错误: 没有找到有效的实验结果")
        sys.exit(1)
    
    print(f"找到 {len(results)} 个实验结果")
    
    # 提取指标数据
    print("提取指标数据...")
    metrics_df = extract_metrics(results)
    
    # 保存指标数据
    metrics_file = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    print(f"指标数据已保存到: {metrics_file}")
    
    # 创建图表
    chart_paths = []
    if args.include_plots or args.save_plots:
        print("生成可视化图表...")
        
        # 创建困惑度对比图表
        if "perplexity" in args.compare_metrics:
            perplexity_path = create_comparison_chart(
                metrics_df, "Perplexity", "困惑度", output_dir, args.save_plots, plots_dir
            )
            if perplexity_path:
                chart_paths.append(perplexity_path)
        
        # 创建多样性对比图表
        if "diversity" in args.compare_metrics:
            diversity_path = create_comparison_chart(
                metrics_df, "Diversity_TTR", "多样性(TTR)", output_dir, args.save_plots, plots_dir
            )
            if diversity_path:
                chart_paths.append(diversity_path)
        
        # 创建情感准确率对比图表
        if "sentiment_accuracy" in args.compare_metrics:
            sentiment_path = create_comparison_chart(
                metrics_df, "Sentiment_Accuracy", "情感准确率", output_dir, args.save_plots, plots_dir
            )
            if sentiment_path:
                chart_paths.append(sentiment_path)
        
        # 创建雷达图
        radar_path = create_radar_chart(metrics_df, output_dir, args.save_plots, plots_dir)
        if radar_path:
            chart_paths.append(radar_path)
    
    # 创建生成文本示例表格
    print("收集生成文本示例...")
    examples_df = create_text_examples_table(results, output_dir)
    
    # 生成分析报告
    print("生成分析报告...")
    report_file = generate_markdown_report(
        metrics_df, examples_df, chart_paths, output_dir
    )
    
    print(f"\n分析完成！")
    print(f"分析报告: {report_file}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()