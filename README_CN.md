# PPLM中文扩展项目

本项目是Plug and Play Language Model (PPLM)的中文扩展版本，特别针对中文情感控制文本生成任务进行了优化和增强。项目包含了基于CPED(Chinese Poetry Emotion Dataset)中文诗歌情感数据集的完整对比实验方案。

## 项目结构

```
PPLMS/
├─ cped_experiment/          # CPED数据集对比实验代码
│  ├─ character_wordbags/    # 角色词袋数据
│  ├─ compare_pplm_models.py # 模型对比实验主程序
│  ├─ extract_character_wordbags.py # 角色词袋提取脚本
│  ├─ run_cped_experiment.py # 实验执行脚本
│  ├─ cped_experiment_config.py # 实验配置文件
│  ├─ analyze_cped_results.py # 结果分析脚本
│  ├─ prepare_cped_dataset.py # 数据集预处理脚本
│  ├─ train_cped_discriminator.py # 判别器训练脚本
│  └─ CPED实验方案总结.md # 实验方案详细文档
│
├─ ppls_model/               # PPLS模型代码
│  └─ generate_ppls.py       # 改进版PPLM生成器
│
├─ paper_code/               # 原始PPLM论文代码
│  ├─ discrim_models/        # 判别器模型
│  ├─ pytorch_pretrained_bert/ # BERT预训练模型
│  └─ wordlists/             # 词袋词表
│
├─ processed_cped/           # 处理后的CPED数据集
│  └─ bow_files/             # 情感词袋文件
│
├─ human_annotation/         # 人工标注数据
├─ logs/                     # 日志文件
├─ output/                   # 实验输出结果
└─ __pycache__/              # Python缓存文件
```

## 核心功能

### 1. 原始PPLM功能
- 基于词袋(Bag of Words)的文本控制
- 基于判别器(Discriminator)的文本控制
- 支持多种情感和主题控制
- 提供完整的超参数调优指南

### 2. 中文扩展功能
- 针对中文文本生成进行了优化
- 支持CPED中文诗歌情感数据集
- 实现了改进版PPLM模型(PPLS)
- 提供完整的中文情感控制实验方案

### 3. 改进版PPLM(PPLS)创新点

1. **权重调整机制**：引入多因子权重计算公式
   ```python
   weight = base_weight * context_adjustment * sentiment_factor
   ```

2. **词袋词冷却机制**：通过`cooldown_duration`控制干预间隔，避免过度干预

3. **上下文感知优化**：根据已生成内容动态调整控制强度

4. **直接logits修改**：移除复杂的梯度计算，提高计算效率

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 基础使用示例

#### PPLMS模型

```bash
python -m ppls_model.generate_ppls --cond_text "你喜欢我吗" --length 20 --stepsize 0.005 --temperature 0.9 --top_k 100 --num_samples 1 --num_iterations 3
```

#### 词袋模式控制

```bash
python -m ppls_model.generate_ppls -B "./user_vocab.txt" --cond_text "今天天气" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --sample
```

#### 判别器模式控制

```bash
python -m ppls_model.generate_ppls -D sentiment --class_label 2 --cond_text "我的狗死了" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

## CPED数据集对比实验

### 实验流程

1. **数据集预处理**
   ```bash
   python cped_experiment/prepare_cped_dataset.py --input_file ./data/cped/train_split.csv --output_dir ./processed_cped
   ```

2. **训练情感判别器**
   ```bash
   python cped_experiment/train_cped_discriminator.py --data_dir ./processed_cped --output_dir ./cped_discriminator
   ```

3. **执行对比实验**
   ```bash
   python cped_experiment/run_cped_experiment.py --experiments all --output_dir ./output/experiments
   ```

4. **分析实验结果**
   ```bash
   python cped_experiment/analyze_cped_results.py --results_dir ./output/experiments --output_dir ./output/analysis
   ```

### 实验配置

实验配置文件`cped_experiment_config.py`定义了以下关键参数：

- **参与对比的模型**：vanilla_gpt2、pplm_bow、pplm_discrim、modified_pplm等
- **生成参数**：采样参数、PPLM控制参数
- **评估指标**：困惑度、多样性、情感控制准确率、生成速度

### 评估指标

1. **困惑度(Perplexity)**：衡量生成文本的流畅度
2. **多样性指标**：词汇多样性、n-gram多样性
3. **情感控制准确率**：生成文本是否符合目标情感
4. **生成速度**：平均生成时间

## 模型对比

| 模型类型 | 优点 | 缺点 |
|---------|------|------|
| vanilla_gpt2 | 生成速度快，文本流畅度高 | 无法控制情感和主题 |
| pplm_bow | 主题控制效果好，计算效率高 | 情感控制不够精细 |
| pplm_discrim | 情感控制精准，效果稳定 | 计算复杂度高 |
| modified_pplm | 结合了PPLM的控制能力和GPT2的流畅度，优化了中文生成 | 需要调整新的超参数 |

## 技术细节

### 中文支持优化
- 使用jieba分词处理中文文本
- 提供中文停用词列表
- 调整n-gram窗口大小适应中文特点
- 优化词袋构建策略

### 性能优化
- 实现了直接logits修改，避免梯度计算
- 引入词袋词冷却机制，减少干预频率
- 优化内存使用，支持更大批量生成

## 引用

```
@inproceedings{
Dathathri2020Plug,
title={Plug and Play Language Models: A Simple Approach to Controlled Text Generation},
author={Sumanth Dathathri and Andrea Madotto and Janice Lan and Jane Hung and Eric Frank and Piero Molino and Jason Yosinski and Rosanne Liu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1edEyBKDS}
}
```

## 许可证

本项目遵循Apache 2.0许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**：本项目是PPLM的扩展版本，主要针对中文文本生成任务进行了优化。原始PPLM项目的更多信息请参考[官方GitHub仓库](https://github.com/uber-research/PPLM)。