#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPED数据集情感判别器训练脚本
用于训练基于BERT的情感分类器，供PPLM使用
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CPED数据集情感判别器训练脚本")
    parser.add_argument("--data_dir", type=str, default="./processed_cped",
                        help="处理后的数据目录")
    parser.add_argument("--output_dir", type=str, default="./cped_discriminator",
                        help="模型输出目录")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese",
                        help="预训练模型名称")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()

class CPEDDataset(Dataset):
    """CPED数据集类"""
    def __init__(self, data_file, tokenizer, max_seq_length, label_map):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_map = label_map
        self.examples = self._load_data(data_file)
    
    def _load_data(self, data_file):
        """加载数据"""
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            # 假设数据格式为TSV: label\ttext
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    label = parts[0]
                    text = '\t'.join(parts[1:])
                    label_id = self.label_map.get(label, 0)  # 默认标签
                    examples.append((text, label_id))
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text, label = self.examples[idx]
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="训练中")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 获取预测
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    
    return total_loss / len(dataloader), accuracy, report, all_predictions, all_labels

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载标签映射
    label_map_path = os.path.join(args.data_dir, "emotion_labels.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print(f"加载情感标签映射: {label_map}")
    else:
        # 如果没有标签映射，创建默认映射
        # 假设常见的情感类别
        label_map = {
            '积极': 0,
            '消极': 1,
            '中性': 2,
            '喜悦': 3,
            '悲伤': 4,
            '愤怒': 5,
            '恐惧': 6,
            '惊讶': 7
        }
        print(f"使用默认情感标签映射: {label_map}")
    
    # 反向映射，用于评估报告
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # 加载预训练模型和分词器
    print(f"加载预训练模型: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_map),
        output_attentions=False,
        output_hidden_states=False
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_file = os.path.join(args.data_dir, "cped_discrim.tsv")
    
    train_dataset = CPEDDataset(train_file, tokenizer, args.max_seq_length, label_map)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    
    # 准备优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_accuracy = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            args.gradient_accumulation_steps
        )
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss, val_accuracy, val_report, _, _ = evaluate(model, val_dataloader, device)
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        print("验证分类报告:")
        print(val_report)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"保存新的最佳模型 (准确率: {best_val_accuracy:.4f})")
            
            # 保存模型
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # 保存训练参数
            training_args = vars(args)
            with open(os.path.join(args.output_dir, "training_args.json"), 'w', encoding='utf-8') as f:
                json.dump(training_args, f, ensure_ascii=False, indent=2)
            
            # 保存标签映射
            with open(os.path.join(args.output_dir, "label_map.json"), 'w', encoding='utf-8') as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")
    print(f"模型保存路径: {args.output_dir}")
    print("\n使用方法:")
    print("1. 将训练好的判别器集成到PPLM中:")
    print("   - 修改run_pplm.py中的DISCRIMINATOR_MODELS_PARAMS配置")
    print("   - 添加新的判别器路径和参数")
    print("2. 使用训练好的判别器生成情感控制文本:")
    print("   python run_pplm.py --discriminating --discriminator model_name --class_label class_id --cond_text '你的条件文本'")

if __name__ == "__main__":
    main()