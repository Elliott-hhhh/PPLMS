#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
import re
import logging
import os
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

# 配置日志系统
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "generate_ppls.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建logger对象
logger = logging.getLogger(__name__)

# 简化导入，避免导入可能导致问题的额外库
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    logger.error(f"导入transformers库时出错: {e}")
    logger.error("请确保已正确安装transformers库: pip install transformers")
    print(f"导入transformers库时出错: {e}")  # 保留这一行用于控制台错误提示
    print("请确保已正确安装transformers库: pip install transformers")  # 保留这一行用于控制台错误提示
    exit(1)

from ..pplm_classification_head import ClassificationHead
import requests
import tempfile
import os
from urllib.parse import urlparse

# 导入情感分析模块
from .words_sentiment import sentiment_analyzer, get_sentiment_factor

class SentimentWord:
    def __init__(self, word, sentiment=0):
        # sentiment: -1=消极, 0=中性, 1=积极
        self.word = word
        self.sentiment = sentiment

# 示例用法
# sentiment_words = [
#     SentimentWord("哈哈哈哈", sentiment=1),  # 积极情绪词
#     SentimentWord("太好了", sentiment=1),
#     SentimentWord("难过", sentiment=-1),
#     SentimentWord("设计", sentiment=0)      # 中性词
# ]

def cached_file(url):
    """Simple implementation to download and cache files"""
    if os.path.exists(url):  # 如果是本地文件路径
        logger.info(f"使用本地文件: {url}")
        return url

    # 如果是URL，则下载
    parsed = urlparse(url)
    if parsed.scheme in ('http', 'https'):
        try:
            logger.info(f"下载文件: {url}")
            response = requests.get(url)
            response.raise_for_status()

            # 创建临时文件保存下载内容
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(response.content)
            temp_file.close()
            logger.info(f"文件下载完成，保存至临时文件: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            logger.error(f"下载文件 {url} 出错: {e}")
            raise e

    return url

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

def analyze_context_sentiment(model_output, tokenizer, device):
    """分析当前上下文的情感倾向
    返回值: -1(消极) 到 1(积极) 之间的浮点数
    """
    # 这里可以使用简单规则或情感分类器
    # 1. 简单规则方法：关键词匹配
    negative_words = ["难过", "悲伤", "痛苦", "糟糕", "失望"]
    positive_words = ["开心", "快乐", "喜欢", "美好", "感谢"]
    
    # 从model_output获取当前生成的文本
    # 假设我们有一个方法可以获取当前上下文
    context_text = ""  # 需要实现从model_output提取
    
    # 简单的情感计算
    sentiment_score = 0
    for word in negative_words:
        if word in context_text:
            sentiment_score -= 0.2
    for word in positive_words:
        if word in context_text:
            sentiment_score += 0.2
    
    # 限制在-1到1之间
    return max(-1.0, min(1.0, sentiment_score))

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def create_perturbation(grad_accum_list):
        """递归创建可微分的扰动张量"""
        if isinstance(grad_accum_list, list):
            return [create_perturbation(item) for item in grad_accum_list]
        else:
            return to_var(torch.from_numpy(grad_accum_list), requires_grad=True, device=device)

def apply_perturbation(past_list, perturbation_list):
        """递归应用扰动到past结构"""
        if isinstance(past_list, (tuple, list)) and isinstance(perturbation_list, list):
            return [apply_perturbation(p, per) for p, per in zip(past_list, perturbation_list)]
        else:
            return past_list + perturbation_list

# 重写perturb_past函数以支持DynamicCache对象
def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.005,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.05,
        device='cuda',
        verbosity_level=REGULAR,
        user_vocab_vectors=None,
        tokenizer=None,
        sentiment_words=None,
        output_so_far=None,
        bow_indices=None  # 添加词袋索引参数
):
    # 在函数开始时进行上下文情感分析
    context_sentiment = 0.0  # 默认为中性
    
    # 如果有上下文，使用情感分析器计算上下文情感分数
    if output_so_far is not None and tokenizer is not None:
        # 尝试从output_so_far解码当前上下文
        try:
            context_text = tokenizer.decode(output_so_far[0], skip_special_tokens=True)
            context_sentiment = sentiment_analyzer.get_context_sentiment_score(context_text)
            if verbosity_level >= VERBOSE:
                logger.info(f"当前上下文情感分数: {context_sentiment}")
        except Exception as e:
            logger.error(f"解析上下文时出错: {e}")
    
    if num_iterations == 0:
        logger.info("Returning early due to num_iterations=0")
        return past, accumulated_hidden, None, []
    
    # 修复：处理DynamicCache对象
    # def create_grad_accumulator_for_dynamic_cache(cache):
    #     """为DynamicCache创建梯度累加器"""
    #     try:
    #         # 尝试直接访问缓存的内部结构
    #         if hasattr(cache, 'cache'):
    #             # 获取缓存内容
    #             cache_dict = cache.cache
    #             grad_accum = {}
    #             for layer_key, layer_cache in cache_dict.items():
    #                 # layer_cache通常是包含key和value两个张量的元组
    #                 if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
    #                     key, value = layer_cache
    #                     grad_accum[layer_key] = (
    #                         np.zeros(key.shape).astype("float32"),
    #                         np.zeros(value.shape).astype("float32")
    #                     )
    #             return grad_accum
    #         else:
    #             # 回退方案：如果无法访问内部结构，返回空字典
    #             return {}
    #     except Exception as e:
    #         print(f"Error creating grad accumulator: {e}")
    #         return {}
    
    # # 修改：获取past结构的序列长度
    # def get_sequence_length(cache):
    #     """获取DynamicCache的序列长度"""
    #     try:
    #         # 尝试使用get_seq_length方法（如果存在）
    #         if hasattr(cache, 'get_seq_length'):
    #             return cache.get_seq_length()
    #         # 尝试直接从缓存内容计算
    #         elif hasattr(cache, 'cache') and cache.cache:
    #             # 获取第一个层的key张量
    #             first_layer_cache = next(iter(cache.cache.values()))
    #             if isinstance(first_layer_cache, tuple) and len(first_layer_cache) > 0:
    #                 # 假设key张量的倒数第二个维度是序列长度
    #                 return first_layer_cache[0].shape[-2]
    #         return 0
    #     except Exception:
    #         return 0
    
    # # 创建梯度累加器
    # grad_accumulator = create_grad_accumulator_for_dynamic_cache(past)
    
    # if accumulated_hidden is None:
    #     accumulated_hidden = 0
    
    # if decay:
    #     decay_mask = torch.arange(
    #         0.,
    #         1.0 + SMALL_CONST,
    #         1.0 / (window_length)
    #     )[1:]
    # else:
    #     decay_mask = 1.0
    
    # # 获取当前序列长度
    # curr_length = get_sequence_length(past)
    
    # 为简单起见，我们将修改模型调用方式，不再直接扰动past
    # 而是通过梯度更新模型参数的方式来实现类似效果
    loss_per_iter = []
    new_accumulated_hidden = None
    
    # 由于DynamicCache的复杂性，我们简化实现
    # 对于每次迭代，我们直接使用模型生成logits并计算损失
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            logger.info("Iteration {}".format(i + 1))
        # 计算一个假的损失值
        loss_per_iter.append(0.0)
        # 直接使用原始past调用模型
        # try:
        #     # 修改模型调用方式，适配新版transformers
        #     model_output = model(last, past_key_values=past, output_hidden_states=True)
            
        #     if hasattr(model_output, 'logits'):
        #         all_logits = model_output.logits
        #         all_hidden = model_output.hidden_states
        #     else:
        #         all_logits = model_output[0]
        #         all_hidden = model_output[2] if len(model_output) > 2 else None
            
        #     hidden = all_hidden[-1]
        #     new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
            
        #     logits = all_logits[:, -1, :]
        #     probs = F.softmax(logits, dim=-1)
            
        #     loss = 0.0
        #     loss_list = []
            
        #     # 计算词袋损失（带情感匹配）
        #     if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
        #         if user_vocab_vectors:
        #             for one_hot_vocab in user_vocab_vectors:
        #                 vocab_logits = torch.mm(probs, torch.t(one_hot_vocab))
        #                 vocab_loss = -torch.log(torch.sum(vocab_logits))
        #                 loss += vocab_loss * 0.5
        #                 loss_list.append(vocab_loss)
        #         elif one_hot_bows_vectors and sentiment_words:
        #             for i, one_hot_bow in enumerate(one_hot_bows_vectors):
        #                 if i < len(sentiment_words):
        #                     word_sentiment = sentiment_words[i].sentiment
        #                     # 计算情感匹配度
        #                     sentiment_match = 1.0 if abs(context_sentiment - word_sentiment) < 0.5 else 0.0
                            
        #                     if sentiment_match > 0:
        #                         bow_logits = torch.mm(probs, torch.t(one_hot_bow))
        #                         bow_loss = -torch.log(torch.sum(bow_logits))
        #                         loss += bow_loss * sentiment_match
        #                         loss_list.append(bow_loss)
            
        #     # 添加KL散度损失
        #     if kl_scale > 0 and unpert_logits is not None:
        #         kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(unpert_logits, dim=-1))
        #         loss += kl_scale * kl_loss
            
        #     loss_per_iter.append(loss.item())
            
        #     # 由于无法直接扰动DynamicCache，我们跳过梯度计算和更新
        #     # 在实际应用中，可能需要修改模型内部实现或使用适配器
            
        # except Exception as e:
        #     print(f"Error during model forward pass: {e}")
        #     break
    
    # 返回原始past（因为无法直接修改DynamicCache）
    return past, accumulated_hidden, [], loss_per_iter

def top_p_filter(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

def apply_ngram_penalty(probs, output_so_far, penalty=0.5, ngram_size=3):
    """对n-gram重复施加更强的惩罚"""
    if output_so_far is not None and output_so_far.shape[1] >= ngram_size:
        recent_tokens = output_so_far[0, -ngram_size+1:].tolist()
        for token_id in set(recent_tokens):
            probs[0, token_id] *= penalty
        probs = probs / torch.sum(probs)
    return probs
def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_file(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                 logger.warning("class_label {} not in class_vocab".format(class_label))
                 logger.warning("available values are: {}".format(params["class_vocab"]))
                 logger.warning("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                logger.warning("class_label {} not in class_vocab".format(class_label))
                logger.warning("available values are: {}".format(params["class_vocab"]))
                logger.warning("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_file(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_special_tokens=False)
             for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None, None

    one_hot_bows_vectors = []
    bow_words_info = []  # 存储词袋中词的信息（原始词和对应的one-hot向量）
    
    for single_bow in bow_indices:
        # 过滤掉空的编码
        single_bow = list(filter(lambda x: len(x) >= 1, single_bow))
        
        # 不再只取第一个token，而是为每个词创建单独的one-hot向量
        for token_list in single_bow:
            if len(token_list) >= 1:
                # 为每个词创建一个one-hot向量，包含所有token
                one_hot_bow = torch.zeros(1, tokenizer.vocab_size).to(device)
                for token_id in token_list:  # 使用所有token
                    if token_id < tokenizer.vocab_size:
                        one_hot_bow[0, token_id] = 1.0
                one_hot_bows_vectors.append(one_hot_bow)
                
                # 尝试恢复原始词（可能不完全准确，但可以用于情感分析）
                word_text = tokenizer.decode(token_list, skip_special_tokens=True)
                bow_words_info.append((word_text, one_hot_bow))
    
    # 返回one-hot向量和词袋信息
    return one_hot_bows_vectors, bow_words_info
# 替换原有词袋加载逻辑
def get_user_vocab_indices(tokenizer, user_vocab_path="user_vocab.txt"):
    try:
        with open(user_vocab_path, "r", encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(words)} words from user_vocab.txt: {words}")
        # 为每个词创建单独的token列表
        return [[tokenizer.encode(word, add_special_tokens=False)] for word in words]
    except Exception as e:
        logger.error(f"Error loading user vocab: {e}")
        return []


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.005,
        temperature=0.9,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.95,
        kl_scale=0.05,
        verbosity_level=REGULAR,
        user_vocab_path="user_vocab.txt",
        sentiment_words=None,
        sentiment_beta=0.8,  # 添加情感扰动系数参数
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            logger.info("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            logger.info("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            logger.info("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level,
        user_vocab_path=user_vocab_path
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        unpert_gen_tok_text, pert_gen_tok_text, loss_in_time = generate_text_pplm(
                model=model,
                tokenizer=tokenizer,
                context=context,
                device=device,
                perturb=True,
                bow_indices=bow_indices,
                classifier=classifier,
                class_label=class_id,
                loss_type=loss_type,
                length=length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                verbosity_level=verbosity_level,
                user_vocab_path = user_vocab_path,  # 新增
                sentiment_words=sentiment_words,
                sentiment_beta=sentiment_beta  # 传递情感扰动系数参数
            )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


# 在采样之前添加token过滤
def filter_special_tokens(probs, tokenizer, special_tokens_to_remove=None):
    if special_tokens_to_remove is None:
        # 移除常见的特殊token
        special_tokens_to_remove = ['____', '[PAD]', '<|endoftext|>']

    # 获取特殊token的ID
    special_token_ids = []
    for token in special_tokens_to_remove:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:  # 只处理单个token
            special_token_ids.append(token_ids[0])

    # 将特殊token的概率设为0
    for token_id in special_token_ids:
        if token_id < probs.shape[-1]:
            probs[:, token_id] = 0

    # 重新归一化
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return probs

def apply_repeat_penalty(probs, output_so_far, penalty=0.9):
    """对已出现的n-gram施加惩罚"""
    if output_so_far is not None and output_so_far.shape[1] > 1:
        # 获取最近的几个token
        recent_tokens = output_so_far[0, -1:].tolist()
        for token_id in set(recent_tokens):  # 对每个唯一的最近token施加惩罚
            probs[0, token_id] *= penalty
        # 重新归一化
        probs = probs / torch.sum(probs)
    return probs

def apply_repetition_penalty(logits, output_so_far, repetition_penalty=1.2):
    """对已生成的 token 施加重复惩罚（改进版，防止语言崩坏）"""
    if output_so_far is None or output_so_far.size(1) == 0:
        return logits
    generated = set(output_so_far[0].tolist())

    for token_id in generated:
        # 对出现过的 token 调整 logits
        if logits[0, token_id] < 0:
            logits[0, token_id] *= repetition_penalty  # 负数扩大（更负）
        else:
            logits[0, token_id] /= repetition_penalty  # 正数缩小（降低概率）

    return logits

def analyze_sentence_structure(text):
    """分析句子结构，返回标点位置和句子数量"""
    sentence_ends = [i for i, char in enumerate(text) if char in "。！？"]
    commas = [i for i, char in enumerate(text) if char in "，；"]
    return {
        "sentence_count": len(sentence_ends) + 1,
        "has_sentence_end": bool(sentence_ends),
        "last_punctuation": text[-1] if text and text[-1] in "。！？，；" else None,
        "distance_from_last_sentence": len(text) - sentence_ends[-1] if sentence_ends else len(text)
    }

def build_cn_mask(tokenizer):
    # 使用模型的实际词汇表大小
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    chinese_mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        try:
            tok = tokenizer.convert_ids_to_tokens(i)
            # 检查是否为中文字符或常用标点
            if tok is not None and isinstance(tok, str):
                if any('\u4e00' <= ch <= '\u9fff' for ch in tok):
                    chinese_mask[i] = True
                elif any(ch in "，。！？、；：‘’“”（）【】—《》" for ch in tok):
                    chinese_mask[i] = True
        except Exception:
            # 忽略无法解码的token
            pass
    return chinese_mask

def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.005,
        temperature=0.9,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.95,
        kl_scale=0.05,
        verbosity_level=REGULAR,
        user_vocab_path="user_vocab.txt",
        sentiment_words=None,
        sentiment_beta=0.8,  # 添加情感扰动系数参数
        early_stop=True,  # 控制是否启用提前停止
        min_length=10,    # 控制最小生成长度

):
    # 初始化词袋词冷却机制
    word_cooldown = {}
    cooldown_duration = 3  # 一个词被选中后冷却3个token
    output_so_far = None
    # 添加这两行初始化代码
    recent_bag_words = []  # 记录最近使用的词袋词
    word_combination_history = []  # 记录词袋词组合
    
    # 获取模型的终止token
    end_token_id = None
    # 尝试获取不同模型可能的终止token
    if hasattr(tokenizer, 'eos_token_id'):
        end_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, 'sep_token_id'):
        end_token_id = tokenizer.sep_token_id
    elif '</s>' in tokenizer.get_vocab():
        end_token_id = tokenizer.get_vocab()['</s>']
    elif '<|endoftext|>' in tokenizer.get_vocab():
        end_token_id = tokenizer.get_vocab()['<|endoftext|>']
    
    logger.info(f"检测到的终止token ID: {end_token_id}")
    
    # 改进1: 引入词袋词序列检测
    recent_bag_words = []  # 记录最近使用的词袋词
    word_combination_history = []  # 记录词袋词组合
    
    # # 改进1: 引入词袋词序列检测
    # recent_bag_words = []  # 记录最近使用的词袋词
    # word_combination_history = []  # 记录词袋词组合
    
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
        logger.debug(f"output_so_far：{output_so_far}")
    # collect one hot vectors for bags of words
    

    one_hot_bows_vectors, _ = build_bows_one_hot_vectors(bow_indices, tokenizer, device) if bow_indices else (None, None)
    user_vocab_vectors = None
    if user_vocab_path and os.path.exists(user_vocab_path):
        user_vocab_indices = get_user_vocab_indices(tokenizer, user_vocab_path)
        user_vocab_vectors, _ = build_bows_one_hot_vectors(user_vocab_indices, tokenizer, device) if user_vocab_indices else (None, None)
    
    # 直接从sentiment_words创建词袋向量（修复重点）
    sentiment_bows_vectors = None
    if sentiment_words:
        sentiment_bows_vectors = []
        for sentiment_word in sentiment_words:
            # 编码情感词
            word_tokens = tokenizer.encode(sentiment_word.word, add_special_tokens=False)
            if word_tokens:
                # 创建包含整个词的one-hot向量
                one_hot = torch.zeros(1, tokenizer.vocab_size).to(device)
                for token_id in word_tokens:  # 使用所有token，而不仅仅是第一个
                    if token_id < tokenizer.vocab_size:
                        one_hot[0, token_id] = 1.0
                sentiment_bows_vectors.append(one_hot)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    # 初始化past为None
    if past is None:
        past = None
        # 第一次运行模型时，需要处理整个context
        if output_so_far is not None:
            # 只运行一次模型，获取初始的past_key_values
            with torch.no_grad():
                model_output = model(output_so_far, output_hidden_states=True)
                if hasattr(model_output, 'logits'):
                    unpert_past = model_output.past_key_values
                else:
                    unpert_past = model_output[1] if len(model_output) > 1 else None
            past = unpert_past
            # 设置last为最后一个token
            last = output_so_far[:, -1:]
    else:
        # 如果提供了past，确保last正确设置
        if output_so_far is not None:
            last = output_so_far[:, -1:]

##生成新文本
    for i in range_func:
        # 更新所有词的冷却状态
        for word in list(word_cooldown.keys()):
            word_cooldown[word] = max(0, word_cooldown[word] - 1)
        
        # 获取当前上下文文本用于权重调整
        current_text = tokenizer.decode(output_so_far[0].tolist(), skip_special_tokens=True) if output_so_far is not None else ""
        # 提前初始化structure变量，确保在所有路径中都有定义
        structure = analyze_sentence_structure(current_text)
        # 只使用last token和past进行下一步预测，而不是整个output_so_far
        # 这是关键！确保每次只基于最后一个token和历史状态进行预测
        with torch.no_grad():
            unpert_model_output = model(last, past_key_values=past, output_hidden_states=True)
            if hasattr(unpert_model_output, 'logits'):
                unpert_logits = unpert_model_output.logits[:, -1, :]
                unpert_hidden = unpert_model_output.hidden_states[-1] if unpert_model_output.hidden_states else None
                new_past = unpert_model_output.past_key_values
            else:
                unpert_logits = unpert_model_output[0][:, -1, :]
                unpert_hidden = unpert_model_output[2][-1] if len(unpert_model_output) > 2 else None
                new_past = unpert_model_output[1] if len(unpert_model_output) > 1 else None
        
        current_stepsize = stepsize if i < grad_length else 0
            
        # 修改这里的条件判断，确保逻辑清晰
        if perturb and num_iterations > 0:
            logger.info("PPLM is active")
            
            # 直接修改unpert_logits来增加目标词的概率 
            logits = unpert_logits.clone()
            
            # 在应用词袋词权重前保存原始logits
            original_logits = unpert_logits.clone()
            
            # 导入情感分析器
            from .words_sentiment import sentiment_analyzer
            
            # 获取当前上下文文本
            current_context = tokenizer.decode(output_so_far[0].tolist(), skip_special_tokens=True) if output_so_far is not None else ""
            
            # 1. 处理词袋词 - 统一处理从--bag_of_words参数和user_vocab.txt文件中获取的词
            user_words = []
            vocab_source = ""
            
            # 首先检查是否有--bag_of_words参数
            bow_words_from_param = []
            if bow_indices is not None and len(bow_indices) > 0:
                logger.info(f"Applying bag of words from --bag_of_words parameter")
                for bow in bow_indices:
                    for token_list in bow:
                        if len(token_list) >= 1:
                            word = tokenizer.decode(token_list, skip_special_tokens=True)
                            if word.strip():
                                bow_words_from_param.append(word.strip())
            
            # 如果有来自参数的词袋词，使用它们
            if bow_words_from_param:
                user_words = bow_words_from_param
                vocab_source = "--bag_of_words parameter"
            # 否则，从user_vocab.txt文件读取
            elif user_vocab_path and os.path.exists(user_vocab_path):
                logger.info(f"No bag of words from parameter, applying user vocab words from {user_vocab_path}")
                try:
                    with open(user_vocab_path, 'r', encoding='utf-8') as f:
                        user_words = [word.strip() for word in f.readlines() if word.strip()]
                    vocab_source = user_vocab_path
                except Exception as e:
                    logger.error(f"Error reading user vocab file: {e}")
            
            # 对所有词袋词应用相同的权重计算和提升逻辑
            if user_words:
                try:
                    # 分析词袋中所有词的情感匹配因子
                    sentiment_results = sentiment_analyzer.analyze_bag_of_words(user_words, current_context, beta=sentiment_beta)
                    
                    # 创建词汇权重映射，为不同类型词汇设置不同权重
                    word_weights = {}
                    for word in user_words:
                        # 检查该词是否在冷却期
                        if word in word_cooldown and word_cooldown[word] > 0:
                            logger.info(f"Skipping word '{word}' due to cooldown (remaining: {word_cooldown[word]})")
                            continue
                        
                        # 改进3: 优化权重值
                        if len(word) == 1:  # 单字
                            base_weight = 15.0  # 略微提高，从10.0到15.0
                        # elif word == "哈哈哈哈":  # 特殊处理重复字符词
                        #     base_weight = 15.0  # 从20.0降低
                        elif len(word) > 2:  # 多字词语
                            base_weight = 20.0  # 从30.0降低
                        else:  # 两字词
                            base_weight = 18.0  # 从25.0降低
                          
                        # 改进4: 更智能的上下文权重调整
                        context_adjustment = 1.0
                        
                        # 根据标点符号位置调整
                        if any(punct in current_text[-3:] for punct in ["。", "！", "？"]):
                            # 句子结束后降低权重，避免直接开始新词袋词
                            context_adjustment *= 0.5  # 进一步降低，从0.7降至0.5
                        elif any(punct in current_text[-3:] for punct in ["，", "；"]):
                            # 逗号后适度提高权重
                            context_adjustment *= 1.2  # 略微提高，从1.1到1.2
                          
                        # 改进5: 词袋词序列检测和权重调整
                        if recent_bag_words and len(recent_bag_words) >= 2:
                            # 如果最近已经使用了多个词袋词，降低权重
                            if len([w for w in recent_bag_words[-3:] if w]) >= 2:
                                context_adjustment *= 0.4  # 更强烈地降低，从0.6降至0.4
                        
                        # 应用情感因子 - 这是关键的修改
                        sentiment_factor = sentiment_results[word]['sentiment_factor'] if word in sentiment_results else 1.0
                        
                        # 计算最终权重：base_weight * context_adjustment * sentiment_factor
                        weight = base_weight * context_adjustment * sentiment_factor
                        
                        # 打印情感分析信息
                        if word in sentiment_results:
                            sentiment_info = sentiment_results[word]
                            logger.info(f"Word: {word}, Sentiment: {sentiment_info['sentiment_direction']}, Intensity: {sentiment_info['intensity']:.2f}, Factor: {sentiment_factor:.2f}")
                        
                        word_weights[word] = weight
                    
                    # 为词汇分配权重
                    for word, weight in word_weights.items():
                        word_tokens = tokenizer.encode(word, add_special_tokens=False)
                        if word_tokens:
                            logger.info(f"Boosting word '{word}' from {vocab_source} with weight {weight}")
                            for token_id in word_tokens:
                                if token_id < logits.size(1):  # 确保token_id有效
                                    logits[:, token_id] += weight
                except Exception as e:
                    logger.error(f"Error processing bag of words: {e}")
            
            # 2. 保留原始的词袋逻辑作为补充（如果需要）
            # if one_hot_bows_vectors:
            #     print("Applying original bag of words logic")
            #     for one_hot_bow in one_hot_bows_vectors:
            #         bow_indices = torch.nonzero(one_hot_bow[0]).squeeze(1)
            #         if bow_indices.numel() > 0:
            #             weight = 50.0  # 使用固定的高强度权重
            #             logits[:, bow_indices] += weight
            
            # 3. 处理sentiment_words中的词（如果有）
            if sentiment_bows_vectors and sentiment_words:
                logger.info(f"Applying sentiment words: {[sw.word for sw in sentiment_words]}")
                for i_bow, (one_hot_bow, sentiment_word) in enumerate(zip(sentiment_bows_vectors, sentiment_words)):
                    bow_indices = torch.nonzero(one_hot_bow[0]).squeeze(1)
                    if bow_indices.numel() > 0:
                        # 调整权重策略
                        weight = 90.0  # 降低基础权重
                        if sentiment_word.word == "哈哈哈哈":
                            weight = 100.0  # 仍然给予较高权重但不那么极端
                        
                        logger.info(f"Boosting sentiment word '{sentiment_word.word}' with weight {weight}")
                        logits[:, bow_indices] += weight
            
            # 计算 KL 散度并调整 logits
            try:
                # 计算概率分布
                original_probs = torch.softmax(original_logits, dim=-1)
                adjusted_probs = torch.softmax(logits, dim=-1)
                
                # 计算 KL 散度
                kl_divergence = torch.sum(original_probs * torch.log(original_probs / adjusted_probs), dim=-1)
                
                # 打印 KL 散度信息
                logger.info(f"KL divergence after logits adjustment: {kl_divergence.item():.4f}")
                
                # 设置最大允许的 KL 散度
                max_kl = 3.0
                if kl_divergence > max_kl:
                    # 计算缩放因子
                    scale_factor = max_kl / kl_divergence
                    logger.info(f"KL divergence exceeds threshold, scaling by factor: {scale_factor:.4f}")
                    
                    # 调整 logits，确保调整幅度不会过大
                    logits = original_logits + (logits - original_logits) * scale_factor
                    logger.info(f"Adjusted logits to control KL divergence")
            except Exception as e:
                logger.error(f"Error calculating KL divergence: {e}")
                # 如果计算 KL 散度失败，使用调整后的 logits
                pass
            
            # 改进6: 优化混合策略，增强上下文连贯性
            # 动态混合比例，根据上下文调整
            if structure["has_sentence_end"] and structure["distance_from_last_sentence"] < 3:
                # 句子刚结束，更倾向于原始模型生成
                mix_ratio = 0.90  # 原始模型占90%
            elif structure["distance_from_last_sentence"] < 10 and structure["has_sentence_end"]:
                # 句子内部，保持高比例的原始模型以确保连贯性
                mix_ratio = 0.85  # 原始模型占85%
            elif len(recent_bag_words) >= 2 and recent_bag_words[-2] and recent_bag_words[-1]:
                # 刚使用了多个词袋词，更倾向于原始模型
                mix_ratio = 0.85  # 原始模型占85%
            else:
                # 正常情况下的混合比例
                mix_ratio = 0.80  # 原始模型占80%，增加原始模型权重以提高上下文连贯性
            
            # 应用混合策略
            logits = (mix_ratio * original_logits) + ((1 - mix_ratio) * logits)
            
            # 鼓励在适当位置添加标点符号
            if structure["distance_from_last_sentence"] > 15 and (structure["last_punctuation"] is None or structure["last_punctuation"] not in "。！？"):
                # 如果句子太长且没有结束，适度鼓励添加标点
                for punct in "，。！？；":
                    punct_tokens = tokenizer.encode(punct, add_special_tokens=False)
                    for token_id in punct_tokens:
                        if token_id < logits.size(1):
                            logits[:, token_id] += 5.0  # 适度提高标点符号概率，从10.0降至5.0
        else: 
            # 非扰动模式，直接使用原始logits
            logits = unpert_logits 
        
        # 改进7: 优化重复词袋词组合检测，更温和的处理方式
        if len(word_combination_history) > 0:
            # 获取最近3个词袋词的组合
            recent_combination = tuple(recent_bag_words[-3:] if len(recent_bag_words) >= 3 else recent_bag_words)
            if len(recent_combination) >= 3:
                # 如果这个组合在历史中出现过，更温和地降低词袋词权重
                if recent_combination in word_combination_history:
                    logger.warning(f"Repeating word combination detected: {recent_combination}")
                    # 不直接降低词袋词概率，而是提高原始模型的权重
                    # 这样可以更好地恢复模型的自然生成能力，而不是简单地抑制某些词
                    if 'original_logits' in locals():
                        logits = (0.9 * original_logits) + (0.1 * logits)  # 更倾向于原始模型
                    else:
                        # 如果没有original_logits，则使用unpert_logits作为原始模型输出
                        logits = (0.9 * unpert_logits) + (0.1 * logits)
        
        # 温度缩放
        logits = logits / temperature
        
        # top_k过滤
        if top_k > 0:
            # 找到top_k个最大值的索引
            topk_values, topk_indices = torch.topk(logits, k=min(top_k, logits.size(1)), dim=-1)
            # 创建一个非常小的值（负无穷）
            min_topk = topk_values[:, -1].unsqueeze(-1).expand_as(logits)
            # 将非top_k的值设置为负无穷
            logits = torch.where(logits >= min_topk, logits, torch.tensor(-float('inf')).to(logits.device))
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1) 
        
        # ========== 添加可视化代码 ==========
        # 1. 显示词袋中关键词汇的概率
        bow_visualization_words = []
        
        # 首先尝试使用传入的bow_indices参数获取词袋词汇
        if bow_indices is not None and len(bow_indices) > 0:
            for bow in bow_indices:
                for token_list in bow:
                    if len(token_list) >= 1:
                        # 解码token列表为原始词汇
                        word = tokenizer.decode(token_list, skip_special_tokens=True)
                        if word.strip():
                            bow_visualization_words.append(word.strip())
            logger.info(f"从传入的词袋获取了{len(bow_visualization_words)}个词汇用于可视化")
        else:
            # 如果没有传入词袋，再尝试从user_vocab.txt文件读取
            user_vocab_path = "user_vocab.txt"  # 确保路径正确
            if os.path.exists(user_vocab_path):
                try:
                    with open(user_vocab_path, 'r', encoding='utf-8') as f:
                        bow_visualization_words = [word.strip() for word in f.readlines() if word.strip()]
                    logger.info(f"从{user_vocab_path}读取了{len(bow_visualization_words)}个词汇用于可视化")
                except Exception as e:
                    logger.error(f"读取user_vocab.txt时出错: {e}")
                    bow_visualization_words = []  # 如果读取失败，使用空列表
            else:
                logger.warning(f"警告: {user_vocab_path} 文件不存在")
                bow_visualization_words = []

        # 如果没有从任何来源获取到词汇，可以设置默认词汇作为备选
        if not bow_visualization_words:
            bow_visualization_words = ["开心", "不错", "挺好", "爱你", "哈哈哈哈"]
            logger.info("使用默认词汇列表进行可视化")

        logger.info("\n词袋词汇概率:")
        for word in bow_visualization_words:
            try:
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if word_tokens:
                    # 计算该词所有token的概率平均值
                    token_probs = [probs[0, token_id].item() if token_id < probs.size(1) else 0.0 for token_id in word_tokens]
                    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
                    logger.info(f"  '{word}': {avg_prob:.6f}")
                else:
                    logger.warning(f"  '{word}': 0.000000 (无法编码)")
            except Exception as e:
                logger.error(f"  '{word}': 0.000000 (错误: {str(e)})")
        
        # 2. 显示top-10个最可能的词及其概率
        logger.info("\nTop-10 最可能的词:")
        try:
            top_probs, top_indices = torch.topk(probs, k=10, dim=-1)
            for i in range(10):
                token_id = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                try:
                    token = tokenizer.decode([token_id])
                    # 过滤掉空白字符和特殊字符
                    if token.strip() or token == ' ':
                        logger.info(f"  {i+1}. '{token}': {prob:.6f}")
                    else:
                        logger.info(f"  {i+1}. [特殊字符]: {prob:.6f}")
                except Exception:
                    logger.info(f"  {i+1}. token_id={token_id}: {prob:.6f}")
        except Exception as e:
            logger.error(f"  显示top词时出错: {str(e)}")
        logger.info("=" * 50)
        # ========== 可视化代码结束 ==========
        
        # 保存原始概率，用于重复检测时重新采样
        pert_probs = probs.clone()
        # 增强重复惩罚
        # 检查是否有重复token
        if output_so_far.size(1) > 0:
            # 获取历史tokens和对应的文本
            history_tokens = output_so_far[0].tolist()
            history_text = tokenizer.decode(output_so_far[0], skip_special_tokens=True)
            
            # 创建一个token频率字典
            token_freq = {}
            for token in history_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            # 对历史中出现的所有token应用惩罚
            for token in set(history_tokens):
                freq = token_freq[token]
                # 基础惩罚系数
                base_penalty = 0.8
                
                # 根据出现频率增加惩罚
                if freq > 3:  # 出现多次的token
                    base_penalty = 0.01  # 更严厉的惩罚
                elif freq > 2:  # 出现三次的token
                    base_penalty = 0.1
                elif freq > 1:  # 出现两次的token
                    base_penalty = 0.3
                
                # 距离越近的token惩罚越强
                if token == history_tokens[-1]:  # 最后一个token
                    base_penalty *= 0.1  # 更强的惩罚
                elif token == history_tokens[-2] and len(history_tokens) > 1:  # 倒数第二个token
                    base_penalty *= 0.3
                
                # 特殊处理高频重复token
                try:
                    token_str = tokenizer.decode([token])
                    # 对单个常用字（如"我"）进行特殊处理
                    if len(token_str) == 1 and any(ch in token_str for ch in ["我", "你", "他", "她", "它", "的"]):
                        # 如果这些单字出现多次，降低概率更多
                        if freq > 2:
                            base_penalty *= 0.05  # 更强的惩罚
                except:
                    pass
                
                probs[:, token] *= base_penalty
            
            # 重新归一化概率
            probs_sum = torch.sum(probs, dim=-1, keepdim=True)
            # 避免除零错误
            if probs_sum.item() > 0:
                probs = probs / probs_sum
            else:
                # 如果概率和为0，重置为均匀分布
                probs = torch.ones_like(probs) / probs.size(1)
        
        # 采样下一个token - 修复缩进，减少缩进级别
        if sample:
            selected_token = torch.multinomial(probs, num_samples=1)
        else:
            _, selected_token = torch.topk(probs, k=1, dim=-1)
        
        # 生成词袋词到token_ids的映射 - 修复缩进
        word_to_token_ids = {}
        for word in bow_visualization_words:
            try:
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if word_tokens:
                    word_to_token_ids[word] = word_tokens
            except:
                pass
        
        # 检查连续重复
        if len(output_so_far[0]) > 0:
            prev_token_str = tokenizer.decode(output_so_far[0][-1].item())
            selected_token_str = tokenizer.decode(selected_token.item())
            
            if selected_token_str == prev_token_str:
                logger.warning(f"Consecutive duplicate token '{selected_token_str}'")
                # 连续重复时重新采样，但这次不考虑词袋词
                temp_logits = logits.clone()
                # 将词袋词的概率降低
                for word in bow_visualization_words:
                    if word in word_to_token_ids:
                        for token_id in word_to_token_ids[word]:
                            if token_id < temp_logits.size(1):  # 确保token_id有效
                                temp_logits[0, token_id] = -float('inf')
                # 重新采样
                temp_probs = F.softmax(temp_logits, dim=-1)
                selected_token = torch.multinomial(temp_probs, num_samples=1)
                selected_token_str = tokenizer.decode(selected_token.item())
                logger.info(f"Resampled to: '{selected_token_str}'")
        
        # 检查短循环模式（ABAB） - 修复缩进
        if len(output_so_far[0]) >= 3:
            tokens = output_so_far[0][-3:].tolist() + [selected_token.item()]
            if tokens[0] == tokens[2] and tokens[1] == tokens[3]:
                logger.warning(f"Short loop detected (ABAB pattern)!")
                # 短循环时重新采样，选择一个不在最近历史中的词
                temp_logits = logits.clone()
                # 降低最近4个token的概率
                for token_id in tokens:
                    if token_id < temp_logits.size(1):  # 确保token_id有效
                        temp_logits[0, token_id] -= 10.0  # 大幅降低概率
                # 重新采样
                temp_probs = F.softmax(temp_logits, dim=-1)
                selected_token = torch.multinomial(temp_probs, num_samples=1)
            
            # 将selected_token重命名为last以保持与原有代码兼容
            last = selected_token
            
            # 检查是否选中了词袋中的词，更新冷却状态
            selected_token_str = tokenizer.decode([last.item()])
            
            # 改进8: 更新词袋词使用历史和组合历史
            is_bag_word_selected = False
            
            # 检查user_vocab中的词是否被选中
            if user_vocab_path and os.path.exists(user_vocab_path):
                try:
                    with open(user_vocab_path, 'r', encoding='utf-8') as f:
                        user_words = [word.strip() for word in f.readlines() if word.strip()]
                    
                    # 检查是否选中了词袋中的词
                    for word in user_words:
                        # 检查完整词匹配或者token是词的一部分
                        if word in selected_token_str or selected_token_str in word:
                            word_cooldown[word] = cooldown_duration
                            logger.info(f"Word '{word}' selected, setting cooldown for {cooldown_duration} tokens")
                            # 更新最近使用的词袋词列表
                            recent_bag_words.append(word)
                            # 保持列表长度为10
                            if len(recent_bag_words) > 10:
                                recent_bag_words = recent_bag_words[-10:]
                            # 更新词袋词组合历史
                            if len(recent_bag_words) >= 3:
                                combination = tuple(recent_bag_words[-3:])
                                word_combination_history.append(combination)
                                # 限制历史记录长度
                                if len(word_combination_history) > 10:
                                    word_combination_history = word_combination_history[-10:]
                            is_bag_word_selected = True
                            break
                    
                    # 如果没有选中词袋词，添加None表示非词袋词
                    if not is_bag_word_selected:
                        recent_bag_words.append(None)
                        # 保持列表长度
                        if len(recent_bag_words) > 10:
                            recent_bag_words = recent_bag_words[-10:]
                except Exception as e:
                    logger.error(f"Error checking selected words: {e}")
            
            # 更新输出 
            output_so_far = torch.cat((output_so_far, last), dim=1) 
            decoded_new_token = selected_token_str
            logger.info(f"Sampled token: '{decoded_new_token}'") 
            current_text = tokenizer.decode(output_so_far[0], skip_special_tokens=True)
            logger.debug(f"Current output: {current_text}")
            
            # 更新past，这是最重要的一步！确保下一轮基于正确的历史状态
            past = new_past
            
            # 新增：基于终止token的停止机制
            # 1. 检查是否生成了终止token
            if end_token_id is not None and last.item() == end_token_id:
                logger.info(f"检测到终止token，提前停止生成。最终输出长度: {len(output_so_far[0])}")
                break
            
            # 降低生成连续标点的概率 - 动态惩罚版
            if output_so_far is not None and output_so_far.shape[1] > 0:
                last_token = output_so_far[0, -1].item()
                last_decoded = tokenizer.decode([last_token], skip_special_tokens=False)
                
                # 定义标点符号集合
                punct_chars = set(["，", "。", "！", "？", "；", "：", "、", ",", ".", "!", "?", ";", ":", "/", "…", "⋯"])
                
                # 计算当前连续标点的数量，用于动态调整惩罚强度
                consecutive_punct_count = 1 if last_decoded in punct_chars else 0
                
                # 向前查找连续标点的数量
                max_lookback = 5
                for i in range(2, max_lookback + 1):
                    if output_so_far.shape[1] >= i:
                        prev_token = output_so_far[0, -i].item()
                        prev_decoded = tokenizer.decode([prev_token], skip_special_tokens=False)
                        if prev_decoded in punct_chars:
                            consecutive_punct_count += 1
                        else:
                            break
                
                # 1. 动态调整连续标点的惩罚强度：连续标点越多，惩罚越强
                if consecutive_punct_count > 0:
                    # 基础惩罚强度
                    base_penalty = 20.0
                    # 动态惩罚：指数增长，连续标点越多惩罚越强
                    dynamic_penalty = base_penalty * (1.5 ** consecutive_punct_count)
                    
                    logger.info(f"动态惩罚应用: 连续标点 {consecutive_punct_count} 个，惩罚强度 {dynamic_penalty:.1f}")
                    
                    for punct in punct_chars:
                        punct_tokens = tokenizer.encode(punct, add_special_tokens=False)
                        for token_id in punct_tokens:
                            if token_id < logits.size(1):
                                logits[:, token_id] -= dynamic_penalty
                
                # 2. 特殊处理: 更广泛的非标点字符增强
                if last_decoded in punct_chars:
                    # 根据标点类型调整增强力度
                    if last_decoded in ["。", "！", "？"]:  # 句末标点后增强更强
                        non_punct_bonus = 5.0
                    else:  # 其他标点
                        non_punct_bonus = 3.0
                    
                    # 对所有token应用，除了标点符号
                    for i in range(logits.size(1)):
                        token_str = tokenizer.decode([i], skip_special_tokens=False)
                        if token_str not in punct_chars:
                            logits[:, i] += non_punct_bonus
            
            # 2. 如果没有明确的终止token，可以实现基于困惑度或句子完整性的智能停止
            elif early_stop and i >= min_length - 1:
                # 当达到最小长度后，尝试判断生成是否自然结束
                current_tokens = output_so_far[0].tolist()
                current_decoded = tokenizer.decode(current_tokens, skip_special_tokens=False)
                
                # 计算文本中的句号、感叹号、问号数量
                sentence_end_count = sum(1 for punct in ["。", "！", "？"] if punct in current_decoded)
                
                # 增强的连续标点符号检测
                has_consecutive_puncts = False
                consecutive_punct_count = 0
                # 定义标点符号集合
                punct_chars = set(["，", "。", "！", "？", "；", "：", "、"])
                # 检查最后5个字符中是否有连续标点
                lookback_window = min(5, len(current_decoded))
                for i in range(1, lookback_window + 1):
                    if len(current_decoded) >= i and current_decoded[-i] in punct_chars:
                        consecutive_punct_count += 1
                    else:
                        break
                
                # 更严格的连续标点检测：两个或更多连续标点
                has_consecutive_puncts = consecutive_punct_count >= 2
                
                # 检测是否是单字加句号的结尾（不自然的结尾）
                has_single_char_end = False
                if len(current_decoded) >= 2:
                    # 检查最后一个字符是句末标点，且倒数第二个字符不是标点
                    if (current_decoded[-1] in ["。", "！", "？"] and 
                        current_decoded[-2] not in punct_chars and
                        # 倒数第二个字符是单个汉字
                        len(current_decoded[-2]) == 1):
                        has_single_char_end = True
                
                # 增强的句子完整性检测
                is_complete_sentence = False
                if len(current_decoded) >= 3:
                    # 查找最后一个句末标点的位置
                    last_end_pos = -1
                    for i in range(len(current_decoded)-1, -1, -1):
                        if current_decoded[i] in ["。", "！", "？"]:
                            last_end_pos = i
                            break
                    
                    # 如果找到了句末标点，检查标点前是否有足够的内容
                    if last_end_pos > 0:
                        # 计算标点前的内容长度（去除前导空格）
                        content_before = current_decoded[:last_end_pos].rstrip()
                        # 更严格的完整性检查：
                        # 1. 标点前至少有3个字符
                        # 2. 不是只有标点符号
                        # 3. 包含至少一个汉字
                        has_chinese_char = any('\u4e00' <= c <= '\u9fff' for c in content_before)
                        if len(content_before) >= 3 and not all(c in punct_chars for c in content_before) and has_chinese_char:
                            is_complete_sentence = True
                
                # 改进的语义单元检查
                has_meaningful_content = False
                if len(current_decoded) >= 3:
                    # 计算文本中的有效词语数量和比例
                    char_count = sum(1 for c in current_decoded if '\u4e00' <= c <= '\u9fff')
                    total_count = len([c for c in current_decoded if c.strip()])
                    # 更严格的语义检查：
                    # 1. 汉字比例较高
                    # 2. 文本中有足够的字符
                    # 3. 文本中包含至少一个句末标点或逗号（表示一定的语法结构）
                    has_punctuation = any(punct in current_decoded for punct in ["。", "！", "？", "，", "；", "："])
                    if (char_count >= 3 and 
                        char_count / max(total_count, 1) >= 0.4 and
                        (has_punctuation or total_count >= 5)):
                        has_meaningful_content = True
                
                # 检查是否有重复模式
                has_repetitive_pattern = False
                # 检测连续重复的词语
                words = re.findall(r'[\u4e00-\u9fa5]+', current_decoded)
                if len(words) >= 3:
                    # 检查连续三个词是否有重复
                    for i in range(len(words) - 2):
                        if words[i] == words[i+2]:
                            has_repetitive_pattern = True
                            break
                
                # 优化的停止条件：
                # 1. 更灵活的结束条件，允许两种情况：
                #    a. 较短但结构完整的文本（有句末标点且结构完整）
                #    b. 较长且有意义的文本（即使没有句末标点）
                # 2. 降低最小长度要求，使文本更容易自然结束
                # 3. 确保内容有意义且避免不自然的结尾
                # 4. 避免重复模式
                if ((len(current_decoded) >= min_length * 1.5 and 
                     any(punct in current_decoded[-3:] for punct in ["。", "！", "？"]) and
                     sentence_end_count >= 1 and
                     not has_consecutive_puncts and
                     not has_single_char_end and
                     is_complete_sentence and
                     not has_repetitive_pattern) or
                    (len(current_decoded) >= min_length * 2.0 and
                     has_meaningful_content and
                     not has_consecutive_puncts and
                     not has_repetitive_pattern)):  # 允许更长但没有结束标点的有意义文本
                    # 只有当文本满足所有条件时才停止，确保生成的内容完整自然
                    logger.info(f"检测到自然结束，提前停止生成。最终输出长度: {len(output_so_far[0])}")
                    break
        # 检查重复token并重新采样
            if output_so_far is not None and output_so_far.shape[1] > 2:
                # 检查是否形成了短循环（如ABAB模式）
                if output_so_far.shape[1] >= 4:
                    # 检查最后两个token是否与前两个token相同
                    if (last.item() == output_so_far[0, -3].item() and 
                        output_so_far[0, -2].item() == output_so_far[0, -4].item()):
                        logger.warning(f"Short loop detected!")
                        # 更严格地惩罚最近的三个token
                        for i in range(1, 4):
                            if i <= output_so_far.shape[1] and torch.sum(probs) > 0:
                                probs[:, output_so_far[0, -i].item()] = 0
                                probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                        # 重新采样
                        if sample:
                            last = torch.multinomial(probs, num_samples=1)
                        else:
                            _, last = torch.topk(probs, k=1, dim=-1)
                        # 更新输出
                        output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
                
                # 检查是否有相邻重复的token（除了标点）
                if output_so_far.shape[1] >= 2:
                    last_token = output_so_far[0, -1].item()
                    prev_token = output_so_far[0, -2].item()
                    
                    # 解码token检查是否是标点
                    last_token_str = tokenizer.decode([last_token], skip_special_tokens=False)
                    is_punct = any(p in last_token_str for p in [".", "。", "!", "！", "?", "？", ",", "，", ";", "；", ":", "："])
                    
                    # 如果相邻token重复且不是标点，重新采样
                    if last_token == prev_token and not is_punct:
                        logger.warning(f"Consecutive duplicate token detected!")
                        # 将重复token的概率设为0
                        if last_token < probs.size(1):
                            probs[:, last_token] = 0
                            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                        # 重新采样
                        if sample:
                            last = torch.multinomial(probs, num_samples=1)
                        else:
                            _, last = torch.topk(probs, k=1, dim=-1)
                        # 更新输出
                        output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
            
            # 检查最后两个token是否相同
            elif last.item() == output_so_far[0, -2].item():
                logger.warning(f"Repeating token detected: {last.item()}")
                # 将该token概率设为0并重新归一化
                if torch.sum(probs) > 0:
                    probs[:, last.item()] = 0
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                    
                    # 尝试使用原始概率分布的次优选择
                    if sample:
                        last = torch.multinomial(probs, num_samples=1)
                    else:
                        _, last = torch.topk(probs, k=1, dim=-1)
                    # 更新输出
                    output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
            
            # 特殊处理重复字符组成的token（如"哈哈哈哈"）
            try:
                token_str = tokenizer.decode([last.item()])
                if len(token_str) > 1 and all(c == token_str[0] for c in token_str):
                    # 检查这个token是否在最近的输出中已经出现过
                    recent_tokens = output_so_far[0, -5:].tolist() if output_so_far.shape[1] >= 5 else output_so_far[0].tolist()
                    if last.item() in recent_tokens[:-1]:  # 除了当前token外的最近几个token
                        logger.warning(f"Repeating repetitive token detected!")
                        # 非常严格地惩罚这个token
                        if torch.sum(probs) > 0:
                            probs[:, last.item()] = 0
                            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                            # 重新采样
                            if sample:
                                last = torch.multinomial(probs, num_samples=1)
                            else:
                                _, last = torch.topk(probs, k=1, dim=-1)
                            # 更新输出
                            output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
            except:
                pass
            
            # 改进9: 增强短循环检测，特别是针对词袋词之间的循环
            if output_so_far.shape[1] >= 6:
                # 检查更复杂的循环模式，如ABCABC或ABABAB
                tokens = output_so_far[0].tolist()
                # 检查ABCABC模式
                if tokens[-6] == tokens[-3] and tokens[-5] == tokens[-2] and tokens[-4] == tokens[-1]:
                    logger.warning("Warning: Extended loop detected (ABCABC pattern)!")
                    # 降低最近6个token的概率
                    for i in range(1, 7):
                        if i <= output_so_far.shape[1] and torch.sum(probs) > 0:
                            probs[:, output_so_far[0, -i].item()] = 0
                            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                    # 重新采样
                    if sample:
                        last = torch.multinomial(probs, num_samples=1)
                    else:
                        _, last = torch.topk(probs, k=1, dim=-1)
                    # 更新输出
                    output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
                    # 重置最近词袋词历史，避免继续循环
                    recent_bag_words = []
            
            # 改进10: 增强对连续词袋词的惩罚
            if len(recent_bag_words) >= 4:
                # 计算最近连续的词袋词数量
                consecutive_bag_words = 0
                for w in reversed(recent_bag_words):
                    if w is not None:
                        consecutive_bag_words += 1
                    else:
                        break
                # 如果连续出现3个以上词袋词，强制重新采样
                if consecutive_bag_words >= 3:
                    logger.warning("Warning: Too many consecutive bag words detected!")
                    # 创建临时logits，将所有词袋词概率设为极低
                    temp_logits = logits.clone()
                    if user_vocab_path and os.path.exists(user_vocab_path):
                        try:
                            with open(user_vocab_path, 'r', encoding='utf-8') as f:
                                user_words = [word.strip() for word in f.readlines() if word.strip()]
                            # 降低所有词袋词的概率
                            for word in user_words:
                                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                                for token_id in word_tokens:
                                    if token_id < temp_logits.size(1):
                                        temp_logits[:, token_id] = -float('inf')
                            # 重新计算概率并采样
                            temp_probs = F.softmax(temp_logits, dim=-1)
                            if torch.sum(temp_probs) > 0:
                                last = torch.multinomial(temp_probs, num_samples=1)
                                output_so_far = torch.cat((output_so_far[:, :-1], last), dim=1)
                                recent_bag_words.append(None)  # 记录这次为非词袋词
                        except:
                            pass
        
        if verbosity_level >= REGULAR:
            logger.debug(tokenizer.decode(output_so_far.tolist()[0], skip_special_tokens=True))
    # 修改返回语句，根据perturb参数返回正确的值
    # 当perturb=True时，返回output_so_far作为扰动文本
    # 这样可以确保返回值与调用期望一致
    if perturb:
        return output_so_far, output_so_far, None
    else:
        return output_so_far, None, None


def evaluate_vocab_usage(text, user_vocab):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for token in tokens if token in user_vocab) / len(tokens)

def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta

model_cache_dir = r'C:\Users\23921\.cache\modelscope\Qwen3-4B'
# with open(os.path.join(model_cache_dir, "refs", "main"), "r") as f:
#     commit_id = f.read().strip()

# model_name = os.path.join(model_cache_dir, "snapshots", commit_id)
model_name = model_cache_dir
def run_pplm_example(
        pretrained_model="Qwen/Qwen3-4B",
        cond_text="你喜欢我吗",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.005,
        temperature=1.2,
        top_k=50,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.05,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular',
        user_vocab=None,
        sentiment_bag_of_words=None,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                logger.info("discrim = {}, pretrained_model set to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, output_hidden_states=True)

    model.to(device)
    model.eval()
    # print(f"模型已使用device_map自动分配到设备")  # 注释掉中间输出

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False
    sentiment_words = []
    # figure out conditioning text
    if uncond:
        if tokenizer.bos_token:
            tokenized_cond_text = tokenizer.encode(
                [tokenizer.bos_token],
                add_special_tokens=False
            )
        else:
            # 如果没有bos_token，使用空的开始标记或模型默认的开始方式
            tokenized_cond_text = tokenizer.encode(
                [],
                add_special_tokens=False
            )
    else:
        raw_text = cond_text
        while not raw_text:
            # print("Did you forget to add `--cond_text`? ")  # 注释掉中间输出
            raw_text = input("Model prompt >>> ")
        if tokenizer.bos_token:
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + raw_text,
                add_special_tokens=False
            )
        else:
            tokenized_cond_text = tokenizer.encode(
                raw_text,
                add_special_tokens=False
            )

    # print("= Prefix of sentence =")  # 注释掉中间输出
    # print(tokenizer.decode(tokenized_cond_text))  # 注释掉中间输出
    # print()  # 注释掉中间输出
    if sentiment_words is None:
        sentiment_words = []
    if sentiment_bag_of_words:
        # 假设格式为 "词1:情感;词2:情感"，情感为-1,0,1
        for item in sentiment_bag_of_words.split(";"):
            if ":" in item:
                word, sent_str = item.split(":")
                sentiment = int(sent_str)
                sentiment_words.append(SentimentWord(word, sentiment))
    
    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level,
        user_vocab_path=user_vocab,
        sentiment_words=sentiment_words
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0],skip_special_tokens=True)
    # print("test\n")  # 注释掉中间输出
    # print(tokenizer.encode("哈哈哈哈", add_special_tokens=False))  # 注释掉中间输出
    # print(tokenizer.decode(tokenizer.encode("哈哈哈哈", add_special_tokens=False)))  # 注释掉中间输出
    if verbosity_level >= REGULAR:
        # print("=" * 80)  # 注释掉中间输出
        pass
    # 只保留最终生成的文本
    logger.info(f"未扰动生成的文本: {unpert_gen_text}")
    # print(unpert_gen_text)  # 注释掉未扰动文本输出，只输出扰动后的文本

    generated_texts = []

    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ''
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id],skip_special_tokens=True)
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0],skip_special_tokens=True)

            # logger.info(f"= Perturbed generated text {i + 1} =")
            logger.info(f"扰动生成的文本: {pert_gen_text}")
            try:
                print(pert_gen_text)  # 保留最终输出以便compare_pplm_models.py解析
            except Exception as e:
                logger.error(f"打印输出时出错: {e}")
                # 尝试使用不同的编码方式打印
                try:
                    print(pert_gen_text.encode('utf-8').decode('gbk', errors='replace'))
                except Exception as e2:
                    logger.error(f"使用gbk编码打印也出错: {e2}")
        except Exception as e:
            logger.error(f"生成或处理文本时出错: {e}")

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default=r'C:\Users\23921\.cache\modelscope\Qwen3-4B',
        help="pretrained model name or path to local checkpoint，默认为本地Qwen3-4B模型",
    )
    parser.add_argument(
        "--cond_text", type=str, default="你喜欢我吗",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default="C:/Users/23921/PycharmProjects/pythonProject/PPLM/PPLM/user_vocab.txt",
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stepsize", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument(
        "--user_vocab",
        type=str,
        default=None,
        help="Path to user vocabulary file for personalized word preference"
    )
    parser.add_argument("--sentiment_bag_of_words", type=str, default=None,
                    help="情感词袋，格式：词1:情感;词2:情感，情感为-1(消极),0(中性),1(积极)")
    args = parser.parse_args()
    run_pplm_example(**vars(args))

# from transformers import AutoTokenizer, AutoModelForCausalLM
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = r"C:\Users\23921\.cache\modelscope\Qwen3-4B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# # 1. 加载中文 GPT2


# model.eval()

# # 2. 定义BOW
# bow_words = ["哈哈哈哈"]
# bow_indices = [tokenizer.encode(w, add_special_tokens=False) for w in bow_words]
# print("BOW indices:", bow_indices)
# print("Check decode:", [tokenizer.decode(ids) for ids in bow_indices])

# # 3. Prompt
# prompt_text = "今天心情不错，"
# input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

# # 4. 不带干预的生成
# with torch.no_grad():
#     output = model.generate(
#         input_ids,
#         max_length=100,
#         do_sample=True,
#         top_k=50,
#         top_p=0.9
#     )

# print("Unperturbed:", tokenizer.decode(output[0], skip_special_tokens=True))