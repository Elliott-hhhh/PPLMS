# PPLM with Chinese Extensions

This repository contains code to run the Plug and Play Language Model (PPLM), with additional extensions for Chinese text generation tasks using the CPED (Chinese Poetry Emotion Dataset).

## Project Structure

```
PPLMS/
├─ cped_experiment/          # CPED dataset comparison experiments
│  ├─ character_wordbags/    # Character word bag data
│  ├─ compare_pplm_models.py # Main model comparison script
│  ├─ extract_character_wordbags.py # Character word bag extraction
│  ├─ run_cped_experiment.py # Experiment execution script
│  ├─ cped_experiment_config.py # Experiment configuration
│  ├─ analyze_cped_results.py # Results analysis script
│  ├─ prepare_cped_dataset.py # Dataset preprocessing
│  ├─ train_cped_discriminator.py # Discriminator training
│  └─ CPED实验方案总结.md # Chinese experiment documentation
│
├─ ppls_model/               # PPLS (Modified PPLM) model
│  ├─ generate_ppls.py       # Enhanced PPLM generator
│  └─ words_sentiment.py     # Sentiment analysis module
│
├─ paper_code/               # Original PPLM paper implementation
│  ├─ discrim_models/        # Discriminator models
│  ├─ pytorch_pretrained_bert/ # BERT pretrained models
│  └─ wordlists/             # Word lists for bag-of-words
│
├─ processed_cped/           # Processed CPED dataset
│  └─ bow_files/             # Sentiment bag-of-words files
│
└─ other directories...      # Original PPLM files
```

## Key Features

### 1. PPLS Model (Modified PPLM)
- **Enhanced Chinese Support**: Optimized for Chinese text generation with Qwen3-4B model
- **Sentiment Control**: Generate text with specific emotional tones using sentiment word bags
- **Personalized Vocabulary**: Support for user-defined vocabulary to guide text generation
- **Context-Aware Optimization**: Dynamically adjusts control strength based on generated content
- **Word Cooling Mechanism**: Prevents over-intervention by controlling word usage frequency
- **Direct Logits Modification**: Improved efficiency by removing complex gradient calculations

### 2. CPED Dataset Experiments
- Complete experimental framework for Chinese sentiment control
- Model comparison between vanilla GPT2, original PPLM, and modified PPLM
- Comprehensive evaluation metrics: perplexity, diversity, sentiment accuracy
- Automated experiment execution and result analysis

### 3. Core Innovations

1. **Weight Adjustment Mechanism**
   ```python
   weight = base_weight * context_adjustment * sentiment_factor
   ```

2. **Word Cooling Mechanism** - Controls intervention frequency to avoid over-control
3. **Context-Aware Optimization** - Dynamically adjusts control strength based on generated content
4. **Direct Logits Modification** - Removes complex gradient calculations for improved efficiency

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage Examples

#### Basic Text Generation

```bash
python -m ppls_model.generate_ppls --cond_text "你喜欢我吗" --length 20 --stepsize 0.005 --temperature 0.9 --top_k 100 --num_samples 1 --num_iterations 3
```

#### Sentiment Control Generation

```bash
python -m ppls_model.generate_ppls --sentiment_bag_of_words "高兴:1;快乐:1;难过:-1;悲伤:-1" --cond_text "今天天气" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --sample
```

#### Personalized Vocabulary Generation

```bash
python -m ppls_model.generate_ppls --user_vocab "path/to/your/vocab.txt" --cond_text "我的梦想是" --length 50 --stepsize 0.04 --num_iterations 5 --num_samples 5
```

## CPED Dataset Experiments

### Experimental Workflow

1. **Dataset Preprocessing**
   ```bash
   python -m cped_experiment.prepare_cped_dataset --input_file ./data/cped/train_split.csv --output_dir ./processed_cped
   ```

2. **Train Sentiment Discriminator**
   ```bash
   python -m cped_experiment.train_cped_discriminator --data_dir ./processed_cped --output_dir ./cped_discriminator
   ```

3. **Run Comparison Experiments**
   ```bash
   python -m cped_experiment.run_cped_experiment --experiments all --output_dir ./output/experiments
   ```

4. **Analyze Results**
   ```bash
   python -m cped_experiment.analyze_cped_results --results_dir ./output/experiments --output_dir ./output/analysis
   ```

### Evaluation Metrics

1. **Perplexity** - Measures text fluency
2. **Diversity Metrics** - Vocabulary diversity, n-gram diversity
3. **Sentiment Control Accuracy** - How well generated text matches target sentiment
4. **Generation Speed** - Average time per generated text

## Model Comparison

| Model Type | Advantages | Disadvantages |
|---------|------|------|
| vanilla_gpt2 | Fast generation, high fluency | No sentiment/topic control |
| pplm_bow | Good topic control, efficient | Less precise sentiment control |
| pplm_discrim | Precise sentiment control | Higher computational complexity |
| modified_pplm | Balances control and fluency, optimized for Chinese | Requires hyperparameter tuning |

## Citation

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

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Additional Resources

- [中文README](README_CN.md) - Chinese version of this documentation
- [CPED实验方案总结.md](cped_experiment/CPED实验方案总结.md) - Detailed Chinese experiment documentation
- [Original PPLM Paper](https://arxiv.org/abs/1912.02164)
- [PPLM Blog Post](https://eng.uber.com/pplm)
- [PPLM Colab Notebook](https://colab.research.google.com/drive/1Ux0Z4-ruiVtJ6jUk98uk6FqfvGHCOYL3)
