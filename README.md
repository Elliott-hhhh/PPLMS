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
│  └─ generate_ppls.py       # Enhanced PPLM generator
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

### 1. Original PPLM Functionality
- Bag-of-Words (BoW) control for text generation
- Discriminator-based control for sentiment and topics
- Support for multiple control objectives
- Comprehensive hyperparameter tuning guide

### 2. Chinese Extensions
- Enhanced support for Chinese text generation
- Integration with CPED (Chinese Poetry Emotion Dataset)
- Modified PPLM model (PPLS) with improved Chinese support
- Complete experimental framework for Chinese sentiment control

### 3. PPLS (Modified PPLM) Innovations

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

#### Bag-of-Words Control

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample
```

#### Discriminator Control

```bash
python run_pplm.py -D sentiment --class_label 2 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

## CPED Dataset Experiments

### Experimental Workflow

1. **Dataset Preprocessing**
   ```bash
   python cped_experiment/prepare_cped_dataset.py --input_file ./data/cped/train_split.csv --output_dir ./processed_cped
   ```

2. **Train Sentiment Discriminator**
   ```bash
   python cped_experiment/train_cped_discriminator.py --data_dir ./processed_cped --output_dir ./cped_discriminator
   ```

3. **Run Comparison Experiments**
   ```bash
   python cped_experiment/run_cped_experiment.py --experiments all --output_dir ./output/experiments
   ```

4. **Analyze Results**
   ```bash
   python cped_experiment/analyze_cped_results.py --results_dir ./output/experiments --output_dir ./output/analysis
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