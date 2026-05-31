# Memory-Efficient LLMs Training with Dynamic Sparsity: From Stability to Practical Scaling


This is the offical implementation for paper titled [Memory-Efficient LLMs Training with Dynamic Sparsity: From Stability to Practical Scaling]().

## Abstract

Dynamic Sparse Training (DST) offers a promising paradigm for improving the training and inference efficiency of deep neural networks; however, we find that in large language model training, DST can suffer from optimization instability, manifested as loss spikes after topology updates. In this work, we show that the naive use of standard Adam-based optimizers leads to a cold-start issue for newly regrown parameters, resulting in excessively large updates and disrupted training dynamics. To address this issue, we propose Sparse Memory-Efficient Training (SMET), which stabilizes DST with optimizer warm-up and improves training progress through density-aware learning-rate scaling. SMET further reduces memory consumption by storing gradients and optimizer states only for active parameters. We provide a theoretical analysis of the update behaviors under SMET, showing improved optimization stability.
Extensive experiments demonstrate that SMET enables stable, scalable, and memory-efficient sparse pre-training of LLMs, paving the way for sparse training as a practical alternative to dense training.

## Requirements
- torch==2.1.0+cu118
- transformers==4.38.0
- huggingface-hub==0.27.0
- datasets==2.19.1

## Usage

```python
from galore_torch import SPAM
# define param groups as spam_params and non_spam_params
param_groups = [{'params': non_spam_params}, 
                {'params': spam_params, 'density': 1.0}]
optimizer = SPAM(param_groups, lr=0.001,warmup_steps=150,threshold=5000,DeltaT=500)
```

### Example 1: Training LLaMA-130M 

```
torchrun --standalone --nproc_per_node=4 \
    torchrun_main_unit.py \
    --wandb_mode disabled \
    --model_config configs/llama_130m.json \
    --density 0.5 \
    --val_dir None \
    --data_dir None \
    --lr 2e-3 \
    --optimizer adamdst \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --epochs 1 \
    --eval_every 2000 \
    --dtype bfloat16 \
```

### Example 2: Training LLaMA-350M 

```
torchrun --standalone --nproc_per_node=4 \
    torchrun_main_unit.py \
    --wandb_mode disabled \
    --model_config configs/llama_350m.json \
    --density 0.5 \
    --val_dir None \
    --data_dir None \
    --lr 2e-3 \
    --optimizer adamdst \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --epochs 1 \
    --eval_every 2000 \
    --dtype bfloat16 \
```

## Acknowledgement
This repository is build upon the  [GaLore](https://github.com/jiaweizzhao/GaLore) repository. Thanks for the great work!
