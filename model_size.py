import os
import time
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoTokenizer, default_data_collator
import datasets
import datasets.distributed
import wandb
from tqdm import tqdm
from loguru import logger
from datasets import load_from_disk

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from sparselearning.core import Masking

transformers.logging.set_verbosity_error()


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--loss_every", type=int, default=10)

    parser.add_argument("--total_training_steps", type=int, default=10_000,
                        help="The total of **update steps** to train for. "
                             "Notice that epochs is taken into account.")
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", type=str2bool, default=True, help="Do not save the model")

    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--single_gpu", type=str2bool, default=False, help="Disable DDP and use single GPU")
    parser.add_argument("--console_log", type=str, default="default")

    parser.add_argument('--wandb_used', type=str2bool, default=False, help="Use wandb or not")
    parser.add_argument('--wandb_mode', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma separated list of tags for wandb. Example: 'tag1,tag2' ")
    parser.add_argument("--print_grad_norm", type=str2bool, default=True, help="Print gradient norm")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")

    # Sparsity args
    parser.add_argument('--density', type=float, default=1.0,
                        help="The density of the sparse network. This is the final density if using a non-constant --density_decay.")
    parser.add_argument('--dense_embedding', action='store_true', default=False,
                        help='Leave embedding layer dense. Default: False.')
    parser.add_argument('--ddt', action='store_true', default=False,
                        help='Enable dynamic dense training. Default: False.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N',
                        help='how many iterations to train between mask update')
    parser.add_argument('--growth', type=str, default='random',
                        help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude',
                        help='Pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--reinit', type=str, default='no',
                        help='Weight reinitialization mode. Choose from: no, zero, original.')
    parser.add_argument('--redistribution', type=str, default='none',
                        help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune_rate', type=float, default=0.50, help='The pruning rate.')
    parser.add_argument('--prune_rate_decay', type=str, default='cosine',
                        help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--density_decay', type=str, default='constant',
                        help='The decay schedule for the density. If not constant, will start training with density=1 and decay to --density. Default: constant. Choose from: constant, linear, cosine.')
    parser.add_argument('--initial_density', type=float, default=0.999,
                        help='The initial density for the density decay schedule. Only used when density_decay!=constant. Default: 0.999.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='Multi_Output', help='sparse initialization')
    parser.add_argument('--mix', type=float, default=0.0)
    parser.add_argument('--temperature_decay', type=str, default='constant',
                        help='The decay schedule for the temperature. Choose from: constant, linear.')
    parser.add_argument('--temperature', type=float, default=3,
                        help='The temperature for soft sampling of pruning. (This is the final temperature if using a non-constant --temperature_decay.)')
    parser.add_argument('--init_temperature', type=float, default=1,
                        help='The initial temperature for the temperature decay schedule. Only used when --temperature_decay != constant.')

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args




def build_dataloader(args, global_rank, world_size, tokenizer, epoch):
    logger.info(f"Loading streaming dataset from: {args.data_dir}")
    dataset = datasets.load_dataset("arrow", data_dir=args.data_dir, split="train", streaming=True)

    logger.info(f"Shuffling streaming dataset with seed = {epoch}")
    dataset = dataset.shuffle(seed=epoch)

    # Apply DDP sharding
    if not args.single_gpu:
        logger.info(f"Sharding dataset: rank {global_rank} of {world_size}")
        dataset = datasets.distributed.split_dataset_by_node(
            dataset, rank=global_rank, world_size=world_size
        )

    # Wrap in iterable dataset that applies tokenizer dynamically
    dataset = PreprocessedIterableDataset(
        data=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # already batched inside PreprocessedIterableDataset
        num_workers=args.workers,
        pin_memory=True,
    )

    return dataloader


def main(args):
    start_script_time = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Global rank {global_rank}, local rank {local_rank}, world size {world_size}")

    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0 and args.wandb_used:
        wandb.init(project="dst_llms", name=args.run_name, mode=args.wandb_mode, tags=args.tags)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

    model_config = AutoConfig.from_pretrained(args.model_config)

    model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    args.total_training_steps = int(args.epochs * args.num_training_steps)

    # print params and trainable params
    logger.info(f"\n{model}\n")

    logger.info(f"=== Model Size for {args.model_config} ===")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")



if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
