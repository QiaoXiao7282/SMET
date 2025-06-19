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
from sparselearning.optimizer import AdamDST

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
    parser.add_argument("--activation_checkpointing", type=str2bool, default=False, help="")
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
    parser.add_argument("--tags", type=str, default=None, help="Comma separated list of tags for wandb. Example: 'tag1,tag2' ")
    parser.add_argument("--print_grad_norm", type=str2bool, default=True, help="Print gradient norm")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")

    ## optimizer
    parser.add_argument('--op_decay_steps', type=float, default=20, help='decay steps for the regrow weights.')
    parser.add_argument('--op_decay_max', type=float, default=1.0, help='decay maximum for the regrow weights.')
    parser.add_argument('--accumulate_grad_steps', type=float, default=5, help='accumulate gradient steps.')
    parser.add_argument('--maintain_num', type=float, default=2, help='accumulate gradient number.')

    # Sparsity args
    parser.add_argument('--density', type=float, default=1.0, help="The density of the sparse network. This is the final density if using a non-constant --density_decay.")
    parser.add_argument('--dense_embedding', type=str2bool, default=True, help='Leave embedding layer dense. Default: False.')
    parser.add_argument('--dense_head', type=str2bool, default=True, help='Leave embedding layer dense. Default: False.')
    parser.add_argument('--am_ratio', type=float, default=1.0, help='Attention/mlp ratio. Default: 1.')

    parser.add_argument('--ddt', action='store_true', default=False, help='Enable dynamic dense training. Default: False.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--reinit', type=str, default='no', help='Weight reinitialization mode. Choose from: no, zero, original.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune_rate', type=float, default=0.50, help='The pruning rate.')
    parser.add_argument('--prune_rate_decay', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--density_decay', type=str, default='constant', help='The decay schedule for the density. If not constant, will start training with density=1 and decay to --density. Default: constant. Choose from: constant, linear, cosine.')
    parser.add_argument('--initial_density', type=float, default=0.999, help='The initial density for the density decay schedule. Only used when density_decay!=constant. Default: 0.999.')
    parser.add_argument('--fix', type=str2bool, default=False, help='Fix topology during training. Default: False.')
    parser.add_argument('--sparse_init', type=str, default='Multi_Output', help='sparse initialization')
    parser.add_argument('--mix', type=float, default=0.0)
    parser.add_argument('--temperature_decay', type=str, default='constant', help='The decay schedule for the temperature. Choose from: constant, linear.')
    parser.add_argument('--temperature', type=float, default=3, help='The temperature for soft sampling of pruning. (This is the final temperature if using a non-constant --temperature_decay.)')
    parser.add_argument('--init_temperature', type=float, default=1, help='The initial temperature for the temperature decay schedule. Only used when --temperature_decay != constant.')

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


def save_model(model, optimizer, scheduler, update_step, global_step, run_config,
               tokens_seen, tokens_seen_before, update_time, args):
    """Save the model and optimizer state."""
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.single_gpu:
        model.save_pretrained(current_model_directory, max_shard_size='100GB')
    else:
        model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

    optimizer_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update_step": update_step,
        "global_step": global_step,
        "config": run_config,
        "wandb": wandb.run.dir,
        "dtype": args.dtype,
    }
    torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

    training_state_checkpoint = {
        "global_step": global_step,
        "update_step": update_step,
        "tokens_seen": tokens_seen,
        "tokens_seen_before": tokens_seen_before,
        "update_time": update_time,
    }
    with open(f"{current_model_directory}/training_state.json", "w") as f:
        json.dump(training_state_checkpoint, f, indent=4)

    # save wandb related info
    if args.wandb_used:
        wandb_info = {
            "wandb_id": wandb.run.id,
        }
        with open(f"{args.save_dir}/wandb.json", "w") as f:
            json.dump(wandb_info, f, indent=4)


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    """Evaluate the current model."""
    _time = time.time()

    # Print dtype of model weights to check
    for name, param in model.named_parameters():
        if "lm_head" in name:
            logger.info(f"Parameter {name} has dtype {param.dtype}")

    # val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True) #DGX
    # val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    # val_data = val_data.shuffle(seed=42)

    val_dir = '/scratch-shared/xiaoq/c4_sampling/c4_filtered_validation_10M'
    val_data = datasets.load_dataset("arrow", data_dir=val_dir, split="validation", streaming=True)

    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns= ["text"],  #["text", "timestamp", "url"],
    )
    # val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)
    dataloader = torch.utils.data.DataLoader(
        val_data_mapped,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    # for batch in val_data_mapped.batch(batch_size=batch_size):
    for batch in dataloader:
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        # batch = default_data_collator(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        # Standard, single model
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def get_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type) for p in parameters
        ]),
        norm_type
    )
    return total_norm.item()

def build_dataloader(args, global_rank, world_size, tokenizer, epoch):

    logger.info(f"Loading streaming dataset from: {args.data_dir}")

    if args.data_dir is None:
        dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        seed_for_shuffle = 32
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
        dataset: datasets.Dataset = dataset.shuffle(seed=seed_for_shuffle)
    else:
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
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    pad_idx = tokenizer.pad_token_id

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    model_config = AutoConfig.from_pretrained(args.model_config)

    model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    local_step = 0

    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

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

    if global_rank == 0:
        if args.wandb_used:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.total_training_steps - local_step, desc="Update steps", ncols=80)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if not args.no_save:
        logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    


    mask = None
    use_sparsity = not args.density == 1
    use_sparsity = use_sparsity or args.ddt  # even if dense, we may want dynamic dense training
    if use_sparsity:
        mask = Masking(preprocess_batched, args=args, pad_idx=pad_idx, global_rank=global_rank, world_size=world_size)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

    else:
        if args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        elif args.optimizer.lower() == "adamdst":
            optimizer = AdamDST(trainable_params, lr=args.lr)
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")

        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.total_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
        )

    # global steps and others are defined above
    update_time = time.time()

    setup_time = time.time() - start_script_time
    print(f'Time to setup: {setup_time}')

    ###############################
    # TRAINING LOOP
    # implement epochs !!!
    ###############################
    for epoch in range(args.epochs):

        update_step = 0
        logger.info(f"=== Starting Epoch {epoch + 1} ===")
        dataloader = build_dataloader(args, global_rank, world_size, tokenizer, epoch)

        for batch_idx, batch in enumerate(dataloader):

            global_step += 1

            if update_step > args.num_training_steps:
                logger.info(f"Reached max number of update steps within epoch (f{args.num_training_steps}). Stopping training.")
                print(f"Rank {global_rank} stopping training.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

            # Standard, single model
            loss = model(**batch, labels=labels).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()


            if global_step % args.gradient_accumulation != 0:
                continue

            # The below code is only executed during the update step, after a full gradient accumulation

            # Check gradient norm validity, avoid data corruption
            grad_norm = get_grad_norm(trainable_params)
            # if math.isnan(grad_norm) or math.isinf(grad_norm):
            #     logger.warning(f"Skipping step {global_step} due to invalid grad_norm = {grad_norm}")
            #     if use_sparsity:
            #         mask.optimizer.zero_grad()
            #     else:
            #         optimizer.zero_grad()
            #     continue

            # add grad clipping
            if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

            ## check loss; global gradient norm !!!
            if args.print_grad_norm:
                # grad_norm = get_grad_norm(trainable_params)
                logger.info(f"After clipping {global_step}: loss = {loss.item():.4f}, grad_norm = {grad_norm:.4f}")


            if global_rank == 0: pbar.update(1)

            if use_sparsity:
                mask.step()  # performs optimizer.step() inside, then applies the mask
                mask.lr_scheduler.step()
                mask.optimizer.zero_grad()

            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            update_step += 1
            local_step += 1
            update_time = time.time() - update_time

            # save checkpoint by save_every
            if not args.no_save:
                if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
                    if use_sparsity:
                        save_model(model, mask.optimizer, mask.lr_scheduler, update_step, global_step, run_config,
                                   tokens_seen, tokens_seen_before, update_time, args)

                    else:
                        save_model(model, optimizer, scheduler, update_step, global_step, run_config,
                               tokens_seen, tokens_seen_before, update_time, args)

            # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at epoch {epoch + 1} / step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
                )
                perplexity = math.exp(total_loss)
                if global_rank == 0:
                    if args.wandb_used:
                        wandb.log({
                            "eval/eval_perplexity": perplexity,
                            "eval/eval_loss": total_loss,
                            "eval/eval_tokens": evaluated_on_tokens,
                            },
                            step=global_step,
                        )
                    logger.info(f"Eval at epoch {epoch + 1} / step {update_step}: "
                                f"eval_loss {total_loss}  "
                                f"perplexity {perplexity}  "
                                f"tokens {evaluated_on_tokens}")

            if use_sparsity:
                lr = mask.optimizer.param_groups[0]["lr"]
            else:
                lr = optimizer.param_groups[0]["lr"]

            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size

            if global_rank == 0:

                if args.wandb_used:
                    wandb.log({
                        "train_loss": loss.item(),
                        "lr": lr,
                        "update_step": update_step,
                        "tokens_seen": tokens_seen,
                        "throughput_tokens": tokens_in_update / update_time,
                        "throughput_examples": args.total_batch_size / update_time,
                        "throughput_batches": batches_in_update / update_time,
                        },
                        step=global_step,
                    )

                if update_step % args.loss_every == 0:

                    logger.info(f"Epoch {epoch + 1} / Step {update_step}: "
                                f"lr {lr} "
                                # f"lr {grad_norm.item()} "
                                f"train_loss {loss.item()}  "
                                f"tokens_seen {tokens_seen}")

            update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        perplexity = math.exp(total_loss)
        if args.wandb_used:
            wandb.log({
                "final_eval/final_eval_perplexity": perplexity,
                "final_eval/final_eval_loss": total_loss,
                "final_eval/final_eval_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
        logger.info(f"Final eval loss: {total_loss} and perplexity: {perplexity} on tokens: {evaluated_on_tokens}")

    if not args.no_save and False:
        current_model_directory = f"{args.save_dir}/model_{args.epochs}_{update_step}_seed{args.seed}"
        if global_rank == 0 and not os.path.exists(current_model_directory):
            logger.info(f"Saving model and optimizer to {current_model_directory}, epoch {args.epochs}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            if args.single_gpu:
                model.save_pretrained(current_model_directory)
            else:
                model.module.save_pretrained(current_model_directory)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "epoch": epoch+1,
                "global_step": global_step,
                "config": run_config,
                # "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "epoch": epoch+1,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
