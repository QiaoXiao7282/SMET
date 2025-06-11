import argparse
from datasets import load_dataset, Dataset, Features, Value
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets
import datasets.distributed
from datasets import load_from_disk
import os
import pyarrow as pa
from datasets.arrow_writer import ArrowWriter
import numpy as np
import torch
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_step", type=int, default=10_000, help="Target total steps count")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenization truncation length")
    parser.add_argument("--total_batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default='/scratch-shared/xiaoq/c4_sampling/c4_filtered_validation_10M', help="Where to save dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    return parser.parse_args()

def sampling_data():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    seed_for_shuffle = 32
    cache_dir = '/projects/0/einf3822/Lu/transformers_cache/HF_HOME'
    print("[INFO] Loading streaming C4...")
    data = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir=cache_dir)
    data = data.shuffle(seed=seed_for_shuffle)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    print_every = 10_000
    shard_idx = 0
    token_count = 0
    sample_count = 0
    shard_token_count = 0

    max_tokens = args.max_step * args.max_length * args.total_batch_size
    step_token_limit = args.max_step * args.max_length * args.total_batch_size

    def start_new_writer(index):
        arrow_path = os.path.join(args.output_dir, f"train-{index:05d}-of-00004.arrow")
        features = Features({"text": Value("string")})
        return ArrowWriter(path=arrow_path, schema=features.arrow_schema)

    writer = start_new_writer(shard_idx)

    for example in tqdm(data, desc="Collecting examples"):
        try:
            text = example.get("text", "").strip()
            if not text:
                continue

            tokens = tokenizer(text, truncation=True, padding="max_length", max_length=args.max_length)["input_ids"]
            num_tokens = len(tokens)

            token_count += num_tokens
            shard_token_count += num_tokens
            sample_count += 1

            writer.write({"text": text})

            if sample_count % print_every == 0:
                print(f"[INFO] Collected {sample_count} samples, approx {token_count:,} tokens...")

            if token_count >= max_tokens:
                break

            # 超过当前 shard 限制，切 shard
            if shard_token_count >= step_token_limit:
                writer.finalize()
                shard_idx += 1
                writer = start_new_writer(shard_idx)
                shard_token_count = 0

        except Exception as e:
            print(f"Skipped corrupted example: {e}")


    writer.finalize()
    print(f"[INFO] Finished: {sample_count:,} samples, ~{token_count:,} tokens.")
    print(f"[DONE] Saved {shard_idx + 1} Arrow files to: {args.output_dir}")


def export_to_arrow(dataset: Dataset, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = ArrowWriter(path=output_path, schema=dataset.data.schema)
    writer.write_table(dataset.data.table)
    writer.finalize()

def count_tokens():
    args = parse_args()

    dataset = datasets.load_dataset("arrow", data_dir=args.output_dir, split="train", streaming=True)
    # dataset = load_from_disk(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    token_count = 0
    for example in tqdm(dataset):
        text = example["text"]
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=args.max_length)["input_ids"]
        token_count += len(tokens)

    print(f"Total tokens: {token_count}")

def sampling_validation_data():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.output_dir = '/scratch-shared/xiaoq/c4_sampling/c4_filtered_validation_10M'
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading streaming validation split from C4...")
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    data = data.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    print_every = 1_000
    target_eval_tokens = 10_000_000  # tokens for evaluation

    arrow_path = os.path.join(args.output_dir, "validation-00000-of-00001.arrow")
    features = Features({"text": Value("string")})
    writer = ArrowWriter(path=arrow_path, schema=features.arrow_schema)

    token_count = 0
    sample_count = 0

    for example in tqdm(data, desc="Collecting validation examples"):
        text = example.get("text", "").strip()
        if not text:
            continue

        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=args.max_length)["input_ids"]
        token_count += len(tokens)

        writer.write({"text": text})
        sample_count += 1

        if sample_count % print_every == 0:
            print(f"[INFO] Collected {sample_count} validation samples, approx {token_count:,} tokens...")

        if token_count >= target_eval_tokens:
            break

    writer.finalize()
    print(f"[INFO] Finished: {sample_count:,} validation samples, ~{token_count:,} tokens.")
    print(f"[DONE] Saved Arrow file: {arrow_path}")


if __name__ == "__main__":
    sampling_data()
    # count_tokens()
    # sampling_validation_data()
