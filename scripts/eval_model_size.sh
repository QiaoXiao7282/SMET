#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 1:00:00
source activate /projects/0/einf3822/.conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/projects/0/einf3822/xiaoq/codes/dst_llms"


export NORM_TYPE="pre"
export POST_NUM=3

# size="350m"
for size in "60m"; do
for density in 1.0; do
for epochs in 1; do
for training_steps in 100; do

model_name='llama'

seed=0
learning_rate=1e-3
batch=256
growth="gradient"      # SET: "random", RigL: "gradient"
prune="magnitude"  # "magnitude_soft" or "magnitude"
temperature=3.0
prune_rate=0.5
update_freq=100

warmup_steps=20

run_name="${size}_s${seed}"

max_length=256
total_batch_size=512

torchrun --nproc_per_node 1 --master_port=29551 model_size.py \
    --wandb_mode disabled \
    --seed $seed \
    --model_config "configs/llama_${size}.json" \
    --density $density \
    --update_frequency $update_freq \
    --growth ${growth} \
    --prune $prune \
    --prune_rate $prune_rate \
    --temperature $temperature \
    --sparse_init uniform \
    --lr $learning_rate \
    --batch_size $batch \
    --total_batch_size $total_batch_size \
    --num_training_steps $training_steps \
    --epochs $epochs \
    --warmup_steps $warmup_steps \
    --dtype bfloat16 \
    --grad_clipping 0.0 \
    --run_name $run_name > model_size_log.txt 2>&1

done
done
done
done