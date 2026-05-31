#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus=4
#SBATCH -t 24:00:00

set -euo pipefail

########################################
# Environment
########################################

source activate env


########################################
# Experiment settings
########################################

size="1b"
density=0.5
training_steps=100000
batch=128
learning_rate=1.0e-3

optimizer="adamdst"

model_name="llama"


########################################
# Output directory
########################################

output_dir="output/model_${model_name}${size}_c4_f_bs${total_batch_size}_step${training_steps}_d${density}_lr${learning_rate}_steps${training_steps}_op${optimizer}"

mkdir -p "${output_dir}/checkpoints"

log_file="${output_dir}/log.txt"

########################################
# Run
########################################

echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[job] SLURM_NODELIST=${SLURM_NODELIST}"
echo "[job] output_dir=${output_dir}"
echo "[job] log_file=${log_file}"

torchrun \
    --standalone \
    --nproc_per_node=4 \
    torchrun_main_unit.py \
    --wandb_mode disabled \
    --model_config "configs/llama_${size}.json" \
    --density "${density}" \
    --val_dir None \
    --data_dir None \
    --lr "${learning_rate}" \
    --optimizer "${optimizer}" \
    --batch_size "${batch}" \
    --total_batch_size "${total_batch_size}" \
    --num_training_steps "${training_steps}" \
    --epochs 1 \
    --eval_every 2000 \
    --dtype bfloat16 \
    --save_dir "${output_dir}/checkpoints" \
    > "${log_file}" 2>&1