#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH --cpus-per-task=18

source activate /projects/0/einf3822/.conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/projects/0/einf3822/xiaoq/codes/dst_llms"

max_step=$((40 * 1000))
max_length=256
total_batch_size=512

output_dir=/scratch-shared/xiaoq/c4_sampling/c4_filtered_maxlength${max_length}_bs${total_batch_size}_step${max_step}_arrow_shuffle32_offline
#output_dir=/scratch-shared/xiaoq/c4_sampling/c4_filtered_validation_10M

mkdir -p ${output_dir}

python -u data_sampling.py \
    --max_step $max_step \
    --max_length ${max_length} \
    --total_batch_size ${total_batch_size} \
    --output_dir ${output_dir} > ${output_dir}/log.txt 2>&1
