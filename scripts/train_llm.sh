#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 40:00:00
source activate /projects/0/einf3822/.conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/projects/0/einf3822/xiaoq/codes/dst_llms"


export NORM_TYPE="pre"
export POST_NUM=3

# size="350m"
for size in "60m"; do
for density in 1.0; do
for epochs in 3; do
for training_steps in 10000; do

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

data_dir="/scratch-shared/xiaoq/c4_sampling/c4_filtered_maxlength${max_length}_bs${total_batch_size}_step${training_steps}_arrow"
#data_dir=None

output_dir="/scratch-shared/xiaoq/dst_llms/model_${model_name}${size}_c4_f_l${max_length}_bs${total_batch_size}_step${training_steps}_g${growth}_p${prune}${prune_rate}_f${update_freq}_d${density}_wp${warmup_steps}_lr${learning_rate}_ep${epochs}_steps${training_steps}"

mkdir -p ${output_dir}/checkpoints


torchrun --nproc_per_node 1 --master_port=29511 torchrun_main.py \
    --wandb_mode disabled \
    --seed $seed \
    --model_config "configs/llama_${size}.json" \
    --density $density \
    --data_dir ${data_dir} \
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
    --run_name $run_name \
    --save_dir "${output_dir}/checkpoints" > ${output_dir}/log.txt 2>&1

done
done
done
done