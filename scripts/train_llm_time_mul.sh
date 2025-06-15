#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus=4
#SBATCH -t 20:00:00
source activate /projects/0/einf3822/.conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/projects/0/einf3822/xiaoq/codes/dst_llms"


export NORM_TYPE="pre"
export POST_NUM=3

get_free_port() {
  while :; do
    port=$((12000 + RANDOM % 10000))
    if ! lsof -i:$port >/dev/null 2>&1; then
      echo $port
      return
    fi
  done
}

master_port=$(get_free_port)

# size="350m"
for size in "480m"; do
for epochs in 2; do
for training_steps in 10000; do


# Set density based on model size
if [[ "$size" == "60m" ]]; then
  density=1.0
elif [[ "$size" == "100m" ]]; then
  density=0.5
elif [[ "$size" == "250m" ]]; then
  density=0.25
elif [[ "$size" == "480m" ]]; then
  density=0.125
else
  echo "Unknown model size: $size"
  exit 1
fi


model_name='llama'

seed=0
learning_rate=1e-3
batch=128
growth="gradient"      # SET: "random", RigL: "gradient"
prune="magnitude"  # "magnitude_soft" or "magnitude"
temperature=3.0
prune_rate=0.1
update_freq=100

warmup_steps=1000

run_name="${size}_s${seed}"

max_length=256
total_batch_size=512

data_dir="/scratch-shared/xiaoq/c4_sampling/c4_filtered_maxlength${max_length}_bs${total_batch_size}_step${training_steps}_arrow_shuffle32"
#data_dir=None

output_dir="/scratch-shared/xiaoq/dst_llms/model_${model_name}${size}_c4_f_l${max_length}_bs${total_batch_size}_step${training_steps}_g${growth}_p${prune}${prune_rate}_f${update_freq}_d${density}_wp${warmup_steps}_lr${learning_rate}_ep${epochs}_steps${training_steps}_exfl"

mkdir -p ${output_dir}/checkpoints


torchrun --nproc_per_node 4 --master_port=${master_port} torchrun_main.py \
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
