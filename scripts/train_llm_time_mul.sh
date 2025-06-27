#!/bin/bash
#SBATCH --account=p200910
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=160G
#SBATCH --qos=default

source activate /project/home/p200910/conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/home/users/u103022/codes/DST_LLMs"

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

export NORM_TYPE="pre"
export POST_NUM=3

CMD="

# size="350m"
for size in "250m"; do
for epochs in 1; do
for training_steps in 10000; do
for prune_rate in 0.2; do
for update_freq in 100; do

density=0.25

model_name="llama"

seed=0

acc_grad_steps=5
maintain_times=2
reinit="zero"

optimizer="adamdst"
learning_rate=1e-3
batch=64
growth="gradient"      # SET: "random", RigL: "gradient"
prune="magnitude"  # "magnitude_soft" or "magnitude"
temperature=3.0

fix=False
prune_rate_decay="cosine"  # constant, cosine, WSD
am_ratio=1.0

sparse_init="uniform"  # fixed_ERK; uniform; uniform_ratio

warmup_steps=1000

run_name="\${size}_s\${seed}"

max_length=256
total_batch_size=512

data_dir="/project/home/p200910/data/llms/c4_sampling/c4_filtered_maxlength\${max_length}_bs\${total_batch_size}_step\${training_steps}_arrow_shuffle32"
#data_dir=None

output_dir="/project/home/p200910/models/dst_llms_mul/model_\${model_name}\${size}_c4_f_l\${max_length}_bs\${total_batch_size}_step\${training_steps}_g\${growth}\${acc_grad_steps}\${maintain_times}_reinit\${reinit}_p\${prune}\${prune_rate}_\${prune_rate_decay}_f\${update_freq}_d\${density}_init\${sparse_init}_am\${am_ratio}_fix\${fix}_wp\${warmup_steps}_lr\${learning_rate}_ep\${epochs}_steps\${training_steps}_op\${optimizer}new_nodecay_exfl"

mkdir -p "\${output_dir}/checkpoints"

log_file="\${output_dir}/log.txt"

echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    torchrun_main.py \
    --wandb_mode disabled \
    --seed "\${seed}" \
    --model_config "configs/llama_\${size}.json" \
    --density "\${density}" \
    --data_dir "\${data_dir}" \
    --update_frequency "\${update_freq}" \
    --growth "\${growth}" \
    --accumulate_grad_steps "\${acc_grad_steps}" \
    --maintain_num "\${maintain_times}" \
    --reinit "\${reinit}" \
    --prune "\${prune}" \
    --prune_rate "\${prune_rate}" \
    --prune_rate_decay "\${prune_rate_decay}" \
    --temperature "\${temperature}" \
    --sparse_init "\${sparse_init}" \
    --am_ratio "\${am_ratio}" \
    --fix "\${fix}" \
    --lr "\${learning_rate}" \
    --optimizer "\${optimizer}" \
    --batch_size "\${batch}" \
    --total_batch_size "\${total_batch_size}" \
    --num_training_steps "\${training_steps}" \
    --epochs "\${epochs}" \
    --warmup_steps "\${warmup_steps}" \
    --dtype bfloat16 \
    --grad_clipping 0.0 \
    --run_name "\${run_name}" \
    --save_dir "\${output_dir}/checkpoints" > "\${log_file}" 2>&1


done
done
done
done
done
"

srun bash -c "$CMD"
echo "[sbatch-master] task finished"