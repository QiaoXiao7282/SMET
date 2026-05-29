#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --gpus=2
#SBATCH -t 20:00:00
source activate /projects/env

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

export NORM_TYPE="pre"
export POST_NUM=3

export size=$1

# Set density based on model size
export density=$2

export training_steps=$3
export batch=$4

export model_name="llama"
export seed=0
export learning_rate=$5

export optimizer=adamdst
export weight_decay=0.0
export growth="random"      # SET: "random", RigL: "gradient"
export prune="magnitude"  # "magnitude_soft" or "magnitude"
export temperature=3.0

export prune_rate=$6
export update_freq=$7

export acc_grad_steps=1  ## gradient_acc
export maintain_times=2  ## gradient_acc
export reinit='zero'

export fix=False
export prune_rate_decay="cosine"  # constant, cosine, WSD
export am_ratio=1.0
export sparse_init="uniform"  # fixed_ERK; uniform; uniform_ratio

export warmup_steps=500
export run_name="${size}_s${seed}"

export resume=False
export save_step=150

export max_length=256
export total_batch_size=512

export op_decay_steps=$8
export blocksize=$9

export lr_scale=False
export init_sparse=False

export reset_steps=True
export mask_momentum=True

export output_dir="/projects/llm_logs/mul_nodes_lr_add_dst_optimizer_unit/model_${model_name}${size}_c4_f_l${max_length}_bs${total_batch_size}_step${training_steps}_g${growth}$_reinit${reinit}_p${prune}${prune_rate}_${prune_rate_decay}_f${update_freq}_d${density}_init${sparse_init}_am${am_ratio}_fix${fix}_wp${warmup_steps}_lr${learning_rate}_steps${training_steps}_op${optimizer}wd${weight_decay}_nodes${SLURM_NNODES}_opdecay${op_decay_steps}_resetstep${reset_steps}_maskmom${mask_momentum}_dfl"

mkdir -p "${output_dir}/checkpoints"

export log_file="${output_dir}/log1.txt"

CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

echo \" \${data_dir} \${run_name}\"

torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    torchrun_main_unit.py \
    --wandb_mode disabled \
    --seed "\${seed}" \
    --model_config "configs/llama_\${size}.json" \
    --density "\${density}" \
    --val_dir None \
    --data_dir None \
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
    --weight_decay "\${weight_decay}" \
    --batch_size "\${batch}" \
    --total_batch_size "\${total_batch_size}" \
    --num_training_steps "\${training_steps}" \
    --epochs 1 \
    --warmup_steps "\${warmup_steps}" \
    --eval_every 2000 \
    --dtype bfloat16 \
    --grad_clipping 0.0 \
    --init_sparse "\${init_sparse}" \
    --blocksize "\${blocksize}" \
    --lr_scale "\${lr_scale}" \
    --op_decay_steps "\${op_decay_steps}" \
    --run_name "\${run_name}" \
    --resume "\${resume}" \
    --save_step "\${save_step}" \
    --reset_steps "\${reset_steps}" \
    --mask_momentum "\${mask_momentum}" \
    --compare_update False \
    --save_dir "\${output_dir}/checkpoints"

"
#srun bash -c "
srun bash -c "$CMD" > "${log_file}" 2>&1

