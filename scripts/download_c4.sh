#!/bin/bash -l
# Run this script with: sbatch scripts/download_c4.sh

#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH --cpus-per-task=18

source activate /projects/0/einf3822/.conda/dst_llm_py10

export PYTHONPATH="${PYTHONPATH}:/projects/0/einf3822/xiaoq/codes/dst_llms"


cache_dir="/projects/0/einf3822/Lu/transformers_cache/HF_HOME"

# Download C4 and trigger split generation
python -c "from datasets import load_dataset; load_dataset('allenai/c4','en', cache_dir='$cache_dir')"
