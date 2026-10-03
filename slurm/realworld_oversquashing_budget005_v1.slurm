#!/bin/bash
#SBATCH --job-name=rw_osq_b005
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-119%2
#SBATCH --output=logs/rw_osq_b005_%A_%a.out
#SBATCH --error=logs/rw_osq_b005_%A_%a.err

set -euo pipefail

cd /work/log1/$USER/new_test
source ~/venvs/gnn-thesis-gpu/bin/activate

export PYTHONPATH="/work/log1/$USER/new_test:${PYTHONPATH:-}"
export TMPDIR=/work/log1/$USER/tmp
export TORCH_HOME=/work/log1/$USER/cache/torch
export MPLCONFIGDIR=/work/log1/$USER/cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
    logs \
    "$TMPDIR" \
    "$TORCH_HOME" \
    "$MPLCONFIGDIR" \
    runs/realworld_oversquashing_budget005_v1

python \
    scripts/run_realworld_oversquashing_v1.py \
    --grid configs/realworld_oversquashing_budget005_v1.csv \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --out-dir runs/realworld_oversquashing_budget005_v1 \
    --epochs 2400 \
    --device cuda
