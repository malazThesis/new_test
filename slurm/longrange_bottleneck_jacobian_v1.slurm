#!/bin/bash
#SBATCH --job-name=lrb_jac_v1
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-239%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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
    runs/longrange_bottleneck_jacobian_v1

python \
    scripts/run_longrange_bottleneck_jacobian_v1.py \
    --grid configs/longrange_bottleneck_jacobian_v1.csv \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --out-dir runs/longrange_bottleneck_jacobian_v1 \
    --device cuda
