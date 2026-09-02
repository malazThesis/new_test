#!/bin/bash
#SBATCH --job-name=rw_hom_low
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-499%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source ~/venvs/gnn-thesis-gpu/bin/activate

cd /work/log1/$USER/gnn_thesis_cluster

export PYTHONPATH="/work/log1/$USER/gnn_thesis_cluster:${PYTHONPATH:-}"

export TMPDIR=/work/log1/$USER/tmp
export PIP_CACHE_DIR=/work/log1/$USER/cache/pip
export TORCH_HOME=/work/log1/$USER/cache/torch
export MPLCONFIGDIR=/work/log1/$USER/cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GRID="${GRID:-configs/realworld_homophily_lowlabel_main.csv}"

OUT_DIR="${OUT_DIR:-runs/realworld_homophily_lowlabel_main}"

mkdir -p \
    "$TMPDIR" \
    "$PIP_CACHE_DIR" \
    "$TORCH_HOME" \
    "$MPLCONFIGDIR" \
    logs \
    "$OUT_DIR"

echo "========================================"
echo "GRID=$GRID"
echo "OUT_DIR=$OUT_DIR"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "========================================"

python scripts/run_csbm.py \
    --grid "$GRID" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --out-dir "$OUT_DIR" \
    --epochs 200 \
    --device cuda \
    --track-oversmoothing \
    --oversmoothing-epochs "0,1,2,4,8,16,32,50,100,150,200" \
    --oversmoothing-max-nodes 2048 \
    --oversmoothing-max-edges 100000
