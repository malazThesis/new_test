#!/bin/bash
#SBATCH --job-name=rw_rep_parity
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-559%2
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

GRID="configs/realworld_homophily_replicated_parity.csv"
OUT_DIR="runs/realworld_homophily_replicated_parity"

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  logs \
  "$OUT_DIR"

python scripts/run_csbm.py \
  --grid "$GRID" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --out-dir "$OUT_DIR" \
  --epochs 2400 \
  --device cuda \
  --track-oversmoothing \
  --oversmoothing-epochs "0,1,2,4,8,16,32,50,100,150,200,300,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400" \
  --oversmoothing-max-nodes 2048 \
  --oversmoothing-max-edges 100000
