#!/bin/bash
#SBATCH --job-name=rw_l8_lrsens
#SBATCH --partition=log_gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-299%2
#SBATCH --output=logs/rw_l8_lrsens_%A_%a.out
#SBATCH --error=logs/rw_l8_lrsens_%A_%a.err

set -euo pipefail

cd /work/log1/$USER/gnn_thesis_cluster

source ~/venvs/gnn-thesis-gpu/bin/activate

GRID="configs/realworld_homophily_l8_lr_sensitivity_h01.csv"
OUT_DIR="runs/realworld_homophily_l8_lr_sensitivity_h01"

mkdir -p "$OUT_DIR"
mkdir -p logs

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
