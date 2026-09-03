#!/bin/bash
#SBATCH --job-name=rw_pn_ref_v2
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-99%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd /work/log1/$USER/gnn_thesis_cluster
source ~/venvs/gnn-thesis-gpu/bin/activate

export PYTHONPATH="/work/log1/$USER/gnn_thesis_cluster:${PYTHONPATH:-}"
export TMPDIR=/work/log1/$USER/tmp
export PIP_CACHE_DIR=/work/log1/$USER/cache/pip
export TORCH_HOME=/work/log1/$USER/cache/torch
export MPLCONFIGDIR=/work/log1/$USER/cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GRID="configs/realworld_pairnorm_component_reference_matched_lr01.csv"
BASE_OUT="runs/realworld_pairnorm_component_reference_matched_lr01_v2"

GRAPH_SEED=$(
python - "$GRID" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import csv
import re
import sys

grid = sys.argv[1]
task_id = int(sys.argv[2])

with open(grid, newline="") as f:
    rows = list(csv.DictReader(f))

row = rows[task_id]

m = re.search(
    r"seed(\d+)\.pt$",
    row["data_path"],
)

if m is None:
    raise RuntimeError(
        row["data_path"]
    )

print(int(m.group(1)))
PY
)

OUT_DIR="${BASE_OUT}/graph_seed${GRAPH_SEED}"

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
    --oversmoothing-epochs "2400" \
    --oversmoothing-max-nodes 2048 \
    --oversmoothing-max-edges 100000
