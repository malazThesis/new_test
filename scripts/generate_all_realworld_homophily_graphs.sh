#!/bin/bash
set -euo pipefail

cd /work/log1/$USER/gnn_thesis_cluster
source ~/venvs/gnn-thesis-gpu/bin/activate

mkdir -p logs/realworld_homophily_rewiring

for DATASET in PubMed Roman-empire
do
    if [ "$DATASET" = "PubMed" ]; then
        SLUG="pubmed"
    else
        SLUG="roman_empire"
    fi

    for TARGET in 0.1 0.5 0.9
    do
        case "$TARGET" in
            0.1) HSLUG="h01" ;;
            0.5) HSLUG="h05" ;;
            0.9) HSLUG="h09" ;;
        esac

        for SEED in 1 2 3 4 5
        do
            LOW="realworld_data/homophily_controlled/${SLUG}/lowlabel/${SLUG}_${HSLUG}_seed${SEED}.pt"
            REP="realworld_data/homophily_controlled/${SLUG}/replicated/${SLUG}_${HSLUG}_seed${SEED}.pt"

            LOG="logs/realworld_homophily_rewiring/${SLUG}_${HSLUG}_seed${SEED}.log"

            echo
            echo "============================================================"
            echo "$DATASET target=$TARGET seed=$SEED"
            echo "============================================================"

            if [ -f "$LOW" ] && [ -f "$REP" ]; then
                echo "Already exists; skipping."
                continue
            fi

            python \
                scripts/make_realworld_homophily_graphs.py \
                --dataset "$DATASET" \
                --target "$TARGET" \
                --seed "$SEED" \
                --max-proposals 20000000 \
                2>&1 | tee "$LOG"
        done
    done
done
