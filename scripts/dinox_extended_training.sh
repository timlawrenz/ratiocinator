#!/usr/bin/env bash
set -euo pipefail

# Extended training experiments for DINO-X
# Tests whether cm=0.999 retrieval improves with longer training
# Also runs best recipe (cm=0.9) longer for comparison

cd /workspace/DINO-X

INDEX_CSV="data/processed/lidc-idri/_index/index.csv"
SPLIT_MANIFEST="data/processed/_splits/val10_seed42.json"
RESULTS_DIR="data/experiment_results"
RESULTS_FILE="$RESULTS_DIR/extended_results.jsonl"

mkdir -p "$RESULTS_DIR"

run_extended_arm() {
    local name="$1"
    local cm="$2"
    local steps="$3"
    local extra_args="${4:-}"

    local suffix="${name}"
    local ts=$(date +%Y%m%d_%H%M%S)
    local run_dir="data/runs/${ts}_${suffix}"

    echo ""
    echo "================================================================"
    echo "  ARM: $name  ($(date))"
    echo "  Config: cm=$cm, steps=$steps"
    echo "================================================================"

    # Train
    echo "[TRAIN] Starting $name ($steps steps)..."
    python scripts/phase5_big_run.py \
        --index-csv "$INDEX_CSV" \
        --split-manifest "$SPLIT_MANIFEST" \
        --run-suffix "$suffix" \
        --config vit-small \
        --loss-type dino \
        --gram-weight 1.0 \
        --koleo-weight 0.1 \
        --lr 2e-4 \
        --teacher-temp 0.04 \
        --ema 0.996 \
        --center-momentum "$cm" \
        --amp \
        --batch-size 64 \
        --max-steps "$steps" \
        --train-seed 42 \
        $extra_args 2>&1 | tail -3

    # Find checkpoint
    local CKPT=$(find data/runs -path "*${suffix}*" -name "checkpoint_final_*.pth" | sort | tail -1)
    if [ -z "$CKPT" ]; then
        echo "[ERROR] No checkpoint found for $name"
        echo "{\"arm\": \"$name\", \"error\": \"no checkpoint\"}" >> "$RESULTS_FILE"
        return 1
    fi

    echo "[EVAL] Using checkpoint: $CKPT"
    (python scripts/phase5_view_retrieval_eval.py \
        --checkpoint "$CKPT" \
        --split-manifest "$SPLIT_MANIFEST" \
        --index-csv "$INDEX_CSV" \
        --n 2048 --seed 42 || true) 2>&1 | tail -5

    # Collect metrics
    local RETR=$(find data/runs -path "*${suffix}*" -name "view_retrieval_*.json" | sort | tail -1)
    if [ -n "$RETR" ] && [ -f "$RETR" ]; then
        local top1=$(python3 -c "import json; d=json.load(open('$RETR')); print(d.get('retrieval_top1', 0))")
        local ratio=$(python3 -c "import json; d=json.load(open('$RETR')); print(d.get('retrieval_ratio', 0))")
        echo "[METRICS] $name: top1=$top1, ratio=$ratio"
        echo "{\"arm\": \"$name\", \"retrieval_top1\": $top1, \"retrieval_ratio\": $ratio, \"steps\": $steps, \"center_momentum\": $cm}" >> "$RESULTS_FILE"
    else
        echo "[WARN] No retrieval JSON for $name"
        echo "{\"arm\": \"$name\", \"error\": \"no retrieval json\", \"steps\": $steps}" >> "$RESULTS_FILE"
    fi

    # Get final loss from TensorBoard
    local RUN_DIR=$(find data/runs -maxdepth 1 -type d -name "*${suffix}*" | sort | tail -1)
    if [ -n "$RUN_DIR" ]; then
        python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$RUN_DIR')
ea.Reload()
tags = ea.Tags().get('scalars', [])
loss_tags = [t for t in tags if 'loss' in t.lower()]
if loss_tags:
    vals = ea.Scalars(loss_tags[0])
    print(f'[LOSS] {len(vals)} steps, final={vals[-1].value:.4f}')
" 2>/dev/null || true
    fi
}

echo "=========================================="
echo " EXTENDED TRAINING EXPERIMENTS"
echo " $(date)"
echo "=========================================="

# Experiment 1: cm=0.999 for 5000 steps (breakthrough config, 2.5x longer)
run_extended_arm "ext_cm999_5k" "0.999" "5000"

# Experiment 2: cm=0.999 for 10000 steps (5x longer, see if retrieval catches up)
run_extended_arm "ext_cm999_10k" "0.999" "10000"

# Experiment 3: best recipe cm=0.9 for 5000 steps (comparison baseline)
run_extended_arm "ext_cm09_5k" "0.9" "5000"

# Experiment 4: cm=0.9 for 10000 steps
run_extended_arm "ext_cm09_10k" "0.9" "10000"

echo ""
echo "=========================================="
echo " ALL EXTENDED TRAINING COMPLETE"
echo " $(date)"
echo "=========================================="
echo "Results: $RESULTS_FILE"
cat "$RESULTS_FILE"
