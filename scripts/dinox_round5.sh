#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/DINO-X
INDEX="data/processed/lidc-idri/_index/index.csv"
SPLIT="data/processed/_splits/val10_seed42.json"
RESULTS_DIR="data/experiment_results"
RESULTS_FILE="$RESULTS_DIR/round5_results.jsonl"
LOG_FILE="$RESULTS_DIR/round5_experiments.log"

cd "$REPO"
mkdir -p "$RESULTS_DIR"

BASE_ARGS="--index-csv $INDEX --split-manifest $SPLIT --loss-type dino --gram-weight 1.0 --koleo-weight 0.1 --teacher-temp 0.04 --ema 0.996 --center-momentum 0.999 --amp --train-seed 42"

declare -A ARMS
ARMS[vits_100k]="--config vit-small --lr 2e-4 --batch-size 64 --max-steps 100000 --run-suffix vits_100k"
ARMS[vitl_20k_lr5e5]="--config vit-large --lr 5e-5 --batch-size 16 --max-steps 20000 --run-suffix vitl_20k_lr5e5"

ARM_ORDER=(vits_100k vitl_20k_lr5e5)

echo "==========================================" | tee "$LOG_FILE"
echo " ROUND 5: ViT-S 100K + ViT-L 20K" | tee -a "$LOG_FILE"
echo " $(date -u)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

for arm in "${ARM_ORDER[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "  ARM: $arm  ($(date -u))" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    args="${ARMS[$arm]}"
    config=$(echo "$args" | grep -oP "(?<=--config )\S+")
    steps=$(echo "$args" | grep -oP "(?<=--max-steps )\S+")
    lr=$(echo "$args" | grep -oP "(?<=--lr )\S+")
    batch=$(echo "$args" | grep -oP "(?<=--batch-size )\S+")
    echo "  Config: $config, steps=$steps, lr=$lr, batch=$batch" | tee -a "$LOG_FILE"

    echo "[TRAIN] Starting $arm..." | tee -a "$LOG_FILE"
    train_output=$(python scripts/phase5_big_run.py $BASE_ARGS $args 2>&1) || true
    echo "$train_output" | tail -5 | tee -a "$LOG_FILE"

    run_dir=$(ls -dt data/runs/*${arm}* 2>/dev/null | head -1)
    ckpt=""
    if [ -n "$run_dir" ]; then
        ckpt=$(ls -t "$run_dir"/checkpoint_final_*.pth 2>/dev/null | head -1)
    fi

    if [ -z "$ckpt" ]; then
        echo "[ERROR] No checkpoint found for $arm" | tee -a "$LOG_FILE"
        echo "{\"arm\": \"$arm\", \"config\": \"$config\", \"steps\": $steps, \"lr\": \"$lr\", \"ratio\": 0, \"top1\": 0, \"passed\": \"error\"}" >> "$RESULTS_FILE"
        continue
    fi

    final_loss=$(echo "$train_output" | grep -oP "Final loss: \K[\d.]+" || echo "0")
    gram_health=$(echo "$train_output" | grep -oP "Gram: mean=\K[\d.]+" || echo "N/A")
    echo "Training complete. Final loss: $final_loss, Gram: $gram_health" | tee -a "$LOG_FILE"

    echo "[EVAL] Using checkpoint: $ckpt" | tee -a "$LOG_FILE"
    eval_output=$(python scripts/phase5_view_retrieval_eval.py \
        --index-csv "$INDEX" \
        --split-manifest "$SPLIT" \
        --checkpoint "$ckpt" \
        --n 2048 \
        --seed 42 2>&1) || true
    echo "$eval_output" | tee -a "$LOG_FILE"

    ratio=$(echo "$eval_output" | grep -oP "ratio=\K[\d.]+" || echo "0")
    top1=$(echo "$eval_output" | grep -oP "top1=\K[\d.]+" || echo "0")
    top5=$(echo "$eval_output" | grep -oP "top5=\K[\d.]+" || echo "0")
    passed=$(echo "$eval_output" | grep -oP "passed=\K\w+" || echo "unknown")

    echo "[RESULT] $arm: ratio=$ratio, top1=$top1, top5=$top5, passed=$passed, loss=$final_loss, gram=$gram_health" | tee -a "$LOG_FILE"
    echo "{\"arm\": \"$arm\", \"config\": \"$config\", \"steps\": $steps, \"lr\": \"$lr\", \"ratio\": $ratio, \"top1\": $top1, \"top5\": $top5, \"passed\": \"$passed\", \"loss\": $final_loss, \"gram\": \"$gram_health\"}" >> "$RESULTS_FILE"

    # Clean intermediate checkpoints
    if [ -n "$run_dir" ]; then
        echo "[CLEANUP] Removing intermediate checkpoints..." | tee -a "$LOG_FILE"
        find "$run_dir" -name "checkpoint_*.pth" ! -name "checkpoint_final_*" -delete 2>/dev/null || true
        # Also clean tensorboard events to save space (keep last)
        echo "Disk: $(df -h /workspace | tail -1 | awk "{print \$4}") free" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo " ROUND 5 COMPLETE" | tee -a "$LOG_FILE"
echo " $(date -u)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
cat "$RESULTS_FILE" | tee -a "$LOG_FILE"
