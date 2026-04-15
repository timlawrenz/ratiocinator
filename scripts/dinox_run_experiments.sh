#!/bin/bash
# DINO-X Experiment Runner — runs directly on Vast.ai instance
# Usage: bash scripts/dinox_run_experiments.sh [experiment_name]
#
# Experiments: center_momentum, augmentation, all
set -euo pipefail

cd /workspace/DINO-X

INDEX_CSV="data/processed/lidc-idri/_index/index.csv"
SPLIT_MANIFEST="data/processed/_splits/val10_seed42.json"
RESULTS_DIR="data/experiment_results"
mkdir -p "$RESULTS_DIR"

# Common args for all experiments
BASE_ARGS="--config vit-small --max-steps 2000 --batch-size 64 --accumulation-steps 4 \
  --lr 2e-4 --warmup-steps 200 --teacher-temp 0.04 --ema 0.996 \
  --gram-weight 1.0 --koleo-weight 0.1 --loss-type dino \
  --ckpt-every 500 --amp \
  --index-csv $INDEX_CSV --split-manifest $SPLIT_MANIFEST --train-seed 42"

run_arm() {
    local name="$1"
    local extra_args="$2"
    local suffix="$3"

    echo ""
    echo "================================================================"
    echo "  ARM: $name  ($(date -u))"
    echo "================================================================"

    # Training
    echo "[TRAIN] Starting $name..."
    python scripts/phase5_big_run.py $BASE_ARGS $extra_args --run-suffix "$suffix" 2>&1 | tail -20
    echo "[TRAIN] $name complete"

    # Eval
    CKPT=$(find data/runs -path "*${suffix}*" -name "checkpoint_final_*.pth" | sort | tail -1)
    if [ -z "$CKPT" ]; then
        echo "[EVAL] No checkpoint found for $name — skipping eval"
        echo "{\"arm\": \"$name\", \"error\": \"no checkpoint\"}" >> "$RESULTS_DIR/results.jsonl"
        return
    fi

    echo "[EVAL] Using checkpoint: $CKPT"
    python scripts/phase5_view_retrieval_eval.py \
        --checkpoint "$CKPT" \
        --split-manifest "$SPLIT_MANIFEST" \
        --index-csv "$INDEX_CSV" \
        --n 2048 --seed 42 2>&1 | tail -5

    # Collect metrics
    RETR=$(find data/runs -path "*${suffix}*" -name "view_retrieval_*.json" | sort | tail -1)
    if [ -n "$RETR" ] && [ -f "$RETR" ]; then
        python3 -c "
import json
r = json.load(open('$RETR'))
result = dict(arm='$name', retrieval_top1=r.get('top1',-1), retrieval_ratio=r.get('ratio_vs_random',-1))
print('METRICS:' + json.dumps(result))
with open('$RESULTS_DIR/results.jsonl', 'a') as f:
    f.write(json.dumps(result) + '\n')
"
    else
        echo "[EVAL] No eval results for $name"
        echo "{\"arm\": \"$name\", \"error\": \"no eval results\"}" >> "$RESULTS_DIR/results.jsonl"
    fi
}

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Center Momentum Ablation
# Tests: center_momentum ∈ {0.9, 0.95, 0.99, 0.999}
# Using winning recipe: DINO + Gram(1.0) + KoLeo(0.1)
# ══════════════════════════════════════════════════════════════════════
run_center_momentum() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  EXPERIMENT: Center Momentum Ablation (4 arms)             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    for cm in 0.9 0.95 0.99 0.999; do
        safe_cm=$(echo "$cm" | tr '.' '_')
        run_arm "cm_${safe_cm}" "--center-momentum $cm" "cm_${safe_cm}"
    done

    echo ""
    echo "=== Center Momentum Results ==="
    cat "$RESULTS_DIR/results.jsonl" | grep '"cm_' || echo "(no results)"
}

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Augmentation Ablation
# Tests different augmentation strategies for medical imaging
# Requires patching the dataset transform in phase5_big_run.py
# ══════════════════════════════════════════════════════════════════════
run_augmentation() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  EXPERIMENT: Augmentation Ablation (5 arms)                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    ORIG_FILE="scripts/phase5_big_run.py"
    BACKUP="scripts/phase5_big_run.py.bak"
    cp "$ORIG_FILE" "$BACKUP"

    # Arm A: Spatial only (RandomResizedCrop + HFlip, no normalize — oh wait, normalize is needed)
    echo "[PATCH] Arm A: Spatial only"
    cp "$BACKUP" "$ORIG_FILE"
    # Default transform is already spatial-only (RandomResizedCrop + HFlip + Normalize)
    # This IS the current default — so this is our baseline
    run_arm "aug_spatial" "" "aug_spatial"

    # Arm B: Spatial + ColorJitter (grayscale-safe: brightness + contrast only)
    echo "[PATCH] Arm B: Spatial + ColorJitter"
    cp "$BACKUP" "$ORIG_FILE"
    python3 -c "
import re
code = open('$ORIG_FILE').read()
old = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),'''
new = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),'''
code = code.replace(old, new)
open('$ORIG_FILE', 'w').write(code)
"
    run_arm "aug_colorjitter" "" "aug_colorjitter"

    # Arm C: Spatial + GaussianBlur
    echo "[PATCH] Arm C: Spatial + GaussianBlur"
    cp "$BACKUP" "$ORIG_FILE"
    python3 -c "
code = open('$ORIG_FILE').read()
old = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),'''
new = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),'''
code = code.replace(old, new)
open('$ORIG_FILE', 'w').write(code)
"
    run_arm "aug_blur" "" "aug_blur"

    # Arm D: Spatial + ColorJitter + GaussianBlur (full augmentation)
    echo "[PATCH] Arm D: Spatial + ColorJitter + GaussianBlur"
    cp "$BACKUP" "$ORIG_FILE"
    python3 -c "
code = open('$ORIG_FILE').read()
old = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),'''
new = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),'''
code = code.replace(old, new)
open('$ORIG_FILE', 'w').write(code)
"
    run_arm "aug_full" "" "aug_full"

    # Arm E: Aggressive crop (scale 0.2-1.0 instead of 0.5-1.0)
    echo "[PATCH] Arm E: Aggressive crop scale"
    cp "$BACKUP" "$ORIG_FILE"
    python3 -c "
code = open('$ORIG_FILE').read()
code = code.replace('scale=(0.5, 1.0)', 'scale=(0.2, 1.0)')
open('$ORIG_FILE', 'w').write(code)
"
    run_arm "aug_aggcrop" "" "aug_aggcrop"

    # Restore original
    cp "$BACKUP" "$ORIG_FILE"
    rm "$BACKUP"

    echo ""
    echo "=== Augmentation Results ==="
    cat "$RESULTS_DIR/results.jsonl" | grep '"aug_' || echo "(no results)"
}

# ══════════════════════════════════════════════════════════════════════
# Main dispatch
# ══════════════════════════════════════════════════════════════════════
EXPERIMENT="${1:-all}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DINO-X Experiment Runner                                   ║"
echo "║  Instance: $(hostname)                                      ║"
echo "║  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  ║"
echo "║  Started: $(date -u)                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Verify data is ready
if [ ! -f "$INDEX_CSV" ]; then
    echo "ERROR: Data not ready — $INDEX_CSV not found"
    exit 1
fi
echo "Data OK: $(wc -l < $INDEX_CSV) entries in index"

case "$EXPERIMENT" in
    center_momentum) run_center_momentum ;;
    augmentation)    run_augmentation ;;
    all)
        run_center_momentum
        run_augmentation
        ;;
    *) echo "Unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE — $(date -u)                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "=== Full Results ==="
cat "$RESULTS_DIR/results.jsonl"
