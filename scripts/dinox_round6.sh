#!/bin/bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# DINO-X Round 6: ViT-L 100K + Augmentation 3-Seed Validation
# ══════════════════════════════════════════════════════════════════════════════

cd /workspace/DINO-X
RESULTS_DIR="data/runs/round6"
mkdir -p "$RESULTS_DIR"

INDEX="data/processed/lidc-idri/_index/index.csv"
SPLIT="data/processed/_splits/val10_seed42.json"
ORIG_FILE="scripts/phase5_big_run.py"

# Fix SimCLR fp16 bug if not already fixed
if grep -q 'masked_fill_(mask, -9e15)' "$ORIG_FILE" 2>/dev/null; then
    echo "[FIX] Applying SimCLR fp16 masked_fill fix (-9e15 -> -1e4)"
    sed -i 's/masked_fill_(mask, -9e15)/masked_fill_(mask, -1e4)/g' "$ORIG_FILE"
fi

# Ensure tensorboard is installed
pip install -q tensorboard 2>/dev/null || true

run_arm() {
    local name="$1"
    local extra_args="$2"
    local suffix="$3"
    echo ""
    echo "================================================================"
    echo "  ARM: $name"
    echo "  Started: $(date -u)"
    echo "================================================================"

    python3 scripts/phase5_big_run.py \
        --index-csv "$INDEX" \
        --split-manifest "$SPLIT" \
        --run-suffix "$suffix" \
        --center-momentum 0.999 \
        --amp \
        $extra_args 2>&1 | tee "$RESULTS_DIR/${name}.log"

    local ecode=$?
    echo "[ARM $name] Exit code: $ecode"

    # Find the latest run directory
    local latest_run
    latest_run=$(ls -dt data/runs/*"$suffix"* 2>/dev/null | head -1)
    if [ -n "$latest_run" ]; then
        echo "[ARM $name] Run dir: $latest_run"
        # Run eval
        local ckpt
        ckpt=$(ls -t "$latest_run"/checkpoint_final_*.pth 2>/dev/null | head -1)
        if [ -z "$ckpt" ]; then
            ckpt=$(ls -t "$latest_run"/checkpoint_*.pth 2>/dev/null | head -1)
        fi
        if [ -n "$ckpt" ]; then
            echo "[ARM $name] Evaluating: $ckpt"
            (python3 scripts/phase5_view_retrieval_eval.py \
                --index-csv "$INDEX" \
                --split-manifest "$SPLIT" \
                --checkpoint "$ckpt" \
                --seed 42 --n 4096 2>&1 || true) | tee -a "$RESULTS_DIR/${name}.log"
        fi
    fi

    # Extract metrics from log
    python3 -c "
import re, json
log = open('$RESULTS_DIR/${name}.log').read()
m = dict(arm='$name')
for pat, key in [
    (r'Final loss[:\s]*([\d.]+)', 'final_loss'),
    (r'top1_acc[:\s]*([\d.]+)', 'top1'),
    (r'top5_acc[:\s]*([\d.]+)', 'top5'),
    (r'ratio[:\s]*([\d.]+)', 'ratio'),
    (r'passed[:\s]*(true|false)', 'passed'),
]:
    match = re.search(pat, log, re.IGNORECASE)
    if match:
        v = match.group(1)
        try: v = float(v)
        except: pass
        m[key] = v
print('METRICS:' + json.dumps(m))
" 2>/dev/null || true

    echo "[ARM $name] Completed: $(date -u)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: ViT-Large 100K steps
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT 1: ViT-Large 100K (lr=5e-5, batch=16, cm=0.999)    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

run_arm "vitl_100k" \
    "--config vit-large --lr 5e-5 --batch-size 16 --max-steps 100000 --grad-checkpoint --ckpt-every 20000" \
    "vitl_100k"

# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Augmentation 3-seed validation
# Spatial-only vs ColorJitter, 10K steps each, 3 seeds
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT 2: Augmentation 3-Seed Validation (10K steps)       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Backup original file
cp "$ORIG_FILE" "${ORIG_FILE}.bak_round6"

for SEED in 42 123 7; do
    echo ""
    echo "──── SEED=$SEED ────"

    # Arm A: Spatial-only (default, no patch needed)
    cp "${ORIG_FILE}.bak_round6" "$ORIG_FILE"
    # Re-apply fp16 fix
    sed -i 's/masked_fill_(mask, -9e15)/masked_fill_(mask, -1e4)/g' "$ORIG_FILE" 2>/dev/null || true

    run_arm "aug_spatial_s${SEED}" \
        "--config vit-small --lr 2e-4 --batch-size 64 --max-steps 10000 --train-seed $SEED --gram-weight 1.0 --koleo-weight 0.1" \
        "aug_spatial_s${SEED}"

    # Arm B: Spatial + ColorJitter
    cp "${ORIG_FILE}.bak_round6" "$ORIG_FILE"
    sed -i 's/masked_fill_(mask, -9e15)/masked_fill_(mask, -1e4)/g' "$ORIG_FILE" 2>/dev/null || true
    python3 -c "
code = open('$ORIG_FILE').read()
old = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),'''
new = '''transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),'''
code = code.replace(old, new)
open('$ORIG_FILE', 'w').write(code)
"
    run_arm "aug_cj_s${SEED}" \
        "--config vit-small --lr 2e-4 --batch-size 64 --max-steps 10000 --train-seed $SEED --gram-weight 1.0 --koleo-weight 0.1" \
        "aug_cj_s${SEED}"
done

# Restore original
cp "${ORIG_FILE}.bak_round6" "$ORIG_FILE"
sed -i 's/masked_fill_(mask, -9e15)/masked_fill_(mask, -1e4)/g' "$ORIG_FILE" 2>/dev/null || true
rm "${ORIG_FILE}.bak_round6"

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ROUND 6 COMPLETE                                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
grep "^METRICS:" "$RESULTS_DIR"/*.log 2>/dev/null || echo "(check logs in $RESULTS_DIR/)"
echo ""
echo "Finished: $(date -u)"
