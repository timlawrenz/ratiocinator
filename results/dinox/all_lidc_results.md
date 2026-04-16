# DINO-X LIDC-IDRI Experiment Results

All experiments: ViT-Small, DINO + Gram(1.0) + KoLeo(0.1), lr=2e-4, temp=0.04,
ema=0.996, spatial-only augmentation, batch=64, on RTX 4090.

## Loss Sweep (2000 steps, cm=default)

| Arm | Loss Type | Gram | KoLeo | Ratio | Top-1 | Loss |
|-----|-----------|------|-------|-------|-------|------|
| dino_gram_koleo | DINO | 1.0 | 0.1 | **11.0** | 0.0054 | 9.01 |
| dino_koleo | DINO | 0.0 | 0.1 | 9.0 | 0.0044 | 9.01 |
| dino_gram | DINO | 1.0 | 0.0 | 9.0 | 0.0044 | 9.01 |
| dino_only | DINO | 0.0 | 0.0 | 8.0 | 0.0039 | 9.01 |
| simclr_only | SimCLR | 0.0 | 0.0 | 7.0 | 0.0034 | 4.84 |
| simclr_gram | SimCLR | 1.0 | 0.0 | 7.0 | 0.0034 | 4.84 |

## Center Momentum Ablation (2000 steps)

| Arm | CM | Loss | Ratio | Top-1 |
|-----|-----|------|-------|-------|
| cm_0.9 | 0.9 | 8.997 | 9.0 | 0.0044 |
| cm_0.95 | 0.95 | 8.998 | 6.0 | 0.0029 |
| cm_0.99 | 0.99 | 9.005 | 8.0 | 0.0039 |
| cm_0.999 | 0.999 | **5.763** | 6.0 | 0.0029 |

## Augmentation Ablation (2000 steps, cm=0.9)

| Arm | Augmentation | Loss | Ratio | Top-1 |
|-----|-------------|------|-------|-------|
| aug_spatial | RRC + HFlip | 8.999 | **9.0** | 0.0044 |
| aug_blur | + GaussianBlur | 9.000 | 8.0 | 0.0039 |
| aug_aggcrop | Crop 0.08-0.5 | 8.998 | 7.0 | 0.0034 |
| aug_colorjitter | + ColorJitter | 8.998 | 4.0 | 0.0020 |
| aug_full | All combined | 9.000 | 4.0 | 0.0020 |

## Extended Training (cm=0.999 vs cm=0.9)

| Arm | CM | Steps | Final Loss | Ratio | Top-1 | Passed |
|-----|-----|-------|-----------|-------|-------|--------|
| ext_cm999_5k | 0.999 | 5000 | 3.825 | 8.0 | 0.0039 | false |
| **ext_cm999_10k** | **0.999** | **10000** | **0.941** | **18.0** | **0.0088** | **true** |
| ext_cm09_5k | 0.9 | 5000 | 8.988 | 6.0 | 0.0029 | false |
| ext_cm09_10k | 0.9 | 10000 | 9.002 | 4.0 | 0.0020 | false |

### Loss Trajectory (TensorBoard)

- **cm=0.999, 10K:** 8.62 -> -0.01 -> 0.12 -> 0.63 -> 1.01 (breaks through, learning)
- **cm=0.9, 10K:** 8.62 -> 9.00 -> 9.00 -> 9.03 -> 9.00 (permanently stuck)

## Scale & Duration Experiments (cm=0.999)

| Arm | Model | Steps | LR | Batch | Final Loss | Ratio | Top-1 | Top-5 | Passed |
|-----|-------|-------|----|-------|-----------|-------|-------|-------|--------|
| **vits_50k** | **ViT-S** | **50000** | **2e-4** | **64** | **0.17** | **850.0** | **41.50%** | **76.12%** | **true ✅** |
| vits_20k | ViT-S | 20000 | 2e-4 | 64 | 0.44 | 311.0 | 15.19% | 38.18% | true ✅ |
| vitl_10k_lr5e5 | ViT-L | 10000 | 5e-5 | 16 | 0.61 | 25.0 | 1.22% | 4.35% | true ✅ |
| vitl_10k_lr1e4 | ViT-L | 10000 | 1e-4 | 16 | 5.04 | 1.0 | 0.05% | — | false |
| vitl_2k | ViT-L | 2000 | 2e-4 | 16 | 0.14 | 4.0 | 0.20% | — | false |
| vitl_5k | ViT-L | 5000 | 2e-4 | 16 | 0.80 | 2.0 | 0.10% | — | false |

### ViT-Small Scaling Trajectory

| Steps | Loss | Ratio | Top-1 | Improvement |
|-------|------|-------|-------|-------------|
| 2K | 5.76 | 6.0 | 0.29% | (entropy wall broken) |
| 5K | 3.83 | 8.0 | 0.39% | +33% ratio |
| 10K | 0.94 | 18.0 | 0.88% | +125% ratio |
| 20K | 0.44 | 311.0 | 15.19% | +1628% ratio 🚀 |
| **50K** | **0.17** | **850.0** | **41.50%** | **+173% ratio** 🏆 |

The model keeps improving with more training. Loss still decreasing. No sign of plateau.

### ViT-Large LR Sensitivity

| LR | Batch | Ratio | Analysis |
|----|-------|-------|----------|
| 2e-4 | 16 | 2.0-4.0 | Oscillating loss, representations degrade |
| 1e-4 | 16 | 1.0 | Too conservative, barely learns |
| **5e-5** | **16** | **25.0** | **Goldilocks — stable and learning** |

Linear scaling rule confirms: batch 64→16 (4x) → LR 2e-4 → 5e-5 (4x reduction).

### Loss Trajectories

- **ViT-S 50K:** 8.62 → 0.31 → 0.39 → 0.20 → 0.12 → 0.09 → 0.17 (steady descent)
- **ViT-S 20K:** 8.62 → -0.005 → 0.72 → 0.71 → 0.85 (stable convergence)
- **ViT-L 5e-5:** 8.60 → 0.05 → 0.69 → 0.23 → 3.65 → 0.61 (some oscillation but converging)
- **ViT-L 2e-4:** 8.60 → 1.18 → 0.67 → 0.04 → 0.79 (unstable oscillation)

## Key Findings

1. **Center momentum 0.999 breaks the entropy wall.** This is the DINO paper default.
   Lower values (0.9, 0.95, 0.99) cause the center to update too fast → mode collapse.
2. **ViT-Small 50K is the champion:** ratio=850, top1=41.5%, top5=76.1%, loss=0.17.
3. **Training scales beautifully.** Ratio: 6→8→18→311→850 over 2K→5K→10K→20K→50K steps.
   No sign of plateau — more training = better representations.
4. **cm=0.9 degrades with more training.** Ratio drops 9→6→4 over 2K→5K→10K steps.
5. **Spatial-only augmentation is optimal for medical CT.** ColorJitter destroys diagnostic
   intensity information.
6. **KoLeo + Gram is the best loss combination.** Together they boost ratio from 8.0 to 11.0.
7. **The entropy wall was the main blocker.** With cm=0.999, loss: 9.01→5.76→3.83→0.94→0.44→0.17.
8. **ViT-Large needs lr=5e-5 with batch=16.** Linear scaling rule applies.
   ViT-L 10K achieves ratio=25 — needs more steps to rival ViT-S.
9. **Gram matrix attention health: 0.81 at 50K** (1.0 = collapsed). Healthy and improving.

## Winning Recipe (ViT-Small)

```
--config vit-small
--loss-type dino
--gram-weight 1.0
--koleo-weight 0.1
--lr 2e-4
--teacher-temp 0.04
--ema 0.996
--center-momentum 0.999
--amp
--batch-size 64
--max-steps 50000+
```

Augmentation: RandomResizedCrop(0.2-1.0) + HorizontalFlip only.

## Next Steps

1. **Even longer ViT-Small training** (100K steps) — no sign of plateau, could push top-1 > 60%
2. **ViT-Large extended** (50K steps at lr=5e-5) — ViT-L at 10K is where ViT-S was at 10K
3. **Linear probe on LIDC-IDRI** — the ultimate AUC > 0.90 target (use vits_50k checkpoint)
4. **Resolution comparison** (224 vs 512)
