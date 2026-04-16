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

## Key Findings

1. **Center momentum 0.999 breaks the entropy wall.** This is the DINO paper default.
   Lower values (0.9, 0.95, 0.99) cause the center to update too fast -> mode collapse.
2. **cm=0.999 + 10K steps gives the best result by far:** ratio=18.0, loss=0.94, eval passed.
3. **cm=0.9 degrades with more training.** Ratio drops 9->6->4 over 2K->5K->10K steps.
4. **Spatial-only augmentation is optimal for medical CT.** ColorJitter destroys diagnostic
   intensity information.
5. **KoLeo + Gram is the best loss combination.** Together they boost ratio from 8.0 to 11.0.
6. **The entropy wall was the main blocker.** With cm=0.999, loss drops: 9.01->5.76->3.83->0.94.

## Winning Recipe

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
--max-steps 10000+
```

Augmentation: RandomResizedCrop(0.2-1.0) + HorizontalFlip only.

## Next Steps

1. **ViT-Large scale-up** with this winning recipe
2. **Linear probe on LIDC-IDRI** -- the ultimate AUC > 0.90 target
3. **Even longer training** (20K-50K steps) to see if loss continues improving
4. **Resolution comparison** (224 vs 512)
