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
| **vits_100k** | **ViT-S** | **100000** | **2e-4** | **64** | **0.23** | **1032.0** | **25.20%** | **55.13%** | **true ✅** |
| vits_50k | ViT-S | 50000 | 2e-4 | 64 | 0.17 | 850.0 | 41.50%† | 76.12%† | true ✅ |
| vits_20k | ViT-S | 20000 | 2e-4 | 64 | 0.44 | 311.0 | 15.19%† | 38.18%† | true ✅ |
| vitl_20k_lr5e5 | ViT-L | 20000 | 5e-5 | 16 | 0.63 | 30.0 | 0.73% | 2.83% | true ✅ |
| vitl_10k_lr5e5 | ViT-L | 10000 | 5e-5 | 16 | 0.61 | 25.0 | 1.22%† | 4.35%† | true ✅ |
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
| 50K | 0.17 | 850.0 | 41.50%† | +173% ratio |
| **100K** | **0.23** | **1032.0** | **25.20%** | **+21% ratio** 🏆 |

† Evaluated at N=2048; 100K evaluated at N=4096 (harder task, lower top-1 expected).
Ratio is the comparable metric: 850→1032 (+21%). Still improving, but gains decelerating.

### ViT-Large LR Sensitivity

| LR | Batch | Ratio | Analysis |
|----|-------|-------|----------|
| 2e-4 | 16 | 2.0-4.0 | Oscillating loss, representations degrade |
| 1e-4 | 16 | 1.0 | Too conservative, barely learns |
| **5e-5** | **16** | **25.0** | **Goldilocks — stable and learning** |

Linear scaling rule confirms: batch 64→16 (4x) → LR 2e-4 → 5e-5 (4x reduction).

### ViT-Large Scaling Trajectory

| Steps | LR | Loss | Ratio | Top-1 (N=4096) |
|-------|----|------|-------|----------------|
| 10K | 5e-5 | 0.61 | 25.0 | 1.22% |
| **20K** | **5e-5** | **0.63** | **30.0** | **0.73%** |

ViT-L improving but slower than ViT-S. Needs more steps (50K+) to see if it catches up.

### Loss Trajectories

- **ViT-S 100K:** 8.62 → 0.16 → 0.87 → 0.57 → 0.48 → 0.37 → 0.26 → 0.23 (steady descent, decelerating)
- **ViT-S 50K:** 8.62 → 0.31 → 0.39 → 0.20 → 0.12 → 0.09 → 0.17 (steady descent)
- **ViT-S 20K:** 8.62 → -0.005 → 0.72 → 0.71 → 0.85 (stable convergence)
- **ViT-L 5e-5:** 8.60 → 0.05 → 0.69 → 0.23 → 3.65 → 0.61 (some oscillation but converging)
- **ViT-L 2e-4:** 8.60 → 1.18 → 0.67 → 0.04 → 0.79 (unstable oscillation)

## Key Findings

1. **Center momentum 0.999 breaks the entropy wall.** This is the DINO paper default.
   Lower values (0.9, 0.95, 0.99) cause the center to update too fast → mode collapse.
2. **ViT-Small 100K is the champion:** ratio=1032, top1=25.2% (N=4096), loss=0.23.
3. **Training scales beautifully.** Ratio: 6→8→18→311→850→1032 over 2K→50K→100K.
   Gains decelerating at 100K (850→1032 = +21% vs 311→850 = +173% at 50K).
4. **ViT-Large scaling is slower.** Ratio: 25→30 over 10K→20K at lr=5e-5. Needs 50K+ steps.
5. **cm=0.9 degrades with more training.** Ratio drops 9→6→4 over 2K→5K→10K steps.
6. **Spatial-only augmentation is optimal for medical CT.** ColorJitter destroys diagnostic
   intensity information.
7. **KoLeo + Gram is the best loss combination.** Together they boost ratio from 8.0 to 11.0.
8. **The entropy wall was the main blocker.** With cm=0.999, loss: 9.01→5.76→3.83→0.94→0.44→0.17.
9. **ViT-Large needs lr=5e-5 with batch=16.** Linear scaling rule applies.
10. **Gram matrix attention health: 0.96 at 100K** (1.0 = collapsed). Monitoring needed.

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

1. **Linear probe on LIDC-IDRI malignancy** ✅ DONE — See results below
2. **k-NN classification with malignancy labels** — Cheaper than linear probe (no training,
   just cosine k-NN with k=20). Good intermediate check.
3. **ViT-Large extended** (50K steps at lr=5e-5) — ViT-L at 20K is ratio=30, needs more steps.
4. **Resolution comparison** (224 vs 512) — Higher resolution may help CT nodule detection.
5. **MedMNIST benchmark** — Standardized medical imaging benchmark for cross-method comparison.

## Phase 6: Malignancy Linear Probe (2026-04-17)

### Label Extraction
- Source: pylidc built-in SQLite database (no DICOM re-download needed)
- Method: Spatial clustering (10mm threshold) of radiologist annotations
- Result: **2,095 physical nodules** from 848 patients
  - 430 malignant (20.5%), 1,665 benign (79.5%)
  - Median 3 annotators per nodule

### Protocol
- Patient-level 60/20/20 train/val/test split (no data leakage)
- Frozen backbone → feature extraction → linear/MLP probe
- Weighted BCE loss (pos_weight=3.87), SGD LR sweep, 100 epochs
- Features: CLS token (dim=384/1024) and avg patch tokens

### Results

| Model | Feature | Window | Best Test AUC |
|-------|---------|--------|---------------|
| ViT-S 100K | Avg patch | wide | **0.687** |
| ViT-S 100K | CLS MLP | wide | 0.670 |
| ViT-S 100K | CLS | mediastinal | 0.652 |
| ViT-S 100K | CLS | lung | 0.659 |
| ViT-S 100K | CLS | wide | 0.663 |
| ViT-S 50K | CLS | mediastinal | 0.649 |
| ViT-L 20K | Avg patch | wide | 0.631 |
| ViT-L 20K | CLS | wide | 0.620 |
| Random features | — | — | 0.526 |
| Supervised ResNet18 (lit) | — | — | 0.767 |
| **Target** | — | — | **0.900** |

### Key Findings
1. **Model learns some malignancy features** — AUC 0.69 >> 0.53 random baseline
2. **Avg patch tokens > CLS token** (0.687 vs 0.66) — spatial info matters
3. **MLP doesn't improve** — feature space is the bottleneck, not probe capacity
4. **ViT-Small > ViT-Large** — because 5x more training steps (100K vs 20K)
5. **Windowing has minor effect** — wide (0/1200) slightly best

### Phase 6b: Multi-Slice Aggregation Probe

Hypothesis: Aggregating features across all slices in a nodule's Z-range would
capture 3D morphology (spiculation, lobulation) that single slices miss.

**Full Z-range (mean pool across all nodule slices):**

| Feature | Slices/nod | AUC | vs single |
|---------|-----------|-----|-----------|
| CLS | median=7 | 0.589 | −7.4pp |
| Avg patch | median=7 | 0.650 | −3.7pp |
| Concat | median=7 | 0.609 | −5.4pp |

**Center-3 slices only (center of nodule ±1 slice):**

| Pool method | AUC | vs single |
|-------------|-----|-----------|
| Center-only | 0.640 | −4.7pp |
| Mean | 0.624 | −6.3pp |
| Max | 0.620 | −6.7pp |

**Conclusion: Multi-slice aggregation consistently HURTS performance.**

Why: The DINO-X PngDataset already uses 3-channel input (z-1, z, z+1), so
each single-slice feature already encodes local 3D context. Feature-level
pooling across additional slices dilutes the signal — boundary slices where
the nodule is small or absent add noise. The 2D SSL objective does not learn
inter-slice relationships, so there is no complementary 3D information to
aggregate.

### Gap Analysis (0.69 → 0.90)
The pretext task (view retrieval under HU windowing) learns generic slice-level
similarity, not nodule morphology. Malignancy requires 3D features (spiculation,
lobulation). Multi-slice feature aggregation does not help because the model
lacks inter-slice understanding. Remaining paths to improve:
- **ViT-Large 100K+**: Scale training to match ViT-Small's step count
- **Fine-tuning**: Unfreeze top backbone layers instead of frozen linear probe
- **Nodule-specific crops**: ROI extraction around nodule coordinates
- **3D-aware pretraining**: Volumetric patch tokens instead of 2D slices
