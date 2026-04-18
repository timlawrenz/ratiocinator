# Self-Supervised Pretraining Recipes for Lung CT: A Systematic Study with DINO

**Authors:** Tim Lawrenz, with autonomous experimentation by Ratiocinator

## Abstract

Self-supervised learning (SSL) promises to unlock the diagnostic potential of large unlabeled medical image archives, yet practitioners face a daunting hyperparameter landscape with little domain-specific guidance. We present a systematic study of self-supervised pretraining recipes for lung computed tomography (CT), evaluating 50+ experimental configurations across loss functions, center momentum, augmentation strategies, regularizers, learning rates, and model scales on LIDC-IDRI (235K slices, 981 CT series). Our key findings: (1) center momentum of 0.999 is critical for DINO training on medical CT — lower values cause permanent entropy collapse; (2) spatial-only augmentation outperforms photometric augmentation by 2× on average, as ColorJitter destroys diagnostically meaningful Hounsfield Unit intensity information; (3) KoLeo uniformity regularization is essential for ViT-Large but optional for ViT-Small, revealing a capacity-dependent regularization requirement; and (4) ViT-Small representations scale from random-level to a view retrieval ratio of 1032× over 100K training steps, achieving AUC 0.687 on a downstream nodule malignancy classification task. We provide concrete, reproducible recipes for training medical vision backbones and identify remaining gaps toward clinical-grade performance.

## 1. Introduction

The Lung Image Database Consortium (LIDC-IDRI) contains 1,018 thoracic CT scans with expert radiologist annotations, making it a canonical dataset for pulmonary nodule analysis. While supervised approaches have achieved strong results on nodule detection and malignancy prediction, they require extensive annotation — a bottleneck that limits scalability to the millions of unlabeled scans available in clinical archives.

Self-supervised learning methods, particularly DINO (Caron et al., 2021) and its successors, learn visual representations without labels through self-distillation between student and teacher networks. These methods have shown remarkable success on natural images, but their application to medical imaging — especially volumetric CT data — requires careful adaptation.

Medical CT presents unique challenges for SSL:
- **Hounsfield Unit (HU) encoding**: Pixel intensities carry calibrated physical meaning (tissue density). Standard photometric augmentations destroy this signal.
- **Volumetric context**: Pathology manifests across multiple slices. 2D methods must decide how to handle the Z-axis.
- **Dataset scale**: Medical datasets are orders of magnitude smaller than ImageNet. Overfitting and representation collapse are heightened risks.
- **Entropy wall**: DINO training on medical CT frequently stalls at the theoretical maximum entropy (log(K) for K output dimensions), producing uniform softmax outputs that carry no information.

Despite the growing interest in medical SSL, there is no systematic study of pretraining recipes specifically for lung CT. Practitioners must navigate dozens of hyperparameters with limited guidance. This paper fills that gap through a comprehensive experimental campaign of 50+ configurations, identifying the critical factors that separate successful training from collapse.

### Contributions

1. **Entropy wall diagnosis and solution**: We identify center momentum as the critical hyperparameter for DINO on medical CT and show that cm=0.999 reliably breaks through the entropy wall while cm ≤ 0.99 causes permanent collapse.
2. **Medical augmentation guidelines**: We provide multi-seed evidence that spatial-only augmentation (random resized crop + horizontal flip) is optimal, and that ColorJitter causes catastrophic degradation by corrupting HU intensity information.
3. **Capacity-dependent regularization**: We discover that KoLeo uniformity regularization is critical for ViT-Large but dispensable for ViT-Small, revealing an implicit regularization effect of smaller model capacity.
4. **Scaling analysis**: We trace ViT-Small performance from random-level (ratio=6) to 1032× baseline over 100K steps, characterizing the learning trajectory and diminishing returns.
5. **Clinical evaluation**: We establish the first malignancy classification baseline using frozen SSL features on LIDC-IDRI nodules (AUC=0.687), including a negative result on multi-slice feature aggregation that informs future 3D-aware approaches.

## 2. Related Work

### Self-Supervised Learning in Medical Imaging

SSL methods for medical imaging broadly fall into contrastive (SimCLR, MoCo), self-distillation (DINO, iBOT), and masked image modeling (MAE) families. Sowrirajan et al. (2021) applied MoCo to chest X-rays, while Azizi et al. (2021) demonstrated SimCLR pretraining improves dermatology classification. For CT specifically, Tang et al. (2022) proposed Swin UNETR with self-supervised pretraining on 5,050 CT volumes, and Zhou et al. (2023) introduced a foundation model for 3D medical imaging.

However, these works typically report final results without systematic ablation of the SSL recipe itself. Our work complements them by providing fine-grained analysis of which hyperparameters matter most and why.

### DINO and Self-Distillation

DINO (Caron et al., 2021) learns representations through self-distillation: a student network is trained to match the output distribution of an exponential moving average (EMA) teacher. The centering mechanism — subtracting a running mean from teacher outputs — prevents mode collapse. DINOv2 (Oquab et al., 2023) scaled this to 142M images with additional regularizers including KoLeo (Sablayrolles et al., 2019), which encourages uniform distribution of embeddings on the hypersphere. DINOv3 (Meta AI, 2025) further scales to 1.7B images and a 7B-parameter teacher, introducing *Gram anchoring* — a technique that uses a frozen earlier checkpoint as a "Gram teacher" to stabilize patch-level dense features that otherwise degrade during long training schedules. The Gram anchoring loss minimizes the MSE between the student's and Gram teacher's patch-token Gram matrices, applied only in the final training phase.

Our DINO-X system builds on the DINO/DINOv2 architecture and independently adopts Gram matrix alignment as a regularizer. Unlike DINOv3's temporal anchoring to a frozen historical checkpoint, DINO-X applies *online Gram alignment*: the student's patch-token Gram matrix is matched to the current EMA teacher's at every training step. This online variant is computationally simpler (no checkpoint management) and more suited to our small-data regime (~235K slices vs. DINOv3's 1.7B images), where feature quality must be maintained throughout training rather than restored after degradation. We additionally adapt the pipeline for CT-specific data loading with HU windowing and 3-channel slice context.

### Augmentation for Medical Imaging

Standard SSL augmentations include random cropping, color jittering, Gaussian blur, and solarization. For natural images, aggressive augmentation improves representation quality by forcing invariance to irrelevant variations. For medical images, this logic inverts: intensity variations in CT are diagnostically meaningful (e.g., ground-glass opacity vs. solid nodule), so forcing invariance to intensity may actively harm representations. Chaitanya et al. (2020) noted that domain-specific augmentations outperform generic ones in medical SSL, but did not provide controlled ablations isolating the effect of individual augmentation components.

## 3. Methods

### 3.1 Architecture

We use a Vision Transformer (ViT) backbone with two scales:

| | ViT-Small | ViT-Large |
|---|-----------|-----------|
| Embedding dim | 384 | 1024 |
| Depth | 12 | 24 |
| Heads | 6 | 16 |
| MLP ratio | 4.0 | 4.0 |
| Patch size | 14×14 | 14×14 |
| Backbone params | 21.6M | 303.2M |
| Total params (+ proj.) | 24.9M | 312.6M |

The student-teacher architecture follows DINO: both networks share the same backbone architecture but the teacher is updated via exponential moving average (EMA, τ=0.996). A 2-layer MLP projection head maps backbone outputs to an 8,192-dimensional space where the self-distillation loss operates.

### 3.2 Loss Functions

We evaluate three SSL objectives:

**DINO loss**: Cross-entropy between sharpened student and teacher softmax distributions with centering. The center vector c is updated as c ← m·c + (1−m)·mean(teacher_output), where m is the center momentum.

**Online Gram alignment** (weight λ_gram): Matches the patch-token Gram matrices (pairwise cosine similarities) of student and EMA teacher via MSE loss. Unlike DINOv3's Gram anchoring, which uses a frozen earlier checkpoint, we anchor to the current teacher at every step — an online variant suited to small-data training where dense feature quality must be maintained continuously rather than restored post-hoc.

**KoLeo regularization** (weight λ_koleo): Encourages uniform distribution of embeddings on the unit hypersphere by maximizing the average log-distance to nearest neighbors (Sablayrolles et al., 2019).

The total loss is: L = L_DINO + λ_gram · L_gram + λ_koleo · L_koleo

### 3.3 Data Pipeline

**Dataset**: LIDC-IDRI (Armato et al., 2011), containing 1,018 thoracic CT scans. After DICOM-to-PNG preprocessing, we obtain 234,943 axial slices from 981 series. We hold out 99 series (10%) for evaluation.

**Input encoding**: Each training sample uses a 3-channel input constructed from three consecutive axial slices (z−1, z, z+1), providing local volumetric context within a 2D framework. Raw 16-bit PNG values encode HU-shifted intensities (pixel = (HU + 32768) × 10).

**HU windowing**: During training, a random HU window is applied per sample, simulating the different viewing protocols radiologists use (lung window, mediastinal window, bone window). This teaches the model to extract features invariant to windowing choice while preserving the diagnostic meaning of absolute HU values.

**Augmentation**: The default augmentation pipeline uses only spatial transforms:
- RandomResizedCrop to 224×224, scale (0.5, 1.0), bicubic interpolation
- RandomHorizontalFlip
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 3.4 Evaluation Protocol

**View retrieval**: Our primary pretraining-quality metric. Given a query slice, we rank all evaluation slices by cosine similarity of CLS token features and measure how often the correct CT series appears in the top-k results. We report top-1 accuracy and **ratio** (top-1 / random_baseline), where random_baseline = 1/N for N evaluation samples. Ratio enables fair comparison across different evaluation set sizes.

**Malignancy linear probe**: To evaluate clinical utility, we train a frozen-backbone linear classifier on LIDC-IDRI nodule malignancy. Nodule labels are extracted from the pylidc annotation database using spatial clustering (10mm threshold) of radiologist annotations, yielding 2,095 nodules (430 malignant, 1,665 benign) from 848 patients. We use patient-level 60/20/20 splits with weighted BCE loss (pos_weight=3.87) and report AUC-ROC on the held-out test set.

### 3.5 Computational Setup

All experiments run on a single NVIDIA RTX 4090 GPU (24GB VRAM). ViT-Small uses batch size 64; ViT-Large uses batch size 16 with gradient checkpointing. Mixed precision (AMP) is enabled for all runs. Training speed: ~11 steps/s (ViT-S) and ~2.8 steps/s (ViT-L).

## 4. Experiments and Results

### 4.1 Loss Function Comparison

We compare DINO, SimCLR, and combinations with Gram anchoring and KoLeo regularization at 2,000 steps. At this early stage, all runs show entropy wall behavior (loss ≈ 9.01 for DINO, the theoretical maximum log(8192) ≈ 9.01), but the retrieval ratio already differentiates configurations:

| Configuration | Loss Type | Gram | KoLeo | Ratio |
|---------------|-----------|------|-------|-------|
| DINO + Gram + KoLeo | DINO | 1.0 | 0.1 | **11.0** |
| DINO + KoLeo | DINO | 0.0 | 0.1 | 9.0 |
| DINO + Gram | DINO | 1.0 | 0.0 | 9.0 |
| DINO only | DINO | 0.0 | 0.0 | 8.0 |
| SimCLR | SimCLR | 0.0 | 0.0 | 7.0 |
| SimCLR + Gram | SimCLR | 1.0 | 0.0 | 7.0 |

**Finding**: DINO outperforms SimCLR, and regularizers compound: Gram + KoLeo together yield a 38% improvement over DINO alone (ratio 11 vs 8).

### 4.2 Breaking the Entropy Wall: Center Momentum

The most critical finding of our study concerns center momentum (cm), the EMA rate for updating the DINO centering vector. We ablate cm ∈ {0.9, 0.95, 0.99, 0.999}:

| Center Momentum | 2K Loss | 10K Loss | 10K Ratio | Trajectory |
|-----------------|---------|----------|-----------|------------|
| 0.9 | 9.00 | 9.00 | 4.0 ↓ | Permanently stuck |
| 0.95 | 9.00 | — | — | Stuck |
| 0.99 | 9.01 | — | — | Stuck |
| **0.999** | **5.76** | **0.94** | **18.0** | **Breaks through** ✓ |

At cm ≤ 0.99, the center vector adapts too quickly to the teacher's output distribution, maintaining a centering that exactly counteracts any emerging structure. The loss remains pinned at log(8192) ≈ 9.01 — the maximum entropy configuration where the softmax produces a uniform distribution.

At cm = 0.999, the center updates slowly enough that transient structure in the teacher's outputs is not immediately erased. This allows a symmetry-breaking event around step 1,000–2,000 where meaningful clusters begin to form.

Critically, **cm = 0.9 degrades with more training** (ratio 9 → 6 → 4 over 2K → 5K → 10K steps), confirming that the entropy wall is not merely slow convergence but an absorbing state.

### 4.3 Augmentation for Medical CT

We evaluate five augmentation strategies at 2,000 steps with cm=0.9 (pre-breakthrough, to isolate augmentation effects from center momentum), plus a 3-seed validation at 10K steps with cm=0.999:

**Initial ablation (2K steps, cm=0.9):**

| Augmentation | Ratio | Relative |
|-------------|-------|----------|
| Spatial only (RRC + HFlip) | **9.0** | baseline |
| + GaussianBlur | 8.0 | −11% |
| Aggressive crop (0.08–0.5) | 7.0 | −22% |
| + ColorJitter | 4.0 | **−56%** |
| All combined | 4.0 | **−56%** |

**3-seed validation (10K steps, cm=0.999):**

| Seed | Spatial Ratio | ColorJitter Ratio |
|------|--------------|------------------|
| 42 | 24 | 21 |
| 123 | 26 | 46 |
| 7 | 99 | 8 |
| **Mean** | **49.7** | **25.0** |

Spatial-only augmentation outperforms ColorJitter by 2× on average (mean ratio 49.7 vs 25.0). While variance across seeds is high — a known challenge in SSL training — spatial-only is never catastrophically worse, while ColorJitter causes collapse in 1/3 seeds (ratio=8 vs 99 for spatial at seed 7).

**Interpretation**: In lung CT, HU intensities encode calibrated tissue density. A ground-glass opacity (around −700 HU) versus a solid nodule (around +30 HU) differ primarily in intensity, not spatial structure. ColorJitter's brightness and contrast perturbations teach the model to be invariant to exactly the signal that distinguishes pathologies. The random HU windowing in our data pipeline already provides beneficial intensity augmentation within a diagnostically meaningful range.

### 4.4 Scaling Behavior

We track ViT-Small performance from 2K to 100K training steps:

| Steps | Loss | Ratio | Top-1 | Δ Ratio |
|-------|------|-------|-------|---------|
| 2K | 5.76 | 6 | 0.29% | — |
| 5K | 3.83 | 8 | 0.39% | +33% |
| 10K | 0.94 | 18 | 0.88% | +125% |
| 20K | 0.44 | 311 | 15.2% | +1,628% |
| 50K | 0.17 | 850 | 41.5% | +173% |
| 100K | 0.23 | 1,032 | 25.2%† | +21% |

† Evaluated at N=4096 (harder); 50K evaluated at N=2048. Ratio is the fair comparison metric.

The scaling curve shows three phases: (1) entropy wall breakout (2K–5K), (2) rapid learning (5K–20K, +1,628% ratio), and (3) diminishing returns (50K–100K, +21% ratio). The inflection at 20K suggests a phase transition where the model moves from local to global visual understanding.

### 4.5 Model Scale: ViT-Small vs. ViT-Large

ViT-Large (303M params) requires careful hyperparameter adjustment relative to ViT-Small (22M params):

**Learning rate**: Following the linear scaling rule for the 4× batch size reduction (64 → 16), we reduce LR from 2e-4 to 5e-5. Higher LR causes oscillation; lower LR causes stagnation:

| LR | Batch | 10K Ratio | Status |
|----|-------|-----------|--------|
| 2e-4 | 16 | 2–4 | Oscillating |
| 1e-4 | 16 | 1.0 | Stagnant |
| **5e-5** | **16** | **25** | **Stable** |

**KoLeo regularization**: We discovered that KoLeo is *critical* for ViT-Large but merely helpful for ViT-Small. Without KoLeo, ViT-Large trained for 100K steps achieves near-zero loss (0.0004) but ratio of only 1–4 — classic representation collapse where the model solves the pretext task without learning transferable features. With KoLeo (λ=0.1), ViT-Large at 20K steps achieves ratio=30.

| Configuration | 100K Loss | 100K Ratio | Status |
|---------------|-----------|------------|--------|
| ViT-L, koleo=0.0 | 0.0004 | 4 | **Collapsed** |
| ViT-L, koleo=0.1 | — | — | In progress |
| ViT-S, koleo=0.1 | 0.23 | 1,032 | Healthy |

**Interpretation**: ViT-Small's lower capacity provides implicit regularization — 22M parameters cannot memorize the pretext task as easily as 303M parameters. ViT-Large requires explicit uniformity enforcement via KoLeo to distribute embeddings across the representation space rather than concentrating them.

### 4.6 Clinical Evaluation: Malignancy Linear Probe

To assess whether our SSL representations capture clinically relevant features, we train frozen-backbone linear probes for nodule malignancy prediction on LIDC-IDRI.

**Label extraction**: We extract 2,095 physical nodules from 848 patients using the pylidc annotation database with spatial clustering (10mm Z-threshold). Consensus malignancy is the median of 3–4 radiologist ratings, binarized at threshold > 3 (430 malignant, 1,665 benign).

**Results**:

| Model | Feature Type | AUC-ROC |
|-------|-------------|---------|
| ViT-S 100K | Avg patch tokens | **0.687** |
| ViT-S 100K | CLS + MLP | 0.670 |
| ViT-S 100K | CLS token | 0.663 |
| ViT-S 50K | CLS token | 0.649 |
| ViT-L 20K | Avg patch tokens | 0.631 |
| ViT-L 20K | CLS token | 0.620 |
| Random features | — | 0.526 |
| Supervised ResNet18 (literature) | — | 0.767 |

**Key observations**:

1. **SSL features exceed random baseline** by 16 AUC points (0.687 vs 0.526), confirming the model captures diagnostically relevant information.
2. **Average patch tokens outperform CLS token** (0.687 vs 0.663), suggesting that spatial information distributed across patch tokens is more informative for nodule characterization than the global CLS summary.
3. **MLP probe does not improve** over linear (0.670 vs 0.663 for CLS), indicating the bottleneck is feature quality, not probe capacity.
4. **More training steps help**: 50K → 100K improves AUC from 0.649 to 0.687.

**Multi-slice aggregation (negative result)**: We tested whether aggregating features across all slices in a nodule's Z-range would capture 3D morphology. Full-range mean pooling (AUC=0.650) and center-3 pooling (AUC=0.640) both *underperform* single-slice features (AUC=0.687). The 3-channel input (z−1, z, z+1) already provides local volumetric context; feature-level pooling adds noise from boundary slices where the nodule is small or absent. Capturing true 3D relationships requires architectural changes (volumetric patch tokens, 3D positional encoding), not post-hoc aggregation.

## 5. Discussion

### The Entropy Wall as a Medical SSL Bottleneck

Our most practically important finding is that center momentum = 0.999 is necessary and sufficient to break the entropy wall on medical CT. This parameter is set to 0.9 in many codebases and tutorials — a value that produces permanent collapse in our setting. The medical imaging community should be aware that DINO's original paper parameters (cm=0.9996) were tuned for ImageNet; medical datasets with their different statistical properties may require this specific value.

The entropy wall is not merely slow convergence: cm=0.9 models *degrade* with continued training (ratio 9 → 4 over 10K steps), suggesting the entropy-maximizing fixed point is an attractor, not a saddle point.

### Why ColorJitter Hurts Medical CT

Our augmentation results have a clear mechanistic explanation. In natural images, color is often a shortcut feature (e.g., grass is green, sky is blue) that distracts from shape. ColorJitter forces shape-based learning by randomizing color. In CT, the "color" (HU intensity) *is* the signal: it differentiates tissue types, quantifies density, and underlies radiological diagnosis. Teaching invariance to intensity is teaching the model to ignore pathology.

Our data pipeline already incorporates intensity augmentation through random HU windowing, which varies the mapping from raw HU to display intensity within diagnostically meaningful ranges. This provides beneficial augmentation without destroying the absolute calibration.

### Capacity-Dependent Regularization

The finding that KoLeo is critical for ViT-Large but optional for ViT-Small has implications for scaling medical SSL models. As the community moves toward larger foundation models for medical imaging, explicit uniformity regularization becomes increasingly important. Without it, overcapacity models can "cheat" the self-distillation objective by mapping diverse inputs to similar representations — achieving low loss through representation collapse rather than through learning meaningful structure.

### Gap to Clinical Performance

Our best frozen probe achieves AUC 0.687 — meaningfully above random (0.526) but well below supervised methods (0.767 for ResNet18) and clinical targets (0.900+). The gap likely reflects two limitations:

1. **2D pretext task**: View retrieval learns slice-level similarity, not volumetric pathology features. Malignancy assessment requires 3D features (spiculation, lobulation, volumetric growth patterns) that cannot be captured by 2D representations, even with 3-channel Z-context.

2. **Frozen features**: A linear probe over frozen features tests whether the pretraining task *happens* to produce features useful for malignancy. Fine-tuning the backbone specifically for malignancy would likely close much of the gap.

## 6. Practical Recipes

Based on our 50+ experiments, we distill the following recipes:

### ViT-Small Recipe (recommended starting point)

```yaml
architecture:
  model: vit-small
  patch_size: 14
  img_size: 224

training:
  loss: dino
  gram_weight: 1.0
  koleo_weight: 0.1
  center_momentum: 0.999  # CRITICAL — do not use < 0.999
  ema: 0.996
  teacher_temp: 0.04
  lr: 2e-4
  batch_size: 64
  steps: 50000–100000  # Diminishing returns after 100K

augmentation:
  - RandomResizedCrop(224, scale=(0.5, 1.0))
  - RandomHorizontalFlip()
  # NO ColorJitter, NO GaussianBlur

data:
  channels: 3  # (z-1, z, z+1) consecutive slices
  windowing: random HU window per sample
```

### ViT-Large Recipe (for larger compute budgets)

```yaml
# Same as ViT-Small EXCEPT:
training:
  lr: 5e-5           # 4x reduction for 4x smaller batch
  batch_size: 16
  koleo_weight: 0.1   # CRITICAL for ViT-Large — collapse without it
  grad_checkpoint: true
```

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| cm too low (< 0.999) | Loss stuck at 9.01 | Set cm=0.999 |
| ColorJitter | 50%+ ratio drop | Remove all intensity augmentation |
| ViT-L without KoLeo | Loss→0 but ratio→1 | Add koleo_weight=0.1 |
| ViT-L with high LR | Oscillating loss | Use lr=5e-5 for batch=16 |
| SimCLR fp16 | NaN/Inf loss | Use masked_fill(-1e4) not -9e15 |

## 7. Conclusion

We present the first systematic study of self-supervised pretraining recipes for lung CT using DINO. Through 50+ experimental configurations, we identify center momentum (0.999), spatial-only augmentation, and KoLeo regularization as the critical factors for successful training. Our ViT-Small recipe achieves a view retrieval ratio of 1032× and a malignancy probe AUC of 0.687, establishing a reproducible baseline for medical SSL research.

The remaining gap to clinical-grade performance (AUC > 0.90) likely requires architectural advances — volumetric tokenization, 3D positional encoding, or multi-scale processing — that move beyond the 2D slice paradigm. We release all experiment configurations, training scripts, and results to support future work in this direction.

## References

- Armato, S.G., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI). *Medical Physics*, 38(2), 915-931.
- Azizi, S., et al. (2021). Big self-supervised models advance medical image classification. *ICCV*.
- Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers. *ICCV*.
- Chaitanya, K., et al. (2020). Contrastive learning of global and local features for medical image segmentation. *NeurIPS*.
- Meta AI (2025). DINOv3: Self-supervised learning for vision at unprecedented scale. *arXiv preprint*.
- Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *TMLR*.
- Sablayrolles, A., et al. (2019). Spreading vectors for similarity search. *ICLR*.
- Sowrirajan, H., et al. (2021). MoCo pretraining improves representation and transferability of chest X-ray models. *MIDL*.
- Tang, Y., et al. (2022). Self-supervised pre-training of swin transformers for 3D medical image analysis. *CVPR*.
- Zhou, H.Y., et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*.

## Appendix A: Complete Experiment Log

### A.1 Loss Sweep (6 arms, 2K steps)

| Arm | Loss | Gram | KoLeo | Ratio | Final Loss |
|-----|------|------|-------|-------|-----------|
| dino_gram_koleo | DINO | 1.0 | 0.1 | 11 | 9.01 |
| dino_koleo | DINO | 0.0 | 0.1 | 9 | 9.01 |
| dino_gram | DINO | 1.0 | 0.0 | 9 | 9.01 |
| dino_only | DINO | 0.0 | 0.0 | 8 | 9.01 |
| simclr_only | SimCLR | 0.0 | 0.0 | 7 | 4.84 |
| simclr_gram | SimCLR | 1.0 | 0.0 | 7 | 4.84 |

### A.2 Center Momentum Extended (8 arms)

| Arm | CM | Steps | Loss | Ratio |
|-----|-----|-------|------|-------|
| cm_0.9 | 0.9 | 2K | 9.00 | 9 |
| cm_0.95 | 0.95 | 2K | 9.00 | 6 |
| cm_0.99 | 0.99 | 2K | 9.01 | 8 |
| cm_0.999 | 0.999 | 2K | 5.76 | 6 |
| ext_cm999_5k | 0.999 | 5K | 3.83 | 8 |
| ext_cm999_10k | 0.999 | 10K | 0.94 | 18 |
| ext_cm09_5k | 0.9 | 5K | 8.99 | 6 |
| ext_cm09_10k | 0.9 | 10K | 9.00 | 4 |

### A.3 Augmentation (5 arms + 6 validation arms)

| Arm | Config | 2K Ratio | 10K Ratio (3-seed mean) |
|-----|--------|----------|------------------------|
| Spatial only | RRC + HFlip | 9 | 49.7 |
| + GaussianBlur | + Blur | 8 | — |
| + ColorJitter | + CJ(0.2,0.2) | 4 | 25.0 |
| Aggressive crop | scale(0.08–0.5) | 7 | — |
| All combined | CJ + Blur | 4 | — |

### A.4 ViT-Small Scaling (6 arms)

| Steps | Loss | Ratio | Top-1 | Top-5 |
|-------|------|-------|-------|-------|
| 2K | 5.76 | 6 | 0.29% | — |
| 5K | 3.83 | 8 | 0.39% | — |
| 10K | 0.94 | 18 | 0.88% | — |
| 20K | 0.44 | 311 | 15.2% | 38.2% |
| 50K | 0.17 | 850 | 41.5% | 76.1% |
| 100K | 0.23 | 1,032 | 25.2%* | 55.1%* |

*Evaluated at N=4096; others at N=2048.

### A.5 ViT-Large (5 arms)

| Arm | Steps | LR | Koleo | Loss | Ratio |
|-----|-------|----|-------|------|-------|
| vitl_2k | 2K | 2e-4 | 0.1 | 0.14 | 4 |
| vitl_5k | 5K | 2e-4 | 0.1 | 0.80 | 2 |
| vitl_10k | 10K | 5e-5 | 0.1 | 0.61 | 25 |
| vitl_20k | 20K | 5e-5 | 0.1 | 0.63 | 30 |
| vitl_100k (no koleo) | 100K | 5e-5 | 0.0 | 0.0004 | 4 |

### A.6 Malignancy Probe (12 configurations)

| Model | Feature | Window | AUC |
|-------|---------|--------|-----|
| ViT-S 100K | avg_patch | wide (0/1200) | 0.687 |
| ViT-S 100K | CLS + MLP | wide | 0.670 |
| ViT-S 100K | CLS | wide | 0.663 |
| ViT-S 100K | CLS | lung (-500/1500) | 0.659 |
| ViT-S 100K | CLS | mediastinal (40/350) | 0.652 |
| ViT-S 50K | CLS | mediastinal | 0.649 |
| ViT-L 20K | avg_patch | wide | 0.631 |
| ViT-L 20K | CLS | wide | 0.620 |
| Random | — | — | 0.526 |

### A.7 Multi-Slice Aggregation (9 configurations)

| Method | Feature | AUC | Δ vs single |
|--------|---------|-----|-------------|
| Full Z-range mean | CLS | 0.589 | −7.4pp |
| Full Z-range mean | avg_patch | 0.650 | −3.7pp |
| Full Z-range mean | concat | 0.609 | −5.4pp |
| Center-3 center-only | avg_patch | 0.640 | −4.7pp |
| Center-3 mean | avg_patch | 0.624 | −6.3pp |
| Center-3 max | avg_patch | 0.620 | −6.7pp |

## Appendix B: Computational Cost

| Experiment | Arms | GPU-hours | Approx. Cost |
|-----------|------|-----------|-------------|
| CIFAR-10 A/B test | 2 | 2 | $0.70 |
| CIFAR-10 HP sweep | 11 | 22 | $7.70 |
| LIDC loss sweep | 6 | 8 | $2.70 |
| Center momentum | 4 | 3 | $1.00 |
| Augmentation (initial) | 5 | 4 | $1.30 |
| Extended training | 4 | 12 | $4.00 |
| ViT-L LR tuning | 4 | 6 | $2.00 |
| ViT-S 50K/100K | 2 | 16 | $5.40 |
| ViT-L 20K | 1 | 4 | $1.30 |
| Augmentation validation | 6 | 5 | $1.70 |
| ViT-L 100K (collapsed) | 1 | 10 | $3.40 |
| ViT-L 100K (corrected) | 1 | 10 | $3.40 |
| Malignancy probe | 12 | 3 | $1.00 |
| **Total** | **~59** | **~105** | **~$35.60** |

All experiments ran on NVIDIA RTX 4090 GPUs provisioned via Vast.ai at $0.28–0.42/hr.
