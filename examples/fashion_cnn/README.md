# fashion_cnn — Ratiocinator GPU Test Bench

A self-contained PyTorch CNN training script for testing the Ratiocinator
pipeline on Vast.ai GPU instances. Trains on FashionMNIST (auto-downloaded).

## Why this project?

| Property | Value |
|----------|-------|
| Runtime (GPU) | ~2-3 min |
| Runtime (CPU) | ~8-10 min |
| Dependencies | PyTorch + torchvision (in Docker image) |
| Baseline val_acc | ~0.87 |
| Achievable val_acc | ~0.93+ |
| Tunable knobs | 15+ |
| Score key | `val_loss` (minimize) |

The defaults are **intentionally suboptimal** — plain SGD, no augmentation,
no batch norm, no schedule, small network. Plenty of room for improvement.

## Running with Ratiocinator

```bash
ratiocinator search \
    --repo examples/fashion_cnn \
    --command "python train.py" \
    --score-key val_loss \
    --topic "CNN architecture and training optimization for image classification"
```

Or with Vast.ai:

```bash
ratiocinator search \
    --repo examples/fashion_cnn \
    --command "python train.py" \
    --score-key val_loss \
    --vast \
    --topic "CNN architecture and training optimization for image classification"
```

## What the LLM can improve

### High impact (expect 2-4% accuracy gain each)
- **Optimizer**: `sgd` → `adamw` with weight decay 1e-4
- **Batch normalization**: `USE_BATCHNORM = True`
- **Data augmentation**: `USE_AUGMENTATION = True`
- **LR schedule**: `constant` → `cosine` or `onecycle`
- **Wider network**: `CONV_CHANNELS = [32, 64]` or `[32, 64, 128]`

### Medium impact (1-2% gain)
- **Kernel size**: 5 → 3 (more appropriate for 28×28)
- **Dropout**: 0 → 0.2-0.5
- **Learning rate**: 0.01 → 0.001 with Adam, or 0.1 with SGD+momentum
- **Momentum**: 0.0 → 0.9 (if staying with SGD)
- **Activation**: `relu` → `gelu` or `silu`
- **FC hidden**: 128 → 256

### Advanced (requires code changes beyond constants)
- Add residual connections
- Replace flatten with global average pooling
- Add mixup or cutmix
- Add EMA of model weights
- Add warmup to the LR schedule

## Metrics output

```
METRICS:{"train_loss": 0.3842, "val_loss": 0.4210, "train_acc": 0.8615,
         "val_acc": 0.8712, "best_val_loss": 0.4110, "best_val_acc": 0.8752,
         "params": 35914, "elapsed_seconds": 142.5}
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRAIN_STEPS` | 10 | Number of epochs |
| `DATA_DIR` | `/tmp/fashion_data` | Where to cache FashionMNIST |
