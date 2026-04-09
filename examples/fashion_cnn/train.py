"""PyTorch CNN on FashionMNIST — Ratiocinator test bench.

A self-contained training script designed to test the Ratiocinator pipeline
on Vast.ai GPU instances. Uses only PyTorch + torchvision (both pre-installed
in the pytorch/pytorch Docker image).

Runs in ~2-3 minutes on GPU, ~8-10 minutes on CPU.

The defaults are intentionally suboptimal — there are 15+ improvements an LLM
can discover through code diffs:

  Architecture:
  - Only 2 conv layers with 16/32 channels (wider/deeper helps)
  - No batch normalization (big win for CNNs)
  - No residual/skip connections
  - Global average pooling would be better than flatten
  - Kernel size 5 is large for 28x28 images (try 3)

  Optimization:
  - Plain SGD (Adam or AdamW would converge faster)
  - Learning rate 0.01 is conservative
  - No learning rate schedule (cosine annealing helps a lot)
  - No weight decay / L2 regularization
  - No gradient clipping
  - No warmup

  Regularization:
  - No dropout
  - No data augmentation (random crop, horizontal flip)
  - No label smoothing
  - No mixup / cutmix

  Training:
  - Batch size 64 (larger batches + LR scaling may help)
  - No EMA of model weights

Outputs METRICS:{json} for the Ratiocinator orchestrator.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Hyperparameters (tunable via code diffs) ─────────────────────────
SEED = 42
EPOCHS = int(os.environ.get("TRAIN_STEPS", "10"))
LEARNING_RATE = 0.01
OPTIMIZER = "sgd"  # sgd, adam, adamw
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
BATCH_SIZE = 64
LR_SCHEDULE = "constant"  # constant, cosine, step, onecycle
LR_STEP_SIZE = 5
LR_GAMMA = 0.5
DROPOUT = 0.0
LABEL_SMOOTHING = 0.0
USE_BATCHNORM = False
USE_AUGMENTATION = False
GRADIENT_CLIP = 0.0  # 0 = no clipping

# Architecture
CONV_CHANNELS = [16, 32]
KERNEL_SIZE = 5
FC_HIDDEN = 128
POOL_TYPE = "max"  # max, avg
ACTIVATION = "relu"  # relu, gelu, silu, mish

# ─── Data ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/fashion_data")


def get_activation():
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }[ACTIVATION]


def get_transforms():
    base = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    if USE_AUGMENTATION:
        train_tf = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            *base,
        ])
    else:
        train_tf = transforms.Compose(base)
    val_tf = transforms.Compose(base)
    return train_tf, val_tf


# ─── Model ────────────────────────────────────────────────────────────

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        act_cls = get_activation()
        layers = []
        in_ch = 1
        for out_ch in CONV_CHANNELS:
            layers.append(nn.Conv2d(in_ch, out_ch, KERNEL_SIZE, padding=KERNEL_SIZE // 2))
            if USE_BATCHNORM:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_cls())
            if POOL_TYPE == "max":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AvgPool2d(2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            feat = self.features(dummy)
            flat_size = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, FC_HIDDEN),
            act_cls(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─── Training ─────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_tf, val_tf = get_transforms()
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=train_tf)
    val_ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    # Model
    model = FashionCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Optimizer
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        )

    # LR schedule
    if LR_SCHEDULE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif LR_SCHEDULE == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA,
        )
    elif LR_SCHEDULE == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE * 10,
            epochs=EPOCHS, steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = None

    t0 = time.time()
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            optimizer.step()

            if LR_SCHEDULE == "onecycle":
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        if scheduler is not None and LR_SCHEDULE != "onecycle":
            scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:>2}/{EPOCHS} | "
            f"lr={lr_now:.1e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s on {device}")
    print(f"Best: val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}")
    metrics = json.dumps({
        'train_loss': round(train_loss, 6),
        'val_loss': round(val_loss, 6),
        'train_acc': round(train_acc, 4),
        'val_acc': round(val_acc, 4),
        'best_val_loss': round(best_val_loss, 6),
        'best_val_acc': round(best_val_acc, 4),
        'params': param_count,
        'elapsed_seconds': round(elapsed, 2),
    })
    print(f"METRICS:{metrics}")


if __name__ == "__main__":
    train()
