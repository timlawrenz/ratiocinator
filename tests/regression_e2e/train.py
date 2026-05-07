"""Tiny MLP training script for regression testing.

Trains a 2-layer MLP on random data for ~30 seconds, outputs METRICS lines.
Used to verify the provider interface works end-to-end on real infrastructure.
"""

import json
import os
import time

import torch
import torch.nn as nn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Config from environment (provider injects these)
    hidden_dim = int(os.environ.get("HIDDEN_DIM", "256"))
    lr = float(os.environ.get("LR", "0.001"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    train_steps = int(os.environ.get("TRAIN_STEPS", "200"))

    print(f"Config: hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}, steps={train_steps}")

    # Model
    model = nn.Sequential(
        nn.Linear(128, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    start = time.time()
    total_loss = 0.0
    for step in range(1, train_steps + 1):
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            avg_loss = total_loss / step
            elapsed = time.time() - start
            print(f"Step {step}/{train_steps} | loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | elapsed={elapsed:.1f}s")

    # Final metrics
    final_loss = total_loss / train_steps
    elapsed = time.time() - start

    # Compute accuracy on a test batch
    with torch.no_grad():
        x_test = torch.randn(256, 128, device=device)
        y_test = torch.randint(0, 10, (256,), device=device)
        preds = model(x_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item() * 100

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"METRICS:{json.dumps({'final_loss': round(final_loss, 4), 'accuracy_pct': round(accuracy, 2), 'train_time_s': round(elapsed, 1), 'steps': train_steps})}")


if __name__ == "__main__":
    main()
