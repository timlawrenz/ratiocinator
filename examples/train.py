"""Toy MNIST-like training script for testing the ratiocinator pipeline.

Trains a small MLP on synthetic data. Outputs METRICS: JSON line for the
orchestrator to parse.

Respects TRAIN_STEPS env var to control training duration.
"""

import json
import os
import random

STEPS = int(os.environ.get("TRAIN_STEPS", "100"))
LR = float(os.environ.get("LEARNING_RATE", "0.01"))
SEED = 42


def generate_data(n: int = 200, dim: int = 10):
    """Generate linearly separable synthetic data."""
    random.seed(SEED)
    X = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    weights = [random.gauss(0, 1) for _ in range(dim)]
    y = [1 if sum(x * w for x, w in zip(row, weights)) > 0 else 0 for row in X]
    return X, y, weights


def sigmoid(z):
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    import math
    return 1.0 / (1.0 + math.exp(-z))


def train(X, y, lr, steps):
    dim = len(X[0])
    w = [0.0] * dim
    b = 0.0
    n = len(X)

    for step in range(steps):
        total_loss = 0.0
        correct = 0
        for xi, yi in zip(X, y):
            z = sum(xj * wj for xj, wj in zip(xi, w)) + b
            pred = sigmoid(z)
            loss = -(yi * (max(pred, 1e-15).__log__() if hasattr(pred, '__log__') else __import__('math').log(max(pred, 1e-15)))
                     + (1 - yi) * __import__('math').log(max(1 - pred, 1e-15)))
            total_loss += loss
            if (pred > 0.5) == yi:
                correct += 1
            error = pred - yi
            for j in range(dim):
                w[j] -= lr * error * xi[j]
            b -= lr * error

        if (step + 1) % max(1, steps // 5) == 0:
            avg_loss = total_loss / n
            accuracy = correct / n
            print(f"Step {step + 1}/{steps} | loss={avg_loss:.4f} | accuracy={accuracy:.4f}")

    avg_loss = total_loss / n
    accuracy = correct / n
    return w, b, avg_loss, accuracy


def main():
    X, y, _ = generate_data()
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Training for {STEPS} steps with lr={LR}")
    w, b, train_loss, train_acc = train(X_train, y_train, LR, STEPS)

    # Evaluate on validation set
    correct = 0
    for xi, yi in zip(X_val, y_val):
        z = sum(xj * wj for xj, wj in zip(xi, w)) + b
        pred = 1 if sigmoid(z) > 0.5 else 0
        if pred == yi:
            correct += 1
    val_acc = correct / len(y_val)

    print(f"\nFinal: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    print(f"METRICS:{json.dumps({'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})}")


if __name__ == "__main__":
    main()
