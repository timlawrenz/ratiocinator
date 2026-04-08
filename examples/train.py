"""Toy training script for testing the ratiocinator pipeline.

Trains a small MLP on a synthetic nonlinear regression problem.
Outputs METRICS: JSON line for the orchestrator to parse.

The defaults are intentionally suboptimal so the LLM has room to improve:
- Learning rate is moderate (0.01) but could be tuned
- No feature normalization
- Single hidden layer with only 4 units
- No activation function on hidden layer output (linear)

Respects TRAIN_STEPS env var to control training duration.
"""

import json
import math
import os
import random

STEPS = int(os.environ.get("TRAIN_STEPS", "100"))
LR = float(os.environ.get("LEARNING_RATE", "0.001"))
HIDDEN = int(os.environ.get("HIDDEN_SIZE", "4"))
SEED = 42


def generate_data(n: int = 300, dim: int = 5):
    """Generate nonlinear regression data: y = sin(x1*x2) + x3^2 + noise."""
    random.seed(SEED)
    X = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    y = []
    for row in X:
        target = math.sin(row[0] * row[1]) + row[2] ** 2 + 0.3 * random.gauss(0, 1)
        y.append(target)
    return X, y


def relu(z):
    return max(0.0, z)


def forward(X_row, w_hidden, b_hidden, w_out, b_out):
    """Forward pass through a single hidden layer MLP."""
    hidden = []
    for j in range(len(w_hidden)):
        z = sum(x * w for x, w in zip(X_row, w_hidden[j])) + b_hidden[j]
        hidden.append(z)  # NOTE: no activation — this limits learning capacity
    output = sum(h * w for h, w in zip(hidden, w_out)) + b_out
    return output, hidden


def train(X, y, lr, steps, hidden_size):
    dim = len(X[0])
    random.seed(SEED + 1)

    # Initialize weights (small random)
    w_hidden = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(hidden_size)]
    b_hidden = [0.0] * hidden_size
    w_out = [random.gauss(0, 0.1) for _ in range(hidden_size)]
    b_out = 0.0

    n = len(X)

    for step in range(steps):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            pred, hidden = forward(xi, w_hidden, b_hidden, w_out, b_out)
            error = pred - yi
            loss = 0.5 * error ** 2
            total_loss += loss

            # Backprop: output layer
            d_out = error
            for j in range(hidden_size):
                w_out[j] -= lr * d_out * hidden[j]
            b_out -= lr * d_out

            # Backprop: hidden layer (no activation derivative since linear)
            for j in range(hidden_size):
                d_hidden = d_out * w_out[j]
                for k in range(dim):
                    w_hidden[j][k] -= lr * d_hidden * xi[k]
                b_hidden[j] -= lr * d_hidden

        mse = total_loss / n
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"Step {step + 1}/{steps} | mse={mse:.4f}")

    mse = total_loss / n
    return w_hidden, b_hidden, w_out, b_out, mse


def evaluate(X, y, w_hidden, b_hidden, w_out, b_out):
    """Compute MSE on a dataset."""
    total = 0.0
    for xi, yi in zip(X, y):
        pred, _ = forward(xi, w_hidden, b_hidden, w_out, b_out)
        total += 0.5 * (pred - yi) ** 2
    return total / len(X)


def main():
    X, y = generate_data()
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Training MLP: {STEPS} steps, lr={LR}, hidden={HIDDEN}")
    w_h, b_h, w_o, b_o, train_loss = train(X_train, y_train, LR, STEPS, HIDDEN)

    val_loss = evaluate(X_val, y_val, w_h, b_h, w_o, b_o)

    print(f"\nFinal: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    print(f"METRICS:{json.dumps({'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6)})}")


if __name__ == "__main__":
    main()
