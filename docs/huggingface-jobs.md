# HuggingFace Jobs — Provider Guide

This guide covers how to run Ratiocinator experiments on **HuggingFace Jobs** instead of Vast.ai. HF Jobs provides managed GPU containers with native volume mounts for HF Datasets and Buckets.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Data Flow](#data-flow)
- [Volume Architecture](#volume-architecture)
- [Spec Reference](#spec-reference)
- [Flavor Reference](#flavor-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [SDK Internals](#sdk-internals)

---

## Overview

HuggingFace Jobs runs Docker containers on managed GPU infrastructure. Ratiocinator wraps this into a fleet execution model where each experiment arm runs as an independent HF Job.

```
┌─────────────────────────────────────────────────────┐
│                   Ratiocinator                       │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Arm: base   │  │  Arm: lr-hi │  │  Arm: cosine│ │
│  │  (HF Job)    │  │  (HF Job)    │  │  (HF Job)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │          │
│  ┌──────▼──────────────────────────────────▼──────┐ │
│  │            HF Buckets (shared)                  │ │
│  │  /input/   = wrapper scripts (read-only)        │ │
│  │  /output/  = checkpoints + logs (read-write)    │ │
│  │  /data/    = training data (Datasets/Buckets)   │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### When to Use HF Jobs

| Use Case | Best Provider |
|----------|---------------|
| Standard GPU training with HF data | **HF Jobs** |
| Interactive debugging via SSH | Vast.ai |
| Cheapest spot GPU pricing | Vast.ai |
| Zero-ops, managed environment | **HF Jobs** |
| Data already on HuggingFace | **HF Jobs** |
| Large local datasets via rsync | Vast.ai |
| Multi-node distributed training | Vast.ai |

---

## Setup

### 1. Install Dependencies

```bash
pip install -e ".[hf]"

# For development (includes test deps):
pip install -e ".[dev,hf]"
```

### 2. Create an HF Token

Go to https://huggingface.co/settings/tokens and create a **fine-grained** token with:

| Permission | Why |
|-----------|-----|
| **Repositories → Write access** | Upload scripts to HF Buckets |
| **Jobs → Start and manage Jobs** | Submit and monitor jobs |

> ⚠️ A read-only token will NOT work. You'll get `403 Forbidden: missing permissions: job.write`.

### 3. Set Environment Variables

```bash
# Option A: .env file (recommended)
echo 'HF_TOKEN=hf_YOUR_TOKEN_HERE' >> .env
echo 'HF_NAMESPACE=your_hf_username' >> .env

# Option B: export
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export HF_NAMESPACE=your_hf_username

# Option C: huggingface-cli login
huggingface-cli login
```

`HF_NAMESPACE` defaults to the token owner's username if not set.

---

## Quick Start

### Minimal Experiment

Create `experiments/my_test.yaml`:

```yaml
name: my-first-hf-experiment
hardware:
  gpu: "L4"
  hf_flavor: "l4"
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
repo:
  url: https://github.com/your-user/your-repo.git
  branch: main
arms:
  - name: baseline
    command: "python train.py --epochs 5"
metrics:
  protocol: json_line
budget:
  max_dollars: 2.0
  train_timeout_s: 1800
provider: hf
```

Run it:

```bash
ratiocinator fleet run experiments/my_test.yaml --hf
```

### Autonomous Research

Create `specs/my_research.yaml`:

```yaml
topic: "Optimizing learning rate schedules for ViT-S on CIFAR-10"
goal_metric: val_accuracy
maximize: true
repo_url: https://github.com/your-user/your-repo.git
repo_branch: main
base_config_path: config/default.yaml
runner_script: scripts/train.sh
hardware:
  gpu: "L4"
  hf_flavor: "l4"
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
max_iterations: 3
max_dollars: 15.00
train_timeout_s: 3600
provider: hf
```

Run it:

```bash
ratiocinator research specs/my_research.yaml --hf
```

---

## Data Flow

### Reading Training Data

HF Jobs mounts data volumes directly into the container filesystem:

**From HF Datasets:**
```yaml
data:
  source: hf-dataset
  hf_source: "username/my-dataset"
  hf_mount_path: "/data"
```
→ Your training script accesses data at `/data/` inside the container.

**From HF Buckets:**
```yaml
data:
  source: hf-bucket
  hf_source: "username/my-bucket"
  hf_mount_path: "/data"
```

**No data (repo contains everything):**
```yaml
# Simply omit the data section — repo is cloned to /workspace/repo
```

### Writing Checkpoints and Artifacts

Every HF fleet run creates an output bucket at `{namespace}/ratiocinator-{experiment-name}`. Inside the container, `/output/` is a writable mount to this bucket.

Write your checkpoints and artifacts there:

```python
# In your training script:
import torch
torch.save(model.state_dict(), "/output/checkpoints/best_model.pt")

# Write logs
with open("/output/logs/training.log", "w") as f:
    f.write(f"epoch={epoch} loss={loss}\n")
```

These files persist in the HF Bucket after the job completes and can be browsed at `https://huggingface.co/datasets/{namespace}/ratiocinator-{experiment-name}`.

### Metrics Protocol

Training scripts communicate results via stdout. Two protocols:

**JSON line (default):**
```python
import json
metrics = {"loss": 0.42, "accuracy": 0.95, "epoch": 10}
print(f"METRICS:{json.dumps(metrics)}")
```

**Block protocol:**
```
--- RESULTS ---
loss: 0.42
accuracy: 0.95
epoch: 10
--- END RESULTS ---
```

Configure in the spec:
```yaml
metrics:
  protocol: json_line    # or "block"
  json_prefix: "METRICS:"
```

---

## Volume Architecture

Each HF Job container sees this filesystem:

```
/
├── input/                    ← Read-only bucket (wrapper scripts)
│   └── {experiment}/{arm}/
│       └── run.sh           ← Generated wrapper script
├── data/                     ← Training data (optional)
│   └── ...                  ← Contents of HF Dataset or Bucket
├── output/                   ← Writable bucket (artifacts)
│   ├── checkpoints/
│   │   └── model.pt         ← Your checkpoints
│   └── logs/
│       └── train.log        ← Your logs
└── workspace/
    └── repo/                 ← Cloned experiment repository
        ├── train.py
        ├── config/
        └── requirements.txt
```

The wrapper script at `/input/{experiment}/{arm}/run.sh` orchestrates:

1. Install git (if missing)
2. Clone repository → `/workspace/repo/`
3. Install dependencies (pre_install → requirements → verify)
4. Run preflight validation (if configured)
5. Run training command
6. Run post-training validation (if configured)
7. Copy artifacts to `/output/`

---

## Spec Reference

### HardwareSpec

```yaml
hardware:
  gpu: "A100"                   # Human-readable label (informational)
  hf_flavor: "a100-large"      # REQUIRED — HF Job flavor string
  image: "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
```

`hf_flavor` must match exactly one of the [available flavors](#flavor-reference). The system fails fast if not set.

### DataSpec

```yaml
data:
  source: "hf-dataset"         # "hf-dataset" or "hf-bucket"
  hf_source: "user/dataset"    # HF repo ID or bucket name
  hf_mount_path: "/data"       # Mount point in container (default: /data)
```

### DepsSpec

```yaml
deps:
  pre_install:                  # Run before pip install
    - "pip install torch --index-url https://download.pytorch.org/whl/cu124"
    - "apt-get update -qq && apt-get install -y -qq g++"
  requirements: "requirements.txt"
  exclude_from_requirements:    # Regex patterns to skip
    - "^torch"
  verify: "python -c 'import torch; print(torch.cuda.is_available())'"
```

### ArmSpec

```yaml
arms:
  - name: baseline
    command: "python train.py --lr 0.001"
  - name: high-lr
    command: "python train.py --lr 0.01"
    env:                         # Per-arm environment variables
      CUDA_LAUNCH_BLOCKING: "0"
      WANDB_DISABLED: "true"
```

### PreflightSpec

Quick sanity check before the full training run:

```yaml
preflight:
  command: "python train.py --epochs 1 --max_steps 5"
  timeout_s: 60
  check_metrics: true           # Also verify METRICS: output appears
```

### ValidationSpec

Post-training ground-truth validation:

```yaml
validation:
  command: "python validate.py --output /workspace/output"
  timeout_s: 120
  required_metrics:             # Arm fails if these are absent
    - real_accuracy
  prefix: "val_"                # Namespace: produces val_real_accuracy
```

### BudgetSpec

```yaml
budget:
  max_dollars: 10.0             # Hard spend cap (Python-enforced)
  train_timeout_s: 3600         # Max training time per arm
```

---

## Flavor Reference

All available HF Job flavors and their pricing (as of April 2026):

| Flavor | GPU | VRAM | vCPU | RAM | $/hr |
|--------|-----|------|------|-----|------|
| `cpu-basic` | — | — | 2 | 16 GB | $0.01 |
| `cpu-upgrade` | — | — | 8 | 32 GB | $0.03 |
| `cpu-xl` | — | — | 16 | 64 GB | $1.00 |
| `cpu-performance` | — | — | 32 | 128 GB | $1.90 |
| `t4-small` | T4 | 16 GB | 4 | 15 GB | $0.40 |
| `t4-medium` | T4 | 16 GB | 8 | 30 GB | $0.60 |
| `l4` | L4 | 24 GB | 8 | 30 GB | $0.80 |
| `4xl4` | 4×L4 | 96 GB | 48 | 192 GB | $3.80 |
| `l40s` | L40S | 48 GB | 12 | 48 GB | $1.80 |
| `4xl40s` | 4×L40S | 192 GB | 48 | 192 GB | $8.30 |
| `8xl40s` | 8×L40S | 384 GB | 96 | 384 GB | $23.50 |
| `a10g-small` | A10G | 24 GB | 4 | 15 GB | $1.00 |
| `a10g-large` | A10G | 24 GB | 12 | 46 GB | $1.50 |
| `2xa10g-large` | 2×A10G | 48 GB | 24 | 92 GB | $3.00 |
| `4xa10g-large` | 4×A10G | 96 GB | 48 | 184 GB | $5.00 |
| `a100-large` | A100 | 80 GB | 12 | 142 GB | $2.50 |
| `4xa100` | 4×A100 | 320 GB | 48 | 568 GB | $10.00 |
| `8xa100` | 8×A100 | 640 GB | 96 | 1136 GB | $20.00 |

Use `cpu-basic` ($0.01/hr) for testing specs before committing to GPU time.

---

## Advanced Usage

### Multiple Data Sources

Mount both a dataset and a bucket:

```yaml
data:
  source: hf-dataset
  hf_source: "user/training-data"
  hf_mount_path: "/data"
# Note: additional volumes can be added via the HFFleetExecutor API
```

### Re-running Failed Arms

```bash
# Run only arms 2 and 4 (0-indexed)
ratiocinator fleet run experiments/my_experiment.yaml --hf --arms 2,4
```

### Checking Results

```bash
# Show results table
ratiocinator fleet status

# Results are also saved to .ratiocinator/results/experiments.json
```

### Dry Run

```bash
# Print the plan without launching jobs
ratiocinator fleet run experiments/my_experiment.yaml --hf --dry-run
```

### Custom Output Location

```bash
ratiocinator fleet run experiments/my_experiment.yaml --hf --results-file ./my-results.json
```

---

## Troubleshooting

### Common Errors

**`403 Forbidden: missing permissions: job.write`**

Your HF token is read-only. Create a new fine-grained token at https://huggingface.co/settings/tokens with **Jobs → Start and manage Jobs** permission.

**`403 Forbidden: xet-write-token`**

Token lacks bucket write access. Enable **Repositories → Write access** on the token.

**`hf_flavor is required when using the HF provider`**

Add `hf_flavor: "l4"` (or another flavor) to the `hardware:` section of your spec YAML.

**`'dict' object has no attribute 'to_dict'`**

This was a bug in volume handling — update to the latest code. Volumes must be `huggingface_hub.Volume` objects when passed to the SDK.

**`git: command not found` in job logs**

The wrapper script auto-installs git for minimal images (`python:*-slim`). If using a custom image, ensure git is available or add `"apt-get update && apt-get install -y git"` to `deps.pre_install`.

**Job stuck in UNKNOWN stage**

Normal — this means the job is being scheduled. The poller will continue checking every 15 seconds. HF Jobs typically move to RUNNING within 10–30 seconds.

**Metrics not parsed from successful job**

Ensure your training script:
1. Has `PYTHONUNBUFFERED=1` set (wrapper scripts do this automatically)
2. Prints metrics with the exact prefix: `METRICS:{"key": value}`
3. Each METRICS line is on its own line (not concatenated with other output)

**Job shows ERROR with exit code 137**

Exit code 137 = killed by OOM. Use a larger flavor (more RAM/VRAM) or reduce batch size.

### Debugging Tips

1. **Check job logs:** The most useful debugging tool. Logs are fetched automatically after the job completes.

2. **Use preflight:** Add a `preflight:` section to catch setup errors (missing modules, bad paths) before the full training run.

3. **Start with cpu-basic:** Test your spec with `hf_flavor: "cpu-basic"` ($0.01/hr) before moving to GPUs.

4. **Sentry integration:** If configured, HF executor reports crashes, attaches logs, and tracks per-arm metrics in Sentry.

---

## SDK Internals

### Method Mapping

The `huggingface_hub.HfApi` method names don't always match what you'd expect. Our `HFClient` wraps them:

| HFClient method | Actual `HfApi` call | Return type |
|----------------|---------------------|-------------|
| `run_job(...)` | `api.run_job(...)` | Job ID (str) |
| `get_job(id)` | `api.inspect_job(job_id=id)` | `HFJobInfo` |
| `get_job_logs(id)` | `api.fetch_job_logs(job_id=id)` | str (newline-joined) |
| `cancel_job(id)` | `api.cancel_job(job_id=id)` | None |
| `list_jobs(ns)` | `api.list_jobs(namespace=ns)` | list[HFJobInfo] |
| `create_bucket(id)` | `api.create_bucket(id, exist_ok=True)` | None |
| `upload_to_bucket(...)` | `api.batch_bucket_files(id, add=[(local, remote)])` | None |

All `HfApi` calls use **keyword-only arguments** for IDs.

### Job Lifecycle

```
UNKNOWN → PENDING → STARTING → RUNNING → COMPLETED
                                        → ERROR (failed)
                                        → CANCELLED
```

- `UNKNOWN`: Just submitted, not yet scheduled
- `PENDING`: In queue, waiting for resources
- `STARTING`: Container is being prepared
- `RUNNING`: Training in progress
- `COMPLETED`: Success — metrics available in logs
- `ERROR`: Failed — check logs for stack traces

### Token Auto-Discovery

`HFClient(token="")` auto-discovers tokens in this order:
1. `HF_TOKEN` environment variable
2. Token stored by `huggingface-cli login` in `~/.huggingface/token`
3. Token stored in `HF_HOME` (custom cache directory)

### Wrapper Script Generation

The executor generates a self-contained bash script per arm. Example:

```bash
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Ensure git is available
if ! command -v git &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
fi

# Arm-specific environment variables
export LR='0.001'
export BATCH_SIZE='32'

# Clone experiment repository
git clone --depth 1 --branch main https://github.com/user/repo.git /workspace/repo
cd /workspace/repo

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt

# Pre-flight validation
echo '--- Preflight: baseline ---'
python train.py --epochs 1 --max_steps 5
echo '--- Preflight complete ---'

# Training
echo '--- Training: baseline ---'
python train.py --lr 0.001 --epochs 10 --data-dir /data
TRAIN_EXIT=$?
echo '--- Training complete (exit $TRAIN_EXIT) ---'

# Post-training validation (only if training succeeded)
if [ $TRAIN_EXIT -eq 0 ]; then
  echo '--- Validation: baseline ---'
  python validate.py --output /workspace/output
  echo '--- Validation complete ---'
fi

# Copy artifacts to output
cp -r /workspace/repo/results/* /output/ 2>/dev/null || true
exit $TRAIN_EXIT
```
