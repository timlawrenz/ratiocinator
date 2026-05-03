---
name: ratiocinator
description: >-
  Run parallel GPU experiments and autonomous research loops from any ML project.
  Use when asked to run ablation studies, hyperparameter sweeps, training experiments
  on cloud GPUs (Vast.ai or HuggingFace Jobs), or to orchestrate an autonomous
  ideate→execute→analyse research cycle. Handles provisioning, data staging,
  dependency installation, metric collection, and paper synthesis.
---

## Install

```bash
pip install ratiocinator
```

## Workflow

```bash
ratiocinator init              # 1. Scaffold workspace (idempotent)
# write spec YAML in research/specs/
ratiocinator fleet run <spec>  # 2. Execute experiment on GPU cloud
ratiocinator fleet status      # 3. Review results
```

## Step 1: Initialize Workspace

Run from the root of the target project:

```bash
ratiocinator init
```

Creates `research/specs/`, `research/results/`, `.ratiocinator/` and updates `.gitignore`.

## Step 2: Write an Experiment Spec

Create a YAML file in `research/specs/`. Minimal example:

```yaml
name: lr-sweep

hardware:
  gpu: "RTX 4090"
  max_dph: 0.50
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

arms:
  - name: lr-1e3
    command: "python train.py --lr 0.001"
  - name: lr-1e2
    command: "python train.py --lr 0.01"

metrics:
  protocol: json_line

budget:
  max_dollars: 5.00
  train_timeout_s: 1800
```

If `repo:` is omitted, git remote URL, branch, and commit are auto-detected.

See [references/schemas.md](references/schemas.md) for `ExperimentSpec` and `ResearchSpec` schema reference (common fields with defaults).

## Step 3: Run

```bash
# Vast.ai (default)
ratiocinator fleet run research/specs/lr_sweep.yaml

# HuggingFace Jobs
ratiocinator fleet run research/specs/lr_sweep.yaml --hf

# Subset of arms
ratiocinator fleet run research/specs/lr_sweep.yaml --arms 0,2

# Autonomous research loop (ideate → execute → analyse → paper)
ratiocinator research research/specs/my_research.yaml
```

## Step 4: Check Results

```bash
ratiocinator fleet status
```

Results persist to `.ratiocinator/results/experiments.json`.

## CLI Reference

| Command | Purpose |
|---------|---------|
| `ratiocinator init` | Scaffold workspace directories and .gitignore |
| `ratiocinator fleet run <spec.yaml>` | Run parallel experiments from YAML spec |
| `ratiocinator fleet run <spec.yaml> --hf` | Run on HuggingFace Jobs |
| `ratiocinator fleet run <spec.yaml> --arms 0,2` | Run specific arms |
| `ratiocinator fleet status` | Show results from previous runs |
| `ratiocinator research <spec.yaml>` | Autonomous loop: ideate → execute → analyse → paper |
| `ratiocinator research <spec.yaml> --hf` | Autonomous research on HuggingFace |
| `ratiocinator run --task "..." --repo ./` | Single experiment (propose → run → report) |
| `ratiocinator ask "..."` | One-off LLM query |
| `ratiocinator publish --output-dir <dir>` | Upload artifacts to HuggingFace Hub |

Global options: `--config <path>`, `-v/--verbose`.

## Metrics Protocol

Training scripts output metrics to stdout. Two protocols:

**JSON line** (default):
```
METRICS:{"train_loss": 0.33, "avg_iter_per_sec": 4.27}
```

**Block**:
```
--- RESULTS ---
avg_iter_per_sec: 4.268
final_loss: 0.331
--- END RESULTS ---
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VAST_API_KEY` | For Vast.ai | Vast.ai API key |
| `HF_TOKEN` | For HF Jobs | HuggingFace token (fine-grained, write + jobs) |
| `HF_NAMESPACE` | No | HF org/user for jobs (defaults to token owner) |

## Integration Notes

- **Workspace location:** Run `ratiocinator init` from the target project root
- **Git auto-detection:** Omit `repo:` from spec and it detects origin URL, branch, commit
- **Safety limits:** Budget caps (`max_dollars`) are Python-enforced hard stops
- **Idempotent init:** Safe to call multiple times
- **No custom scripts needed:** Fleet framework handles provisioning, cloning, deps, data, training, cleanup
