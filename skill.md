# Ratiocinator — AgentSkill Definition

> Autonomous scientific discovery pipeline for running GPU experiments and synthesizing papers.

## Overview

Ratiocinator is a globally installable CLI tool that orchestrates ML research from inside any target project. It provisions ephemeral GPU instances, runs parallel experiment arms, collects metrics, and optionally generates research papers from results.

**Install:** `pip install ratiocinator`

## Standard Workflow

```
ratiocinator init              # 1. Scaffold workspace
# write spec YAML              # 2. Define experiment
ratiocinator fleet run <spec>  # 3. Execute on GPU cloud
ratiocinator fleet status      # 4. Review results
```

### Step 1: Initialize Workspace

```bash
ratiocinator init
```

Creates:
- `research/specs/` — experiment and research spec YAML files
- `research/results/` — output artifacts (gitignored)
- `.ratiocinator/` — runtime state and config (gitignored)

Also appends `.ratiocinator/` and `research/results/` to `.gitignore`.

### Step 2: Write an Experiment Spec

Create a YAML file in `research/specs/` describing your experiment. See the schema sections below.

### Step 3: Run the Experiment

```bash
# Run on Vast.ai (default)
ratiocinator fleet run research/specs/my_experiment.yaml

# Run on HuggingFace Jobs
ratiocinator fleet run research/specs/my_experiment.yaml --hf

# Run specific arms only
ratiocinator fleet run research/specs/my_experiment.yaml --arms 0,2,4

# Fully autonomous research loop
ratiocinator research research/specs/my_research.yaml
```

### Step 4: Check Results

```bash
ratiocinator fleet status
```

Results are persisted to `.ratiocinator/results/experiments.json` by default.

---

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
| `ratiocinator search` | Best-First Tree Search over code modifications |
| `ratiocinator ask --prompt "..."` | One-off LLM query |
| `ratiocinator publish` | Upload artifacts to HuggingFace Hub |

Global options: `--config <path>` (config JSON), `-v/--verbose` (debug logging).

---

## ExperimentSpec Schema

The `ExperimentSpec` YAML defines a parallel experiment. Used with `ratiocinator fleet run`.

```yaml
# Required
name: my-ablation-study

hardware:
  gpu: "RTX 4090"           # GPU model (use spaces, e.g. "RTX 4090")
  num_gpus: 1               # GPUs per instance
  max_dph: 0.50             # Max dollars-per-hour bid
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
  hf_flavor: "a100-large"  # Required for HF Jobs provider
  batch_size: 32            # Default BATCH_SIZE env var for all arms

# Optional: auto-detected from git context if omitted
repo:
  url: https://github.com/user/repo.git
  branch: main
  commit: abc123            # Pin to specific commit

# Optional: data provisioning
data:
  source: s3-presigned      # s3-presigned | rsync | local | none | hf-dataset | hf-bucket
  urls_file: data-urls.txt  # For s3-presigned
  target: /workspace/data   # Remote path for data
  hf_source: "user/dataset" # For hf-dataset or hf-bucket
  hf_mount_path: "/data"    # Mount point inside container

# Optional: dependency installation
deps:
  pre_install:
    - "pip install torch --index-url https://download.pytorch.org/whl/cu130"
    - "apt-get install -y g++"
  requirements: requirements.txt
  exclude_from_requirements:
    - "^torch"
  verify: "python -c 'import torch; print(torch.cuda.is_available())'"

# Required: experiment arms
arms:
  - name: baseline
    command: "python train.py --lr 0.001"
    description: "Baseline learning rate"
    env:
      CUDA_LAUNCH_BLOCKING: "0"
    batch_size: 64           # Override hardware.batch_size for this arm
  - name: high-lr
    command: "python train.py --lr 0.01"
    description: "Higher learning rate"

# Required: how to extract metrics from training stdout
metrics:
  protocol: json_line        # json_line | block
  json_prefix: "METRICS:"   # For json_line protocol
  # For block protocol:
  # start_marker: "--- RESULTS ---"
  # end_marker: "--- END RESULTS ---"

# Optional: quick sanity check before full training
preflight:
  command: "python train.py --epochs 1 --max_steps 5"
  timeout_s: 60
  check_metrics: true        # Verify METRICS: output appears

# Optional: post-training ground-truth validation
validation:
  command: "python validate.py --output /workspace/output"
  timeout_s: 120
  required_metrics:
    - real_accuracy
  prefix: "val_"            # Namespace validation metrics

# Required: cost/time limits
budget:
  max_dollars: 10.00
  train_timeout_s: 1800
  download_timeout_s: 7200

# Optional: provider selection (default: vast)
provider: vast              # vast | hf
```

### Metrics Protocol

Training scripts must output metrics to stdout. Two protocols:

**JSON line** (default):
```
METRICS:{"train_loss": 0.33, "avg_iter_per_sec": 4.27}
```

**Block** (for structured output):
```
--- RESULTS ---
avg_iter_per_sec: 4.268
final_loss: 0.331
--- END RESULTS ---
```

---

## ResearchSpec Schema

The `ResearchSpec` YAML defines an autonomous research loop. Used with `ratiocinator research`.

```yaml
topic: "Improving training throughput for DiT on RTX 4090"
goal_metric: avg_iter_per_sec
maximize: true

repo_url: https://github.com/user/repo.git
repo_branch: main
repo_local_path: /home/user/repo
base_config_path: production/config.yaml
runner_script: scripts/run_arm.sh

hardware:
  gpu: "RTX 4090"
  num_gpus: 1
  max_dph: 0.50
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

data:
  source: rsync
  rsync_server: "root@host:/data/path"

deps:
  pre_install:
    - "pip install torch --index-url https://download.pytorch.org/whl/cu130"
  requirements: requirements.txt

metrics:
  protocol: json_line

max_iterations: 3           # Max ideation→execute→analyse cycles
max_dollars: 30.00          # Hard budget cap (Python-enforced)
train_timeout_s: 3600

paper_title: "My Research Paper"  # Omit to skip synthesis
```

The autonomous loop:
1. **Ideate** — LLM proposes experiment arms
2. **Translate** — generates ExperimentSpec configs
3. **Execute** — FleetExecutor runs arms on GPU
4. **Analyse** — LLM reviews results, decides if another iteration needed
5. **Synthesise** — generates paper with automated review (if paper_title set)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VAST_API_KEY` | For Vast.ai | Vast.ai API key |
| `HF_TOKEN` | For HF Jobs | HuggingFace token (fine-grained, write + jobs) |
| `HF_NAMESPACE` | No | HF org/user for jobs (defaults to token owner) |

---

## Quick Start Example

```bash
# Install
pip install ratiocinator

# Initialize workspace in your ML project
cd /path/to/your/ml-project
ratiocinator init

# Create a minimal experiment spec
cat > research/specs/lr_sweep.yaml << 'EOF'
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
  - name: lr-5e3
    command: "python train.py --lr 0.005"

metrics:
  protocol: json_line

budget:
  max_dollars: 5.00
  train_timeout_s: 1800
EOF

# Run (repo URL/branch auto-detected from git context)
export VAST_API_KEY=your-key
ratiocinator fleet run research/specs/lr_sweep.yaml

# Check results
ratiocinator fleet status
```

---

## Integration Notes for Agents

- **Workspace location:** Always run `ratiocinator init` from the root of the target project
- **Git context auto-detection:** If `repo:` is omitted from spec YAML, the CLI auto-detects the git remote URL, branch, and HEAD commit from the current directory
- **Results persistence:** Results go to `.ratiocinator/results/experiments.json` (gitignored)
- **Safety limits:** Budget caps (`max_dollars`) are Python-enforced hard stops — not LLM-controllable
- **Idempotent init:** `ratiocinator init` is safe to call multiple times
- **No custom scripts needed:** The fleet framework handles provisioning, cloning, deps, data, training, and cleanup
