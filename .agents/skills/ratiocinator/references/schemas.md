# Ratiocinator Spec Schemas

## ExperimentSpec

Used with `ratiocinator fleet run`. Defines a parallel experiment.

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
    - "torch"
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

## ResearchSpec

Used with `ratiocinator research`. Defines an autonomous research loop.

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
5. **Synthesise** — generates paper with automated review (if `paper_title` set)
