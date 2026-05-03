# Ratiocinator Spec Schemas

> Common fields and defaults. For all available fields see the Pydantic models in
> the [ratiocinator source](https://github.com/timlawrenz/ratiocinator/blob/main/src/ratiocinator/fleet/spec.py)
> (`fleet/spec.py`: ExperimentSpec, HardwareSpec, etc.).

## ExperimentSpec

Used with `ratiocinator fleet run`. Defines a parallel experiment.

```yaml
# Required
name: my-ablation-study

hardware:
  gpu: "RTX 4090"           # GPU model (use spaces, e.g. "RTX 4090")
  num_gpus: 1               # GPUs per instance (default: 1)
  min_cpu_ram_gb: 64        # Minimum system RAM in GB (default: 64)
  min_pcie_bw: 20.0         # Minimum PCIe bandwidth in GB/s (default: 20.0)
  max_dph: 0.50             # Max dollars-per-hour bid
  disk_gb: 200.0            # Disk space in GB (default: 200.0)
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
  hf_flavor: "a100-large"  # Required for HF Jobs provider
  batch_size: 32            # Default BATCH_SIZE env var for all arms

# Optional: auto-detected from git context if omitted
repo:
  url: https://github.com/user/repo.git
  branch: main
  commit: abc123            # Pin to specific commit (default: "")
  clone_depth: 1            # Git clone depth (default: 1, use 0 for full history)
  remote_path: /workspace/experiment  # Clone destination on instance

# Optional: data provisioning
data:
  source: s3-presigned      # s3-presigned | rsync | local | none | hf-dataset | hf-bucket
  urls_file: data-urls.txt  # For s3-presigned: file with one presigned URL per line
  target: /workspace/data   # Remote path for data (default: /workspace/data)
  rsync_server: ""          # For rsync: remote host:path
  rsync_port: 22            # For rsync: SSH port (default: 22)
  max_shards: null          # For rsync: limit number of shards (default: null = all)
  local_path: ""            # For local: path on orchestrator machine to SCP
  hf_source: "user/dataset" # For hf-dataset or hf-bucket
  hf_mount_path: "/data"    # Mount point inside container (default: /data)

# Optional: dependency installation
deps:
  pre_install:
    - "pip install torch --index-url https://download.pytorch.org/whl/cu130"
    - "apt-get install -y g++"
  requirements: requirements.txt
  exclude_from_requirements:
    - "torch"               # Plain package name prefix, not a regex
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
  json_prefix: "METRICS:"   # For json_line protocol (default: "METRICS:")
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
  download_timeout_s: 7200  # Data download timeout in seconds (default: 7200)
  boot_timeout_s: 600       # Instance boot/SSH timeout in seconds (default: 600)
  instance_ttl_s: 3600      # Hard TTL for entire instance lifetime (default: 3600)

# Optional: provider selection (default: vast)
provider: vast              # vast | hf
```

## ResearchSpec

Used with `ratiocinator research`. Defines an autonomous research loop where the LLM proposes concrete arms each iteration.

```yaml
# Required
name: my-research
description: "Improving training throughput for DiT on RTX 4090"

# Required: target repository
repo:
  url: https://github.com/user/repo.git
  branch: main

hardware:
  gpu: "RTX 4090"
  num_gpus: 1
  max_dph: 0.50
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
  hf_flavor: "a100-large"   # Required for HF Jobs provider

data:
  source: rsync
  rsync_server: "root@host:/data/path"

deps:
  pre_install:
    - "pip install torch --index-url https://download.pytorch.org/whl/cu130"
  requirements: requirements.txt

metrics:
  protocol: json_line

# Research-specific fields
base_command: "python train.py"  # Base command; LLM injects config_overrides as env vars
base_config_path: ""             # Optional path to base config file
num_arms: 6                      # Arms proposed per iteration
iterations: 3                    # Max ideation→execute→analyse cycles
score_key: loss                  # Metric key to optimise
maximize: false                  # Set true to maximise score_key (default: minimise)
provider: vast                   # vast | hf

budget:
  max_dollars: 30.00             # Hard budget cap (Python-enforced)
  train_timeout_s: 3600
```

The autonomous loop:
1. **Ideate** — LLM proposes experiment arms
2. **Translate** — generates ExperimentSpec configs
3. **Execute** — FleetExecutor runs arms on GPU
4. **Analyse** — LLM reviews results, decides if another iteration needed
5. **Synthesise** — generates paper with automated review (triggered via `ratiocinator synthesize`)
