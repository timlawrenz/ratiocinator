# AGENTS.md — Ratiocinator

> Instructions for AI coding agents working on this repository.

## What This Project Is

Ratiocinator is an **autonomous research pipeline** that uses LLMs and tree search to run scientific experiments without human intervention. It proposes hypotheses, modifies code, provisions GPU instances, runs training, collects results, and writes papers.

The typical workflow is: **ideate → search → execute → synthesize → publish**.

This is a *meta-research tool* — it does not do ML training itself. It orchestrates training runs of *other* repos (e.g., a text-to-image model) by modifying their code and measuring outcomes.

## Architecture Overview

```
src/ratiocinator/
├── cli.py              # Click CLI — the user's entry point
├── config.py           # Pydantic config with env var overrides
├── experiment.py       # Single experiment loop (propose → apply → run → report)
├── observability.py    # Sentry SDK init (traces, logs, crash reporting)
├── llm/
│   └── client.py       # Async LiteLLM wrapper with task-based model routing
├── search/
│   ├── tree.py         # SQLite-backed experiment tree (persistent, crash-safe)
│   ├── bfts.py         # Best-First Tree Search algorithm with error recovery
│   └── parallel.py     # Parallel multi-node expansion via Vast.ai
├── ideation/
│   ├── retriever.py    # arXiv paper fetching + FAISS semantic search
│   ├── novelty.py      # LLM-based novelty filter for hypotheses
│   └── grounded.py     # Literature-grounded hypothesis generation
├── infra/
│   ├── vast_client.py  # Async Vast.ai HTTP API client (httpx)
│   ├── hf_client.py    # Async HuggingFace Jobs + Buckets client (huggingface_hub)
│   ├── vast_runner.py  # High-level runner: provision → transfer → execute → destroy
│   ├── remote.py       # RemoteExecutor: SSH/rsync/SCP with retries + Sentry spans
│   ├── bootstrap.py    # Generates onstart shell scripts for Vast.ai instances
│   ├── webhook.py      # HTTP server for receiving metric pushes from fleet
│   └── safety.py       # Budget caps + TTL enforcement (NOT LLM-controllable)
├── fleet/
│   ├── spec.py         # ExperimentSpec + PreflightSpec + ValidationSpec
│   ├── executor.py     # FleetExecutor: Vast.ai parallel orchestrator with Sentry observability
│   ├── hf_executor.py  # HFFleetExecutor: HuggingFace Jobs parallel orchestrator
│   ├── hf_data.py      # HF volume building + script upload utilities
│   ├── data.py         # DataProvisioner: pluggable data staging (S3, rsync, local)
│   └── results.py      # ResultStore: persistent JSON with merge semantics
├── orchestration/
│   ├── __init__.py     # Package init
│   └── coordinator.py  # ResearchCoordinator: autonomous ideation→fleet→analysis loop
└── synthesis/
    ├── plotting.py     # Matplotlib/seaborn chart generation from experiment trees
    ├── paper.py        # LaTeX/Markdown paper generator (LLM-filled sections)
    ├── reviewer.py     # LLM-as-judge automated review with revision cycles
    └── publisher.py    # HuggingFace Hub artifact upload
```

## Build, Test, Lint

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Full install (includes arXiv retrieval + plotting + publishing)
pip install -e ".[dev,ideation,synthesis]"

# Install with HuggingFace Jobs support
pip install -e ".[dev,hf]"

# Run all tests (~336 tests, <20s)
pytest tests/ -v

# Run a specific test file
pytest tests/test_fleet_spec.py -v

# Lint (must pass before committing)
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/
```

**Ruff rules:** `E, F, I, N, UP, B, SIM, RUF` — notably `N806` (variable naming in functions) and `N818` (exception classes need `Error` suffix). Line length is 100.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `ratiocinator research <spec.yaml>` | **Autonomous research loop:** ideation → fleet → analysis → paper |
| `ratiocinator fleet run <spec.yaml>` | **Declarative parallel experiments** from YAML spec |
| `ratiocinator fleet run <spec.yaml> --hf` | **Declarative experiments on HuggingFace Jobs** |
| `ratiocinator fleet status` | Show results from previous fleet runs |
| `ratiocinator run` | Single experiment: LLM proposes → sandbox executes → metrics reported |
| `ratiocinator search` | Best-First Tree Search over code modifications (`--local`, `--vast`) |
| `ratiocinator synthesize` | Full pipeline: search → plots → paper → review → publish |
| `ratiocinator ask` | One-off LLM prompt for quick queries |
| `ratiocinator vast-run` | Single ad-hoc experiment on a Vast.ai instance |
| `ratiocinator publish` | Upload artifacts to HuggingFace Hub |

The most important commands:
- **`research`** — fully autonomous, chains everything end-to-end
- **`fleet run`** — when you already know what arms to test

## Autonomous Research (`research` command)

The `research` command runs an autonomous loop via `ResearchCoordinator`:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Ideation   │────▶│  Translate   │────▶│   Execute    │────▶│   Analyse    │
│ (LLM proposes│     │ (Generate    │     │ (FleetExecutor│    │ (LLM reviews │
│  experiment  │     │  YAML configs│     │  runs arms on │    │  results,    │
│  arms)       │     │  + commands) │     │  Vast.ai GPU) │    │  decides     │
└──────────────┘     └──────────────┘     └──────────────┘     │  next steps) │
       ▲                                                        └──────┬───────┘
       │                                                               │
       │              ┌──────────────┐                                 │
       └──────────────│  Iterate?    │◀────────────────────────────────┘
                      └──────┬───────┘
                             │ No
                      ┌──────▼───────┐
                      │  Synthesise  │
                      │  (Paper +    │
                      │   Review)    │
                      └──────────────┘
```

The coordinator:
- Uses the LLM for ideation (proposing arms) and analysis (deciding whether to iterate)
- Uses FleetExecutor for execution (no custom SSH scripts)
- Uses the synthesis pipeline for paper generation
- Feeds prior results back to the LLM for iterative refinement
- Saves state after each iteration for crash recovery
- Enforces `_total_cost >= max_dollars` as a Python-enforced hard stop (not an LLM suggestion)

### ResearchSpec (YAML)

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
    - "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
  requirements: requirements.txt
metrics:
  protocol: json_line
  json_prefix: "METRICS:"
max_iterations: 3
max_dollars: 30.00
train_timeout_s: 3600
paper_title: "My Research Paper"  # Omit to skip synthesis
```

### Launch

```bash
ratiocinator research specs/my_research.yaml
ratiocinator research specs/my_research.yaml --data-server root@host:/data
```

## Key Design Decisions

### 1. Model-agnostic via LiteLLM

All LLM calls go through `llm/client.py` → `litellm.acompletion`. Config routes tasks to models:
- `config.llm.coding` → coding tasks (e.g., `deepseek/deepseek-coder`, `ollama/qwen3-coder:30b`)
- `config.llm.generalist` → synthesis, review, ideation (e.g., `ollama/gemma4:31b`)

The user typically runs local inference via Ollama or vLLM. Never hardcode model names — always use config routing.

`LLMResponse` supports `str()` conversion (returns `.content`), so it can be used safely in string contexts without causing `TypeError`.

### 2. Safety limits are NOT LLM-controllable

`SafetyController` enforces:
- `max_dollars_per_run` — hard spend cap per experiment
- `instance_ttl_seconds` — maximum instance lifetime

These are enforced in Python orchestrator code, never delegated to the LLM. This is a fundamental design invariant. Do not add code paths that let LLM outputs bypass budget or TTL limits.

The coordinator additionally checks `_total_cost >= max_dollars` after each iteration — this is a Python-enforced hard stop, not an LLM suggestion.

### 3. Everything async

All I/O-bound operations use `async/await`:
- LLM calls: `await client.complete()`
- Vast.ai API: `await vast_client.search_offers()`
- SSH/rsync: via `asyncio.create_subprocess_exec`
- Fleet execution: `await executor.run()`
- Coordinator loop: `await coordinator.run()`

The CLI bridge is `asyncio.run(_async_impl(...))`.

### 4. Experiment trees use SQLite

`search/tree.py` persists experiments to SQLite with `row_factory` for crash recovery. Use `--fresh` flag to clear stale state. The tree survives process crashes — this is intentional and important.

### 5. Docker-sandboxed by default, Vast.ai for GPU

Four execution backends:
- `SandboxRunner` — Docker containers (default, local)
- `LocalRunner` — subprocess on the host (`--local` flag)
- `VastRunner` — ephemeral Vast.ai GPU instances (`--vast` flag)
- `FleetExecutor` — parallel multi-arm experiments (fleet framework)

### 6. Declarative experiment specs (fleet framework)

The fleet framework eliminates custom scripts. Define experiments as YAML:

```yaml
name: my-ablation
hardware:
  gpu: "RTX 4090"
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
repo:
  url: git@github.com:user/repo.git
  branch: experiment/branch
data:
  source: s3-presigned
  urls_file: data-urls.txt
deps:
  pre_install:
    - "pip install torch --index-url https://download.pytorch.org/whl/cu130"
  requirements: requirements.txt
arms:
  - name: baseline
    command: "python train.py --config baseline.yaml"
  - name: optimized
    command: "python train.py --config optimized.yaml"
    env:
      CUDA_LAUNCH_BLOCKING: "0"
metrics:
  protocol: json_line    # or "block" for marker-delimited output
preflight:               # Optional: quick sanity check before full training
  command: "python train.py --epochs 1 --batch_size 2 --max_steps 5"
  timeout_s: 60
  check_metrics: true
validation:              # Optional: post-training ground-truth validation
  command: "python validate.py --output /workspace/output"
  timeout_s: 120
  required_metrics:
    - real_validity_pct
  prefix: "val_"
budget:
  max_dollars: 10.0
  train_timeout_s: 1800
```

Then run: `ratiocinator fleet run experiment.yaml`

### 7. Pre-flight validation (catch failures early)

The `preflight` section runs a quick dry-run before committing to the full training budget. It catches missing modules, bad data, broken args — problems that would otherwise waste a full Vast.ai provision cycle.

```yaml
preflight:
  command: "python train.py --epochs 1 --batch_size 2 --max_steps 5"
  timeout_s: 60
  check_metrics: true  # Also verify that METRICS: output appears
```

- Runs between data download and training in `_run_arm()`
- If preflight exits non-zero, the arm fails immediately (no training)
- If `check_metrics: true` and no metrics in output, the arm also fails
- Arm env vars are propagated to the preflight command
- Field is optional and defaults to `None`

### 8. Post-training validation (ground-truth metrics)

The `validation` section runs a separate command after training succeeds. Its metrics are merged into (and can override) the training metrics. This prevents false-positive proxy metrics — e.g., a heuristic reporting 99.5% validity while real parser-based validation shows 0%.

```yaml
validation:
  command: "ruby -c generated/*.rb | python count_valid.py"
  timeout_s: 120
  required_metrics:
    - real_validity_pct
  prefix: "val_"
```

- **Only runs on training success** — skipped if training exits non-zero
- **Metrics merge** — validation metrics are added to the result dict
- **Override** — same-named metrics are overwritten (use `prefix` to avoid)
- **Prefix namespacing** — `prefix: "val_"` produces `val_validity`, keeping training metrics intact
- **Required metrics** — arm fails if specified metrics are absent from validation output
- Field is optional and defaults to `None`

## Metrics Protocol

Training scripts communicate results back to the orchestrator via stdout. Two protocols are supported:

### JSON line protocol (default)
```
METRICS:{"train_loss": 0.33, "avg_iter_per_sec": 4.27, "peak_vram_gb": 5.49}
```

### Block protocol (for scripts with structured output)
```
--- RESULTS ---
avg_iter_per_sec: 4.268
peak_vram_gb: 5.49
final_loss: 0.331
--- END RESULTS ---
```

The protocol is configured in the experiment spec's `metrics` section. The parser is in `fleet/spec.py::parse_metrics()`.

Both preflight and validation steps use the same metrics protocol as configured in the spec.

## Configuration

Pydantic models in `config.py` with sensible defaults. Load order:
1. `.ratiocinator/config.json` (if exists in CWD)
2. Explicit `--config path.json` flag
3. Environment variables override specific fields:
   - `VAST_API_KEY` → `config.vast.api_key`
   - `HF_TOKEN` → `config.hf.token` (HuggingFace Jobs + publishing)
   - `HF_NAMESPACE` → `config.hf.namespace` (HF org/user for Jobs)
   - `HF_REPO_ID` → `config.publish.repo_id`
   - `SENTRY_DSN` → observability DSN override
   - `SENTRY_ENVIRONMENT` → `development` | `fleet` | `production`

## Results & Artifacts Storage

All runtime data lives under `.ratiocinator/` in the **current working directory**. This directory is gitignored — experiment results are never committed to the repository.

```
.ratiocinator/
├── config.json                  # Optional config overrides
├── search.db                    # SQLite experiment tree (search command)
├── results/
│   ├── experiments.json         # Fleet results (FleetExecutor default)
│   ├── <research-name>.json     # Research coordinator results
│   └── *.log                    # Per-arm execution logs
└── output/
    ├── paper.md                 # Generated paper (synthesize command)
    └── plots/                   # Generated charts
```

**Where to run ratiocinator from:** Run it from the root of the project you're experimenting on. Results land in `<that-project>/.ratiocinator/` and stay out of git. If `.ratiocinator/` isn't in that project's `.gitignore`, add it.

**Overriding the default path:** Both `fleet run` and `fleet status` accept `--results-file` to point at a custom location. The `research` command accepts `--results-file` similarly.

**Published artifacts** go to HuggingFace Hub via `ratiocinator publish`, not to the local filesystem.

## Vast.ai Integration — What You Need to Know

These are hard-won lessons from production use:

1. **Image must have SSH daemon.** Use `pytorch/pytorch:*` images. `python:3.11-slim` lacks sshd and `/workspace` and will fail.
2. **Boot time is 2-10 minutes.** Budget at least 3600s wall clock for any search using `--vast`.
3. **API redirects.** `cloud.vast.ai` → `console.vast.ai`. httpx needs `follow_redirects=True`.
4. **Torch version matters.** `torch.optim.Muon` requires PyTorch 2.11.0+ (NOT 2.7.0). When a specific torch version is needed, `pip uninstall torch torchvision -y` first, then install from the correct index URL. Note: PyTorch 2.11+ images use PEP 668 externally-managed Python — set `PIP_BREAK_SYSTEM_PACKAGES=1` env var.
5. **GPU name uses spaces.** Vast.ai API: `"RTX 4090"` (with space), not `"RTX_4090"` (underscore).
6. **CUDA version filtering.** Use `min_cuda_version` in HardwareSpec. PyTorch cu130 needs CUDA 13.0+ drivers.
7. **Bandwidth matters.** `min_inet_down >= 2000` Mbps prevents stalls on large data downloads.
8. **Rate limits.** Stagger instance creation by ~5s per arm to avoid API rate limits.
9. **Presigned URLs expire.** S3 presigned URLs have a TTL (~12h default). Regenerate before launching long experiments.
10. **g++ required for torch.compile.** Add `apt-get install -y g++` to `pre_install` deps.

## HuggingFace Jobs Integration

HuggingFace Jobs is supported as a parallel infrastructure provider alongside Vast.ai. Use `--hf` flag or set `provider: hf` in spec YAML.

### Key Differences from Vast.ai

| Aspect | Vast.ai | HuggingFace Jobs |
|--------|---------|------------------|
| Access model | Raw VM with SSH | Managed container, no SSH |
| Provisioning | Marketplace search → instance | `run_job(image=, command=, flavor=)` |
| Data staging | rsync / SCP / wget | Volume mounts (datasets, buckets) |
| Cleanup | `destroy_instance()` | Automatic (managed) |
| GPU selection | Any GPU on marketplace | Fixed flavors (`a100-large`, `l4`, etc.) |
| Cost model | Market-rate $/hr, variable | Fixed $/hr per flavor |
| Debugging | SSH into instance | Logs-only via API + Sentry |

### When to Use HF Jobs vs Vast.ai

**Choose HF Jobs when:**
- You want zero-ops provisioning (no SSH keys, no security groups)
- Your data is already on HuggingFace (Datasets or Buckets)
- You need reproducible, managed environments
- You want automatic cleanup — no orphaned instances
- Your experiments fit standard GPU flavors

**Choose Vast.ai when:**
- You need SSH access for interactive debugging
- You need non-standard GPU configurations or spot pricing
- You want to choose specific GPU models from the marketplace
- You need rsync/SCP for large local datasets
- You need fine-grained control over the instance lifecycle

### Usage

```bash
# Fleet run with HF Jobs
ratiocinator fleet run experiments/my_experiment.yaml --hf

# Autonomous research with HF
ratiocinator research specs/my_research.yaml --hf

# Check results (same command for both providers)
ratiocinator fleet status
```

### Spec YAML for HF

Minimal HF spec:

```yaml
name: my-hf-experiment
hardware:
  gpu: "A100"
  hf_flavor: "a100-large"   # REQUIRED for HF provider
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
repo:
  url: https://github.com/user/repo.git
  branch: main
arms:
  - name: baseline
    command: "python train.py"
metrics:
  protocol: json_line
budget:
  max_dollars: 5.0
  train_timeout_s: 3600
provider: hf
```

Full spec with data, deps, preflight, and validation:

```yaml
name: full-hf-experiment
hardware:
  gpu: "A100"
  hf_flavor: "a100-large"
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

repo:
  url: https://github.com/user/repo.git
  branch: experiment/my-branch
  remote_path: /workspace/repo     # Where to clone inside container

data:
  source: hf-dataset               # "hf-dataset" or "hf-bucket"
  hf_source: "user/my-dataset"     # HF repo ID or bucket name
  hf_mount_path: "/data"           # Mount point inside container

deps:
  pre_install:
    - "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    - "apt-get update -qq && apt-get install -y -qq g++"
  requirements: requirements.txt
  exclude_from_requirements:
    - "^torch"                     # Skip torch — installed via pre_install
  verify: "python -c 'import torch; print(torch.cuda.is_available())'"

arms:
  - name: baseline
    command: "python train.py --lr 0.001 --data-dir /data"
  - name: high-lr
    command: "python train.py --lr 0.01 --data-dir /data"
    env:
      CUDA_LAUNCH_BLOCKING: "0"
  - name: cosine-schedule
    command: "python train.py --lr 0.005 --scheduler cosine --data-dir /data"

metrics:
  protocol: json_line              # or "block" for marker-delimited output
  json_prefix: "METRICS:"

preflight:
  command: "python train.py --lr 0.001 --data-dir /data --epochs 1 --max_steps 5"
  timeout_s: 60
  check_metrics: true

validation:
  command: "python validate.py --output /workspace/output"
  timeout_s: 120
  required_metrics:
    - real_accuracy
  prefix: "val_"

budget:
  max_dollars: 10.0
  train_timeout_s: 3600

provider: hf                       # "vast" (default) or "hf"
```

### Data with HF Volumes

HF Jobs mounts data directly into containers — no SSH, rsync, or download scripts needed.

**HF Datasets** (read-only, versioned):
```yaml
data:
  source: hf-dataset
  hf_source: "username/my-dataset"   # Any HF dataset repo
  hf_mount_path: "/data"             # Available at /data inside container
```

**HF Buckets** (read-write, mutable):
```yaml
data:
  source: hf-bucket
  hf_source: "username/my-bucket"    # HF bucket name
  hf_mount_path: "/data"
```

**Output bucket** (auto-created): Every HF fleet run creates an output bucket at `{namespace}/ratiocinator-{experiment-name}`. The job writes to `/output/` and files persist in the bucket after the job completes. This is where checkpoints, logs, and artifacts go.

**Volume architecture in a typical job:**
```
Container filesystem:
  /input/          ← Read-only bucket with wrapper scripts (auto-managed)
  /data/           ← Your training data (HF Dataset or Bucket mount)
  /output/         ← Writable bucket for checkpoints + artifacts
  /workspace/repo/ ← Cloned experiment repository
```

### Available HF Flavors

| Flavor | GPU | VRAM | $/hr |
|--------|-----|------|------|
| `cpu-basic` | — | — | $0.01 |
| `cpu-upgrade` | — | — | $0.03 |
| `t4-small` | T4 | 16 GB | $0.40 |
| `t4-medium` | T4 | 16 GB | $0.60 |
| `l4` | L4 | 24 GB | $0.80 |
| `4xl4` | 4×L4 | 96 GB | $3.80 |
| `l40s` | L40S | 48 GB | $1.80 |
| `4xl40s` | 4×L40S | 192 GB | $8.30 |
| `8xl40s` | 8×L40S | 384 GB | $23.50 |
| `a10g-small` | A10G | 24 GB | $1.00 |
| `a10g-large` | A10G | 24 GB | $1.50 |
| `2xa10g-large` | 2×A10G | 48 GB | $3.00 |
| `4xa10g-large` | 4×A10G | 96 GB | $5.00 |
| `a100-large` | A100 | 80 GB | $2.50 |
| `4xa100` | 4×A100 | 320 GB | $10.00 |
| `8xa100` | 8×A100 | 640 GB | $20.00 |

Pricing is tracked in `HF_FLAVOR_PRICING` dict in `infra/hf_client.py` for budget estimation.

### HF Token Setup

HF Jobs requires a **fine-grained token** with write permissions:

1. Go to https://huggingface.co/settings/tokens → **Create new token**
2. Select **Fine-grained** token type
3. Enable these permissions:
   - **Repositories → Write access** (for bucket uploads)
   - **Jobs → Start and manage Jobs** (for job submission)
4. Set the token:
   ```bash
   # Option A: Environment variable
   export HF_TOKEN=hf_...

   # Option B: .env file in project root
   echo "HF_TOKEN=hf_..." >> .env

   # Option C: HuggingFace CLI login (stored in cache)
   huggingface-cli login
   ```

The client auto-discovers tokens in this order:
1. Explicit `HF_TOKEN` environment variable
2. Token stored via `huggingface-cli login` in HF cache

### Architecture Notes

#### Separate executors, not ABC

`HFFleetExecutor` is parallel to `FleetExecutor`, not derived from it. The SSH-interactive (Vast.ai) vs managed-job (HF) execution models are fundamentally different — forcing them into a shared ABC would create a leaky abstraction. Each executor owns its full lifecycle.

#### Wrapper script pattern

HF Jobs runs a single Docker command. We generate a self-contained bash wrapper script per arm that replicates FleetExecutor's step-by-step logic:

```
clone repo → install deps → preflight → train → validate → copy artifacts
```

Scripts are uploaded to an HF Bucket with unique paths `{experiment}/{arm}/run.sh`, then mounted read-only at `/input/`. The wrapper also:
- Sets `PYTHONUNBUFFERED=1` for reliable metrics capture
- Auto-installs `git` if the base image lacks it
- Propagates arm-specific environment variables
- Captures training exit codes for conditional validation
- Exports `RATIOCINATOR_STATE_PATH=/output/<exp>/<arm>/state.json` and creates the parent directory; training scripts should periodically dump `{"step": N, "loss": L, "eta_seconds": E, ...}` here so the orchestrator can monitor progress via the bucket API even when `fetch_job_logs` is degraded

#### Bucket-backed heartbeat (state.json)

`fetch_job_logs` is known to degrade or silently fail at times, leaving the orchestrator blind. Each HF arm therefore writes a small `state.json` heartbeat file to its mounted output bucket:

- The wrapper script exports `RATIOCINATOR_STATE_PATH` pointing at `/output/<experiment>/<arm>/state.json` (a path on the writable bucket FUSE mount)
- Training scripts opt in by atomically writing JSON like `{"step": 1234, "loss": 0.42, "eta_seconds": 600}` to that path every N steps
- `HFFleetExecutor._poll_job` interleaves `client.download_from_bucket()` reads with status polls (every `HEARTBEAT_POLL_INTERVAL_S` ≈ 30s) and emits `fleet.hf.heartbeat` Sentry breadcrumbs + `fleet.arm.heartbeat.{step,loss}` metrics
- On terminal success, if `parse_metrics(logs, ...)` returns nothing (logs missing/truncated), the executor falls back to `state.json` as the source of truth for final metrics

This keeps observability working when the logs API is unhealthy, without changing how the user's training code is structured.

#### Volume mounts for data

HF Jobs natively mounts Datasets and Buckets into containers. No SSH, rsync, or download scripts needed. The executor builds volume lists from the spec's `data` section and passes them directly to the Jobs API.

#### Async-over-sync

The `huggingface_hub` Python SDK is synchronous. All calls are wrapped in `asyncio.to_thread()` so they don't block the event loop. This allows concurrent arm polling without thread pool exhaustion.

#### Budget tracking

Cost is computed as `duration_s × hourly_rate / 3600` using the `HF_FLAVOR_PRICING` dict. The coordinator checks `_total_cost >= max_dollars` after each iteration as a Python-enforced hard stop.

### HF Client API Reference

The `HFClient` in `infra/hf_client.py` wraps `huggingface_hub.HfApi`:

```python
async with HFClient(token="hf_...") as client:
    # Submit a job
    job_id = await client.run_job(
        image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
        command=["bash", "/input/run.sh"],
        flavor="a100-large",
        timeout="2h",
        env={"LR": "0.001"},
        volumes=[...],               # Volume objects or dicts
        labels={"arm": "baseline"},
        namespace="my-org",
    )

    # Poll status
    info = await client.get_job(job_id)    # → HFJobInfo
    print(info.stage)                       # HFJobStage.RUNNING

    # Fetch logs (blocks until available)
    logs = await client.get_job_logs(job_id)  # → str (newline-joined)

    # Cancel a running job
    await client.cancel_job(job_id)

    # List jobs
    jobs = await client.list_jobs(namespace="my-org")

    # Bucket operations
    await client.create_bucket("user/my-bucket", private=True)
    await client.upload_to_bucket("user/my-bucket", "/tmp/file.sh", "remote/path.sh")
```

**Important SDK method mappings** (the `huggingface_hub` API names differ from what you might expect):

| Our method | Underlying `HfApi` call | Notes |
|-----------|-------------------------|-------|
| `get_job()` | `inspect_job(job_id=)` | Keyword-only `job_id` arg |
| `get_job_logs()` | `fetch_job_logs(job_id=)` | Returns iterable of chunks, we join with `\n` |
| `cancel_job()` | `cancel_job(job_id=)` | Keyword-only `job_id` arg |
| `run_job()` | `run_job(...)` | Volumes must be `Volume` objects, not dicts |
| `upload_to_bucket()` | `batch_bucket_files(bucket_id, add=[(local, remote)])` | NOT `upload_file(repo_type="bucket")` |
| `create_bucket()` | `create_bucket(bucket_id, private=, exist_ok=True)` | Instance method, not module-level |

**Job stages:** `HFJobStage` enum maps to plain string values returned by the API:
- Non-terminal: `PENDING`, `STARTING`, `RUNNING`, `UPDATING`, `UNKNOWN`
- Terminal (success): `COMPLETED`
- Terminal (failure): `ERROR`, `FAILED`, `CANCELLED`, `DELETED`

Note: The HF API returns `"ERROR"` for failed jobs (not `"FAILED"`). Both are handled as terminal failure states.

### HFFleetExecutor Lifecycle

For each arm, `HFFleetExecutor` follows this sequence:

```
1. Generate wrapper script (_build_wrapper_script)
     └── clone → deps → preflight → train → validate → artifacts
2. Upload script to HF Bucket (upload_arm_script)
3. Build volume list (_build_volumes)
     └── /input (scripts), /data (optional), /output (artifacts)
4. Submit HF Job (client.run_job)
5. Poll until terminal (_poll_job, every 15s)
6. Fetch logs (client.get_job_logs)
7. Parse metrics (parse_metrics from spec)
8. Handle result:
     └── COMPLETED → extract metrics
     └── ERROR/FAILED → extract error, report crash to Sentry
     └── CANCELLED → mark as cancelled
9. Emit Sentry metrics (duration, exit code, cost)
```

Arms run in parallel via `asyncio.gather()`. Job creation is staggered by 2s.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token with write + jobs permissions |
| `HF_NAMESPACE` | No | HF username or org to run jobs under (defaults to token owner) |

### Dependencies

Install with: `pip install -e ".[hf]"` (adds `huggingface-hub>=1.9.0`)

For development: `pip install -e ".[dev,hf]"`

### Gotchas

| Problem | Solution |
|---------|----------|
| `hf_flavor is required` error | Set `hf_flavor` explicitly in spec YAML |
| `403 Forbidden: missing permissions: job.write` | Token needs **Jobs** permission — create a fine-grained token |
| `403 Forbidden: xet-write-token` | Token needs **Repositories → Write access** permission |
| No SSH debugging | Check job logs via `client.get_job_logs()`; use Sentry crash reporting |
| Private repos | HF Jobs can't clone private repos via SSH — use public repos or HF-hosted repos |
| `huggingface_hub` not installed | Install with `pip install -e ".[hf]"` |
| `'dict' object has no attribute 'to_dict'` | Volumes must be `huggingface_hub.Volume` objects, not dicts |
| Job logs have no newlines | Fixed — we join `fetch_job_logs()` chunks with `\n` |
| `git: command not found` in job | Wrapper scripts auto-install git; or use an image that includes it |
| Job stage `UNKNOWN` on first poll | Normal — job is being scheduled. Non-terminal, polling continues. |
| `ERROR` stage not recognized | Fixed — `HFJobStage.ERROR` is mapped as terminal failure |
| Metrics not parsed from logs | Ensure `PYTHONUNBUFFERED=1` is set (wrapper scripts do this automatically) |

### HF Jobs Best Practices for Agents

When writing code or specs that target HF Jobs, remember these key paradigms:

#### Serverless / Black-Box Execution

HF Jobs are NOT interactive VMs — you cannot SSH in. Unlike Vast.ai where you have full shell access, HF Jobs are **fire-and-forget containers**:

- **No SSH debugging.** All information comes from job logs and bucket artifacts.
- **No mid-run intervention.** Once submitted, the job runs to completion or failure.
- **Use preflight for validation.** Catch setup errors before committing GPU time.
- **Emit verbose logs.** Your training script's stdout becomes the only debugging surface.

#### hf://buckets Protocol

Use `hf://buckets/{owner}/{bucket}/{path}` for programmatic bucket access:

```python
# Orchestrator-side: download final artifacts
path = await client.download_artifact(
    bucket_name="my-org/ratiocinator-experiment",
    remote_path="experiment/arm/checkpoints/best.pt",
    local_path="/tmp/best.pt",
)

# Orchestrator-side: sync a directory to a bucket
await client.sync_to_bucket(
    local_dir="/tmp/scripts/",
    bucket_path="hf://buckets/my-org/my-bucket/scripts/",
)
```

Requires `huggingface_hub>=1.9.0` (enforced by `HFClient._get_api()`).

#### Label-Based Job Management

Jobs are tagged with `{experiment, arm, arm_index}` labels at submission. Use these for:

- **Orphan cleanup:** `cleanup_hf_orphans(client, experiment, arm_name)` cancels stuck jobs
- **Job filtering:** `list_jobs()` + filter by labels to find specific arms
- **Fleet restart:** `ratiocinator fleet restart` uses labels to safely cancel and resubmit

#### Data Access — Volumes, Not Downloads

HF Jobs mount data via FUSE — no rsync/SCP/wget needed:

```yaml
# CORRECT: use volume mounts
data:
  source: hf-bucket
  hf_source: "org/training-data"
  hf_mount_path: "/data"

# INCORRECT: don't use download-based data staging for HF
# data:
#   source: s3-presigned  ← This is for Vast.ai only
```

Write checkpoints to `/output/` — they persist in the bucket across preemptions.

#### Preemption Handling

HF containers may be preempted and restarted. The wrapper script handles this automatically:

1. Detects restart via nonce-based sentinel in `/output/`
2. Finds latest `checkpoint_*.pt` file
3. Exports `RATIOCINATOR_RESUME_CHECKPOINT` env var

**Training scripts MUST:**
- Save checkpoints to `/output/{experiment}/{arm}/checkpoints/checkpoint_step{N:08d}.pt`
- Check `os.environ.get("RATIOCINATOR_RESUME_CHECKPOINT")` on startup
- Resume from the checkpoint if present

#### Debugging HF Job Failures

1. Check logs: `await client.get_job_logs(job_id)` (or `ratiocinator fleet status`)
2. Check `state.json`: `await client.download_from_bucket(bucket, "exp/arm/state.json")`
3. Check Sentry: `fleet.hf.preemption` breadcrumbs, `fleet.arm.*` metrics
4. Browse artifacts: `https://huggingface.co/datasets/{bucket_prefix or namespace}/ratiocinator-{experiment}` (bucket ID resolved by `HFFleetExecutor._resolve_output_bucket()`)

## Observability

Sentry is integrated throughout the system (sentry-sdk >= 2.35.0):

### Core setup
- `observability.py` — idempotent `init_sentry()`, auto-discovers git SHA for release tags
- `enable_logs=True` — stdlib `logging` auto-forwarded to Sentry Logs
- DSN is hardcoded for the project but overridable via `SENTRY_DSN` env var

### Fleet-level observability
- **Sentry spans** on: LLM calls, Vast.ai API, sandbox runs, SSH/rsync, fleet execution, preflight, validation
- **Per-arm breadcrumbs** via `fleet_breadcrumb()` at each stage: provision, boot, clone, deps, data, preflight, training, validation, cleanup
- **Per-arm metrics** via `fleet_metric()`: `fleet.arm.duration`, `fleet.arm.exit_code`, `fleet.arm.cost`
- **Remote crash reporting** via `_report_remote_crash()`: parses tracebacks from SSH stderr into proper Sentry exception frames, attaches stderr/stdout as Sentry attachments
- **Scope isolation**: each arm runs in `sentry_sdk.new_scope()` to prevent tag pollution between concurrent `asyncio.gather()` arms

### API notes (sentry-sdk 2.x)
- `sentry_sdk.metrics.distribution(name, value, unit=, attributes=)` — positional args, not `key=/value=/tags=`
- `sentry_sdk.add_attachment(bytes=, filename=, content_type=)` — not `event["attachments"]`
- `sentry_sdk.new_scope()` — context manager for per-arm scope isolation

## Test Patterns

- **All tests use pytest + pytest-asyncio** with `asyncio_mode = "auto"` (no manual `@pytest.mark.asyncio` needed)
- **Heavy mocking** — external services (Vast.ai, LiteLLM, Docker, SSH) are always mocked
- **`AsyncMock`** for async callables, `MagicMock` for sync
- **No real network calls** in the test suite
- **Test files mirror source structure**: `test_fleet_spec.py` tests `fleet/spec.py`, `test_fleet_executor.py` tests `fleet/executor.py`, `test_coordinator.py` tests `orchestration/coordinator.py`, etc.
- **Use `tmp_path` for all file outputs** — including `fleet_config.log_dir`, results files, etc. Never write test artifacts to `results/`.

Example test structure:
```python
class TestExperimentSpecFromYAML:
    def test_minimal_spec(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump({...}))
        spec = ExperimentSpec.from_yaml(spec_file)
        assert spec.name == "test-experiment"

class TestValidation:
    @pytest.fixture
    def validation_spec(self):
        return ExperimentSpec(
            name="validation-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=120,
                required_metrics=["real_validity_pct"],
            ),
        )

    @pytest.mark.asyncio
    async def test_validation_merges_metrics(self, validation_spec, ...):
        # Mock remote.run side_effect: hwinfo, clone, train, validation
        mock_remote.run = AsyncMock(side_effect=[...])
        results = await executor.run(arm_indices=[0])
        assert results[0].metrics["real_validity_pct"] == 12.5
```

## Fleet Framework Deep Dive

This is the most important abstraction for running experiments. Here's how the pieces fit:

### ExperimentSpec (`fleet/spec.py`)
Pydantic model loaded from YAML. Contains: `HardwareSpec`, `DataSpec`, `RepoSpec`, `DepsSpec`, list of `ArmSpec`, `MetricsSpec`, `BudgetSpec`, optional `PreflightSpec`, optional `ValidationSpec`. Also provides `parse_metrics(stdout)` to extract metrics from training output.

### DataProvisioner (`fleet/data.py`)
Abstract base class with four implementations:
- `S3PresignedProvisioner` — downloads from presigned URLs (most common)
- `RsyncProvisioner` — rsync from a data server
- `LocalProvisioner` — SCP from the orchestrator machine
- `NullProvisioner` — no-op (data already on instance or not needed)

Factory: `create_provisioner("s3-presigned", urls_file="urls.txt")`

### RemoteExecutor (`infra/remote.py`)
Unified SSH/rsync/SCP interface with:
- Retry logic with exponential backoff
- Sentry span instrumentation
- Configurable timeouts per operation
- `wait_for_ssh()`, `run()`, `rsync_to()`, `scp_to()`, `scp_from()`

### ResultStore (`fleet/results.py`)
JSON-file-backed store with merge semantics:
- Success overrides failure for the same (experiment, arm) pair
- First success wins (no clobbering)
- `get_best(metric_key)`, `compare_arms()`, `export_table()`

### FleetExecutor (`fleet/executor.py`)
The main orchestrator. For each arm:
1. Provision Vast.ai instance
2. Wait for boot + SSH
3. Clone repo + gather GPU hardware info
4. Install dependencies (pre_install → requirements → verify)
5. Download data via DataProvisioner
6. **Pre-flight validation** (if `spec.preflight` is set) — quick sanity check
7. Run training command
8. Parse metrics from stdout
9. **Post-training validation** (if `spec.validation` is set, training succeeded) — ground-truth check, metrics merge
10. Emit Sentry metrics (duration, exit code, cost)
11. Destroy instance (always, even on failure — `finally` block)

Arms run in parallel via `asyncio.gather()`. Instance creation is staggered by 5s. Each arm runs in an isolated Sentry scope to prevent tag pollution.

### ResearchCoordinator (`orchestration/coordinator.py`)
Top-level autonomous orchestrator. For the `research` command:
1. Load context (base training config, arXiv papers)
2. **Ideate** — LLM proposes experiment arms as JSON
3. **Translate** — generates ExperimentSpec with proper env var propagation
4. **Execute** — FleetExecutor runs all arms in parallel
5. **Analyse** — LLM reviews results, sets `should_iterate` flag
6. **Iterate or converge** — repeat from step 2 if budget allows
7. **Synthesise** — generates paper with automated review cycles

## When You're Asked to Run an Experiment

### Option A: Fully Autonomous (recommended)

1. **Create a ResearchSpec YAML** in `specs/` — define topic, repo, hardware, data, budget
2. **Run**: `ratiocinator research specs/my_research.yaml --data-server root@host:/data`
3. **Or with HF Jobs**: `ratiocinator research specs/my_research.yaml --hf`
4. The coordinator handles everything: ideation, config generation, fleet execution, analysis, iteration

### Option B: Manual (when you know the arms)

1. **Define the experiment as YAML** — create a spec in `experiments/` using the `ExperimentSpec` schema
2. **Ensure the target repo has the right branch** with training scripts that output metrics in one of the supported protocols
3. **Stage data** — if data is on S3, generate presigned URLs and put them in a file. For HF, use HF Datasets or Buckets.
4. **Run**: `ratiocinator fleet run experiments/my_experiment.yaml` (or `--hf` for HuggingFace)
5. **Re-run failures**: `ratiocinator fleet run experiments/my_experiment.yaml --arms 4,5,6`
6. **Check results**: `ratiocinator fleet status`

Do NOT write custom orchestrator scripts. The fleet framework and research coordinator exist to prevent that.

## Common Gotchas

| Problem | Solution |
|---------|----------|
| `torch.optim.Muon` not found | Need PyTorch 2.11.0+ (NOT 2.7). Use `pre_install` in deps spec. Set `PIP_BREAK_SYSTEM_PACKAGES=1` for 2.11+ images. |
| SSH timeout on Vast.ai | Instance may still be pulling Docker image. Boot takes 2-10 min. |
| `JSONDecodeError: Invalid control character` | Use `json.loads(text, strict=False)` for LLM output. |
| Local LLMs wrap sections in full LaTeX docs | `paper.py::_clean_section()` strips these. |
| Stale `search.db` from prior runs | Use `--fresh` flag or delete `.ratiocinator/search.db`. |
| httpx log spam at INFO | Set `logging.getLogger("httpx").setLevel(logging.WARNING)` |
| Data download timeout | Increase `download_timeout_s` in budget spec; check URL validity. |
| `python:3.11-slim` on Vast.ai | Use `pytorch/pytorch:*` images instead — they have sshd. |
| `torch.compile` crashes: no C++ compiler | Add `apt-get install -y g++` to deps `pre_install`. |
| GPU name mismatch in offers | Use `"RTX 4090"` (space), not `"RTX_4090"` (underscore). |
| Heuristic metrics mask real failures | Use `validation:` section with real parser-based validation. |
| Missing modules waste GPU time | Use `preflight:` section to catch import errors early. |
| `LLMResponse` used as string raises `TypeError` | `LLMResponse.__str__()` returns `.content` — safe to use. |
| `FleetExecutor` expects `FleetConfig`, not `VastConfig` | Construct `FleetConfig(api_key=, ssh_key=, ...)` explicitly. |

## Conventions

- **Imports:** `from __future__ import annotations` at the top of every file
- **Type hints:** Use `str | None` not `Optional[str]`; use `list[str]` not `List[str]`
- **Pydantic for config/specs:** All structured data uses Pydantic `BaseModel`
- **Dataclasses for internal state:** `ResearchSpec`, `IterationResult`, `ResearchReport` use `@dataclass`
- **Async by default:** All I/O operations are async. Sync wrappers use `asyncio.run()`
- **Error suffix:** Exception classes must end with `Error` (ruff N818)
- **Variable naming:** lowercase in functions (ruff N806), `CamelCase` for classes only
- **Sentry optional:** Always guard with `try: import sentry_sdk except ImportError: sentry_sdk = None`
- **Logging:** Use `logger = logging.getLogger(__name__)` — never `print()` for operational output
- **Click CLI:** Lazy imports inside command functions to keep startup fast
- **No secrets in code:** API keys come from env vars or config files, never hardcoded
- **YAML specs:** Experiment definitions go in `experiments/` (known arms) or `specs/` (autonomous research)
