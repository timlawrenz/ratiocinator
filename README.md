# Ratiocinator

**Distributed Scientific Discovery Pipeline using Agentic Tree Search**

Ratiocinator automatically proposes hypotheses, modifies source code, runs experiments on GPU instances, and synthesizes LaTeX papers — all driven by LLMs and Best-First Tree Search.

## Features

- **Agentic Tree Search** — Best-First Search over code modifications with automatic error recovery
- **Literature-Grounded Ideation** — arXiv RAG pipeline with novelty filtering
- **Multi-Provider GPU Execution** — Run experiments on **Vast.ai** or **HuggingFace Jobs** with safety controls
- **Declarative Experiments** — Define ablation studies as YAML specs, run with one command
- **Paper Synthesis** — Auto-generate LaTeX papers with plots, then review and revise
- **HuggingFace Publishing** — Push artifacts to HuggingFace Hub datasets
- **Model Agnostic** — Works with any LiteLLM-compatible backend (Ollama, OpenAI, etc.)

## Quick Start

```bash
pip install -e ".[dev,ideation,synthesis]"

# For HuggingFace Jobs support:
pip install -e ".[dev,hf]"
```

Create a `.env` file:

```
VAST_API_KEY=your_vast_api_key
HF_TOKEN=your_huggingface_token
HF_NAMESPACE=your_hf_username
```

### Run a local experiment

```bash
ratiocinator search --repo ./examples --command "python train.py" --local
```

### Run on Vast.ai

```bash
ratiocinator vast-run \
  --repo-url https://github.com/user/repo.git \
  --command "cd examples && python train.py" \
  --no-install
```

### Run on HuggingFace Jobs

```bash
ratiocinator fleet run experiments/my_experiment.yaml --hf
```

### Run the full autonomous pipeline

```bash
# On Vast.ai (default)
ratiocinator research specs/my_research.yaml

# On HuggingFace Jobs
ratiocinator research specs/my_research.yaml --hf
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ratiocinator research <spec.yaml>` | Autonomous research loop: ideation → fleet → analysis → paper |
| `ratiocinator fleet run <spec.yaml>` | Declarative parallel experiments from YAML spec |
| `ratiocinator fleet run <spec.yaml> --hf` | Same, on HuggingFace Jobs instead of Vast.ai |
| `ratiocinator fleet status` | Show results from previous fleet runs |
| `ratiocinator run` | Single experiment: propose → execute → report |
| `ratiocinator search` | Tree search over code modifications |
| `ratiocinator synthesize` | Full pipeline: search → plots → paper → review |
| `ratiocinator vast-run` | Run experiment on Vast.ai GPU instance |
| `ratiocinator publish` | Upload artifacts to HuggingFace Hub |
| `ratiocinator ask` | One-off LLM prompt |

## Experiment Specs (YAML)

Define experiments declaratively. Here's a minimal example:

```yaml
name: lr-ablation
hardware:
  gpu: "RTX 4090"
  hf_flavor: "l4"                    # Required for --hf
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
repo:
  url: https://github.com/user/repo.git
  branch: main
arms:
  - name: lr-1e3
    command: "python train.py --lr 0.001"
  - name: lr-1e4
    command: "python train.py --lr 0.0001"
metrics:
  protocol: json_line
budget:
  max_dollars: 5.0
  train_timeout_s: 3600
```

Training scripts emit metrics as: `METRICS:{"train_loss": 1.09, "val_loss": 1.77}`

See `examples/fleet/hf_ablation.yaml` for a complete HF example.

## Infrastructure Providers

| Feature | Vast.ai | HuggingFace Jobs |
|---------|---------|------------------|
| GPU access | Marketplace (any GPU) | Fixed flavors |
| Data staging | rsync / SCP / S3 | Volume mounts (Datasets, Buckets) |
| Debugging | SSH into instance | Logs via API + bucket artifacts |
| Cleanup | Manual (auto-destroy on completion) | Automatic (managed) |
| Cost model | Variable market pricing | Fixed per-flavor pricing |
| Preemption | Not applicable (dedicated) | Automatic resume via checkpoints |
| Job identification | Single label string | Structured labels (experiment, arm, arm_index) |
| Flag | `--vast` (default) | `--hf` |

### HF Jobs: Data and Buckets

HF Jobs mount data directly into containers via FUSE — no download scripts needed:

```yaml
data:
  source: hf-bucket              # or "hf-dataset"
  hf_source: "org/my-data"      # HF repo ID or bucket name
  hf_mount_path: "/data"        # Available at /data/ inside the container
```

Output artifacts (checkpoints, logs) are written to `/output/` which maps to an auto-created HF Bucket at `{bucket_prefix or namespace}/ratiocinator-{experiment-name}` (where `bucket_prefix` defaults to `namespace` if not set). These persist across job restarts and can be browsed on the HF Hub.

For programmatic bucket access from the orchestrator, use the `hf://buckets/` protocol (requires `huggingface_hub>=1.9.0`).

### HF Jobs: Preemption

HF containers may be preempted and restarted. Ratiocinator detects this automatically and exports `RATIOCINATOR_RESUME_CHECKPOINT` pointing to the latest checkpoint. Training scripts should check this variable on startup.

See `docs/huggingface-jobs.md` for detailed guidance on preemption handling, label-based querying, and debugging.

## Configuration

JSON config file or sensible defaults. Secrets come from environment variables.

```bash
ratiocinator --config my_config.json search --repo ./my-project
```

```json
{
  "llm": {
    "coding": {"model": "ollama/qwen3-coder:30b", "api_base": "http://localhost:11434"},
    "generalist": {"model": "ollama/gemma3:27b", "api_base": "http://localhost:11434"}
  },
  "search": {"max_depth": 5, "max_nodes": 50},
  "vast": {"max_dph": 0.15, "default_image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"},
  "safety": {"max_dollars_per_run": 10.0, "instance_ttl_seconds": 1800}
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VAST_API_KEY` | Vast.ai API key for remote GPU execution |
| `HF_TOKEN` | HuggingFace token (Jobs, Buckets, publishing) |
| `HF_NAMESPACE` | HuggingFace username or org for Jobs |
| `HF_REPO_ID` | Default HuggingFace dataset repo for publishing |
| `SENTRY_DSN` | Sentry DSN for observability (optional) |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Literature  │────▶│  Tree Search │────▶│  Synthesis  │
│  RAG + arXiv │     │  (Best-First)│     │  (LaTeX)    │
└─────────────┘     └──────┬───────┘     └─────┬──────┘
                           │                    │
                    ┌──────▼───────┐     ┌──────▼──────┐
                    │ FleetExecutor│     │  Auto-Review │
                    │  Vast.ai /   │     │  + Publish   │
                    │  HF Jobs     │     │  (HF Hub)    │
                    └──────────────┘     └─────────────┘
```

## Development

```bash
pip install -e ".[dev,ideation,synthesis,hf]"
pytest tests/ -v          # 336 tests
ruff check src/ tests/    # lint
```

## License

Apache-2.0
