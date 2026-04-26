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
| Debugging | SSH into instance | Logs via API |
| Cleanup | Manual (auto-destroy on completion) | Automatic |
| Cost model | Variable market pricing | Fixed per-flavor pricing |
| Flag | `--vast` (default) | `--hf` |

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
