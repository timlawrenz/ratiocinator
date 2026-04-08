# Ratiocinator

**Distributed Scientific Discovery Pipeline using Agentic Tree Search**

Ratiocinator automatically proposes hypotheses, modifies source code, runs experiments on GPU instances, and synthesizes LaTeX papers — all driven by LLMs and Best-First Tree Search.

## Features

- **Agentic Tree Search** — Best-First Search over code modifications with automatic error recovery
- **Literature-Grounded Ideation** — arXiv RAG pipeline with novelty filtering
- **Distributed Execution** — Run experiments on Vast.ai GPU instances with safety controls
- **Paper Synthesis** — Auto-generate LaTeX papers with plots, then review and revise
- **HuggingFace Publishing** — Push artifacts to HuggingFace Hub datasets
- **Model Agnostic** — Works with any LiteLLM-compatible backend (Ollama, OpenAI, etc.)

## Quick Start

```bash
pip install -e ".[dev,ideation,synthesis]"
```

Create a `.env` file:

```
VAST_API_KEY=your_vast_api_key
HF_TOKEN=your_huggingface_token
```

### Run a local experiment

```bash
ratiocinator search --repo ./examples --command "python train.py" --local
```

### Run the full pipeline

```bash
ratiocinator synthesize \
  --repo ./examples \
  --title "Optimizing Nonlinear Regression" \
  --command "python train.py" \
  --local \
  --publish-to user/my-results
```

### Run on Vast.ai

```bash
ratiocinator vast-run \
  --repo-url https://github.com/user/repo.git \
  --command "cd examples && python train.py" \
  --no-install
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ratiocinator run` | Single experiment: propose → execute → report |
| `ratiocinator search` | Tree search over code modifications |
| `ratiocinator synthesize` | Full pipeline: search → plots → paper → review |
| `ratiocinator vast-run` | Run experiment on Vast.ai GPU instance |
| `ratiocinator publish` | Upload artifacts to HuggingFace Hub |
| `ratiocinator ask` | One-off LLM prompt |

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
| `HF_TOKEN` | HuggingFace token for publishing artifacts |
| `HF_REPO_ID` | Default HuggingFace dataset repo for publishing |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Literature  │────▶│  Tree Search │────▶│  Synthesis  │
│  RAG + arXiv │     │  (Best-First)│     │  (LaTeX)    │
└─────────────┘     └──────┬───────┘     └─────┬──────┘
                           │                    │
                    ┌──────▼───────┐     ┌──────▼──────┐
                    │   Sandbox    │     │  Auto-Review │
                    │ Docker/Local │     │  + Publish   │
                    │  / Vast.ai   │     │  (HF Hub)    │
                    └──────────────┘     └─────────────┘
```

## Metrics Protocol

Training scripts output metrics as: `METRICS:{"train_loss": 1.09, "val_loss": 1.77}`

The pipeline parses this to drive tree search decisions.

## Development

```bash
pip install -e ".[dev,ideation,synthesis]"
pytest tests/ -q          # 105 tests
ruff check src/ tests/    # lint
```

## License

Apache-2.0
