# Copilot Instructions

## Project Overview

Ratiocinator is an auto-researcher: a distributed scientific discovery pipeline that uses Agentic Tree Search to propose hypotheses, modify source code, run experiments on ephemeral GPU instances, and synthesize LaTeX papers. The design document is `reatiocinator.md`.

## Architecture

The system has four major components:

1. **Command Center** — Local orchestrator running LiteLLM, an arXiv RAG database, and Best-First Tree Search state management.
2. **Dispatcher** — Commits experimental code branches to GitHub and provisions Vast.ai GPU instances via `onstart` scripts.
3. **Fleet** — Ephemeral Vast.ai GPU containers that train models to a step budget, upload artifacts, and signal completion.
4. **Compiler** — Converges results, generates plots (matplotlib/seaborn), and compiles LaTeX papers with automated LLM-as-a-judge review cycles.

## Module Map

```
src/ratiocinator/
├── cli.py              # Click CLI (run, search, ask commands)
├── config.py           # Pydantic config (LLM, sandbox, search, safety)
├── experiment.py       # Single experiment loop
├── llm/client.py       # Async LiteLLM wrapper with model routing
├── sandbox/runner.py   # Docker sandbox runner
├── search/
│   ├── tree.py         # SQLite-backed experiment tree
│   ├── bfts.py         # Best-First Tree Search algorithm
│   └── parallel.py     # Parallel search over Vast.ai instances
├── ideation/
│   ├── retriever.py    # arXiv paper retrieval + semantic search
│   ├── novelty.py      # Novelty filter (LLM-based assessment)
│   └── grounded.py     # Literature-grounded hypothesis generation
├── infra/
│   ├── vast_client.py  # Async Vast.ai API client (httpx)
│   ├── bootstrap.py    # Instance onstart script generator
│   ├── webhook.py      # HTTP server for metric pushes
│   └── safety.py       # Budget caps, TTL enforcement, cleanup
└── synthesis/
    ├── plotting.py     # Matplotlib chart generation
    ├── paper.py        # LaTeX paper generator (LLM-filled sections)
    ├── reviewer.py     # Automated reviewer with revision cycles
    └── publisher.py    # HuggingFace Hub artifact upload
```

## Key Design Decisions

- **Model agnostic via LiteLLM** — All LLM calls go through a LiteLLM abstraction layer pointing at local inference (vLLM, Ollama) or API providers. Coding tasks route to coding models (e.g., DeepSeek-Coder); synthesis tasks route to generalist models.
- **Docker-sandboxed experiments** — Every training run executes inside a Docker container. Crashes are isolated; the orchestrator catches exit codes, feeds stack traces back to the LLM, and retries.
- **Hard-coded safety limits** — `MAX_DOLLARS_PER_PAPER` spend cap and strict TTL timeouts are enforced outside LLM control to prevent runaway costs.
- **Git-ops experiment tracking** — Each hypothesis gets its own branch. Code diffs, configs, and winning checkpoints are committed for full reproducibility.
- **Artifact pipeline** — Checkpoints and logs upload to HuggingFace Hub, tagged with git commit hashes.

## Build, Test, and Lint

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Full install (with optional deps)
pip install -e ".[dev,ideation,synthesis]"

# Run all tests (82 tests)
pytest tests/ -v

# Run a single test
pytest tests/test_llm.py::test_complete_routes_coding_task -v

# Lint
ruff check src/ tests/

# Lint with auto-fix
ruff check --fix src/ tests/
```

## Conventions

- **Metrics protocol:** Training scripts output `METRICS:{"key": value}` on stdout for the orchestrator to parse.
- **Async LLM calls:** All LLM interactions use `async/await` via `litellm.acompletion`.
- **Config-driven routing:** LLM model selection is config-based, not hardcoded. Coding tasks route to `config.llm.coding`, everything else to `config.llm.generalist`.
- **Pydantic config:** All configuration uses Pydantic models with sensible defaults. Load from JSON file or use defaults.
- **Safety limits are not LLM-controllable:** Budget caps and TTLs are enforced in Python orchestrator code, never delegated to the LLM.
- **SQLite persistence:** Experiment trees use SQLite with `row_factory` for crash recovery.
- **Ruff rules:** E, F, I, N, UP, B, SIM, RUF — N818 requires Error suffix on exceptions.
