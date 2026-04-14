# AGENT.md — Ratiocinator

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
│   ├── vast_runner.py  # High-level runner: provision → transfer → execute → destroy
│   ├── remote.py       # RemoteExecutor: SSH/rsync/SCP with retries + Sentry spans
│   ├── bootstrap.py    # Generates onstart shell scripts for Vast.ai instances
│   ├── webhook.py      # HTTP server for receiving metric pushes from fleet
│   └── safety.py       # Budget caps + TTL enforcement (NOT LLM-controllable)
├── fleet/
│   ├── spec.py         # ExperimentSpec: YAML-driven declarative experiment definition
│   ├── executor.py     # FleetExecutor: parallel orchestrator for multi-arm experiments
│   ├── data.py         # DataProvisioner: pluggable data staging (S3, rsync, local)
│   └── results.py      # ResultStore: persistent JSON with merge semantics
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

# Run all tests (~192 tests, <15s)
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
| `ratiocinator run` | Single experiment: LLM proposes → sandbox executes → metrics reported |
| `ratiocinator search` | Best-First Tree Search over code modifications (`--local`, `--vast`) |
| `ratiocinator synthesize` | Full pipeline: search → plots → paper → review → publish |
| `ratiocinator ask` | One-off LLM prompt for quick queries |
| `ratiocinator fleet run <spec.yaml>` | **Declarative parallel experiments** from YAML spec |
| `ratiocinator fleet status` | Show results from previous fleet runs |
| `ratiocinator vast-run` | Single ad-hoc experiment on a Vast.ai instance |
| `ratiocinator publish` | Upload artifacts to HuggingFace Hub |

The most important command for real experiments is `fleet run` — it replaces the need for custom orchestrator scripts.

## Key Design Decisions

### 1. Model-agnostic via LiteLLM

All LLM calls go through `llm/client.py` → `litellm.acompletion`. Config routes tasks to models:
- `config.llm.coding` → coding tasks (e.g., `deepseek/deepseek-coder`, `ollama/qwen3-coder:30b`)
- `config.llm.generalist` → synthesis, review, ideation (e.g., `ollama/gemma4:31b`)

The user typically runs local inference via Ollama or vLLM. Never hardcode model names — always use config routing.

### 2. Safety limits are NOT LLM-controllable

`SafetyController` enforces:
- `max_dollars_per_run` — hard spend cap per experiment
- `instance_ttl_seconds` — maximum instance lifetime

These are enforced in Python orchestrator code, never delegated to the LLM. This is a fundamental design invariant. Do not add code paths that let LLM outputs bypass budget or TTL limits.

### 3. Everything async

All I/O-bound operations use `async/await`:
- LLM calls: `await client.complete()`
- Vast.ai API: `await vast_client.search_offers()`
- SSH/rsync: via `asyncio.create_subprocess_exec`
- Fleet execution: `await executor.run()`

The CLI bridge is `asyncio.run(_async_impl(...))`.

### 4. Experiment trees use SQLite

`search/tree.py` persists experiments to SQLite with `row_factory` for crash recovery. Use `--fresh` flag to clear stale state. The tree survives process crashes — this is intentional and important.

### 5. Docker-sandboxed by default, Vast.ai for GPU

Three execution backends share the `RunResult` interface:
- `SandboxRunner` — Docker containers (default, local)
- `LocalRunner` — subprocess on the host (`--local` flag)
- `VastRunner` — ephemeral Vast.ai GPU instances (`--vast` flag)
- `FleetExecutor` — parallel multi-arm experiments (fleet framework)

### 6. Declarative experiment specs (fleet framework)

The fleet framework eliminates custom scripts. Define experiments as YAML:

```yaml
experiment: my-ablation
hardware:
  gpu: RTX_4090
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
repo:
  url: git@github.com:user/repo.git
  branch: experiment/branch
data:
  source: s3-presigned
  urls_file: data-urls.txt
deps:
  pre_install: "pip install torch --index-url https://download.pytorch.org/whl/cu130"
  requirements: requirements.txt
arms:
  - name: baseline
    command: "python train.py --config baseline.yaml"
  - name: optimized
    command: "python train.py --config optimized.yaml"
metrics:
  protocol: json_line    # or "block" for marker-delimited output
budget:
  max_dollars: 10.0
```

Then run: `ratiocinator fleet run experiment.yaml`

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

## Configuration

Pydantic models in `config.py` with sensible defaults. Load order:
1. `.ratiocinator/config.json` (if exists in CWD)
2. Explicit `--config path.json` flag
3. Environment variables override specific fields:
   - `VAST_API_KEY` → `config.vast.api_key`
   - `HF_TOKEN` → `config.publish.hf_token`
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
4. **Torch version matters.** `torch.optim.Muon` requires PyTorch 2.7.0+. When a specific torch version is needed, `pip uninstall torch torchvision -y` first, then install from the correct index URL.
5. **Rate limits.** Stagger instance creation by ~5s per arm to avoid API rate limits.
6. **Presigned URLs expire.** S3 presigned URLs have a TTL (~12h default). Regenerate before launching long experiments.

## Observability

Sentry is integrated throughout:
- `observability.py` — idempotent `init_sentry()`, auto-discovers git SHA for release tags
- Spans on: LLM calls, Vast.ai API, sandbox runs, SSH/rsync, fleet execution
- Remote crash reporting: parses tracebacks from SSH stderr into Sentry exception frames
- `enable_logs=True` — stdlib `logging` auto-forwarded to Sentry Logs
- DSN is hardcoded for the project but overridable via `SENTRY_DSN` env var

## Test Patterns

- **All tests use pytest + pytest-asyncio** with `asyncio_mode = "auto"` (no manual `@pytest.mark.asyncio` needed)
- **Heavy mocking** — external services (Vast.ai, LiteLLM, Docker, SSH) are always mocked
- **`AsyncMock`** for async callables, `MagicMock` for sync
- **No real network calls** in the test suite
- **Test files mirror source structure**: `test_fleet_spec.py` tests `fleet/spec.py`, `test_vast_runner.py` tests `infra/vast_runner.py`, etc.

Example test structure:
```python
class TestExperimentSpecFromYAML:
    def test_minimal_spec(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump({...}))
        spec = ExperimentSpec.from_yaml(spec_file)
        assert spec.name == "test-experiment"

class TestResultStoreMerge:
    @pytest.fixture
    def store(self, tmp_path):
        return ResultStore(str(tmp_path / "results.json"))

    def test_success_overrides_failure(self, store):
        store.save("exp", "arm1", {..., "exit_code": 1})
        store.save("exp", "arm1", {..., "exit_code": 0})
        results = store.get_experiment("exp")
        assert results[0]["exit_code"] == 0
```

## Fleet Framework Deep Dive

This is the most important abstraction for running experiments. Here's how the pieces fit:

### ExperimentSpec (`fleet/spec.py`)
Pydantic model loaded from YAML. Contains: `HardwareSpec`, `DataSpec`, `RepoSpec`, `DepsSpec`, list of `ArmSpec`, `MetricsSpec`, `BudgetSpec`. Also provides `parse_metrics(stdout)` to extract metrics from training output.

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
3. Clone repo
4. Install dependencies (pre_install → requirements → verify)
5. Download data via DataProvisioner
6. Run training command
7. Parse metrics from stdout
8. Save results
9. Destroy instance (always, even on failure)

Arms run in parallel via `asyncio.gather()`. Instance creation is staggered by 5s.

## When You're Asked to Run an Experiment

Follow this workflow:

1. **Define the experiment as YAML** — create a spec in `examples/fleet/` using the `ExperimentSpec` schema
2. **Ensure the target repo has the right branch** with training scripts that output metrics in one of the supported protocols
3. **Stage data** — if data is on S3, generate presigned URLs and put them in a file
4. **Run**: `ratiocinator fleet run examples/fleet/my_experiment.yaml`
5. **Re-run failures**: `ratiocinator fleet run examples/fleet/my_experiment.yaml --arms 4,5,6`
6. **Check results**: `ratiocinator fleet status`

Do NOT write custom 800-line orchestrator scripts. The fleet framework exists to prevent that.

## Common Gotchas

| Problem | Solution |
|---------|----------|
| `torch.optim.Muon` not found | Need PyTorch 2.7.0+. Use `pre_install` in deps spec. |
| SSH timeout on Vast.ai | Instance may still be pulling Docker image. Boot takes 2-10 min. |
| `JSONDecodeError: Invalid control character` | Use `json.loads(text, strict=False)` for LLM output. |
| Local LLMs wrap sections in full LaTeX docs | `paper.py::_clean_section()` strips these. |
| Stale `search.db` from prior runs | Use `--fresh` flag or delete `.ratiocinator/search.db`. |
| httpx log spam at INFO | Set `logging.getLogger("httpx").setLevel(logging.WARNING)` |
| Data download timeout | Increase `download_timeout_s` in budget spec; check URL validity. |
| `python:3.11-slim` on Vast.ai | Use `pytorch/pytorch:*` images instead — they have sshd. |

## Conventions

- **Imports:** `from __future__ import annotations` at the top of every file
- **Type hints:** Use `str | None` not `Optional[str]`; use `list[str]` not `List[str]`
- **Pydantic for config/specs:** All structured data uses Pydantic `BaseModel`
- **Async by default:** All I/O operations are async. Sync wrappers use `asyncio.run()`
- **Error suffix:** Exception classes must end with `Error` (ruff N818)
- **Variable naming:** lowercase in functions (ruff N806), `CamelCase` for classes only
- **Sentry optional:** Always guard with `try: import sentry_sdk except ImportError: sentry_sdk = None`
- **Logging:** Use `logger = logging.getLogger(__name__)` — never `print()` for operational output
- **Click CLI:** Lazy imports inside command functions to keep startup fast
- **No secrets in code:** API keys come from env vars or config files, never hardcoded
