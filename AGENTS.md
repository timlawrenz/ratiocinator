# AGENTS.md — Ratiocinator

> Instructions for AI coding agents **developing** Ratiocinator itself.
>
> **Using Ratiocinator as a tool?** See the [AgentSkill](/.agents/skills/ratiocinator/SKILL.md) instead — it has the `init` → write spec → `fleet run` → `status` workflow for running experiments from any project.

## What This Project Is

Ratiocinator is an **autonomous research pipeline** that uses LLMs and tree search to run scientific experiments without human intervention. It proposes hypotheses, modifies code, provisions GPU instances, runs training, collects results, and writes papers.

The typical workflow is: **ideate → search → execute → synthesize → publish**.

This is a *meta-research tool* — it does not do ML training itself. It orchestrates training runs of *other* repos (e.g., a text-to-image model) by modifying their code and measuring outcomes.

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

## Further Reading

- **[docs/architecture.md](docs/architecture.md)** — Module map, design decisions, fleet framework deep dive, HF/Vast.ai integration internals, observability
- **[docs/huggingface-jobs.md](docs/huggingface-jobs.md)** — HuggingFace Jobs user guide (preemption, labels, debugging)
- **[README.md](README.md)** — Consumer-facing installation, quick start, CLI reference

## AgentSkill

Ratiocinator ships an [AgentSkills](https://agentskills.io) definition at `.agents/skills/ratiocinator/SKILL.md`. It provides the `name`/`description` frontmatter agents use for skill discovery, plus step-by-step instructions for the `init` → write spec → `fleet run` → `status` workflow. Common-field reference for `ExperimentSpec` and `ResearchSpec` YAML schemas is in `.agents/skills/ratiocinator/references/schemas.md` and loaded on demand via progressive disclosure.

**When to use the AgentSkill vs this file:**
- **AgentSkill** — You are an agent *using* Ratiocinator to run experiments in another project
- **This file (AGENTS.md)** — You are an agent *developing* Ratiocinator's source code
