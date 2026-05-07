# Creating a New Compute Provider

This guide explains how to add a new compute backend to Ratiocinator. Each provider lives in its own directory and inherits from the `ComputeProvider` abstract base class, which handles all shared orchestration automatically.

## How It Works — The Template Method Pattern

`ComputeProvider` is an **abstract base class** (Python's equivalent of a Ruby abstract class). It provides a concrete `run()` method that handles all the orchestration:

```
run() [inherited — you do NOT override this]
  ├── Select arms (all or by index)
  ├── Deduplicate (skip identical configs)
  ├── If dry_run → call _print_dry_run() [overridable]
  ├── For each arm, in parallel:
  │     └── call _run_arm()  ← YOU IMPLEMENT THIS
  ├── Record results to ResultStore
  └── Log summary
```

**You only implement `_run_arm()`** — one arm in, one result out. The base class handles everything else.

```python
# What you inherit for free (never rewrite these):
# - Arm selection by index
# - Config hash deduplication
# - Parallel asyncio.gather dispatch
# - Result persistence to JSON
# - Duration logging

# What you implement:
async def _run_arm(self, spec, arm_idx, arm, launch_order) -> ArmResult:
    # Provision → setup → train → collect metrics → cleanup
    ...
```

## Directory Structure

```
src/ratiocinator/fleet/providers/
├── __init__.py            # Auto-imports all providers
├── INTERFACE.md           # ← You are reading this
├── vast/                  # Reference: SSH-based cloud GPU
│   ├── __init__.py
│   ├── config.py
│   └── executor.py
├── hf/                    # Reference: API-based container jobs
│   ├── __init__.py
│   ├── config.py
│   └── executor.py
├── docker/                # Reference: Local Docker (simplest example)
│   ├── __init__.py
│   ├── config.py
│   └── executor.py
└── your_provider/         # ← Your new provider
    ├── __init__.py
    ├── config.py
    └── executor.py
```

---

## Step 1: Create Your Directory

```bash
mkdir -p src/ratiocinator/fleet/providers/my_provider
```

---

## Step 2: Define Your Config (`config.py`)

Inherit from both `pydantic.BaseModel` (for validation/serialization) and `ProviderConfig` (for the interface contract).

```python
"""My provider configuration."""

from __future__ import annotations

from pydantic import BaseModel

from ratiocinator.fleet.provider import ProviderConfig


class MyProviderConfig(BaseModel, ProviderConfig):
    """Runtime configuration for My Provider.

    Document every field — this is the public API for users.
    """

    api_key: str
    region: str = "us-east-1"
    instance_type: str = "gpu.large"
    max_concurrent: int = 5
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"

    def validate(self) -> None:
        """Required by ProviderConfig. Raise ValueError if invalid."""
        if not self.api_key:
            raise ValueError("MyProviderConfig requires api_key")
```

---

## Step 3: Implement the Provider (`executor.py`)

Inherit from `ComputeProvider` and implement `_run_arm()`.

```python
"""My provider implementation."""

from __future__ import annotations

import time

from ratiocinator.fleet.provider import (
    ComputeProvider,
    ProviderCapability,
    ProviderMeta,
    register_provider,
)
from ratiocinator.fleet.providers.my_provider.config import MyProviderConfig
from ratiocinator.fleet.results import ArmResult
from ratiocinator.fleet.spec import ArmSpec, ExperimentSpec, parse_metrics


@register_provider("my-provider")
class MyProvider(ComputeProvider):
    """Execute experiment arms on My Cloud Platform.

    Inherits: run(), _print_dry_run() (default), result storage
    Implements: _run_arm(), validate_spec(), estimate_cost()
    """

    # Class-level metadata — describes this provider's capabilities
    _META = ProviderMeta(
        name="my-provider",
        display_name="My Cloud Platform",
        capabilities=(
            ProviderCapability.COST_TRACKING
            | ProviderCapability.MULTI_GPU
        ),
        requires_ssh_key=False,
        requires_api_token=True,
    )

    def __init__(self, config: MyProviderConfig) -> None:
        self.config = config

    @property
    def meta(self) -> ProviderMeta:
        return self._META

    # ------------------------------------------------------------------
    # REQUIRED: The one method you must implement
    # ------------------------------------------------------------------

    async def _run_arm(
        self,
        spec: ExperimentSpec,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        """Execute one arm. Called in parallel by the inherited run()."""
        result = ArmResult(
            experiment=spec.name,
            arm_name=arm.name,
            description=arm.description,
        )
        arm_start = time.monotonic()

        try:
            # 1. Provision compute
            instance = await self._provision(spec, arm, launch_order)

            # 2. Setup environment (clone repo, install deps)
            await self._setup(instance, spec)

            # 3. Run training command
            command = spec.resolve_command(arm)
            env = spec.resolve_arm_env(arm)
            stdout, exit_code = await self._execute(instance, command, env, spec.budget.train_timeout_s)

            # 4. Collect metrics
            result.exit_code = exit_code
            result.metrics = parse_metrics(stdout, spec.metrics)
            if exit_code != 0:
                result.error = f"Training failed (exit {exit_code})"

        except Exception as exc:
            result.exit_code = -1
            result.error = str(exc)
        finally:
            # 5. ALWAYS clean up
            result.duration_seconds = time.monotonic() - arm_start
            await self._cleanup(instance)

        return result

    # ------------------------------------------------------------------
    # OPTIONAL: Override for provider-specific behavior
    # ------------------------------------------------------------------

    async def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        """Warn about incompatible spec settings."""
        warnings = []
        if spec.data.source == "rsync":
            warnings.append("My provider doesn't support rsync")
        return warnings

    async def estimate_cost(
        self, spec: ExperimentSpec, arm_indices: list[int] | None = None,
    ) -> float:
        """Estimated cost in USD."""
        n_arms = len(arm_indices) if arm_indices else len(spec.arms)
        return n_arms * 0.50 * (spec.budget.train_timeout_s / 3600.0)
```

---

## Step 4: Package Init (`__init__.py`)

```python
"""My provider — short description."""

from ratiocinator.fleet.providers.my_provider.config import MyProviderConfig
from ratiocinator.fleet.providers.my_provider.executor import MyProvider

__all__ = ["MyProvider", "MyProviderConfig"]
```

---

## Step 5: Register for Auto-Discovery

Add one line to `src/ratiocinator/fleet/providers/__init__.py`:

```python
from ratiocinator.fleet.providers import my_provider  # noqa: F401
```

Done. The `@register_provider("my-provider")` decorator handles registration.

---

## What You Inherit (Don't Reimplement)

| Method | What it does | Override? |
|--------|-------------|-----------|
| `run()` | Arm selection, dedup, parallel dispatch, result storage | **No** (unless wrapping a legacy executor) |
| `_print_dry_run()` | Generic dry-run output | Optional — override for custom formatting |
| `_get_results_path()` | Reads `self.config.results_path` | Optional — only if your config is nonstandard |
| `validate_spec()` | Returns `[]` (all specs valid) | Optional — add provider-specific checks |
| `estimate_cost()` | Returns `0.0` | Optional — add pricing logic |

## What You Must Implement

| Method | Purpose |
|--------|---------|
| `meta` (property) | Return your `_META` class variable |
| `_run_arm()` | Execute one arm: provision → setup → train → metrics → cleanup |

---

## The `_run_arm()` Contract

```python
async def _run_arm(
    self,
    spec: ExperimentSpec,   # Full experiment definition
    arm_idx: int,           # Position in spec.arms
    arm: ArmSpec,           # The specific arm to run
    launch_order: int,      # 0-based launch position (for staggering)
) -> ArmResult:
```

**Rules:**
1. Never raise for expected failures — set `result.error` and `result.exit_code` instead
2. Always clean up resources in a `try/finally`
3. Respect `spec.budget.train_timeout_s`
4. Use `spec.resolve_command(arm)` for the training command
5. Use `spec.resolve_arm_env(arm)` for environment variables (includes BATCH_SIZE)
6. Use `parse_metrics(stdout, spec.metrics)` to extract metrics from output
7. Handle `spec.preflight` (pre-training validation) if applicable
8. Handle `spec.validation` (post-training validation) if applicable

---

## ArmResult — What You Return

```python
@dataclass
class ArmResult:
    experiment: str           # Experiment name
    arm_name: str             # Arm identifier
    description: str          # Human-readable description
    metrics: dict             # Extracted metrics (the science output)
    exit_code: int            # 0=success, >0=train failure, <0=infra failure
    error: str                # Error message (empty on success)
    duration_seconds: float   # Wall-clock time for this arm
    instance_id: int | None   # Provider-specific instance/job ID
    gpu_info: str             # Hardware description string
    estimated_cost: float     # Cost in USD
```

### Exit Code Conventions

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1-127` | Training script failure (process exit code) |
| `-1` | Infrastructure error (boot, timeout, network) |
| `-2` | Cancelled |

---

## ProviderCapability Flags

Declare what your provider supports:

```python
class ProviderCapability(Flag):
    SSH_ACCESS           # Can run ad-hoc SSH commands
    VOLUME_MOUNTS        # Supports attaching data volumes
    REALTIME_LOGS        # Streams stdout during execution
    PREEMPTION_RECOVERY  # Handles spot instance restarts
    COST_TRACKING        # Reports actual spend
    MULTI_GPU            # Multi-GPU instances
    BUNDLED_ARMS         # Multiple arms on one instance
```

---

## Spec Fields Your Provider Should Use

| Field | Purpose |
|-------|---------|
| `spec.hardware.image` | Docker image to launch |
| `spec.hardware.gpu` | GPU type name |
| `spec.repo` | Git repo to clone |
| `spec.deps.pre_install` | Commands before pip install |
| `spec.deps.requirements` | Requirements file path |
| `spec.resolve_command(arm)` | Resolved training command |
| `spec.resolve_arm_env(arm)` | Full env dict (includes BATCH_SIZE) |
| `spec.budget.train_timeout_s` | Max training time |
| `spec.budget.boot_timeout_s` | Max provisioning time |
| `spec.metrics` | How to extract metrics from stdout |
| `spec.preflight` | Optional pre-training check |
| `spec.validation` | Optional post-training check |

---

## Testing Your Provider

```python
import pytest
from ratiocinator.fleet.provider import get_provider_class
from ratiocinator.fleet.providers.my_provider import MyProvider, MyProviderConfig


class TestRegistration:
    def test_registered(self):
        assert get_provider_class("my-provider") is MyProvider

    def test_meta(self):
        config = MyProviderConfig(api_key="test")
        provider = MyProvider(config)
        assert provider.meta.name == "my-provider"


class TestConfig:
    def test_validate_missing_key(self):
        config = MyProviderConfig(api_key="")
        with pytest.raises(ValueError):
            config.validate()


class TestRunArm:
    @pytest.mark.asyncio
    async def test_dry_run_returns_empty(self, tmp_path):
        config = MyProviderConfig(
            api_key="test",
            results_path=str(tmp_path / "r.json"),
        )
        provider = MyProvider(config)
        spec = ...  # Build a minimal ExperimentSpec
        results = await provider.run(spec, dry_run=True)
        assert results == []
```

---

## Checklist

- [ ] `config.py` — Pydantic model inheriting `ProviderConfig` with `validate()`
- [ ] `executor.py` — Class with `@register_provider("name")` decorator
- [ ] `executor.py` — `_META` class variable (ProviderMeta)
- [ ] `executor.py` — `meta` property returning `_META`
- [ ] `executor.py` — `_run_arm()` implemented
- [ ] `__init__.py` — Exports config + executor
- [ ] `providers/__init__.py` — Import added
- [ ] Tests — Registration, config validation, dry-run
- [ ] Timeout enforcement — Kill after `budget.train_timeout_s`
- [ ] Cleanup — Resources freed in `finally`
- [ ] Metrics — `parse_metrics()` called on stdout
