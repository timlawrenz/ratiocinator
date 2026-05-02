"""Declarative experiment specification.

Defines the `ExperimentSpec` Pydantic model that fully describes a
parallel experiment — hardware requirements, data sources, dependency
installation, experiment arms, metric extraction, and budget limits.

An experiment spec can be loaded from YAML and passed to the
`FleetExecutor` for execution, eliminating the need for custom
orchestrator scripts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class HardwareSpec(BaseModel):
    """GPU instance requirements."""

    gpu: str = "RTX 4090"
    num_gpus: int = 1
    min_cpu_ram_gb: int = 64
    min_pcie_bw: float = 20.0
    max_dph: float = 0.50
    disk_gb: float = 200.0
    image: str = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
    # HuggingFace Jobs: explicit hardware flavor (e.g. "a100-large").
    # Required when using the HF provider; ignored for Vast.ai.
    hf_flavor: str = ""
    # Default batch size for all arms.  Injected as the BATCH_SIZE env var.
    # Per-arm overrides take precedence (see ArmSpec.batch_size).
    batch_size: int | None = None


class DataSpec(BaseModel):
    """Data provisioning configuration."""

    source: Literal[
        "s3-presigned", "rsync", "local", "none",
        "hf-dataset", "hf-bucket",
    ] = "none"
    # For s3-presigned: path to file with one presigned URL per line
    urls_file: str = ""
    # For rsync: remote host:path, port, max shards
    rsync_server: str = ""
    rsync_port: int = 22
    max_shards: int | None = None
    # For local: path on orchestrator machine to sync to the instance
    local_path: str = ""
    # Remote path where data is placed on the instance
    target: str = "/workspace/data"
    # For hf-dataset / hf-bucket: HF repo ID or bucket name
    hf_source: str = ""
    # Mount path inside the HF Jobs container
    hf_mount_path: str = "/data"


class RepoSpec(BaseModel):
    """Source code repository configuration."""

    url: str
    branch: str = "main"
    clone_depth: int = 1
    commit: str = ""  # Pin to a specific commit SHA after cloning
    # Where the repo is cloned on the remote instance
    remote_path: str = "/workspace/experiment"


class DepsSpec(BaseModel):
    """Dependency installation configuration."""

    # Commands run before pip install (e.g., install specific torch version)
    pre_install: list[str] = Field(default_factory=list)
    # Standard requirements file (relative to repo root)
    requirements: str = ""
    # Package names to exclude from requirements (already installed by pre_install)
    exclude_from_requirements: list[str] = Field(default_factory=list)
    # Verification command (must exit 0 to continue)
    verify: str = ""


class ArmSpec(BaseModel):
    """Single experiment arm definition."""

    name: str
    description: str = ""
    config: str = ""
    command: str
    env: dict[str, str] = Field(default_factory=dict)
    # Per-arm batch size override.  Takes precedence over
    # HardwareSpec.batch_size.  Injected as the BATCH_SIZE env var.
    batch_size: int | None = None


class PreflightSpec(BaseModel):
    """Optional pre-flight validation before the full training run."""

    command: str
    timeout_s: int = 60
    check_metrics: bool = False


class ValidationSpec(BaseModel):
    """Optional post-training validation step.

    Runs a separate command after training completes successfully.
    Validation metrics are merged into (and can override) the training
    metrics, with an optional ``prefix`` to namespace them.

    Typical usage: run a real parser or evaluator against model output
    to replace heuristic proxy metrics with ground-truth measurements.
    """

    command: str
    timeout_s: int = 120
    required_metrics: list[str] = Field(default_factory=list)
    prefix: str = ""


class MetricsSpec(BaseModel):
    """How to extract metrics from experiment output."""

    protocol: Literal["block", "json_line"] = "json_line"
    # For "block" protocol: markers surrounding key:value lines
    start_marker: str = "--- RESULTS ---"
    end_marker: str = "--- END RESULTS ---"
    # For "json_line" protocol: prefix before JSON payload
    json_prefix: str = "METRICS:"


class BudgetSpec(BaseModel):
    """Cost and time limits."""

    max_dollars: float = 10.0
    train_timeout_s: int = 1800
    download_timeout_s: int = 7200
    boot_timeout_s: int = 600
    instance_ttl_s: int = 3600


class ExperimentSpec(BaseModel):
    """Complete declarative experiment specification.

    Load from YAML::

        spec = ExperimentSpec.from_yaml("experiment.yaml")

    Or construct programmatically::

        spec = ExperimentSpec(
            name="throughput-ablation",
            hardware=HardwareSpec(gpu="RTX 4090"),
            repo=RepoSpec(url="https://github.com/user/repo.git"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
        )
    """

    name: str
    description: str = ""
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    repo: RepoSpec
    deps: DepsSpec = Field(default_factory=DepsSpec)
    arms: list[ArmSpec]
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    preflight: PreflightSpec | None = None
    validation: ValidationSpec | None = None
    # Infrastructure provider: "vast" (default) or "hf"
    provider: Literal["vast", "hf"] = "vast"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        """Load an experiment spec from a YAML file."""
        import yaml

        text = Path(path).read_text()
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file."""
        import yaml

        Path(path).write_text(
            yaml.dump(
                self.model_dump(mode="json"),
                default_flow_style=False,
                sort_keys=False,
            )
        )

    def get_arm(self, name: str) -> ArmSpec | None:
        """Look up an arm by name."""
        for arm in self.arms:
            if arm.name == name:
                return arm
        return None

    def get_arms_by_index(self, indices: list[int]) -> list[ArmSpec]:
        """Select arms by their positional index."""
        return [self.arms[i] for i in indices if i < len(self.arms)]

    def resolve_command(self, arm: ArmSpec) -> str:
        """Resolve placeholders in an arm's command string.

        Supported placeholders:
            {config} — arm.config
            {name}   — arm.name
            {repo}   — repo.remote_path
            {data}   — data.target
        """
        return arm.command.format(
            config=arm.config,
            name=arm.name,
            repo=self.repo.remote_path,
            data=self.data.target,
        )

    def resolve_batch_size(self, arm: ArmSpec) -> int | None:
        """Return the effective batch size for an arm.

        Per-arm ``batch_size`` takes precedence over the hardware-level
        default.  Returns ``None`` when neither is set.
        """
        if arm.batch_size is not None:
            return arm.batch_size
        return self.hardware.batch_size

    def resolve_arm_env(self, arm: ArmSpec) -> dict[str, str]:
        """Return the arm's env dict with BATCH_SIZE injected when configured.

        Resolution priority for BATCH_SIZE:
        1. Explicit ``arm.env["BATCH_SIZE"]`` (never overwritten)
        2. ``arm.batch_size`` (per-arm override)
        3. ``hardware.batch_size`` (spec-wide default)

        If none of the above is set, the env dict is returned without
        a BATCH_SIZE entry.
        """
        env = dict(arm.env) if arm.env else {}
        if "BATCH_SIZE" not in env:
            batch_size = self.resolve_batch_size(arm)
            if batch_size is not None:
                env["BATCH_SIZE"] = str(batch_size)
        return env


def parse_metrics_block(stdout: str, spec: MetricsSpec) -> dict[str, Any]:
    """Extract metrics from stdout using the block protocol.

    Expects output between start_marker and end_marker with key: value lines.
    """
    results: dict[str, Any] = {}
    in_block = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == spec.start_marker:
            in_block = True
            continue
        if stripped == spec.end_marker:
            break
        if in_block and ":" in stripped:
            key, val = stripped.split(":", 1)
            try:
                results[key.strip()] = float(val.strip())
            except ValueError:
                results[key.strip()] = val.strip()
    return results


def parse_metrics_json_line(stdout: str, spec: MetricsSpec) -> dict[str, Any]:
    """Extract the last METRICS:{json} line from stdout."""
    last_line = ""
    for line in stdout.splitlines():
        if line.startswith(spec.json_prefix):
            last_line = line

    if not last_line:
        return {}

    json_str = last_line[len(spec.json_prefix) :]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return {}


def parse_metrics(stdout: str, spec: MetricsSpec) -> dict[str, Any]:
    """Extract metrics from stdout according to the spec's protocol."""
    if spec.protocol == "block":
        return parse_metrics_block(stdout, spec)
    return parse_metrics_json_line(stdout, spec)


_NAME_SENTINEL = "<arm-name>"


def _canonical_resolved_command(
    arm: ArmSpec, spec: ExperimentSpec | None = None,
) -> str:
    """Resolve an arm's command into a canonical form for hashing.

    Substitutes ``{config}`` with ``arm.config`` and ``{name}`` with a
    fixed sentinel (so arms differing only in ``name`` still hash
    identically).  When ``spec`` is provided, ``{repo}`` and ``{data}``
    are also resolved; otherwise they are left as literal placeholders
    (they're constant across all arms within a single spec, so they
    can't introduce false duplicates between arms in the same run).
    Unknown placeholders cause a fallback to the raw command string.
    """
    repo = spec.repo.remote_path if spec is not None else "{repo}"
    data = spec.data.target if spec is not None else "{data}"
    try:
        return arm.command.format(
            config=arm.config,
            name=_NAME_SENTINEL,
            repo=repo,
            data=data,
        )
    except (KeyError, IndexError):
        return arm.command


def arm_config_hash(
    arm: ArmSpec, spec: ExperimentSpec | None = None,
) -> str:
    """Return a 12-char hash of an arm's training-relevant configuration.

    The hash is computed over a *canonicalized resolved command*
    (placeholders substituted via :func:`_canonical_resolved_command`),
    ``arm.env``, and the *effective* batch size (resolved via
    :meth:`ExperimentSpec.resolve_arm_env` when ``spec`` is provided).

    Two arms that resolve to the same effective command line — e.g. one
    hardcoding ``baseline.yaml`` and another using ``{config}`` with
    ``config='baseline.yaml'`` — hash identically.  Arms differing only
    in ``name`` or ``description`` likewise hash identically, which is
    the desired behaviour for duplicate detection.
    """
    if spec is not None:
        effective_env = spec.resolve_arm_env(arm)
    else:
        effective_env = dict(arm.env) if arm.env else {}
        if "BATCH_SIZE" not in effective_env and arm.batch_size is not None:
            effective_env["BATCH_SIZE"] = str(arm.batch_size)

    relevant = {
        "command": _canonical_resolved_command(arm, spec),
        "env": sorted(effective_env.items()),
    }
    blob = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def find_duplicate_arms(
    arms: list[ArmSpec], spec: ExperimentSpec | None = None,
) -> dict[str, list[str]]:
    """Group arm names by their config hash, returning only collision groups.

    Returns a mapping ``{hash: [arm_name, ...]}`` containing only those
    hashes shared by two or more arms.  An empty dict means all arms have
    unique configurations.  When ``spec`` is provided, placeholders such
    as ``{repo}`` and ``{data}`` are resolved before hashing.
    """
    by_hash: dict[str, list[str]] = {}
    for arm in arms:
        by_hash.setdefault(arm_config_hash(arm, spec), []).append(arm.name)
    return {h: names for h, names in by_hash.items() if len(names) > 1}


def deduplicate_arm_pairs(
    arm_pairs: list[tuple[int, ArmSpec]],
    *,
    skip_duplicates: bool,
    logger: Any | None = None,
    spec: ExperimentSpec | None = None,
) -> list[tuple[int, ArmSpec]]:
    """Detect duplicate arm configs in a list of ``(index, arm)`` pairs.

    Always emits a warning (via ``logger``, if provided) listing the
    duplicate groups.  If ``skip_duplicates`` is True, only the first
    occurrence of each unique config hash is retained; otherwise the
    pairs are returned unchanged (manual-mode behaviour — user may
    intentionally want replication).
    """
    import logging as _logging

    log = logger or _logging.getLogger(__name__)

    if not arm_pairs:
        return arm_pairs

    duplicates = find_duplicate_arms([arm for _, arm in arm_pairs], spec)
    if not duplicates:
        return arm_pairs

    for cfg_hash, names in duplicates.items():
        log.warning(
            "Duplicate arm configs detected (hash=%s): %s",
            cfg_hash, ", ".join(names),
        )

    if not skip_duplicates:
        return arm_pairs

    seen_hashes: set[str] = set()
    unique_pairs: list[tuple[int, ArmSpec]] = []
    skipped: list[str] = []
    for idx, arm in arm_pairs:
        h = arm_config_hash(arm, spec)
        if h in seen_hashes:
            skipped.append(arm.name)
            continue
        seen_hashes.add(h)
        unique_pairs.append((idx, arm))

    if skipped:
        log.warning(
            "Skipping duplicate arms (skip_duplicates=True): %s",
            ", ".join(skipped),
        )
    return unique_pairs
