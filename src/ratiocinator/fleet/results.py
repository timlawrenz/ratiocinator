"""Persistent experiment result storage.

Provides `ResultStore` — an append-only store that persists experiment
results across runs, keyed by (experiment_name, arm_name).  Successful
results override failures on re-runs, enabling incremental completion
of partially-failed experiments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArmResult:
    """Result of a single experiment arm execution."""

    experiment: str
    arm_name: str
    description: str = ""
    instance_id: int | None = None
    gpu_info: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    exit_code: int = -1
    error: str = ""
    duration_seconds: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class ResultStore:
    """Persistent, append-only experiment result store.

    Results are stored in a JSON file, keyed by (experiment, arm_name).
    On re-runs, successful results override failures but not other successes
    (preserving the earliest successful run).

    Usage::

        store = ResultStore("results/experiments.json")
        store.record(result)
        store.record(result2)
        best = store.get_best("throughput-exp", "avg_iter_per_sec", maximize=True)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, dict]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing results from disk."""
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, list):
                # Legacy format: flat list → convert to nested dict
                for entry in raw:
                    exp = entry.get("experiment", "unknown")
                    arm = entry.get("arm_name", entry.get("arm", "unknown"))
                    self._data.setdefault(exp, {})[arm] = entry
            elif isinstance(raw, dict):
                self._data = raw
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not load results from %s — starting fresh", self.path)

    def _save(self) -> None:
        """Persist results to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, default=str) + "\n")

    def record(self, result: ArmResult) -> None:
        """Record an arm result, applying merge semantics.

        Rules:
        - Successful results always override failures.
        - If both prior and new are successful, keep the prior (first success wins).
        - Failures override other failures (latest failure is more relevant).
        """
        exp_data = self._data.setdefault(result.experiment, {})
        prior = exp_data.get(result.arm_name)

        if prior is not None:
            prior_success = prior.get("exit_code", -1) == 0
            if prior_success and not result.success:
                # Don't overwrite a success with a failure
                return
            if prior_success and result.success:
                # Keep the first success
                return

        exp_data[result.arm_name] = asdict(result)
        self._save()
        logger.info(
            "Recorded result: %s/%s (exit=%d)",
            result.experiment, result.arm_name, result.exit_code,
        )

    def record_many(self, results: list[ArmResult]) -> None:
        """Record multiple results in a single save."""
        for result in results:
            exp_data = self._data.setdefault(result.experiment, {})
            prior = exp_data.get(result.arm_name)
            if prior is not None:
                prior_success = prior.get("exit_code", -1) == 0
                if prior_success and not result.success:
                    continue
                if prior_success and result.success:
                    continue
            exp_data[result.arm_name] = asdict(result)
        self._save()

    def get_experiment(self, experiment: str) -> list[dict]:
        """Get all arm results for an experiment, sorted by arm name."""
        exp_data = self._data.get(experiment, {})
        return sorted(exp_data.values(), key=lambda x: x.get("arm_name", ""))

    def get_arm(self, experiment: str, arm_name: str) -> dict | None:
        """Get a specific arm result."""
        return self._data.get(experiment, {}).get(arm_name)

    def get_best(
        self,
        experiment: str,
        metric_key: str,
        *,
        maximize: bool = False,
    ) -> dict | None:
        """Find the arm with the best value for a specific metric.

        Only considers successful arms with the metric present.
        """
        results = self.get_experiment(experiment)
        candidates = []
        for r in results:
            if r.get("exit_code") != 0:
                continue
            metrics = r.get("metrics", {})
            if metric_key in metrics:
                candidates.append((metrics[metric_key], r))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=maximize)
        return candidates[0][1]

    def compare_arms(self, experiment: str) -> list[dict]:
        """Return all arms with their success status and metrics for comparison."""
        results = self.get_experiment(experiment)
        comparison = []
        for r in results:
            comparison.append({
                "arm": r.get("arm_name", ""),
                "description": r.get("description", ""),
                "success": r.get("exit_code") == 0,
                "metrics": r.get("metrics", {}),
                "error": r.get("error", ""),
            })
        return comparison

    def export_table(self, experiment: str, metric_keys: list[str]) -> str:
        """Export a markdown comparison table."""
        results = self.get_experiment(experiment)
        headers = ["Arm", "Status", *metric_keys]
        rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]

        for r in results:
            status = "✓" if r.get("exit_code") == 0 else "✗"
            metrics = r.get("metrics", {})
            values = [
                r.get("arm_name", ""),
                status,
                *[str(metrics.get(k, "—")) for k in metric_keys],
            ]
            rows.append(" | ".join(values))

        return "\n".join(rows)

    @property
    def experiments(self) -> list[str]:
        """List all experiment names."""
        return sorted(self._data.keys())
