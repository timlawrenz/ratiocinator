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
    config_hash: str = ""
    # Cost tracking
    instance_dph: float = 0.0
    """Hourly rate for the instance (dollars per hour)."""
    boot_time_s: float = 0.0
    """Wall-clock seconds spent provisioning the instance and waiting for SSH."""
    estimated_cost: float = 0.0
    """Cost estimated as ``instance_dph * arm_wall_clock / 3600``.

    The wall-clock here is end-to-end (provision + boot + clone + deps +
    train + validate + cleanup), not the training-only
    ``duration_seconds``.  Use ``actual_cost`` when available for ground
    truth.
    """
    actual_cost: float | None = None
    """Cost reported by the Vast.ai billing API; ``None`` if unavailable."""

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

        store = ResultStore(".ratiocinator/results/experiments.json")
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

    def diff_results(
        self,
        experiment: str,
        *,
        metric_keys: list[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-9,
    ) -> dict[str, list[dict]]:
        """Compare arm results to find suspicious patterns.

        Returns a dict with two keys:

        - ``identical_config_diff_metrics`` — pairs of arms that share a
          ``config_hash`` but produced metrics that disagree by more than
          ``rtol`` / ``atol`` (relative + absolute tolerance, isclose
          semantics).  This indicates environment variance (different
          GPUs, nondeterminism, etc).
        - ``different_config_same_metrics`` — pairs of arms with distinct
          ``config_hash`` but metrics that agree within ``rtol`` /
          ``atol``.  This indicates the configuration knob being varied
          did not affect the outcome.

        Numeric comparison uses ``abs(a - b) <= atol + rtol * max(|a|,
        |b|)`` (matching :func:`math.isclose`).  ``atol`` keeps tiny
        absolute differences from being flagged as huge relative
        differences when both values are near zero.

        Only successful arms with non-empty metrics are considered.  If
        ``metric_keys`` is omitted, the intersection of metric names across
        both arms in a candidate pair is used.

        Pairs where either arm is missing a ``config_hash`` (e.g. because
        the result predates the field) are skipped; a single warning is
        logged listing the affected arms so users can re-run them to
        backfill hashes.
        """
        results = [
            r for r in self.get_experiment(experiment)
            if r.get("exit_code") == 0 and r.get("metrics")
        ]

        identical_cfg: list[dict] = []
        different_cfg: list[dict] = []
        missing_hash: set[str] = set()

        for i, a in enumerate(results):
            for b in results[i + 1 :]:
                a_hash = a.get("config_hash") or ""
                b_hash = b.get("config_hash") or ""
                if not a_hash:
                    missing_hash.add(a.get("arm_name", ""))
                if not b_hash:
                    missing_hash.add(b.get("arm_name", ""))
                if not a_hash or not b_hash:
                    continue

                a_metrics = a.get("metrics", {})
                b_metrics = b.get("metrics", {})
                keys = metric_keys or sorted(
                    set(a_metrics.keys()) & set(b_metrics.keys())
                )
                if not keys:
                    continue

                # Compare each shared metric
                diffs: dict[str, tuple[Any, Any]] = {}
                agrees: dict[str, tuple[Any, Any]] = {}
                for k in keys:
                    if k not in a_metrics or k not in b_metrics:
                        continue
                    av, bv = a_metrics[k], b_metrics[k]
                    try:
                        af, bf = float(av), float(bv)
                    except (TypeError, ValueError):
                        # Non-numeric — exact compare
                        if av == bv:
                            agrees[k] = (av, bv)
                        else:
                            diffs[k] = (av, bv)
                        continue
                    # math.isclose-style tolerance
                    if abs(af - bf) <= atol + rtol * max(abs(af), abs(bf)):
                        agrees[k] = (af, bf)
                    else:
                        diffs[k] = (af, bf)

                pair = {
                    "arm_a": a.get("arm_name", ""),
                    "arm_b": b.get("arm_name", ""),
                    "hash_a": a_hash,
                    "hash_b": b_hash,
                    "diffs": diffs,
                    "agrees": agrees,
                }

                if a_hash == b_hash and diffs:
                    identical_cfg.append(pair)
                elif a_hash != b_hash and agrees and not diffs:
                    different_cfg.append(pair)

        if missing_hash:
            logger.warning(
                "diff_results: skipped arms with missing config_hash: %s "
                "(re-run them to backfill)",
                ", ".join(sorted(n for n in missing_hash if n)),
            )

        return {
            "identical_config_diff_metrics": identical_cfg,
            "different_config_same_metrics": different_cfg,
        }
