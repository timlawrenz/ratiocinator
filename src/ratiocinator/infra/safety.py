"""Safety controls: budget enforcement, TTL management, orphan cleanup.

These controls are hard-coded and NOT LLM-controllable, following the
design document's requirement for circuit breakers outside LLM control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ratiocinator.config import SafetyConfig
from ratiocinator.infra.vast_client import VastClient

logger = logging.getLogger(__name__)


@dataclass
class TrackedInstance:
    """An instance being tracked for safety enforcement."""

    instance_id: int
    created_at: float = field(default_factory=time.time)
    estimated_dph: float = 0.0
    actual_cost: float | None = None
    """Final cost reported by the Vast.ai billing API (overrides estimate)."""


class SafetyController:
    """Enforces hard budget and TTL limits on Vast.ai instances.

    All limits are read from SafetyConfig at init and cannot be
    changed by the LLM or at runtime.
    """

    def __init__(self, config: SafetyConfig, client: VastClient) -> None:
        self._max_dollars = config.max_dollars_per_run
        self._ttl = config.instance_ttl_seconds
        self._client = client
        self._tracked: dict[int, TrackedInstance] = {}
        self._total_spent: float = 0.0

    def track(self, instance_id: int, dph: float = 0.0) -> None:
        """Register an instance for safety tracking."""
        self._tracked[instance_id] = TrackedInstance(
            instance_id=instance_id, estimated_dph=dph
        )

    def untrack(self, instance_id: int) -> None:
        """Remove an instance from tracking."""
        self._tracked.pop(instance_id, None)

    def set_actual_cost(self, instance_id: int, actual_cost: float) -> None:
        """Record the Vast.ai-billed cost for an instance.

        When set, ``estimate_spend`` will use this value in place of the
        ``dph * runtime`` estimation for that instance, and the cost is
        carried over to ``_total_spent`` if the instance is later
        untracked.  Pass an actual cost only when the billing API
        returned a value — never substitute the estimate.
        """
        cost = float(actual_cost)
        if cost < 0:
            logger.warning(
                "Ignoring negative actual_cost %.4f for instance %s",
                cost, instance_id,
            )
            cost = 0.0
        tracked = self._tracked.get(instance_id)
        if tracked is None:
            # Unknown instance — bank the cost so budget enforcement
            # still reflects the spend.
            self._total_spent += cost
            return
        tracked.actual_cost = cost

    def estimate_spend(self) -> float:
        """Estimate total spend based on tracked instance runtimes.

        For instances where the Vast.ai billing API has reported an
        ``actual_cost`` (via :meth:`set_actual_cost`), the actual figure
        is used in place of the ``dph * runtime`` estimate.
        """
        total = 0.0
        now = time.time()
        for t in self._tracked.values():
            if t.actual_cost is not None:
                total += t.actual_cost
                continue
            hours = (now - t.created_at) / 3600
            total += hours * t.estimated_dph
        return total + self._total_spent

    def can_launch(self, dph: float = 0.0) -> bool:
        """Check if launching another instance would exceed budget."""
        return self.estimate_spend() + dph < self._max_dollars

    def get_expired(self) -> list[int]:
        """Get instance IDs that have exceeded their TTL."""
        now = time.time()
        return [
            t.instance_id
            for t in self._tracked.values()
            if (now - t.created_at) > self._ttl
        ]

    async def enforce_ttl(self) -> list[int]:
        """Destroy instances that have exceeded their TTL.

        Returns list of destroyed instance IDs.
        """
        expired = self.get_expired()
        destroyed = []
        for instance_id in expired:
            try:
                await self._client.destroy_instance(instance_id)
                self.untrack(instance_id)
                destroyed.append(instance_id)
                logger.warning(
                    "TTL enforced: destroyed instance %s after %ds",
                    instance_id,
                    self._ttl,
                )
            except Exception:
                logger.exception("Failed to destroy expired instance %s", instance_id)
        return destroyed

    async def enforce_budget(self) -> list[int]:
        """If over budget, destroy all tracked instances.

        Returns list of destroyed instance IDs.
        """
        spend = self.estimate_spend()
        if spend < self._max_dollars:
            return []

        logger.error(
            "BUDGET EXCEEDED: $%.2f >= $%.2f — destroying all instances",
            spend,
            self._max_dollars,
        )
        destroyed = []
        for instance_id in list(self._tracked.keys()):
            try:
                await self._client.destroy_instance(instance_id)
                destroyed.append(instance_id)
            except Exception:
                logger.exception("Failed to destroy instance %s", instance_id)
        self._tracked.clear()
        return destroyed

    async def cleanup_all(self) -> list[int]:
        """Destroy all tracked instances (for shutdown)."""
        destroyed = []
        for instance_id in list(self._tracked.keys()):
            try:
                await self._client.destroy_instance(instance_id)
                destroyed.append(instance_id)
            except Exception:
                logger.exception("Failed to destroy instance %s", instance_id)
        self._tracked.clear()
        return destroyed

    @property
    def active_count(self) -> int:
        return len(self._tracked)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self._max_dollars - self.estimate_spend())
