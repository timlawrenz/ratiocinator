"""Compute provider abstraction for fleet execution.

Defines ``ComputeProvider`` — an abstract base class (skeleton) that all
compute backends inherit from.  Think of it like an abstract class in Ruby:
the base class handles orchestration that is identical across providers
(arm selection, deduplication, parallel dispatch, result storage, logging),
and subclasses fill in the provider-specific hooks:

    - ``_run_arm()``       — execute a single arm (the only required hook)
    - ``_print_dry_run()`` — show what would happen without executing
    - ``validate_spec()``  — check spec compatibility with this provider
    - ``estimate_cost()``  — pre-run cost estimation

The research layer (coordinator, tree search, CLI) interacts *only* with
the public ``run()`` method — it never calls provider-specific hooks.

Contract:
    IN:  ExperimentSpec + ProviderConfig (via __init__)
    OUT: list[ArmResult]
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ratiocinator.fleet.results import ArmResult, ResultStore
from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    deduplicate_arm_pairs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider capabilities
# ---------------------------------------------------------------------------


class ProviderCapability(enum.Flag):
    """Capabilities a provider may advertise.

    Used by the orchestrator to adapt behavior (e.g. skip heartbeat
    polling for providers without REALTIME_LOGS, or skip SSH-based
    validation for providers without SSH_ACCESS).
    """

    NONE = 0
    SSH_ACCESS = enum.auto()
    VOLUME_MOUNTS = enum.auto()
    REALTIME_LOGS = enum.auto()
    PREEMPTION_RECOVERY = enum.auto()
    COST_TRACKING = enum.auto()
    MULTI_GPU = enum.auto()
    BUNDLED_ARMS = enum.auto()


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderMeta:
    """Static metadata about a compute provider."""

    name: str
    """Registry key (e.g. ``"vast"``, ``"hf"``, ``"docker"``)."""

    display_name: str
    """Human-readable name (e.g. ``"Vast.ai"``, ``"HuggingFace Jobs"``)."""

    capabilities: ProviderCapability
    """Bitfield of supported capabilities."""

    requires_ssh_key: bool = False
    requires_api_token: bool = True


# ---------------------------------------------------------------------------
# Provider config base
# ---------------------------------------------------------------------------


class ProviderConfig(ABC):
    """Base class for provider-specific runtime configuration.

    Each provider defines its own Pydantic model that also inherits from
    this ABC.  The only requirement is implementing ``validate()``.

    Example::

        class MyProviderConfig(BaseModel, ProviderConfig):
            api_key: str
            region: str = "us-east-1"

            def validate(self) -> None:
                if not self.api_key:
                    raise ValueError("api_key is required")
    """

    @abstractmethod
    def validate(self) -> None:
        """Raise ``ValueError`` if required fields are missing or invalid."""
        ...


# ---------------------------------------------------------------------------
# ComputeProvider — the abstract base class (skeleton)
# ---------------------------------------------------------------------------


class ComputeProvider(ABC):
    """Abstract base class for compute providers.

    This is a **template method** pattern: the base class ``run()``
    handles the orchestration (arm selection, deduplication, parallel
    dispatch, result recording, timing), and subclasses implement the
    provider-specific ``_run_arm()`` hook.

    Subclass checklist:
        1. Set ``_META`` class variable (a ``ProviderMeta`` instance)
        2. Accept a typed ``ProviderConfig`` in ``__init__``
        3. Implement ``_run_arm()`` — execute one arm, return ``ArmResult``
        4. Optionally override ``_print_dry_run()``, ``validate_spec()``,
           ``estimate_cost()``

    Example::

        @register_provider("my-cloud")
        class MyCloudProvider(ComputeProvider):
            _META = ProviderMeta(
                name="my-cloud",
                display_name="My Cloud",
                capabilities=ProviderCapability.COST_TRACKING,
            )

            def __init__(self, config: MyCloudConfig) -> None:
                self.config = config

            @property
            def meta(self) -> ProviderMeta:
                return self._META

            async def _run_arm(self, spec, arm_idx, arm, launch_order):
                # ... provision, run, collect metrics, cleanup ...
                return result
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def meta(self) -> ProviderMeta:
        """Static metadata about this provider."""
        ...

    @abstractmethod
    async def _run_arm(
        self,
        spec: ExperimentSpec,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        """Execute a single experiment arm and return its result.

        This is the core hook that each provider implements.  The base
        class calls it once per arm (in parallel via asyncio.gather).

        Args:
            spec: The full experiment specification.
            arm_idx: Index of this arm in the spec.
            arm: The arm to execute.
            launch_order: 0-based position in the launch sequence
                (use for staggering to avoid rate limits).

        Returns:
            Populated ``ArmResult`` — set ``exit_code``, ``metrics``,
            ``duration_seconds``, ``error``, etc.

        Notes:
            - Do NOT raise for expected failures — set result.error instead.
            - Always clean up resources in a ``try/finally``.
            - Respect ``spec.budget.train_timeout_s``.
            - Use ``spec.resolve_command(arm)`` for the training command.
            - Use ``spec.resolve_arm_env(arm)`` for environment variables.
            - Use ``parse_metrics(stdout, spec.metrics)`` for metric extraction.
        """
        ...

    # ------------------------------------------------------------------
    # Template method — shared orchestration logic
    # ------------------------------------------------------------------

    async def run(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
        *,
        dry_run: bool = False,
        skip_duplicates: bool = False,
    ) -> list[ArmResult]:
        """Execute experiment arms and return results.

        This is the public entry point called by the research layer.
        It handles:
            1. Arm selection (all or subset by index)
            2. Deduplication (warn or skip identical configs)
            3. Dry-run display
            4. Parallel dispatch via ``_run_arm()``
            5. Result recording to ``ResultStore``
            6. Duration/cost logging

        Subclasses should NOT override this method — override
        ``_run_arm()`` instead.
        """
        # 1. Select arms
        arms = (
            spec.get_arms_by_index(arm_indices)
            if arm_indices is not None
            else spec.arms
        )
        arm_pairs: list[tuple[int, ArmSpec]] = [
            (arm_indices[i] if arm_indices else i, arm)
            for i, arm in enumerate(arms)
        ]

        # 2. Deduplicate
        arm_pairs = deduplicate_arm_pairs(
            arm_pairs,
            skip_duplicates=skip_duplicates,
            logger=logger,
            spec=spec,
        )
        if not arm_pairs:
            logger.warning("No arms to execute after deduplication")
            return []

        # 3. Dry run
        if dry_run:
            self._print_dry_run(spec, arm_pairs)
            return []

        # 4. Dispatch all arms in parallel
        run_start = time.monotonic()
        tasks = [
            self._run_arm(spec, arm_idx, arm, j)
            for j, (arm_idx, arm) in enumerate(arm_pairs)
        ]
        results: list[ArmResult] = list(await asyncio.gather(*tasks))

        # 5. Record results
        results_path = self._get_results_path()
        if results_path:
            store = ResultStore(results_path)
            store.record_many(results)

        # 6. Log summary
        total_duration = time.monotonic() - run_start
        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "[%s] Fleet run complete: %d/%d succeeded in %.1fs",
            self.meta.name, succeeded, len(results), total_duration,
        )

        return results

    # ------------------------------------------------------------------
    # Overridable hooks — subclasses MAY override these
    # ------------------------------------------------------------------

    def _print_dry_run(
        self,
        spec: ExperimentSpec,
        arm_pairs: list[tuple[int, ArmSpec]],
    ) -> None:
        """Print what would be executed without launching.

        Override for provider-specific dry-run output (e.g. show offers,
        pricing, flavors).  Default prints a generic summary.
        """
        print(f"\n{'='*60}")
        print(f"DRY RUN: {spec.name} on {self.meta.display_name}")
        print(f"{'='*60}")
        print(f"  Provider: {self.meta.display_name}")
        print(f"  Image:    {spec.hardware.image}")
        print(f"  Timeout:  {spec.budget.train_timeout_s}s")
        print(f"  Arms ({len(arm_pairs)}):")
        for idx, arm in arm_pairs:
            print(f"    [{idx}] {arm.name}: {arm.command[:80]}")
        print(f"{'='*60}\n")

    def _get_results_path(self) -> str:
        """Return path to the results JSON file.

        Override if your config stores this differently.  Return empty
        string to skip result persistence.
        """
        # Convention: most configs have a `results_path` attribute
        config = getattr(self, "config", None)
        if config and hasattr(config, "results_path"):
            return config.results_path
        return ""

    async def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        """Return warnings/errors for the given spec on this provider.

        Override to check provider-specific requirements (e.g.
        ``hf_flavor`` must be set, data source must be ``hf-bucket``).
        Default accepts all specs.
        """
        return []

    async def estimate_cost(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
    ) -> float:
        """Estimate total cost in USD.  Returns ``0.0`` if unknown.

        Override with provider-specific pricing logic.
        """
        return 0.0


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[ComputeProvider]] = {}


def register_provider(name: str):
    """Class decorator that registers a ``ComputeProvider`` under ``name``.

    Usage::

        @register_provider("vast")
        class VastProvider(ComputeProvider):
            ...

    The provider then becomes available via ``get_provider_class("vast")``.
    """

    def decorator(cls: type[ComputeProvider]) -> type[ComputeProvider]:
        if name in _PROVIDERS:
            logger.warning(
                "Provider %r already registered (%s); overwriting with %s",
                name, _PROVIDERS[name].__name__, cls.__name__,
            )
        _PROVIDERS[name] = cls
        return cls

    return decorator


def get_provider_class(name: str) -> type[ComputeProvider]:
    """Look up a registered provider class by its string key.

    Raises:
        ValueError: If no provider is registered under ``name``.
    """
    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys())) or "(none)"
        raise ValueError(
            f"Unknown compute provider '{name}'. Available: {available}"
        )
    return _PROVIDERS[name]


def list_providers() -> dict[str, ProviderMeta]:
    """Return ``{name: ProviderMeta}`` for all registered providers."""
    import contextlib

    result: dict[str, ProviderMeta] = {}
    for name, cls in _PROVIDERS.items():
        with contextlib.suppress(AttributeError):
            result[name] = cls._META  # type: ignore[attr-defined]
    return result


def available_provider_names() -> list[str]:
    """Return sorted list of registered provider keys."""
    return sorted(_PROVIDERS.keys())
