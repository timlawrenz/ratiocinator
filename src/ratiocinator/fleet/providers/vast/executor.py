"""Vast.ai compute provider implementation.

Wraps the existing ``FleetExecutor`` to satisfy the ``ComputeProvider``
protocol.  Since the legacy executor has its own internal orchestration
(bundling, offers, boot polling), this provider overrides ``run()``
entirely and delegates to it.  Future refactors may decompose the
executor into per-arm logic suitable for ``_run_arm()``.
"""

from __future__ import annotations

from ratiocinator.fleet.provider import (
    ComputeProvider,
    ProviderCapability,
    ProviderMeta,
    register_provider,
)
from ratiocinator.fleet.providers.vast.config import VastProviderConfig
from ratiocinator.fleet.results import ArmResult
from ratiocinator.fleet.spec import ArmSpec, ExperimentSpec


@register_provider("vast")
class VastProvider(ComputeProvider):
    """Execute experiment arms on ephemeral Vast.ai GPU instances.

    Delegates to the existing ``FleetExecutor`` for all internal logic
    (instance provisioning, SSH, clone, deps, training, cleanup).

    Note: Because the legacy executor manages its own arm dispatch loop
    (including multi-GPU bundling), this provider overrides ``run()``
    directly rather than using the base class template.  The ``_run_arm()``
    hook is implemented as a no-op placeholder — all real work goes
    through the ``FleetExecutor.run()`` path.
    """

    _META = ProviderMeta(
        name="vast",
        display_name="Vast.ai",
        capabilities=(
            ProviderCapability.SSH_ACCESS
            | ProviderCapability.REALTIME_LOGS
            | ProviderCapability.COST_TRACKING
            | ProviderCapability.MULTI_GPU
            | ProviderCapability.BUNDLED_ARMS
        ),
        requires_ssh_key=True,
        requires_api_token=True,
    )

    def __init__(self, config: VastProviderConfig) -> None:
        self.config = config

    @property
    def meta(self) -> ProviderMeta:
        return self._META

    async def run(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
        *,
        dry_run: bool = False,
        skip_duplicates: bool = False,
    ) -> list[ArmResult]:
        """Delegate to the legacy FleetExecutor."""
        from ratiocinator.fleet.executor import FleetConfig, FleetExecutor

        fleet_config = FleetConfig(
            api_key=self.config.api_key,
            ssh_key=self.config.ssh_key,
            max_concurrent=self.config.max_concurrent,
            stagger_seconds=self.config.stagger_seconds,
            results_path=self.config.results_path,
            log_dir=self.config.log_dir,
        )
        executor = FleetExecutor(spec, fleet_config)
        return await executor.run(
            arm_indices, dry_run=dry_run, skip_duplicates=skip_duplicates,
        )

    async def _run_arm(
        self,
        spec: ExperimentSpec,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        # Not used — run() delegates to legacy executor's own dispatch.
        raise NotImplementedError(
            "VastProvider uses legacy FleetExecutor dispatch; "
            "_run_arm is not called directly."
        )

    async def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        warnings: list[str] = []
        if not spec.repo.url:
            warnings.append("repo.url is required for Vast.ai (need git clone)")
        if spec.data.source in ("hf-dataset", "hf-bucket"):
            warnings.append(
                "HF data sources are not natively supported on Vast.ai; "
                "consider rsync or s3-presigned instead."
            )
        return warnings

    async def estimate_cost(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
    ) -> float:
        n_arms = len(arm_indices) if arm_indices else len(spec.arms)
        hourly = spec.hardware.max_dph
        hours = spec.budget.train_timeout_s / 3600.0
        return n_arms * hourly * hours
