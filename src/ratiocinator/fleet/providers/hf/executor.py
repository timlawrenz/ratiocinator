"""HuggingFace Jobs compute provider implementation.

Wraps the existing ``HFFleetExecutor`` to satisfy the ``ComputeProvider``
protocol.  Like the Vast provider, this overrides ``run()`` directly
because the legacy executor has its own internal orchestration (volume
building, job submission, heartbeat polling).
"""

from __future__ import annotations

from ratiocinator.fleet.provider import (
    ComputeProvider,
    ProviderCapability,
    ProviderMeta,
    register_provider,
)
from ratiocinator.fleet.providers.hf.config import HFProviderConfig
from ratiocinator.fleet.results import ArmResult
from ratiocinator.fleet.spec import ArmSpec, ExperimentSpec


@register_provider("hf")
class HFProvider(ComputeProvider):
    """Execute experiment arms as HuggingFace Jobs.

    Delegates to the existing ``HFFleetExecutor`` for all internal logic
    (wrapper script generation, job submission, polling, log retrieval).

    Note: Overrides ``run()`` to delegate to the legacy executor.
    ``_run_arm()`` is a placeholder — not called directly.
    """

    _META = ProviderMeta(
        name="hf",
        display_name="HuggingFace Jobs",
        capabilities=(
            ProviderCapability.VOLUME_MOUNTS
            | ProviderCapability.PREEMPTION_RECOVERY
            | ProviderCapability.COST_TRACKING
        ),
        requires_ssh_key=False,
        requires_api_token=True,
    )

    def __init__(self, config: HFProviderConfig) -> None:
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
        """Delegate to the legacy HFFleetExecutor."""
        from ratiocinator.fleet.hf_executor import HFFleetConfig, HFFleetExecutor

        hf_config = HFFleetConfig(
            token=self.config.token,
            namespace=self.config.namespace,
            bucket_prefix=self.config.bucket_prefix,
            max_timeout=self.config.max_timeout,
            results_path=self.config.results_path,
            log_dir=self.config.log_dir,
        )
        executor = HFFleetExecutor(spec, hf_config)
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
            "HFProvider uses legacy HFFleetExecutor dispatch; "
            "_run_arm is not called directly."
        )

    async def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        warnings: list[str] = []
        if not spec.hardware.hf_flavor and not self.config.flavor:
            warnings.append(
                "No hf_flavor set in spec or provider config. "
                "HuggingFace Jobs requires an explicit hardware flavor."
            )
        if spec.data.source in ("rsync", "local"):
            warnings.append(
                f"Data source '{spec.data.source}' requires SSH access; "
                "use 'hf-dataset' or 'hf-bucket' for HF Jobs."
            )
        return warnings

    async def estimate_cost(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
    ) -> float:
        from ratiocinator.infra.hf_client import HF_FLAVOR_PRICING

        flavor = spec.hardware.hf_flavor or self.config.flavor
        hourly = HF_FLAVOR_PRICING.get(flavor, 0)
        n_arms = len(arm_indices) if arm_indices else len(spec.arms)
        hours = spec.budget.train_timeout_s / 3600.0
        return n_arms * hourly * hours
