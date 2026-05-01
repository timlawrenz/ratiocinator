"""HuggingFace Jobs fleet executor: parallel experiment orchestration.

Mirrors ``FleetExecutor`` but uses the HuggingFace Jobs API instead
of SSH-based Vast.ai instances.  Each arm runs as an independent HF Job
with a generated wrapper script.

Key differences from Vast.ai executor:
- No SSH access — all setup baked into a single wrapper script
- Data comes via volume mounts (datasets, buckets), not rsync/SCP
- Logs fetched via Jobs API, not captured from SSH stdout
- Automatic cleanup — HF manages the container lifecycle
- Hardware specified by flavor string, not marketplace search
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from dataclasses import dataclass
from typing import Any

from ratiocinator.fleet.hf_data import (
    build_data_volume,
    build_output_volume,
    build_script_volume,
    upload_arm_script,
    volumes_from_dicts,
)
from ratiocinator.fleet.results import ArmResult, ResultStore
from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    arm_config_hash,
    deduplicate_arm_pairs,
    parse_metrics,
)
from ratiocinator.infra.hf_client import (
    HF_FLAVOR_PRICING,
    HFClient,
    HFClientError,
    HFJobStage,
)
from ratiocinator.observability import fleet_breadcrumb, fleet_metric

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

JOB_POLL_INTERVAL_S = 60
JOB_STAGGER_S = 2


@dataclass
class HFFleetConfig:
    """Runtime configuration for HF fleet execution."""

    token: str
    namespace: str = ""
    bucket_prefix: str = ""
    max_timeout: str = "4h"
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"


class HFFleetExecutor:
    """Execute experiment arms in parallel on HuggingFace Jobs.

    Usage::

        spec = ExperimentSpec.from_yaml("experiment.yaml")
        config = HFFleetConfig(token="hf_...")
        executor = HFFleetExecutor(spec, config)
        results = await executor.run()
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        config: HFFleetConfig,
        *,
        result_store: ResultStore | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self.store = result_store or ResultStore(config.results_path)
        self._output_bucket = self._resolve_output_bucket()

    def _resolve_output_bucket(self) -> str:
        """Determine the output bucket name for this experiment."""
        prefix = self.config.bucket_prefix or self.config.namespace
        if not prefix:
            raise ValueError(
                "HFFleetConfig requires either bucket_prefix or namespace "
                "to create output buckets."
            )
        return f"{prefix}/ratiocinator-{self.spec.name}"

    async def run(
        self,
        arm_indices: list[int] | None = None,
        *,
        dry_run: bool = False,
        skip_duplicates: bool = False,
    ) -> list[ArmResult]:
        """Execute selected (or all) experiment arms in parallel.

        Args:
            arm_indices: Indices of arms to run.  None means all arms.
            dry_run: Print plan without launching jobs.
            skip_duplicates: If True, drop arms whose effective config
                hashes to one already scheduled (autonomous mode).
                Otherwise, warn but proceed (manual mode).

        Returns:
            List of ArmResult for each arm executed.
        """
        hw = self.spec.hardware
        if not hw.hf_flavor:
            raise ValueError(
                "HardwareSpec.hf_flavor is required when using the HF provider. "
                "Set it explicitly in your spec YAML (e.g. hf_flavor: a100-large)."
            )

        arms = (
            self.spec.get_arms_by_index(arm_indices)
            if arm_indices is not None
            else self.spec.arms
        )
        arm_pairs = [
            (arm_indices[i] if arm_indices else i, arm)
            for i, arm in enumerate(arms)
        ]

        arm_pairs = self._apply_dedup(arm_pairs, skip_duplicates=skip_duplicates)
        if not arm_pairs:
            logger.warning("No arms to execute after deduplication")
            return []

        if dry_run:
            self._print_dry_run(arm_pairs)
            return []

        async with HFClient(self.config.token) as client:
            # Ensure output bucket exists
            fleet_breadcrumb(
                f"Creating output bucket {self._output_bucket}",
                category="fleet.hf.bucket",
            )
            await client.create_bucket(self._output_bucket, private=True)

            # Upload per-arm wrapper scripts
            for _, arm in arm_pairs:
                script = self._build_wrapper_script(arm)
                await upload_arm_script(
                    client, self._output_bucket,
                    arm.name, script, self.spec.name,
                )

            # Launch all arms in parallel
            tasks = [
                self._run_arm(client, arm_idx, arm, j)
                for j, (arm_idx, arm) in enumerate(arm_pairs)
            ]
            results = await asyncio.gather(*tasks)

            self.store.record_many(list(results))
            return list(results)

    def _apply_dedup(
        self,
        arm_pairs: list[tuple[int, ArmSpec]],
        *,
        skip_duplicates: bool,
    ) -> list[tuple[int, ArmSpec]]:
        """Detect duplicate arm configs and either warn or drop them."""
        return deduplicate_arm_pairs(
            arm_pairs, skip_duplicates=skip_duplicates, logger=logger,
        )

    async def _run_arm(
        self,
        client: HFClient,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        """Run a single experiment arm as an HF Job."""
        result = ArmResult(
            experiment=self.spec.name,
            arm_name=arm.name,
            description=arm.description,
            config_hash=arm_config_hash(arm),
        )
        arm_start = time.monotonic()
        arm_tags = {"experiment": self.spec.name, "arm": arm.name}

        def _scope_ctx():
            if sentry_sdk:
                return sentry_sdk.new_scope()
            from contextlib import nullcontext
            return nullcontext()

        def _span(op: str, desc: str):
            if sentry_sdk:
                return sentry_sdk.start_span(op=op, name=desc)
            from contextlib import nullcontext
            return nullcontext()

        with _scope_ctx() as scope:
            if scope is not None:
                scope.set_tag("arm_name", arm.name)
                scope.set_tag("experiment_name", self.spec.name)
                scope.set_tag("provider", "hf")

            try:
                # Stagger to avoid rate limits
                if launch_order > 0:
                    await asyncio.sleep(launch_order * JOB_STAGGER_S)

                # --- Build volumes ---
                volumes_dicts = self._build_volumes(arm)

                # --- Submit job ---
                fleet_breadcrumb(
                    f"Submitting HF Job for arm {arm.name}",
                    category="fleet.hf.submit",
                    data={"arm": arm.name, "flavor": self.spec.hardware.hf_flavor},
                )
                with _span("hf.job.submit", f"submit {arm.name}"):
                    # Resolve the timeout (HF API accepts "30m", "2h", "9d")
                    budget = self.spec.budget
                    timeout_s = budget.train_timeout_s + budget.boot_timeout_s
                    # Round up to avoid truncating the requested budget
                    if timeout_s >= 86400:
                        days = -(-timeout_s // 86400)  # ceiling division
                        timeout_str = f"{days}d"
                    elif timeout_s >= 3600:
                        hours = -(-timeout_s // 3600)
                        timeout_str = f"{hours}h"
                    else:
                        minutes = max(1, -(-timeout_s // 60))
                        timeout_str = f"{minutes}m"

                    volumes = volumes_from_dicts(volumes_dicts)

                    script_path = f"/input/{self.spec.name}/{arm.name}/run.sh"
                    job_id = await client.run_job(
                        image=self.spec.hardware.image,
                        command=["bash", script_path],
                        flavor=self.spec.hardware.hf_flavor,
                        timeout=timeout_str,
                        env=arm.env or {},
                        labels={
                            "experiment": self.spec.name,
                            "arm": arm.name,
                            "arm_index": str(arm_idx),
                        },
                        volumes=volumes,
                        namespace=self.config.namespace or None,
                    )
                    logger.info("[%s] Submitted HF Job %s", arm.name, job_id)
                    result.instance_id = None  # No Vast instance
                    result.gpu_info = f"HF flavor: {self.spec.hardware.hf_flavor}"

                fleet_breadcrumb(
                    f"HF Job {job_id} submitted for arm {arm.name}",
                    category="fleet.hf.submit",
                    data={"arm": arm.name, "job_id": job_id},
                )

                # --- Poll for completion ---
                with _span("hf.job.poll", f"poll {arm.name}"):
                    final_stage = await self._poll_job(client, job_id, arm.name)

                duration = time.monotonic() - arm_start
                result.duration_seconds = duration

                # --- Fetch logs and extract metrics ---
                with _span("hf.job.logs", f"logs {arm.name}"):
                    try:
                        logs = await client.get_job_logs(job_id)
                    except HFClientError:
                        logs = ""
                        logger.warning("[%s] Could not fetch logs for job %s", arm.name, job_id)

                if final_stage == HFJobStage.COMPLETED:
                    result.exit_code = 0
                    result.metrics = parse_metrics(logs, self.spec.metrics)
                elif final_stage in (HFJobStage.FAILED, HFJobStage.ERROR):
                    result.exit_code = 1
                    result.error = self._extract_error(logs)
                    self._report_crash(arm.name, job_id, logs)
                elif final_stage == HFJobStage.CANCELLED:
                    result.exit_code = -2
                    result.error = "Job was cancelled"
                else:
                    result.exit_code = -1
                    result.error = f"Unexpected terminal stage: {final_stage}"

                # --- Emit metrics ---
                fleet_metric(
                    "fleet.arm.duration", duration,
                    unit="second", tags=arm_tags,
                )
                fleet_metric(
                    "fleet.arm.exit_code", float(result.exit_code),
                    tags=arm_tags,
                )
                hourly = HF_FLAVOR_PRICING.get(self.spec.hardware.hf_flavor, 0)
                cost = duration / 3600 * hourly
                fleet_metric("fleet.arm.cost", cost, unit="dollar", tags=arm_tags)

                # Save logs to disk
                self._save_arm_log(arm.name, logs)

            except Exception as exc:
                result.exit_code = -1
                result.error = str(exc)
                result.duration_seconds = time.monotonic() - arm_start
                logger.error("[%s] HF arm failed: %s", arm.name, exc)

            return result

    async def _poll_job(
        self,
        client: HFClient,
        job_id: str,
        arm_name: str,
    ) -> HFJobStage:
        """Poll an HF Job until it reaches a terminal state."""
        while True:
            try:
                info = await client.get_job(job_id)
            except HFClientError:
                logger.warning("[%s] Poll failed for job %s, retrying...", arm_name, job_id)
                await asyncio.sleep(JOB_POLL_INTERVAL_S)
                continue

            fleet_breadcrumb(
                f"Job {job_id} status: {info.stage.value}",
                category="fleet.hf.poll",
                data={"arm": arm_name, "job_id": job_id, "stage": info.stage.value},
            )

            if info.stage.is_terminal:
                return info.stage

            await asyncio.sleep(JOB_POLL_INTERVAL_S)

    def _build_volumes(self, arm: ArmSpec) -> list[dict[str, Any]]:
        """Build the list of volume mount dicts for an arm's job."""
        volumes: list[dict[str, Any]] = []

        # Script input bucket (contains wrapper script)
        volumes.append(build_script_volume(
            self._output_bucket, mount_path="/input",
        ))

        # Data volume (if configured)
        data = self.spec.data
        if data.source in ("hf-dataset", "hf-bucket") and data.hf_source:
            volumes.append(build_data_volume(
                data.source, data.hf_source, data.hf_mount_path,
            ))

        # Output bucket for artifacts
        volumes.append(build_output_volume(
            self._output_bucket, mount_path="/output",
        ))

        return volumes

    def _build_wrapper_script(self, arm: ArmSpec) -> str:
        """Generate a self-contained bash wrapper script for an arm.

        The script replicates FleetExecutor's step-by-step logic inside
        a single shell script that runs as the HF Job command.
        """
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "# Force unbuffered Python output for reliable metrics capture",
            "export PYTHONUNBUFFERED=1",
            "",
            "# Ensure git is available (some minimal images lack it)",
            "if ! command -v git &>/dev/null; then",
            "  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1",
            "fi",
            "",
        ]

        # Environment variables
        if arm.env:
            lines.append("# Arm-specific environment variables")
            for k, v in arm.env.items():
                lines.append(f"export {k}={shlex.quote(str(v))}")
            lines.append("")

        # Clone repo
        repo = self.spec.repo
        lines.extend([
            "# Clone experiment repository",
            f"git clone --depth {repo.clone_depth} --branch {shlex.quote(repo.branch)} "
            f"{shlex.quote(repo.url)} {shlex.quote(repo.remote_path)}",
            f"cd {shlex.quote(repo.remote_path)}",
        ])
        if repo.commit:
            lines.extend([
                f"git fetch --depth 1 origin {shlex.quote(repo.commit)}",
                f"git checkout {shlex.quote(repo.commit)}",
            ])
        lines.append("")

        # Install dependencies
        deps = self.spec.deps
        if deps.pre_install:
            lines.append("# Pre-install commands")
            for cmd in deps.pre_install:
                lines.append(cmd)
            lines.append("")

        if deps.requirements:
            lines.append("# Install requirements")
            if deps.exclude_from_requirements:
                excl = "|".join(deps.exclude_from_requirements)
                # Strip comments before matching to avoid false positives
                # (e.g., 'torchao' in a comment matching 'torch')
                lines.append(
                    f"sed 's/#.*//' {shlex.quote(deps.requirements)} "
                    f"| grep -vE '^({excl})([=><!\\[\\s]|$)' "
                    f"> /tmp/filtered_requirements.txt || true"
                )
                lines.append("pip install -q -r /tmp/filtered_requirements.txt")
            else:
                lines.append(f"pip install -q -r {shlex.quote(deps.requirements)}")
            lines.append("")

        if deps.verify:
            lines.append("# Verify dependencies")
            lines.append(deps.verify)
            lines.append("")

        # Preflight
        if self.spec.preflight:
            pf = self.spec.preflight
            lines.extend([
                "# Pre-flight validation",
                f"echo '--- Preflight: {arm.name} ---'",
                pf.command,
                "echo '--- Preflight complete ---'",
                "",
            ])

        # Training command (capture exit code without set -e terminating)
        resolved = self.spec.resolve_command(arm)
        lines.extend([
            "# Training",
            f"echo '--- Training: {arm.name} ---'",
            f"{resolved} && TRAIN_EXIT=0 || TRAIN_EXIT=$?",
            "echo \"--- Training complete (exit $TRAIN_EXIT) ---\"",
            "",
        ])

        # Post-training validation
        if self.spec.validation:
            val = self.spec.validation
            lines.extend([
                "# Post-training validation (only if training succeeded)",
                "if [ $TRAIN_EXIT -eq 0 ]; then",
                f"  echo '--- Validation: {arm.name} ---'",
                f"  {val.command}",
                "  echo '--- Validation complete ---'",
                "fi",
                "",
            ])

        # Copy artifacts to output
        lines.extend([
            "# Copy logs to output bucket mount",
            "ARM_OUT=${ARM_OUTPUT_DIR:-/output/" + self.spec.name + "/" + arm.name + "}",
            "mkdir -p \"$ARM_OUT\"",
            "cp -r /tmp/*.log \"$ARM_OUT/\" 2>/dev/null || true",
            "",
            "exit ${TRAIN_EXIT:-0}",
        ])

        return "\n".join(lines) + "\n"

    def _extract_error(self, logs: str) -> str:
        """Extract the last error-like lines from job logs."""
        if not logs:
            return "No logs available"
        lines = logs.strip().splitlines()
        # Return last 10 lines as error context
        return "\n".join(lines[-10:])

    def _report_crash(self, arm_name: str, job_id: str, logs: str) -> None:
        """Report an HF Job failure to Sentry."""
        if not sentry_sdk:
            return

        fleet_breadcrumb(
            f"HF Job {job_id} failed for arm {arm_name}",
            category="fleet.hf.crash",
            level="error",
            data={"arm": arm_name, "job_id": job_id},
        )

        if logs:
            log_tail = "\n".join(logs.strip().splitlines()[-100:])
            sentry_sdk.add_attachment(
                bytes=log_tail.encode("utf-8", errors="replace"),
                filename=f"{arm_name}_hf_logs.txt",
                content_type="text/plain",
            )

        sentry_sdk.capture_event({
            "level": "error",
            "transaction": f"fleet-hf/{self.spec.name}/{arm_name}",
            "tags": {
                "experiment": self.spec.name,
                "arm": arm_name,
                "provider": "hf",
                "job_id": job_id,
            },
            "exception": {
                "values": [{
                    "type": "HFJobFailedError",
                    "value": f"HF Job {job_id} failed for arm {arm_name}",
                    "mechanism": {"type": "hf_jobs", "handled": True},
                }],
            },
        })

    def _save_arm_log(self, arm_name: str, logs: str) -> None:
        """Save arm logs to the log directory."""
        from pathlib import Path

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.spec.name}_{arm_name}_hf.log"
        log_file.write_text(logs)

    def _print_dry_run(
        self, arm_pairs: list[tuple[int, ArmSpec]],
    ) -> None:
        """Print what would be executed without launching."""
        hw = self.spec.hardware
        hourly = HF_FLAVOR_PRICING.get(hw.hf_flavor, 0)
        print(f"\n{'='*60}")
        print(f"DRY RUN: {self.spec.name} on HuggingFace Jobs")
        print(f"{'='*60}")
        print(f"  Image:      {hw.image}")
        print(f"  Flavor:     {hw.hf_flavor}")
        print(f"  $/hr:       ${hourly:.2f}")
        print(f"  Timeout:    {self.spec.budget.train_timeout_s}s")
        print(f"  Namespace:  {self.config.namespace or '(default)'}")
        print(f"  Output:     bucket:{self._output_bucket}")
        print(f"  Arms ({len(arm_pairs)}):")
        for idx, arm in arm_pairs:
            print(f"    [{idx}] {arm.name}: {arm.command[:80]}")
        print(f"{'='*60}\n")
