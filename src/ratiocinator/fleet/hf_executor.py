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
import json
import logging
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
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

# Heartbeat / state-file conventions.
#
# The training script inside the job is expected to periodically dump a
# small JSON blob (``{"step": ..., "loss": ..., "eta_seconds": ..., ...}``)
# to the path advertised via the ``RATIOCINATOR_STATE_PATH`` env var.
# The executor polls this file from the mounted output bucket via the
# HF API, which side-steps ``fetch_job_logs`` (which is known to degrade
# or silently fail at times, leaving the orchestrator blind).
STATE_FILENAME = "state.json"
STATE_ENV_VAR = "RATIOCINATOR_STATE_PATH"
HEARTBEAT_POLL_INTERVAL_S = 30

# Fields the executor surfaces from a heartbeat ``state.json`` blob.
# Other keys are still preserved (used in metric fallback) but are not
# echoed in per-poll Sentry breadcrumbs to keep them concise.
HEARTBEAT_BREADCRUMB_FIELDS = ("step", "loss", "eta_seconds")


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
        # Per-arm nonces (UUID4) generated at script build time.  The nonce
        # is embedded in the wrapper script's sentinel file and state.json so
        # the poll loop can distinguish current-run artifacts from stale
        # leftovers in the persistent output bucket.
        self._run_nonces: dict[str, str] = {}

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
            arm_pairs,
            skip_duplicates=skip_duplicates,
            logger=logger,
            spec=self.spec,
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
            config_hash=arm_config_hash(arm, self.spec),
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

                    # Build effective env (arm env + BATCH_SIZE)
                    job_env = self.spec.resolve_arm_env(arm)

                    script_path = f"/input/{self.spec.name}/{arm.name}/run.sh"
                    job_id = await client.run_job(
                        image=self.spec.hardware.image,
                        command=["bash", script_path],
                        flavor=self.spec.hardware.hf_flavor,
                        timeout=timeout_str,
                        env=job_env,
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
                    if not result.metrics:
                        # Logs may be unavailable/truncated — fall back to
                        # the bucket-backed heartbeat file as the source
                        # of truth for final metrics.
                        state = await self._read_heartbeat(client, arm.name)
                        if state:
                            fleet_breadcrumb(
                                f"Using heartbeat metrics for arm {arm.name}",
                                category="fleet.hf.heartbeat",
                                data={"arm": arm.name},
                            )
                            result.metrics = {
                                k: v for k, v in state.items()
                                if isinstance(v, (int, float, str, bool))
                            }
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
        """Poll an HF Job until it reaches a terminal state.

        Between job-status polls, opportunistically read the bucket-backed
        ``state.json`` heartbeat so the orchestrator has visibility into
        training progress even when ``fetch_job_logs`` is degraded.
        """
        last_step: int | None = None
        # Track whether we have seen a heartbeat written by the current job.
        # The bucket may contain stale state.json from a previous completed
        # run, so the first read is not usable for step-regression detection.
        heartbeat_established = False
        next_status_check = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= next_status_check:
                try:
                    info = await client.get_job(job_id)
                except HFClientError:
                    logger.warning(
                        "[%s] Poll failed for job %s, retrying...",
                        arm_name, job_id,
                    )
                    await asyncio.sleep(JOB_POLL_INTERVAL_S)
                    next_status_check = time.monotonic() + JOB_POLL_INTERVAL_S
                    continue

                fleet_breadcrumb(
                    f"Job {job_id} status: {info.stage.value}",
                    category="fleet.hf.poll",
                    data={
                        "arm": arm_name,
                        "job_id": job_id,
                        "stage": info.stage.value,
                    },
                )

                if info.stage.is_terminal:
                    return info.stage
                next_status_check = now + JOB_POLL_INTERVAL_S

            # Heartbeat read between status polls
            state = await self._read_heartbeat(client, arm_name)
            if state is not None:
                # Only trust state.json written by the current run.
                # The output bucket persists across reruns, so stale
                # payloads from earlier completed jobs may still exist.
                state_nonce = state.get("run_nonce")
                expected_nonce = self._run_nonces.get(arm_name)
                nonce_matches = (
                    expected_nonce is not None
                    and state_nonce == expected_nonce
                )

                # The wrapper script writes preemption info into state.json
                # (including our run_nonce) before training starts.
                if (
                    state.get("preemption_detected")
                    and nonce_matches
                    and not heartbeat_established
                ):
                    ckpt = state.get("resume_checkpoint", "unknown")
                    logger.warning(
                        "[%s] Preemption detected (container restarted). "
                        "Resuming from: %s",
                        arm_name, ckpt,
                    )
                    fleet_breadcrumb(
                        f"Preemption detected for {arm_name}: "
                        f"resuming from {ckpt}",
                        category="fleet.hf.preemption",
                        level="warning",
                        data={
                            "arm": arm_name,
                            "job_id": job_id,
                            "resume_checkpoint": ckpt,
                        },
                    )
                    fleet_metric(
                        "fleet.arm.preemption", 1.0,
                        tags={
                            "experiment": self.spec.name,
                            "arm": arm_name,
                        },
                    )
                    heartbeat_established = True

                step = state.get("step")
                if step is not None and step != last_step:
                    # Detect preemption via step regression, but only once
                    # the heartbeat is established for this job (avoids
                    # false positives from stale state.json left by a
                    # previous completed run in the persistent bucket).
                    if (
                        heartbeat_established
                        and last_step is not None
                        and isinstance(step, (int, float))
                        and isinstance(last_step, (int, float))
                        and step < last_step
                    ):
                        logger.warning(
                            "[%s] Preemption detected: step went from %s "
                            "to %s (container likely restarted)",
                            arm_name, last_step, step,
                        )
                        fleet_breadcrumb(
                            f"Preemption detected for {arm_name}: "
                            f"step {last_step} → {step}",
                            category="fleet.hf.preemption",
                            level="warning",
                            data={
                                "arm": arm_name,
                                "job_id": job_id,
                                "previous_step": last_step,
                                "resumed_step": step,
                            },
                        )
                        fleet_metric(
                            "fleet.arm.preemption", 1.0,
                            tags={
                                "experiment": self.spec.name,
                                "arm": arm_name,
                            },
                        )

                    # Only mark heartbeat as established once we observe
                    # forward progress (step strictly increases from a prior
                    # reading).  The very first heartbeat could be stale data
                    # from a previous run that persists in the output bucket,
                    # so we never trust it as "established" by itself.
                    if (
                        last_step is not None
                        and isinstance(step, (int, float))
                        and isinstance(last_step, (int, float))
                        and step > last_step
                    ):
                        heartbeat_established = True
                    last_step = step
                    fleet_breadcrumb(
                        f"Heartbeat {arm_name}: step={step}",
                        category="fleet.hf.heartbeat",
                        data={
                            "arm": arm_name,
                            "job_id": job_id,
                            **{
                                k: v for k, v in state.items()
                                if k in HEARTBEAT_BREADCRUMB_FIELDS
                            },
                        },
                    )
                    loss = state.get("loss")
                    if isinstance(step, (int, float)):
                        fleet_metric(
                            "fleet.arm.heartbeat.step", float(step),
                            tags={"experiment": self.spec.name, "arm": arm_name},
                        )
                    if isinstance(loss, (int, float)):
                        fleet_metric(
                            "fleet.arm.heartbeat.loss", float(loss),
                            tags={"experiment": self.spec.name, "arm": arm_name},
                        )

            await asyncio.sleep(HEARTBEAT_POLL_INTERVAL_S)

    async def _read_heartbeat(
        self,
        client: HFClient,
        arm_name: str,
    ) -> dict[str, Any] | None:
        """Fetch and parse the bucket-backed heartbeat file for an arm.

        Returns ``None`` if the file does not yet exist or is invalid
        JSON.  Network-level errors are logged and treated as missing
        rather than fatal — heartbeats are advisory.
        """
        remote = self._arm_state_remote_path(arm_name)
        try:
            text = await client.download_from_bucket(self._output_bucket, remote)
        except HFClientError as exc:
            logger.debug(
                "[%s] Heartbeat read failed (%s); treating as missing",
                arm_name, exc,
            )
            return None
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(
                "[%s] Heartbeat file is not valid JSON; ignoring", arm_name,
            )
            return None
        if not isinstance(parsed, dict):
            logger.debug(
                "[%s] Heartbeat JSON is not an object; ignoring", arm_name,
            )
            return None
        return parsed

    def _arm_state_remote_path(self, arm_name: str) -> str:
        """Path of an arm's ``state.json`` within the output bucket."""
        return f"{self.spec.name}/{arm_name}/{STATE_FILENAME}"

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

        # Environment variables (arm-specific + BATCH_SIZE)
        effective_env = self.spec.resolve_arm_env(arm)
        if effective_env:
            lines.append("# Arm-specific environment variables")
            for k, v in effective_env.items():
                lines.append(f"export {k}={shlex.quote(str(v))}")
            lines.append("")

        # Heartbeat: advertise where the training script should dump
        # its periodic state.json.  Reading this file from the bucket
        # is more reliable than parsing stdout via fetch_job_logs.
        state_remote = self._arm_state_remote_path(arm.name)
        state_mount = f"/output/{state_remote}"
        lines.extend([
            "# Heartbeat state path (poll target for the orchestrator)",
            f"export {STATE_ENV_VAR}={shlex.quote(state_mount)}",
            f"mkdir -p {shlex.quote(str(Path(state_mount).parent))}",
            "",
        ])

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

        # Ensure huggingface_hub>=1.9.0 for hf:// fsspec protocol support.
        # Placed AFTER requirements to prevent a later pip install from downgrading.
        if self.spec.data.source in ("hf-dataset", "hf-bucket"):
            lines.append("# Pin huggingface_hub>=1.9.0 for hf:// protocol support")
            lines.append("pip install -q 'huggingface_hub>=1.9.0'")
            lines.append("")

        if deps.verify:
            lines.append("# Verify dependencies")
            lines.append(deps.verify)
            lines.append("")

        # Preemption detection using a nonce-bearing sentinel file.  Each
        # invocation of _build_wrapper_script() embeds a unique run nonce.
        # On first start, the sentinel is either absent or carries a DIFFERENT
        # nonce (left from a previous completed run) — we overwrite it and
        # skip checkpoint search.  On a preemption RESTART, the sentinel
        # contains our nonce (written when the same script ran earlier),
        # so we know the current job was interrupted and look for checkpoints.
        run_nonce = uuid.uuid4().hex
        self._run_nonces[arm.name] = run_nonce
        arm_output = "/output/" + self.spec.name + "/" + arm.name
        sentinel = arm_output + "/_job_started"
        lines.extend([
            "# Preemption/restart detection (nonce-based sentinel)",
            f"mkdir -p {shlex.quote(arm_output)}",
            f"RUN_NONCE={shlex.quote(run_nonce)}",
            f"if [ -f {shlex.quote(sentinel)} ] "
            f"&& [ \"$(cat {shlex.quote(sentinel)})\" = \"$RUN_NONCE\" ]; then",
            f"  RESUME_CKPT=$(find {shlex.quote(arm_output)} "
            "\\( -name 'checkpoint_*.pt' -o -name 'checkpoint_*.pth' \\) "
            "2>/dev/null | sort -V | tail -1)",
            "  if [ -n \"$RESUME_CKPT\" ]; then",
            "    echo \"WARNING: [PREEMPTION DETECTED] Found existing checkpoint"
            " from previous run: $RESUME_CKPT\"",
            "    echo \"WARNING: Container was likely preempted and restarted."
            " Resuming from latest checkpoint.\"",
            "    export RATIOCINATOR_RESUME_CHECKPOINT=\"$RESUME_CKPT\"",
            # Write preemption info into state.json so the orchestrator's
            # heartbeat polling picks it up (the METRICS: protocol only
            # retains the last line, so stdout is unreliable for this).
            "    echo '{\"preemption_detected\": true, "
            "\"resume_checkpoint\": \"'\"$RESUME_CKPT\"'\", "
            "\"run_nonce\": \"'\"$RUN_NONCE\"'\"}'"
            f" > {shlex.quote(state_mount)}",
            "  fi",
            "else",
            f"  echo \"$RUN_NONCE\" > {shlex.quote(sentinel)}",
            "fi",
            "",
        ])

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
