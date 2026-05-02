"""Fleet executor: parallel experiment orchestration on Vast.ai instances.

The `FleetExecutor` takes an `ExperimentSpec` and executes all (or selected)
arms in parallel on ephemeral GPU instances.  It handles:

- Instance provisioning with rate limiting and retry
- SSH readiness checks
- Repository cloning and dependency installation
- Data provisioning (S3 / rsync / local)
- Pre-flight validation (torch version, CUDA, GPU type)
- Experiment execution with timeout enforcement
- Metric extraction from stdout
- Post-training validation (real syntax checks, ground-truth metrics)
- Post-flight cleanup and instance destruction
- Results aggregation to the `ResultStore`
- Sentry observability (spans, crash reporting)
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratiocinator.fleet.data import DataProvisioner, create_provisioner
from ratiocinator.fleet.results import ArmResult, ResultStore
from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    arm_config_hash,
    deduplicate_arm_pairs,
    parse_metrics,
)
from ratiocinator.infra.remote import RemoteExecutor
from ratiocinator.infra.vast_client import InstanceStatus, VastClient
from ratiocinator.observability import fleet_breadcrumb, fleet_metric

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BOOT_POLL_INTERVAL_S = 10
INSTANCE_CREATE_STAGGER_S = 5

# Architecture dump convention (shared with HF executor).
ARCHITECTURE_FILENAME = "resolved_architecture.json"
ARCHITECTURE_ENV_VAR = "RATIOCINATOR_ARCHITECTURE_PATH"


def _parse_remote_traceback(stderr_text: str) -> tuple[list[dict], str, str]:
    """Parse a Python traceback from remote stderr into Sentry-compatible frames."""
    frames = []
    exc_type = "RemoteTrainingError"
    exc_value = (
        stderr_text.strip().splitlines()[-1] if stderr_text.strip() else "Unknown error"
    )

    frame_re = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
    for line in stderr_text.splitlines():
        m = frame_re.match(line)
        if m:
            frames.append({
                "filename": m.group(1),
                "lineno": int(m.group(2)),
                "function": m.group(3),
            })

    last_line = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else ""
    if ": " in last_line and not last_line.startswith(" "):
        exc_type, _, exc_value = last_line.partition(": ")

    return frames, exc_type, exc_value


def _report_remote_crash(
    *,
    experiment: str,
    arm_name: str,
    exit_code: int,
    stderr_text: str,
    stdout_tail: str,
    gpu_info: str,
    instance_id: int | None,
) -> None:
    """Report a remote training crash to Sentry as a structured exception."""
    if not sentry_sdk:
        return

    frames, exc_type, exc_value = _parse_remote_traceback(stderr_text)

    # Add breadcrumb with crash context
    fleet_breadcrumb(
        f"Arm {arm_name} crashed (exit {exit_code}): {exc_type}: {str(exc_value)[:100]}",
        category="fleet.crash",
        level="error",
        data={
            "arm_name": arm_name,
            "exit_code": exit_code,
            "instance_id": instance_id,
        },
    )

    event: dict[str, Any] = {
        "level": "error",
        "transaction": f"fleet/{experiment}/{arm_name}",
        "tags": {
            "experiment": experiment,
            "arm": arm_name,
            "exit_code": str(exit_code),
        },
        "contexts": {
            "fleet": {
                "experiment": experiment,
                "arm_name": arm_name,
                "exit_code": exit_code,
                "instance_id": instance_id,
                "gpu_info": gpu_info[:200] if gpu_info else "",
            },
        },
        "extra": {
            "stderr": stderr_text[-3000:],
            "stdout_tail": stdout_tail,
        },
    }

    if frames:
        event["exception"] = {
            "values": [{
                "type": exc_type,
                "value": exc_value,
                "stacktrace": {"frames": frames},
                "mechanism": {
                    "type": "remote_ssh",
                    "handled": True,
                    "description": f"Remote crash on Vast.ai instance {instance_id}",
                },
            }],
        }
    else:
        event["exception"] = {
            "values": [{
                "type": "RemoteTrainingError",
                "value": f"Arm {arm_name} failed (exit {exit_code}): {stderr_text[-500:]}",
                "mechanism": {"type": "remote_ssh", "handled": True},
            }],
        }

    # Attach the last 100 lines of stdout/stderr for offline debugging
    if stderr_text.strip():
        stderr_tail = "\n".join(stderr_text.strip().splitlines()[-100:])
        sentry_sdk.add_attachment(
            bytes=stderr_tail.encode("utf-8", errors="replace"),
            filename=f"{arm_name}_stderr.txt",
            content_type="text/plain",
        )
    if stdout_tail.strip():
        stdout_lines = "\n".join(stdout_tail.strip().splitlines()[-100:])
        sentry_sdk.add_attachment(
            bytes=stdout_lines.encode("utf-8", errors="replace"),
            filename=f"{arm_name}_stdout.txt",
            content_type="text/plain",
        )

    sentry_sdk.capture_event(event)


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _build_env_prefix(env: dict[str, str] | None) -> str:
    """Build a shell-safe env var prefix string for remote commands.

    Validates key names against ``[A-Za-z_][A-Za-z0-9_]*`` to prevent
    shell injection from malformed or LLM-generated keys.
    """
    if not env:
        return ""
    parts: list[str] = []
    for k, v in env.items():
        if not _ENV_KEY_RE.match(k):
            raise ValueError(
                f"Invalid env var name {k!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
        parts.append(f"{k}={shlex.quote(str(v))}")
    return " ".join(parts) + " "


def _arm_env_with_batch_size(
    arm: ArmSpec, spec: ExperimentSpec,
) -> dict[str, str]:
    """Return the arm's effective env dict including BATCH_SIZE.

    Delegates to :meth:`ExperimentSpec.resolve_arm_env` which is the
    single source of truth for batch-size injection across all providers.
    """
    return spec.resolve_arm_env(arm)


@dataclass
class FleetConfig:
    """Runtime configuration for a fleet execution."""

    api_key: str
    ssh_key: str
    max_concurrent: int = 7
    stagger_seconds: float = INSTANCE_CREATE_STAGGER_S
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"


class FleetExecutor:
    """Executes an experiment spec across parallel Vast.ai instances.

    Usage::

        spec = ExperimentSpec.from_yaml("experiment.yaml")
        config = FleetConfig(api_key="...", ssh_key="~/.ssh/id_rsa")
        executor = FleetExecutor(spec, config)
        results = await executor.run()
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        config: FleetConfig,
        *,
        provisioner: DataProvisioner | None = None,
        result_store: ResultStore | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self.provisioner = provisioner or create_provisioner(
            spec.data.source,
            urls_file=spec.data.urls_file,
            rsync_server=spec.data.rsync_server,
            rsync_port=spec.data.rsync_port,
            max_shards=spec.data.max_shards,
            local_path=spec.data.local_path,
        )
        self.store = result_store or ResultStore(config.results_path)

    async def run(
        self,
        arm_indices: list[int] | None = None,
        *,
        dry_run: bool = False,
        skip_duplicates: bool = False,
    ) -> list[ArmResult]:
        """Execute selected (or all) experiment arms in parallel.

        Args:
            arm_indices: Indices of arms to run. None means all arms.
            dry_run: If True, just find offers and print plan without launching.
            skip_duplicates: If True, arms whose effective configuration hashes
                to one already scheduled are dropped before launch (autonomous
                mode).  If False (default), duplicates are warned about but
                still executed (manual mode — user may want replication).

        Returns:
            List of ArmResult for each arm executed.
        """
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

        async with VastClient(self.config.api_key) as client:
            offers = await self._find_offers(client, len(arm_pairs))
            if not offers:
                logger.error("No matching offers found")
                return []

            if dry_run:
                self._print_dry_run(arm_pairs, offers)
                return []

            # Start Sentry transaction
            txn = self._start_transaction(arm_pairs)

            try:
                tasks = [
                    self._run_arm(
                        client, offers[j % len(offers)],
                        arm_idx, arm, j,
                    )
                    for j, (arm_idx, arm) in enumerate(arm_pairs)
                ]
                results = await asyncio.gather(*tasks)

                # Record results
                self.store.record_many(list(results))

                # Update Sentry transaction
                if txn:
                    succeeded = sum(1 for r in results if r.success)
                    txn.set_data("fleet.succeeded", succeeded)
                    txn.set_data("fleet.failed", len(results) - succeeded)

                return list(results)

            except Exception:
                if txn:
                    txn.set_status("internal_error")
                raise
            finally:
                if txn:
                    txn.__exit__(None, None, None)

    def _apply_dedup(
        self,
        arm_pairs: list[tuple[int, ArmSpec]],
        *,
        skip_duplicates: bool,
    ) -> list[tuple[int, ArmSpec]]:
        """Detect duplicate arm configs and either warn or drop them.

        Always logs a warning listing duplicate groups.  When
        ``skip_duplicates`` is True, only the first occurrence of each
        unique config hash is retained.
        """
        return deduplicate_arm_pairs(
            arm_pairs,
            skip_duplicates=skip_duplicates,
            logger=logger,
            spec=self.spec,
        )

    async def _find_offers(
        self, client: VastClient, num_needed: int,
    ) -> list[dict]:
        """Find GPU offers matching hardware requirements."""
        hw = self.spec.hardware
        offers = await client.search_offers(
            gpu_name=hw.gpu,
            num_gpus=hw.num_gpus,
            max_dph=hw.max_dph,
            limit=50,
        )

        matched = []
        for o in offers:
            if o.get("pcie_bw", 0) < hw.min_pcie_bw:
                continue
            if o.get("cpu_ram", 0) < hw.min_cpu_ram_gb * 1000:
                continue
            matched.append(o)
            if len(matched) >= num_needed:
                break

        if len(matched) < num_needed:
            logger.warning(
                "Found %d offers (need %d) — some arms will share hosts",
                len(matched), num_needed,
            )

        return matched

    async def _run_arm(
        self,
        client: VastClient,
        offer: dict,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        """Run a single experiment arm on a Vast.ai instance."""
        result = ArmResult(
            experiment=self.spec.name,
            arm_name=arm.name,
            description=arm.description,
            config_hash=arm_config_hash(arm, self.spec),
        )
        instance_id = None
        hw = self.spec.hardware
        budget = self.spec.budget
        arm_start = time.monotonic()
        arm_tags = {
            "experiment": self.spec.name,
            "arm": arm.name,
        }

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
            # Set Sentry tags in an isolated scope for this arm
            if scope is not None:
                scope.set_tag("arm_name", arm.name)
                scope.set_tag("experiment_name", self.spec.name)
                scope.set_tag("arm_index", str(arm_idx))

            try:
                # Stagger to avoid rate limits
                if launch_order > 0:
                    await asyncio.sleep(launch_order * self.config.stagger_seconds)

                # --- Provision instance ---
                fleet_breadcrumb(
                    f"Provisioning instance for arm {arm.name}",
                    category="fleet.provision",
                    data={"arm": arm.name, "offer_id": offer.get("id")},
                )
                stage_start = time.monotonic()
                with _span("vm.provision", f"provision {arm.name}") as span:
                    onstart = "#!/bin/bash\necho 'ready' > /tmp/ready\n"
                    instance_id = await client.create_instance(
                        offer_id=offer["id"],
                        image=hw.image,
                        onstart=onstart,
                        label=f"{self.spec.name}-{arm.name}",
                        disk_gb=hw.disk_gb,
                    )
                    result.instance_id = instance_id
                    dph = offer.get("dph_total", 0)
                    result.instance_dph = float(dph or 0.0)
                    logger.info(
                        "[%s] instance %s @ $%.3f/hr", arm.name, instance_id, dph
                    )
                    if span:
                        span.set_data("instance_id", instance_id)
                        span.set_data("offer_dph", dph)

                if scope is not None:
                    scope.set_tag("instance_id", str(instance_id))

                fleet_breadcrumb(
                    f"Instance {instance_id} provisioned in "
                    f"{time.monotonic() - stage_start:.1f}s",
                    category="fleet.provision",
                    data={"arm": arm.name, "instance_id": instance_id, "dph": dph},
                )

                # --- Wait for boot + SSH ---
                fleet_breadcrumb(
                    f"Waiting for boot + SSH on instance {instance_id}",
                    category="fleet.boot",
                    data={"arm": arm.name, "instance_id": instance_id},
                )
                boot_start = time.monotonic()
                with _span("vm.boot", f"boot {arm.name}") as span:
                    ssh_host, ssh_port = await self._wait_for_boot(
                        client, instance_id, budget.boot_timeout_s,
                    )
                    if not ssh_host:
                        result.error = "Instance failed to boot"
                        result.exit_code = -1
                        fleet_breadcrumb(
                            f"Boot failed for arm {arm.name}",
                            category="fleet.boot",
                            level="error",
                            data={"arm": arm.name, "instance_id": instance_id},
                        )
                        return result

                    boot_duration = time.monotonic() - boot_start
                    fleet_metric(
                        "fleet.provision.boot_time", boot_duration,
                        unit="second", tags=arm_tags,
                    )

                    remote = RemoteExecutor(
                        ssh_host, ssh_port, self.config.ssh_key,
                        label=f"{arm.name}@{instance_id}",
                    )

                    ssh_start = time.monotonic()
                    if not await remote.wait_for_ssh():
                        result.error = "SSH never became ready"
                        result.exit_code = -1
                        fleet_breadcrumb(
                            f"SSH never ready for arm {arm.name}",
                            category="fleet.boot",
                            level="error",
                            data={"arm": arm.name, "instance_id": instance_id},
                        )
                        return result

                    ssh_wait_duration = time.monotonic() - ssh_start
                    total_boot = time.monotonic() - boot_start
                    # ``boot_time_s`` is the full provisioning overhead
                    # (boot + SSH-ready), which is what feeds the cost
                    # summary's "boot overhead" line.  Recording it
                    # only after ``wait_for_ssh()`` returns ensures we
                    # don't undercount the wall-clock spent provisioning.
                    result.boot_time_s = total_boot
                    logger.info("[%s] SSH ready: %s:%d", arm.name, ssh_host, ssh_port)
                    fleet_breadcrumb(
                        f"SSH ready after {total_boot:.1f}s",
                        category="fleet.boot",
                        data={
                            "arm": arm.name, "instance_id": instance_id,
                            "ssh_host": ssh_host, "ssh_port": ssh_port,
                            "boot_duration_s": round(boot_duration, 1),
                            "ssh_wait_s": round(ssh_wait_duration, 1),
                        },
                    )
                    fleet_metric(
                        "fleet.provision.ssh_wait", ssh_wait_duration,
                        unit="second", tags=arm_tags,
                    )
                    if span:
                        span.set_data("ssh_host", ssh_host)
                        span.set_data("ssh_port", ssh_port)

                # --- Hardware info ---
                with _span("vm.hwinfo", f"hwinfo {arm.name}"):
                    hw_result = await remote.run(
                        "nvidia-smi --query-gpu=name,memory.total,"
                        "pcie.link.gen.current,"
                        "pcie.link.width.current --format=csv,noheader "
                        "2>/dev/null; "
                        "echo '---'; free -h | head -2; "
                        "lscpu | grep 'Model name'",
                        timeout=30,
                    )
                    result.gpu_info = hw_result.stdout.strip()

                # --- Clone repo ---
                fleet_breadcrumb(
                    f"Cloning repo for arm {arm.name}",
                    category="fleet.clone",
                    data={"arm": arm.name, "repo": self.spec.repo.url},
                )
                stage_start = time.monotonic()
                with _span("git.clone", f"clone {arm.name}"):
                    repo = self.spec.repo
                    clone_result = await remote.run(
                        f"git clone --branch {repo.branch} "
                        f"--depth {repo.clone_depth} "
                        f"{repo.url} {repo.remote_path}",
                        timeout=120,
                    )
                    if not clone_result.success:
                        result.error = (
                            f"Git clone failed: {clone_result.stderr[:500]}"
                        )
                        result.exit_code = -1
                        fleet_breadcrumb(
                            f"Git clone failed for arm {arm.name}",
                            category="fleet.clone",
                            level="error",
                            data={
                                "arm": arm.name,
                                "stderr": clone_result.stderr[:200],
                            },
                        )
                        return result

                fleet_breadcrumb(
                    f"Repo cloned in {time.monotonic() - stage_start:.1f}s",
                    category="fleet.clone",
                    data={"arm": arm.name},
                )

                # --- Install dependencies ---
                fleet_breadcrumb(
                    f"Installing dependencies for arm {arm.name}",
                    category="fleet.deps",
                    data={"arm": arm.name},
                )
                stage_start = time.monotonic()
                with _span("pip.install", f"deps {arm.name}") as span:
                    deps = self.spec.deps

                    # Pre-install commands (e.g., specific torch version)
                    for cmd in deps.pre_install:
                        r = await remote.run(cmd, timeout=600)
                        if span:
                            span.set_data("pre_install_rc", r.exit_code)

                    # Install requirements (excluding pre-installed packages)
                    if deps.requirements:
                        if deps.exclude_from_requirements:
                            excludes = "|".join(
                                f"^{pkg}"
                                for pkg in deps.exclude_from_requirements
                            )
                            install_cmd = (
                                f"cd {repo.remote_path} && "
                                f"sed 's/#.*//' {deps.requirements} "
                                f"| grep -vE '{excludes}([>=<!\\[\\s]|$)' "
                                f"| pip install -q -r /dev/stdin 2>&1 "
                                f"| tail -5"
                            )
                        else:
                            install_cmd = (
                                f"cd {repo.remote_path} && "
                                f"pip install -q -r {deps.requirements} "
                                f"2>&1 | tail -5"
                            )
                        await remote.run(install_cmd, timeout=300)

                fleet_breadcrumb(
                    f"Dependencies installed in "
                    f"{time.monotonic() - stage_start:.1f}s",
                    category="fleet.deps",
                    data={"arm": arm.name},
                )

                # --- Verify dependencies ---
                if self.spec.deps.verify:
                    with _span("pip.verify", f"verify {arm.name}") as span:
                        v_result = await remote.run(
                            f"cd {self.spec.repo.remote_path} && "
                            f"{self.spec.deps.verify}",
                            timeout=30,
                        )
                        if not v_result.success:
                            result.error = (
                                "Dependency verification failed: "
                                f"{v_result.stderr[:500]}"
                            )
                            result.exit_code = -1
                            fleet_breadcrumb(
                                "Dependency verification failed for "
                                f"arm {arm.name}",
                                category="fleet.deps",
                                level="error",
                                data={
                                    "arm": arm.name,
                                    "stderr": v_result.stderr[:200],
                                },
                            )
                            if span:
                                span.set_data(
                                    "verify_stdout", v_result.stdout[:500],
                                )
                            return result

                # --- Provision data ---
                fleet_breadcrumb(
                    f"Downloading data for arm {arm.name}",
                    category="fleet.data",
                    data={"arm": arm.name, "target": self.spec.data.target},
                )
                stage_start = time.monotonic()
                with _span("data.provision", f"data {arm.name}"):
                    ok, err = await self.provisioner.provision(
                        remote, self.spec.data.target,
                        timeout=budget.download_timeout_s,
                    )
                    if not ok:
                        result.error = f"Data provisioning failed: {err}"
                        result.exit_code = -1
                        fleet_breadcrumb(
                            f"Data provisioning failed for arm {arm.name}",
                            category="fleet.data",
                            level="error",
                            data={
                                "arm": arm.name, "error": str(err)[:200],
                            },
                        )
                        return result

                fleet_breadcrumb(
                    f"Data downloaded in "
                    f"{time.monotonic() - stage_start:.1f}s",
                    category="fleet.data",
                    data={"arm": arm.name},
                )

                # --- Pre-flight validation ---
                if self.spec.preflight is not None:
                    pf = self.spec.preflight
                    fleet_breadcrumb(
                        f"Running preflight for arm {arm.name}",
                        category="fleet.preflight",
                        data={"arm": arm.name, "command": pf.command[:100]},
                    )
                    with _span("preflight.run", f"preflight {arm.name}") as span:
                        pf_env_prefix = _build_env_prefix(
                            _arm_env_with_batch_size(arm, self.spec),
                        )

                        pf_result = await remote.run(
                            f"cd {self.spec.repo.remote_path} && "
                            f"{pf_env_prefix}{pf.command}",
                            timeout=pf.timeout_s,
                        )

                        if span:
                            span.set_data("exit_code", pf_result.exit_code)

                        if pf_result.exit_code != 0:
                            pf_stderr = pf_result.stderr or ""
                            result.error = (
                                f"Preflight failed (exit {pf_result.exit_code}): "
                                f"{pf_stderr[-500:]}"
                            )
                            result.exit_code = pf_result.exit_code
                            fleet_breadcrumb(
                                f"Preflight failed for arm {arm.name} "
                                f"(exit {pf_result.exit_code})",
                                category="fleet.preflight",
                                level="error",
                                data={
                                    "arm": arm.name,
                                    "exit_code": pf_result.exit_code,
                                    "stderr_tail": pf_stderr[-200:],
                                },
                            )
                            logger.warning(
                                "[%s] Preflight failed (exit %d)",
                                arm.name, pf_result.exit_code,
                            )
                            return result

                        if pf.check_metrics:
                            pf_metrics = parse_metrics(
                                pf_result.stdout, self.spec.metrics,
                            )
                            if not pf_metrics:
                                result.error = (
                                    "Preflight produced no metrics output"
                                )
                                result.exit_code = -1
                                fleet_breadcrumb(
                                    f"Preflight metrics check failed "
                                    f"for arm {arm.name}",
                                    category="fleet.preflight",
                                    level="error",
                                    data={"arm": arm.name},
                                )
                                logger.warning(
                                    "[%s] Preflight produced no metrics",
                                    arm.name,
                                )
                                return result

                        fleet_breadcrumb(
                            f"Preflight passed for arm {arm.name}",
                            category="fleet.preflight",
                            data={"arm": arm.name},
                        )
                        logger.info("[%s] Preflight passed", arm.name)

                # --- Run experiment ---
                fleet_breadcrumb(
                    f"Starting training for arm {arm.name}",
                    category="fleet.training",
                    data={"arm": arm.name, "command": arm.command[:100]},
                )
                command = self.spec.resolve_command(arm)
                with _span("train.run", f"train {arm.name}") as span:
                    logger.info("[%s] Running: %s", arm.name, command)
                    if span:
                        span.set_data("command", command)
                        span.set_data("gpu_info", result.gpu_info)
                        span.set_data("instance_id", instance_id)

                    # Merge arm-specific env (+ BATCH_SIZE) with command
                    arch_path = (
                        f"{self.spec.repo.remote_path}/"
                        f"{ARCHITECTURE_FILENAME}"
                    )
                    train_env = _arm_env_with_batch_size(arm, self.spec)
                    train_env[ARCHITECTURE_ENV_VAR] = arch_path
                    env_prefix = _build_env_prefix(train_env)

                    run_result = await remote.run(
                        f"cd {self.spec.repo.remote_path} && "
                        f"{env_prefix}{command}",
                        timeout=budget.train_timeout_s,
                    )
                    result.exit_code = run_result.exit_code
                    result.duration_seconds = run_result.duration_seconds

                    # Extract metrics
                    result.metrics = parse_metrics(
                        run_result.stdout, self.spec.metrics,
                    )

                    if span:
                        span.set_data("exit_code", run_result.exit_code)
                        span.set_data("metrics", result.metrics)

                # --- Write local log file ---
                self._write_arm_log(
                    arm.name, run_result.stdout, run_result.stderr,
                )

                # --- Handle failure ---
                if run_result.exit_code != 0:
                    err_detail = (
                        run_result.stderr[-2000:]
                        if run_result.stderr.strip() else ""
                    )
                    out_detail = run_result.stdout[-1000:]
                    result.error = (
                        f"Training failed (exit {run_result.exit_code}): "
                        f"STDERR: {err_detail}\nSTDOUT(tail): {out_detail}"
                        if err_detail else out_detail
                    )
                    logger.warning(
                        "[%s] Failed (exit %d)",
                        arm.name, run_result.exit_code,
                    )

                    fleet_breadcrumb(
                        f"Training failed for arm {arm.name} "
                        f"(exit {run_result.exit_code})",
                        category="fleet.training",
                        level="error",
                        data={
                            "arm": arm.name,
                            "exit_code": run_result.exit_code,
                            "stderr_tail": err_detail[-200:],
                        },
                    )

                    _report_remote_crash(
                        experiment=self.spec.name,
                        arm_name=arm.name,
                        exit_code=run_result.exit_code,
                        stderr_text=err_detail,
                        stdout_tail=out_detail[-500:],
                        gpu_info=result.gpu_info,
                        instance_id=instance_id,
                    )
                else:
                    fleet_breadcrumb(
                        f"Training completed for arm {arm.name}",
                        category="fleet.training",
                        data={"arm": arm.name, "metrics": result.metrics},
                    )
                    logger.info(
                        "[%s] Completed — metrics: %s",
                        arm.name, result.metrics,
                    )

                # --- Post-training validation ---
                if (
                    result.exit_code == 0
                    and self.spec.validation is not None
                ):
                    val = self.spec.validation
                    fleet_breadcrumb(
                        f"Running validation for arm {arm.name}",
                        category="fleet.validation",
                        data={
                            "arm": arm.name,
                            "command": val.command[:100],
                        },
                    )
                    with _span(
                        "validation.run", f"validate {arm.name}",
                    ) as span:
                        val_env = _arm_env_with_batch_size(arm, self.spec)
                        val_env[ARCHITECTURE_ENV_VAR] = arch_path
                        val_env_prefix = _build_env_prefix(val_env)

                        val_result = await remote.run(
                            f"cd {self.spec.repo.remote_path} && "
                            f"{val_env_prefix}{val.command}",
                            timeout=val.timeout_s,
                        )

                        if span:
                            span.set_data("exit_code", val_result.exit_code)

                        # Persist validation output to log
                        self._write_arm_log(
                            f"{arm.name}.validation",
                            val_result.stdout or "",
                            val_result.stderr or "",
                        )

                        if val_result.exit_code != 0:
                            val_stderr = val_result.stderr or ""
                            val_stdout = val_result.stdout or ""
                            # Prefer stderr, fall back to stdout tail
                            error_tail = (
                                val_stderr[-500:]
                                if val_stderr.strip()
                                else val_stdout[-500:]
                            )
                            result.error = (
                                f"Validation failed "
                                f"(exit {val_result.exit_code}): "
                                f"{error_tail}"
                            )
                            result.exit_code = val_result.exit_code
                            fleet_breadcrumb(
                                f"Validation failed for arm {arm.name} "
                                f"(exit {val_result.exit_code})",
                                category="fleet.validation",
                                level="error",
                                data={
                                    "arm": arm.name,
                                    "exit_code": val_result.exit_code,
                                    "stderr_tail": val_stderr[-200:],
                                },
                            )
                            logger.warning(
                                "[%s] Validation failed (exit %d)",
                                arm.name, val_result.exit_code,
                            )
                        else:
                            val_metrics = parse_metrics(
                                val_result.stdout, self.spec.metrics,
                            )
                            # Merge validation metrics into result,
                            # optionally namespaced with a prefix.
                            prefix = val.prefix
                            for k, v in val_metrics.items():
                                key = f"{prefix}{k}" if prefix else k
                                result.metrics[key] = v

                            # Check required metrics against raw
                            # validation output keys (not prefixed,
                            # not merged) so training metrics can't
                            # mask omissions and users specify the
                            # metric names their script actually emits.
                            missing = [
                                m for m in val.required_metrics
                                if m not in val_metrics
                            ]
                            if missing:
                                result.error = (
                                    "Validation missing required metrics: "
                                    + ", ".join(missing)
                                )
                                result.exit_code = -1
                                fleet_breadcrumb(
                                    f"Validation missing metrics for "
                                    f"arm {arm.name}",
                                    category="fleet.validation",
                                    level="error",
                                    data={
                                        "arm": arm.name,
                                        "missing": missing,
                                    },
                                )
                                logger.warning(
                                    "[%s] Validation missing required "
                                    "metrics: %s",
                                    arm.name, missing,
                                )
                            else:
                                fleet_breadcrumb(
                                    f"Validation passed for arm "
                                    f"{arm.name}",
                                    category="fleet.validation",
                                    data={
                                        "arm": arm.name,
                                        "validation_metrics": val_metrics,
                                    },
                                )
                                logger.info(
                                    "[%s] Validation passed — metrics: %s",
                                    arm.name, val_metrics,
                                )

                elif (
                    result.exit_code != 0
                    and self.spec.validation is not None
                ):
                    fleet_breadcrumb(
                        f"Validation skipped for arm {arm.name} "
                        f"(training exit_code={result.exit_code})",
                        category="fleet.validation",
                        data={
                            "arm": arm.name,
                            "reason": "training_failed",
                            "training_exit_code": result.exit_code,
                        },
                    )

                # --- Retrieve architecture dump before instance teardown ---
                # Use run_result (training exit code) rather than result
                # (which may reflect a later validation failure) so we still
                # persist the dump when training succeeded but validation
                # marked the run invalid.
                if run_result.exit_code == 0:
                    try:
                        arch_result = await remote.run(
                            f"cat {arch_path} 2>/dev/null || true",
                            timeout=10,
                        )
                        if (
                            arch_result.exit_code == 0
                            and arch_result.stdout.strip()
                        ):
                            self._write_architecture(arm.name, arch_result.stdout)
                    except Exception:
                        logger.debug(
                            "[%s] Could not retrieve architecture dump",
                            arm.name,
                        )

                # --- Emit Sentry metrics ---
                arm_duration = time.monotonic() - arm_start
                fleet_metric(
                    "fleet.arm.duration", arm_duration,
                    unit="second", tags=arm_tags,
                )
                fleet_metric(
                    "fleet.arm.exit_code", float(result.exit_code),
                    tags=arm_tags,
                )
                dph = offer.get("dph_total", 0)
                cost = dph * (arm_duration / 3600.0)
                result.estimated_cost = float(cost)
                fleet_metric(
                    "fleet.arm.cost", cost, unit="none", tags=arm_tags,
                )

            except Exception as e:
                result.error = str(e)
                result.exit_code = -1
                logger.exception("[%s] Unexpected error", arm.name)
                fleet_breadcrumb(
                    f"Unexpected error in arm {arm.name}: {str(e)[:100]}",
                    category="fleet.error",
                    level="error",
                    data={"arm": arm.name},
                )
                if sentry_sdk:
                    sentry_sdk.capture_exception(e)

            finally:
                # Make sure estimated_cost reflects the wall-clock duration
                # even when the arm aborted before reaching the metrics block.
                if result.estimated_cost == 0.0:
                    result.estimated_cost = result.instance_dph * (
                        (time.monotonic() - arm_start) / 3600.0
                    )

                fleet_breadcrumb(
                    f"Cleaning up arm {arm.name} (instance {instance_id})",
                    category="fleet.cleanup",
                    data={"arm": arm.name, "instance_id": instance_id},
                )
                if instance_id is not None:
                    try:
                        await client.destroy_instance(instance_id)
                        logger.info(
                            "[%s] Destroyed instance %s",
                            arm.name, instance_id,
                        )
                    except Exception:
                        logger.exception(
                            "[%s] Failed to destroy instance %s",
                            arm.name, instance_id,
                        )

                    # Query Vast.ai billing API for actual charges.
                    # Returns None if API unavailable — callers fall back
                    # to estimated cost in that case.
                    try:
                        actual = await client.get_instance_cost(instance_id)
                    except Exception:
                        logger.exception(
                            "[%s] Failed to query actual cost for instance %s",
                            arm.name, instance_id,
                        )
                        actual = None
                    if actual is not None:
                        result.actual_cost = actual
                        delta = actual - result.estimated_cost
                        logger.info(
                            "[%s] Cost: estimated=$%.3f actual=$%.3f delta=$%+.3f",
                            arm.name, result.estimated_cost, actual, delta,
                        )
                        fleet_metric(
                            "fleet.arm.cost.actual", actual,
                            unit="none", tags=arm_tags,
                        )
                        fleet_metric(
                            "fleet.arm.cost.delta", delta,
                            unit="none", tags=arm_tags,
                        )

        return result

    def _write_arm_log(
        self,
        arm_name: str,
        stdout: str,
        stderr: str,
    ) -> Path | None:
        """Write stdout/stderr to a local log file for the arm.

        Creates ``<log_dir>/<experiment>/<arm>.log`` with combined output.
        Returns the log file path, or None on failure.
        """
        try:
            log_dir = Path(self.config.log_dir) / self.spec.name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{arm_name}.log"
            with log_path.open("w", encoding="utf-8") as f:
                if stdout.strip():
                    f.write("=== STDOUT ===\n")
                    f.write(stdout)
                    f.write("\n")
                if stderr.strip():
                    f.write("=== STDERR ===\n")
                    f.write(stderr)
                    f.write("\n")
            logger.info("[%s] Arm log written to %s", arm_name, log_path)
            return log_path
        except Exception:
            logger.debug("[%s] Failed to write arm log", arm_name, exc_info=True)
            return None

    def _write_architecture(self, arm_name: str, content: str) -> Path | None:
        """Persist the remote resolved_architecture.json locally.

        Creates ``<log_dir>/<experiment>/<arm>.resolved_architecture.json``.
        Best-effort — returns None on any failure.
        """
        try:
            # Sanitize arm_name to prevent path traversal
            safe_name = arm_name.replace("/", "_").replace("\\", "_")
            safe_name = safe_name.replace("..", "_")
            log_dir = Path(self.config.log_dir) / self.spec.name
            log_dir.mkdir(parents=True, exist_ok=True)
            arch_path = log_dir / f"{safe_name}.{ARCHITECTURE_FILENAME}"
            arch_path.write_text(content, encoding="utf-8")
            logger.info(
                "[%s] Architecture dump written to %s", arm_name, arch_path,
            )
            return arch_path
        except Exception:
            logger.debug(
                "[%s] Failed to write architecture dump",
                arm_name, exc_info=True,
            )
            return None

    async def _wait_for_boot(
        self,
        client: VastClient,
        instance_id: int,
        timeout_s: int,
    ) -> tuple[str, int]:
        """Poll until the instance is running. Returns (ssh_host, ssh_port)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                info = await client.get_instance(instance_id)
                if (
                    info.status == InstanceStatus.RUNNING
                    and info.ssh_host
                    and info.ssh_port
                ):
                    return info.ssh_host, info.ssh_port
                if info.status in (InstanceStatus.ERROR, InstanceStatus.EXITED):
                    return "", 0
            except Exception:
                pass
            await asyncio.sleep(BOOT_POLL_INTERVAL_S)
        return "", 0

    def _start_transaction(self, arm_pairs: list) -> Any:
        """Start a Sentry transaction if the SDK is available."""
        if not sentry_sdk:
            return None
        txn = sentry_sdk.start_transaction(
            op="fleet", name=f"fleet/{self.spec.name}"
        )
        txn.__enter__()
        txn.set_data("fleet.experiment", self.spec.name)
        txn.set_data("fleet.num_arms", len(arm_pairs))
        txn.set_data("fleet.image", self.spec.hardware.image)
        txn.set_data("fleet.arm_indices", [i for i, _ in arm_pairs])
        return txn

    def _print_dry_run(
        self, arm_pairs: list, offers: list[dict],
    ) -> None:
        """Print what would be launched without actually launching."""
        print(f"DRY RUN: {self.spec.name}")
        print(f"  Arms: {len(arm_pairs)}")
        print(f"  Offers: {len(offers)}")
        print(f"  Image: {self.spec.hardware.image}")
        print()
        for j, (i, arm) in enumerate(arm_pairs):
            o = offers[j % len(offers)]
            print(
                f"  [{i}] {arm.name}: offer {o['id']} "
                f"(${o.get('dph_total', 0):.3f}/hr, "
                f"PCIe {o.get('pcie_bw', 0):.0f} GB/s)"
            )


def print_cost_summary(
    results: list[ArmResult],
    *,
    budget: float | None = None,
) -> None:
    """Print an estimated-vs-actual cost summary for a fleet run.

    Aggregates ``estimated_cost``, ``actual_cost``, and boot overhead
    across the given results.  Falls back to the estimate when the
    Vast.ai billing API did not return a value for an arm.
    """
    if not results:
        return

    total_estimated = sum(float(r.estimated_cost or 0.0) for r in results)
    arms_with_actual = [r for r in results if r.actual_cost is not None]
    arms_without_actual = [r for r in results if r.actual_cost is None]
    n_actual = len(arms_with_actual)
    n_total = len(results)

    actual_known = sum(float(r.actual_cost) for r in arms_with_actual)
    estimated_fallback = sum(
        float(r.estimated_cost or 0.0) for r in arms_without_actual
    )
    total_actual = actual_known + estimated_fallback

    # Boot overhead: dph * boot_time, summed across arms with known dph.
    boot_overhead = sum(
        float(r.instance_dph or 0.0) * float(r.boot_time_s or 0.0) / 3600.0
        for r in results
    )

    print("\n" + "=" * 60)
    print("Cost Summary:")
    print(f"  Estimated: ${total_estimated:.2f}")
    if n_actual == n_total:
        # Full coverage — straightforward "Actual" line
        delta = total_actual - total_estimated
        if total_estimated > 0:
            pct = 100.0 * delta / total_estimated
            print(f"  Actual:    ${total_actual:.2f} ({pct:+.0f}%)")
        else:
            print(f"  Actual:    ${total_actual:.2f}")
    elif n_actual > 0:
        # Partial coverage — be explicit about what's measured vs estimated
        print(
            f"  Actual:    ${actual_known:.2f} "
            f"({n_actual}/{n_total} arms; ${estimated_fallback:.2f} estimated for the rest)"
        )
    else:
        print("  Actual:    (no billing data available — using estimate)")
    if budget is not None and budget > 0:
        used_pct = 100.0 * total_actual / budget
        print(f"  Budget:    ${budget:.2f} ({used_pct:.0f}% used)")
    if total_actual > 0:
        boot_pct = 100.0 * boot_overhead / total_actual
        print(f"  Boot overhead: ${boot_overhead:.2f} ({boot_pct:.0f}% of total)")
    else:
        print(f"  Boot overhead: ${boot_overhead:.2f}")
    print("=" * 60)


def print_results_table(results: list[ArmResult], baseline_metric: str = "") -> None:
    """Print a formatted comparison table of arm results."""
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS")
    print("=" * 90)

    # Determine metric columns from first successful result
    metric_keys: list[str] = []
    for r in results:
        if r.success and r.metrics:
            metric_keys = list(r.metrics.keys())[:5]
            break

    header_parts = ["Arm", "Status", *metric_keys]
    print(" | ".join(f"{h:<20}" for h in header_parts))
    print("-" * 90)

    for r in results:
        status = "OK" if r.success else "ERROR"
        parts = [f"{r.arm_name:<20}", f"{status:<20}"]
        for k in metric_keys:
            val = r.metrics.get(k, "—")
            if isinstance(val, float):
                parts.append(f"{val:<20.4f}")
            else:
                parts.append(f"{val!s:<20}")
        print(" | ".join(parts))

    print("=" * 90)
