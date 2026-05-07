"""Local Docker compute provider implementation.

Runs experiment arms in Docker containers on the local machine.
Useful for development, CI, and environments without cloud access.

This provider demonstrates the **template method pattern**: it inherits
the shared orchestration from ``ComputeProvider.run()`` (arm selection,
deduplication, parallel dispatch, result storage) and only implements
the provider-specific ``_run_arm()`` hook.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from pathlib import Path

from ratiocinator.fleet.provider import (
    ComputeProvider,
    ProviderCapability,
    ProviderMeta,
    register_provider,
)
from ratiocinator.fleet.providers.docker.config import DockerProviderConfig
from ratiocinator.fleet.results import ArmResult
from ratiocinator.fleet.spec import ArmSpec, ExperimentSpec, parse_metrics

logger = logging.getLogger(__name__)


@register_provider("docker")
class DockerProvider(ComputeProvider):
    """Execute experiment arms in local Docker containers.

    Each arm runs in an isolated container with the repo cloned inside.
    No cloud APIs, no SSH — just ``docker run``.

    This is the reference implementation of the template method pattern:
    ``run()`` is inherited from ``ComputeProvider`` (handles dedup,
    dispatch, storage); only ``_run_arm()`` is implemented here.
    """

    _META = ProviderMeta(
        name="docker",
        display_name="Local Docker",
        capabilities=ProviderCapability.MULTI_GPU,
        requires_ssh_key=False,
        requires_api_token=False,
    )

    def __init__(self, config: DockerProviderConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    @property
    def meta(self) -> ProviderMeta:
        return self._META

    # ------------------------------------------------------------------
    # The only required hook — everything else is inherited
    # ------------------------------------------------------------------

    async def _run_arm(
        self,
        spec: ExperimentSpec,
        arm_idx: int,
        arm: ArmSpec,
        launch_order: int,
    ) -> ArmResult:
        """Run a single arm in a Docker container."""
        async with self._semaphore:
            return await self._execute_in_container(spec, arm)

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def _print_dry_run(
        self,
        spec: ExperimentSpec,
        arm_pairs: list[tuple[int, ArmSpec]],
    ) -> None:
        image = self.config.image_override or spec.hardware.image
        print(f"\n{'='*60}")
        print(f"DRY RUN: {spec.name} on Local Docker")
        print(f"{'='*60}")
        print(f"  Image:     {image}")
        print(f"  Runtime:   {self.config.runtime}")
        print(f"  Parallel:  {self.config.max_concurrent}")
        print(f"  Arms ({len(arm_pairs)}):")
        for idx, arm in arm_pairs:
            print(f"    [{idx}] {arm.name}: {arm.command[:80]}")
        print(f"{'='*60}\n")

    async def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        warnings: list[str] = []
        if spec.data.source not in ("none", "local"):
            warnings.append(
                f"Data source '{spec.data.source}' may not work in local "
                "Docker without network access to remote storage."
            )
        return warnings

    async def estimate_cost(
        self,
        spec: ExperimentSpec,
        arm_indices: list[int] | None = None,
    ) -> float:
        return 0.0  # Local execution is free

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _execute_in_container(
        self,
        spec: ExperimentSpec,
        arm: ArmSpec,
    ) -> ArmResult:
        """Build and run the Docker container for one arm."""
        result = ArmResult(
            experiment=spec.name,
            arm_name=arm.name,
            description=arm.description,
        )
        arm_start = time.monotonic()
        image = self.config.image_override or spec.hardware.image
        command = spec.resolve_command(arm)

        # Build env flags
        env_flags: list[str] = []
        effective_env = spec.resolve_arm_env(arm)
        for k, v in effective_env.items():
            env_flags.extend(["-e", f"{k}={v}"])

        # Build docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "--runtime", self.config.runtime,
            "--network", self.config.network,
            *env_flags,
            image,
            "bash", "-c",
            self._build_script(spec, arm, command),
        ]

        logger.info("[%s] docker run %s", arm.name, image)

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=spec.budget.train_timeout_s + spec.budget.boot_timeout_s,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            result.exit_code = proc.returncode or 0
            result.duration_seconds = time.monotonic() - arm_start
            result.metrics = parse_metrics(stdout, spec.metrics)

            if result.exit_code != 0:
                result.error = stderr[-2000:] if stderr else stdout[-1000:]

            self._save_log(spec.name, arm.name, stdout, stderr)

        except TimeoutError:
            result.exit_code = -1
            result.error = "Docker container timed out"
            result.duration_seconds = time.monotonic() - arm_start
        except Exception as exc:
            result.exit_code = -1
            result.error = str(exc)
            result.duration_seconds = time.monotonic() - arm_start

        return result

    def _build_script(
        self, spec: ExperimentSpec, arm: ArmSpec, command: str,
    ) -> str:
        """Build the bash script that runs inside the container."""
        lines = ["set -e"]

        # Clone repo
        repo = spec.repo
        lines.append(
            f"git clone --depth {repo.clone_depth} "
            f"--branch {shlex.quote(repo.branch)} "
            f"{shlex.quote(repo.url)} {shlex.quote(repo.remote_path)}"
        )
        lines.append(f"cd {shlex.quote(repo.remote_path)}")

        if repo.commit:
            lines.append(
                f"git fetch --depth 1 origin {shlex.quote(repo.commit)}"
            )
            lines.append(f"git checkout {shlex.quote(repo.commit)}")

        # Install deps
        for cmd in spec.deps.pre_install:
            lines.append(cmd)
        if spec.deps.requirements:
            lines.append(
                f"pip install -q -r {shlex.quote(spec.deps.requirements)}"
            )

        # Run training command
        lines.append(command)

        return " && ".join(lines)

    def _save_log(
        self, experiment: str, arm_name: str, stdout: str, stderr: str,
    ) -> None:
        """Write combined log to disk."""
        log_dir = Path(self.config.log_dir) / experiment
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{arm_name}_docker.log"
        with log_path.open("w", encoding="utf-8") as f:
            if stdout.strip():
                f.write("=== STDOUT ===\n")
                f.write(stdout)
            if stderr.strip():
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
