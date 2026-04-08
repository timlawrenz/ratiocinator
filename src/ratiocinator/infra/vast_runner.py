"""Vast.ai GPU runner — runs experiments on ephemeral cloud GPU instances.

Implements the same .run() interface as SandboxRunner and LocalRunner,
but provisions a Vast.ai instance, transfers the workspace via SCP,
executes the training command over SSH, and tears down the instance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ratiocinator.config import Config
from ratiocinator.infra.safety import SafetyController
from ratiocinator.infra.vast_client import InstanceStatus, VastClient
from ratiocinator.sandbox.runner import RunResult

logger = logging.getLogger(__name__)

# How long to wait for an instance to boot before giving up
BOOT_TIMEOUT_S = 300
BOOT_POLL_INTERVAL_S = 10

# How long to wait for training to complete
TRAIN_TIMEOUT_S = 1800
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]


class VastRunner:
    """Runs experiments on ephemeral Vast.ai GPU instances.

    Each call to .run() provisions a new instance, transfers the workspace
    via rsync/SCP, executes the command over SSH, and destroys the instance.

    Uses the same RunResult interface as SandboxRunner and LocalRunner.
    """

    def __init__(
        self,
        config: Config,
        ssh_key: Path | None = None,
    ) -> None:
        self.config = config
        self.ssh_key = ssh_key or Path.home() / ".ssh" / "id_rsa"
        self._client: VastClient | None = None
        self._safety: SafetyController | None = None

    def _get_client(self) -> VastClient:
        if self._client is None:
            api_key = self.config.vast.api_key
            if not api_key:
                raise RuntimeError("VAST_API_KEY not set")
            self._client = VastClient(api_key)
        return self._client

    def _get_safety(self) -> SafetyController:
        if self._safety is None:
            self._safety = SafetyController(self.config.safety, self._get_client())
        return self._safety

    def run(
        self,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        """Sync wrapper — only works outside an existing event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.run_async(image, command, repo_path=repo_path, env=env)
            )
        finally:
            loop.close()

    async def run_async(
        self,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        """Async version of run()."""
        return await self._run_async(image, command, repo_path=repo_path, env=env)

    async def _run_async(
        self,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        client = self._get_client()
        safety = self._get_safety()

        start = time.monotonic()
        instance_id = None

        try:
            # 1. Find a GPU offer
            max_dph = self.config.vast.max_dph
            offers = await client.search_offers(max_dph=max_dph, limit=5)
            if not offers:
                return RunResult(
                    exit_code=1, stdout="",
                    stderr=f"No Vast.ai offers found under ${max_dph:.2f}/hr",
                )

            offer = offers[0]
            gpu_name = offer.get("gpu_name", "unknown")
            dph = offer.get("dph_total", 0)
            logger.info("Selected: %s @ $%.3f/hr", gpu_name, dph)

            if not safety.can_launch(dph):
                return RunResult(
                    exit_code=1, stdout="",
                    stderr=f"Budget exceeded: ${safety.estimate_spend():.2f} spent",
                )

            # 2. Launch instance (minimal onstart — just install SSH)
            onstart = "#!/bin/bash\necho 'ready' > /tmp/ratiocinator_ready\n"
            instance_id = await client.create_instance(
                offer_id=offer["id"],
                image=image,
                onstart=onstart,
                label="ratiocinator-search",
                disk_gb=self.config.vast.disk_gb,
            )
            safety.track(instance_id, dph)
            logger.info("Launched instance %s", instance_id)

            # 3. Wait for boot
            ssh_host, ssh_port = await self._wait_for_boot(client, instance_id)
            if not ssh_host:
                return RunResult(
                    exit_code=1, stdout="",
                    stderr="Instance failed to boot within timeout",
                    duration_seconds=time.monotonic() - start,
                )

            logger.info("Instance running: %s:%d", ssh_host, ssh_port)

            # 4. Wait a moment for SSH to be ready
            await self._wait_for_ssh(ssh_host, ssh_port)

            # 5. Transfer workspace via rsync
            if repo_path:
                ok = await self._transfer_workspace(repo_path, ssh_host, ssh_port)
                if not ok:
                    return RunResult(
                        exit_code=1, stdout="",
                        stderr="Failed to transfer workspace to instance",
                        duration_seconds=time.monotonic() - start,
                    )

            # 6. Install deps if configured
            if self.config.vast.install_deps and repo_path:
                await self._install_deps(ssh_host, ssh_port)

            # 7. Run the training command via SSH
            env_exports = ""
            if env:
                env_exports = " ".join(f'{k}="{v}"' for k, v in env.items()) + " "

            remote_cmd = f"cd /workspace && {env_exports}{command}"
            result = await self._ssh_exec(ssh_host, ssh_port, remote_cmd)

            result.duration_seconds = time.monotonic() - start
            return result

        except Exception as e:
            logger.exception("Vast.ai run failed")
            return RunResult(
                exit_code=1, stdout="",
                stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

        finally:
            # Always destroy the instance
            if instance_id is not None:
                try:
                    await client.destroy_instance(instance_id)
                    safety.untrack(instance_id)
                    elapsed = time.monotonic() - start
                    logger.info("Destroyed instance %s after %.0fs", instance_id, elapsed)
                except Exception:
                    logger.exception("Failed to destroy instance %s", instance_id)

    async def _wait_for_boot(
        self, client: VastClient, instance_id: int,
    ) -> tuple[str, int]:
        """Poll until the instance is running. Returns (ssh_host, ssh_port)."""
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                info = await client.get_instance(instance_id)
                if info.status == InstanceStatus.RUNNING and info.ssh_host and info.ssh_port:
                    return info.ssh_host, info.ssh_port
                if info.status in (InstanceStatus.ERROR, InstanceStatus.EXITED):
                    logger.error("Instance %s entered %s state", instance_id, info.status)
                    return "", 0
            except Exception:
                pass
            await asyncio.sleep(BOOT_POLL_INTERVAL_S)
        return "", 0

    async def _wait_for_ssh(self, host: str, port: int, retries: int = 12) -> None:
        """Wait until SSH is accepting connections."""
        for attempt in range(retries):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ssh", *SSH_OPTIONS,
                    "-i", str(self.ssh_key),
                    "-p", str(port),
                    f"root@{host}",
                    "echo ok",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
                if proc.returncode == 0:
                    logger.info("SSH ready after %d attempts", attempt + 1)
                    return
            except Exception:
                pass
            await asyncio.sleep(10)
        logger.warning("SSH not confirmed ready after %d attempts, proceeding anyway", retries)

    async def _transfer_workspace(
        self, repo_path: Path, host: str, port: int,
    ) -> bool:
        """Transfer the workspace to the instance via rsync."""
        # Create /workspace on remote
        await self._ssh_exec(host, port, "mkdir -p /workspace")

        rsync_cmd = [
            "rsync", "-az", "--delete",
            "-e", f"ssh {' '.join(SSH_OPTIONS)} -i {self.ssh_key} -p {port}",
            f"{repo_path}/",
            f"root@{host}:/workspace/",
        ]
        logger.info("Transferring workspace: %s", repo_path)
        proc = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            logger.error("rsync failed: %s", stderr.decode())
            return False
        logger.info("Workspace transferred")
        return True

    async def _install_deps(self, host: str, port: int) -> None:
        """Install Python dependencies on the remote instance."""
        install_cmd = (
            "cd /workspace && "
            "if [ -f requirements.txt ]; then "
            "  pip install -q -r requirements.txt; "
            "elif [ -f pyproject.toml ]; then "
            "  pip install -q -e . 2>/dev/null || echo 'pip install skipped'; "
            "fi"
        )
        result = await self._ssh_exec(host, port, install_cmd)
        if result.exit_code != 0:
            logger.warning("Dep install returned %d: %s", result.exit_code, result.stderr[:500])

    async def _ssh_exec(self, host: str, port: int, command: str) -> RunResult:
        """Execute a command on the remote instance via SSH."""
        ssh_cmd = [
            "ssh", *SSH_OPTIONS,
            "-i", str(self.ssh_key),
            "-p", str(port),
            f"root@{host}",
            command,
        ]
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TRAIN_TIMEOUT_S,
            )
            return RunResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_seconds=time.monotonic() - start,
            )
        except TimeoutError:
            proc.kill()
            return RunResult(
                exit_code=124,
                stdout="",
                stderr=f"SSH command timed out after {TRAIN_TIMEOUT_S}s",
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            return RunResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

    async def close(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
