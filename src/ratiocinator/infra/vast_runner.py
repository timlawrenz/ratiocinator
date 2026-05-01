"""Vast.ai GPU runner — runs experiments on ephemeral cloud GPU instances.

Implements the same .run() interface as SandboxRunner and LocalRunner,
but provisions a Vast.ai instance, transfers the workspace via rsync,
executes the training command over SSH, and tears down the instance.

Uses `RemoteExecutor` for all SSH/rsync operations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ratiocinator.config import Config
from ratiocinator.infra.remote import RemoteExecutor
from ratiocinator.infra.safety import SafetyController
from ratiocinator.infra.vast_client import InstanceStatus, VastClient, VastError
from ratiocinator.sandbox.runner import RunResult

logger = logging.getLogger(__name__)

BOOT_TIMEOUT_S = 600
BOOT_POLL_INTERVAL_S = 10
TRAIN_TIMEOUT_S = 1800


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
        self._ssh_key_registered = False

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

    async def _ensure_ssh_key_registered(self, client: VastClient) -> None:
        """Ensure the local SSH public key is registered on the Vast.ai account."""
        if self._ssh_key_registered:
            return

        pub_key_path = Path(str(self.ssh_key) + ".pub")
        if not pub_key_path.exists():
            logger.warning("No public key at %s — SSH may fail", pub_key_path)
            return

        local_pub = pub_key_path.read_text().strip()
        local_parts = local_pub.split()[:2]

        try:
            registered = await client.list_ssh_keys()
            for key_info in registered:
                stored_key = (
                    key_info.get("public_key", "")
                    or key_info.get("ssh_key", "")
                ).strip()
                stored_parts = stored_key.split()[:2]
                if stored_parts == local_parts:
                    logger.info("SSH key already registered on Vast.ai")
                    self._ssh_key_registered = True
                    return

            logger.info("Registering SSH key on Vast.ai account...")
            await client.add_ssh_key(local_pub)
            self._ssh_key_registered = True
        except VastError as e:
            if "duplicate" in str(e).lower():
                logger.info("SSH key already registered (confirmed by API)")
                self._ssh_key_registered = True
            else:
                logger.warning(
                    "Could not verify/register SSH key — SSH may fail",
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "Could not verify/register SSH key — SSH may fail",
                exc_info=True,
            )

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
        max_instance_retries: int = 3,
    ) -> RunResult:
        client = self._get_client()
        safety = self._get_safety()

        await self._ensure_ssh_key_registered(client)

        start = time.monotonic()

        max_dph = self.config.vast.max_dph
        offers = await client.search_offers(max_dph=max_dph, limit=5)
        if not offers:
            return RunResult(
                exit_code=1, stdout="",
                stderr=f"No Vast.ai offers found under ${max_dph:.2f}/hr",
            )

        last_error = ""
        tried_offer_ids: set[int] = set()

        for attempt in range(max_instance_retries):
            offer = None
            for o in offers:
                if o["id"] not in tried_offer_ids:
                    offer = o
                    break
            if offer is None:
                offers = await client.search_offers(max_dph=max_dph, limit=10)
                offer = next((o for o in offers if o["id"] not in tried_offer_ids), None)
                if offer is None:
                    break

            tried_offer_ids.add(offer["id"])
            result = await self._try_instance(
                client, safety, offer, image, command,
                repo_path=repo_path, env=env, start=start,
            )
            if result is not None:
                return result

            last_error = f"Instance attempt {attempt + 1} failed (host issue)"
            if attempt < max_instance_retries - 1:
                logger.info("Retrying with a different instance...")

        return RunResult(
            exit_code=1, stdout="",
            stderr=f"All {max_instance_retries} instance attempts failed. {last_error}",
            duration_seconds=time.monotonic() - start,
        )

    async def _try_instance(
        self,
        client: VastClient,
        safety: SafetyController,
        offer: dict,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
        start: float,
    ) -> RunResult | None:
        """Try to run on a single Vast.ai instance.

        Returns RunResult on success (including command failure),
        or None if the instance itself was unusable and we should retry.
        """
        instance_id = None
        gpu_name = offer.get("gpu_name", "unknown")
        dph = offer.get("dph_total", 0)
        logger.info("Selected: %s @ $%.3f/hr", gpu_name, dph)

        if not safety.can_launch(dph):
            return RunResult(
                exit_code=1, stdout="",
                stderr=f"Budget exceeded: ${safety.estimate_spend():.2f} spent",
            )

        try:
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

            # Wait for boot
            ssh_host, ssh_port = await self._wait_for_boot(client, instance_id)
            if not ssh_host:
                logger.warning("Instance %s failed to boot, will retry", instance_id)
                return None

            logger.info("Instance running: %s:%d", ssh_host, ssh_port)

            # Create RemoteExecutor for all subsequent operations
            remote = RemoteExecutor(
                ssh_host, ssh_port, str(self.ssh_key),
                label=f"vast-{instance_id}",
            )

            # Wait for SSH
            ssh_ready = await remote.wait_for_ssh()
            if not ssh_ready:
                logger.warning("SSH failed on %s:%d, will retry", ssh_host, ssh_port)
                return None

            # Transfer workspace
            if repo_path:
                rsync_result = await remote.rsync_to(repo_path, "/workspace")
                if not rsync_result.success:
                    return RunResult(
                        exit_code=1, stdout="",
                        stderr="Failed to transfer workspace to instance",
                        duration_seconds=time.monotonic() - start,
                    )

            # Install deps if configured
            if self.config.vast.install_deps and repo_path:
                install_cmd = (
                    "cd /workspace && "
                    "if [ -f requirements.txt ]; then "
                    "  pip install -q -r requirements.txt; "
                    "elif [ -f pyproject.toml ]; then "
                    "  pip install -q -e . 2>/dev/null || echo 'pip install skipped'; "
                    "fi"
                )
                dep_result = await remote.run(install_cmd, timeout=300)
                if dep_result.exit_code != 0:
                    logger.warning(
                        "Dep install returned %d: %s",
                        dep_result.exit_code, dep_result.stderr[:500],
                    )

            # Run the training command
            env_exports = ""
            if env:
                env_exports = " ".join(f'{k}="{v}"' for k, v in env.items()) + " "

            remote_cmd = f"cd /workspace && {env_exports}{command}"
            logger.info("Running: %s", remote_cmd)
            exec_result = await remote.run(remote_cmd, timeout=TRAIN_TIMEOUT_S)

            if exec_result.exit_code != 0:
                logger.warning(
                    "Command exited %d, stderr: %s",
                    exec_result.exit_code, exec_result.stderr[:500],
                )
            else:
                stdout_tail = exec_result.stdout[-200:] if exec_result.stdout else "(empty)"
                logger.info("Command completed, stdout tail: %s", stdout_tail)

            return RunResult(
                exit_code=exec_result.exit_code,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                duration_seconds=time.monotonic() - start,
            )

        except Exception as e:
            logger.exception("Vast.ai run failed")
            return RunResult(
                exit_code=1, stdout="",
                stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

        finally:
            if instance_id is not None:
                destroyed_ok = False
                try:
                    await client.destroy_instance(instance_id)
                    destroyed_ok = True
                    elapsed = time.monotonic() - start
                    logger.info("Destroyed instance %s after %.0fs", instance_id, elapsed)
                except Exception:
                    logger.exception("Failed to destroy instance %s", instance_id)

                # Reconcile actual billed cost from Vast.ai (regardless
                # of whether destroy succeeded — a still-running
                # instance is the strongest reason to keep accurate
                # spend).  Falls through silently when the billing API
                # is unavailable.
                try:
                    actual = await client.get_instance_cost(instance_id)
                except Exception:
                    logger.exception(
                        "Failed to query actual cost for instance %s", instance_id,
                    )
                    actual = None
                if actual is not None:
                    safety.set_actual_cost(instance_id, actual)

                # Only drop the instance from SafetyController tracking
                # when we've confirmed it's gone.  Otherwise leave it
                # tracked so TTL/budget enforcement (or shutdown
                # cleanup) can still find and kill the orphan.
                if destroyed_ok:
                    safety.untrack(instance_id)
                else:
                    logger.warning(
                        "Instance %s left tracked because destroy failed; "
                        "TTL/cleanup will retry",
                        instance_id,
                    )

    async def _wait_for_boot(
        self, client: VastClient, instance_id: int,
    ) -> tuple[str, int]:
        """Poll until the instance is running. Returns (ssh_host, ssh_port)."""
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        last_status = ""
        while time.monotonic() < deadline:
            try:
                info = await client.get_instance(instance_id)
                if info.actual_status != last_status:
                    elapsed = BOOT_TIMEOUT_S - (deadline - time.monotonic())
                    logger.info(
                        "Instance %s: %s (%.0fs)",
                        instance_id, info.actual_status, elapsed,
                    )
                    last_status = info.actual_status
                if info.status == InstanceStatus.RUNNING and info.ssh_host and info.ssh_port:
                    return info.ssh_host, info.ssh_port
                if info.status in (InstanceStatus.ERROR, InstanceStatus.EXITED):
                    logger.error("Instance %s entered %s state", instance_id, info.status)
                    return "", 0
            except Exception:
                pass
            await asyncio.sleep(BOOT_POLL_INTERVAL_S)
        logger.error(
            "Instance %s boot timeout after %ds (last: %s)",
            instance_id, BOOT_TIMEOUT_S, last_status,
        )
        return "", 0

    async def close(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
