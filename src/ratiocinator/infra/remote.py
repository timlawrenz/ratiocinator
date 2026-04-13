"""Unified remote execution over SSH.

Provides `RemoteExecutor` — a single abstraction for SSH commands, file
transfers (rsync/SCP), and connection readiness checks.  All operations
carry Sentry spans when the SDK is available.

Extracted from patterns common to VastRunner and the fleet orchestrator
to eliminate duplication and provide consistent error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ratiocinator.observability import fleet_breadcrumb

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]


@dataclass
class RemoteResult:
    """Result of a remote command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class RemoteExecutor:
    """Executes commands and transfers files on a remote host via SSH.

    All operations use the same SSH key and host/port, carry Sentry spans
    (when the SDK is available), and have configurable timeouts.

    Usage::

        remote = RemoteExecutor("ssh.vast.ai", 22222, "/home/user/.ssh/id_rsa")
        result = await remote.run("nvidia-smi", timeout=30)
        await remote.rsync_to(Path("./repo"), "/workspace/repo")
    """

    def __init__(
        self,
        host: str,
        port: int,
        ssh_key: str | Path,
        *,
        label: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.ssh_key = str(ssh_key)
        self.label = label or f"{host}:{port}"

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run(
        self,
        command: str,
        *,
        timeout: int = 1800,
        span_op: str | None = None,
    ) -> RemoteResult:
        """Execute a shell command on the remote host via SSH.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds before the process is killed.
            span_op: Optional Sentry span operation name.

        Returns:
            RemoteResult with exit code, stdout, stderr, and duration.
        """
        span = self._start_span(span_op or "ssh.exec", f"ssh {self.label}")
        start = time.monotonic()

        ssh_cmd = [
            "ssh", *SSH_OPTIONS,
            "-i", self.ssh_key,
            "-p", str(self.port),
            f"root@{self.host}",
            command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            result = RemoteResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_seconds=time.monotonic() - start,
            )
        except TimeoutError:
            proc.kill()
            result = RemoteResult(
                exit_code=124,
                stdout="",
                stderr=f"Timed out after {timeout}s",
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            result = RemoteResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

        self._finish_span(span, result)
        return result

    # ------------------------------------------------------------------
    # File transfers
    # ------------------------------------------------------------------

    async def rsync_to(
        self,
        local_path: Path,
        remote_path: str,
        *,
        timeout: int = 300,
        delete: bool = True,
        retries: int = 3,
    ) -> RemoteResult:
        """Transfer a local directory to the remote host via rsync.

        Args:
            local_path: Local directory or file to send.
            remote_path: Remote destination path.
            timeout: Per-attempt timeout in seconds.
            delete: Whether to delete extraneous files on remote.
            retries: Number of retry attempts on failure.
        """
        await self.run(f"mkdir -p {remote_path}", timeout=15)

        delete_flag = ["--delete"] if delete else []
        rsync_cmd = [
            "rsync", "-az", *delete_flag,
            "-e", f"ssh {' '.join(SSH_OPTIONS)} -i {self.ssh_key} -p {self.port}",
            f"{local_path}/",
            f"root@{self.host}:{remote_path}/",
        ]

        span = self._start_span("rsync.to", f"rsync→{self.label}:{remote_path}")
        start = time.monotonic()
        last_result = RemoteResult(exit_code=1, stdout="", stderr="no attempts")

        for attempt in range(retries):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *rsync_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
                last_result = RemoteResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode("utf-8", errors="replace"),
                    stderr=stderr_bytes.decode("utf-8", errors="replace"),
                    duration_seconds=time.monotonic() - start,
                )
                if last_result.success:
                    self._finish_span(span, last_result)
                    return last_result

                if attempt < retries - 1:
                    logger.warning(
                        "rsync attempt %d/%d failed (rc=%d): %s — retrying in 10s",
                        attempt + 1, retries, last_result.exit_code,
                        last_result.stderr[:200],
                    )
                    await asyncio.sleep(10)
            except Exception as e:
                last_result = RemoteResult(
                    exit_code=1, stdout="", stderr=str(e),
                    duration_seconds=time.monotonic() - start,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(10)

        self._finish_span(span, last_result)
        return last_result

    async def rsync_from(
        self,
        remote_path: str,
        local_path: Path,
        *,
        timeout: int = 300,
    ) -> RemoteResult:
        """Download a remote directory to the local filesystem via rsync."""
        local_path.mkdir(parents=True, exist_ok=True)

        rsync_cmd = [
            "rsync", "-az",
            "-e", f"ssh {' '.join(SSH_OPTIONS)} -i {self.ssh_key} -p {self.port}",
            f"root@{self.host}:{remote_path}/",
            f"{local_path}/",
        ]

        span = self._start_span("rsync.from", f"rsync←{self.label}:{remote_path}")
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            result = RemoteResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            result = RemoteResult(
                exit_code=1, stdout="", stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

        self._finish_span(span, result)
        return result

    async def scp_to(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        timeout: int = 60,
    ) -> RemoteResult:
        """Copy a single file to the remote host via SCP."""
        scp_cmd = [
            "scp", *SSH_OPTIONS,
            "-i", self.ssh_key,
            "-P", str(self.port),
            str(local_path),
            f"root@{self.host}:{remote_path}",
        ]

        span = self._start_span("scp.to", f"scp→{self.label}:{remote_path}")
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            result = RemoteResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            result = RemoteResult(
                exit_code=1, stdout="", stderr=str(e),
                duration_seconds=time.monotonic() - start,
            )

        self._finish_span(span, result)
        return result

    async def write_remote_script(
        self,
        script_content: str,
        remote_path: str = "/tmp/ratiocinator_script.sh",
        *,
        timeout: int = 60,
    ) -> RemoteResult:
        """Write a script to the remote host and make it executable.

        Creates a local temp file, SCPs it, and chmod +x.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as f:
            f.write(script_content)
            local_script = f.name

        try:
            result = await self.scp_to(local_script, remote_path, timeout=timeout)
            if not result.success:
                return result
            return await self.run(f"chmod +x {remote_path}", timeout=15)
        finally:
            os.unlink(local_script)

    # ------------------------------------------------------------------
    # Connection readiness
    # ------------------------------------------------------------------

    async def wait_for_ssh(
        self,
        *,
        retries: int = 15,
        interval: float = 10.0,
    ) -> bool:
        """Wait until SSH is accepting connections.

        Returns True if SSH became ready within the retry window.
        """
        start = time.monotonic()
        last_stderr = ""
        for attempt in range(retries):
            try:
                result = await self.run("echo ok", timeout=20)
                if result.success:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "%s: SSH ready after %d attempt(s)", self.label, attempt + 1
                    )
                    fleet_breadcrumb(
                        f"SSH ready for {self.label} after {elapsed:.1f}s "
                        f"({attempt + 1} attempt(s))",
                        category="remote.ssh",
                        data={
                            "host": self.host,
                            "port": self.port,
                            "attempts": attempt + 1,
                            "elapsed_s": round(elapsed, 1),
                        },
                    )
                    return True
                last_stderr = result.stderr[:200]
                if attempt % 5 == 4:
                    logger.info(
                        "%s: SSH attempt %d/%d failed (rc=%d): %s",
                        self.label, attempt + 1, retries,
                        result.exit_code, last_stderr,
                    )
            except Exception:
                pass
            await asyncio.sleep(interval)

        elapsed = time.monotonic() - start
        logger.error(
            "%s: SSH not ready after %d attempts (%ds). Last error: %s",
            self.label, retries, int(retries * interval), last_stderr[:300],
        )
        fleet_breadcrumb(
            f"SSH failed for {self.label} after {elapsed:.1f}s "
            f"({retries} attempts). Last error: {last_stderr[:100]}",
            category="remote.ssh",
            level="error",
            data={
                "host": self.host,
                "port": self.port,
                "attempts": retries,
                "elapsed_s": round(elapsed, 1),
                "last_stderr": last_stderr[:200],
            },
        )
        return False

    # ------------------------------------------------------------------
    # Sentry span helpers
    # ------------------------------------------------------------------

    def _start_span(self, op: str, description: str) -> object | None:
        if not sentry_sdk:
            return None
        span = sentry_sdk.start_span(op=op, name=description)
        span.__enter__()
        span.set_data("remote.host", self.host)
        span.set_data("remote.port", self.port)
        span.set_data("remote.label", self.label)
        return span

    def _finish_span(self, span: object | None, result: RemoteResult) -> None:
        if span is None:
            return
        span.set_data("exit_code", result.exit_code)  # type: ignore[union-attr]
        span.set_data("duration_s", result.duration_seconds)  # type: ignore[union-attr]
        if not result.success:
            span.set_status("internal_error")  # type: ignore[union-attr]
        span.__exit__(None, None, None)  # type: ignore[union-attr]
