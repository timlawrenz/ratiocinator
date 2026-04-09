"""Sandbox runners for running experiments in isolation."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

from ratiocinator.config import SandboxConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a sandboxed experiment run."""

    exit_code: int
    stdout: str
    stderr: str
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class SandboxRunner:
    """Runs experiment code inside Docker containers."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._client = None

    @property
    def client(self):
        import docker

        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def run(
        self,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        """Run a command in a Docker container and capture results.

        Args:
            image: Docker image to use (e.g., "python:3.11-slim").
            command: Shell command to execute inside the container.
            repo_path: Local repo to mount at /workspace inside the container.
            env: Environment variables to pass to the container.
        """
        import docker  # noqa: F401 (triggers install check)
        from docker.errors import ContainerError, ImageNotFound

        volumes = {}
        if repo_path:
            volumes[str(repo_path.resolve())] = {"bind": "/workspace", "mode": "rw"}

        container_env = {"PYTHONUNBUFFERED": "1"}
        if env:
            container_env.update(env)

        start = time.monotonic()
        try:
            span = (
                sentry_sdk.start_span(op="sandbox.docker", name=f"docker.run({image})")
                if sentry_sdk
                else None
            )
            if span:
                span.__enter__()
                span.set_data("sandbox.image", image)
                span.set_data("sandbox.command", command)
                span.set_data("sandbox.memory_limit", self.config.memory_limit)

            output = self.client.containers.run(
                image,
                command=f"sh -c {command!r}",
                volumes=volumes,
                environment=container_env,
                working_dir="/workspace" if repo_path else None,
                mem_limit=self.config.memory_limit,
                stdout=True,
                stderr=True,
                remove=True,
                timeout=self.config.timeout_seconds,
            )
            duration = time.monotonic() - start
            stdout = (
                output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
            )
            if span:
                span.set_data("sandbox.exit_code", 0)
                span.set_data("sandbox.duration_s", duration)
                span.__exit__(None, None, None)
            return RunResult(exit_code=0, stdout=stdout, stderr="", duration_seconds=duration)

        except ContainerError as e:
            duration = time.monotonic() - start
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            stdout = e.container.logs().decode("utf-8", errors="replace") if e.container else ""
            if span:
                span.set_data("sandbox.exit_code", e.exit_status)
                span.set_status("internal_error")
                span.__exit__(None, None, None)
            return RunResult(
                exit_code=e.exit_status,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        except ImageNotFound:
            logger.error("Docker image not found: %s", image)
            if span:
                span.set_status("not_found")
                span.__exit__(None, None, None)
            return RunResult(exit_code=127, stdout="", stderr=f"Image not found: {image}")

        except Exception as e:
            duration = time.monotonic() - start
            logger.exception("Sandbox run failed")
            if span:
                span.set_status("internal_error")
                span.__exit__(None, None, None)
            return RunResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )

    def build_image(self, path: Path, tag: str) -> str:
        """Build a Docker image from a directory containing a Dockerfile."""
        logger.info("Building image %s from %s", tag, path)
        _image, _logs = self.client.images.build(path=str(path), tag=tag, rm=True)
        return tag


class LocalRunner:
    """Runs experiments as local subprocesses (no Docker required).

    Useful for smoke testing and development when Docker is unavailable.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def run(
        self,
        image: str,
        command: str,
        *,
        repo_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        """Run a command as a local subprocess.

        The `image` parameter is ignored (kept for interface compatibility).
        """
        import os

        run_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if env:
            run_env.update(env)

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["sh", "-c", command],
                cwd=repo_path,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            duration = time.monotonic() - start
            return RunResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start
            return RunResult(
                exit_code=124,
                stdout=e.stdout or "",
                stderr=f"Timeout after {self.config.timeout_seconds}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return RunResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )
