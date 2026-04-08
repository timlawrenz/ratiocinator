"""Docker-based sandbox for running experiments in isolation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound

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
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
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
        volumes = {}
        if repo_path:
            volumes[str(repo_path.resolve())] = {"bind": "/workspace", "mode": "rw"}

        container_env = {"PYTHONUNBUFFERED": "1"}
        if env:
            container_env.update(env)

        start = time.monotonic()
        try:
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
            return RunResult(exit_code=0, stdout=stdout, stderr="", duration_seconds=duration)

        except ContainerError as e:
            duration = time.monotonic() - start
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            stdout = e.container.logs().decode("utf-8", errors="replace") if e.container else ""
            return RunResult(
                exit_code=e.exit_status,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        except ImageNotFound:
            logger.error("Docker image not found: %s", image)
            return RunResult(exit_code=127, stdout="", stderr=f"Image not found: {image}")

        except Exception as e:
            duration = time.monotonic() - start
            logger.exception("Sandbox run failed")
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
