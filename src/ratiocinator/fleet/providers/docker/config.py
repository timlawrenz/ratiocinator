"""Local Docker provider configuration."""

from __future__ import annotations

from pydantic import BaseModel

from ratiocinator.fleet.provider import ProviderConfig


class DockerProviderConfig(BaseModel, ProviderConfig):
    """Runtime configuration for local Docker execution.

    No cloud credentials required — experiments run on the local machine
    inside Docker containers.

    Tuning:
        runtime: Docker runtime (``"nvidia"`` for GPU, ``"runc"`` for CPU).
        network: Docker network mode.
        max_concurrent: Max parallel containers.
        image_override: Override the image from ``spec.hardware.image``.

    Output:
        results_path: JSON file for persisting experiment results.
        log_dir: Directory for per-arm logs.
    """

    runtime: str = "nvidia"
    network: str = "host"
    max_concurrent: int = 2
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"
    image_override: str = ""

    def validate(self) -> None:
        # No mandatory credentials for local Docker
        pass
