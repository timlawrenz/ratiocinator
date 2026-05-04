"""Vast.ai provider configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ratiocinator.fleet.provider import ProviderConfig


class VastProviderConfig(BaseModel, ProviderConfig):
    """Runtime configuration for the Vast.ai SSH-based provider.

    Credentials:
        api_key: Vast.ai API key (from env ``VAST_API_KEY`` or config).
        ssh_key: Path to SSH private key for instance access.

    Tuning:
        max_concurrent: Max parallel instances to provision.
        stagger_seconds: Delay between instance creation requests.

    Output:
        results_path: JSON file for persisting experiment results.
        log_dir: Directory for per-arm stdout/stderr logs.
    """

    api_key: str
    ssh_key: str = str(Path.home() / ".ssh" / "id_rsa")
    max_concurrent: int = 7
    stagger_seconds: float = 5.0
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("VastProviderConfig requires a non-empty api_key")
        if not self.ssh_key:
            raise ValueError("VastProviderConfig requires a non-empty ssh_key")
