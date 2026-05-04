"""Provider-specific configuration models.

Each compute provider declares its own Pydantic config model here.
These are instantiated from the global ``Config`` object or from
``ExperimentSpec.provider_config`` dicts at runtime.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ratiocinator.fleet.provider import ProviderConfig

# ---------------------------------------------------------------------------
# Vast.ai
# ---------------------------------------------------------------------------


class VastProviderConfig(BaseModel, ProviderConfig):
    """Configuration for the Vast.ai SSH-based provider."""

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


# ---------------------------------------------------------------------------
# HuggingFace Jobs
# ---------------------------------------------------------------------------


class HFProviderConfig(BaseModel, ProviderConfig):
    """Configuration for the HuggingFace Jobs provider."""

    token: str
    namespace: str = ""
    bucket_prefix: str = ""
    max_timeout: str = "4h"
    flavor: str = ""
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"

    def validate(self) -> None:
        if not self.token:
            raise ValueError("HFProviderConfig requires a non-empty token")


# ---------------------------------------------------------------------------
# Local Docker (proof-of-concept)
# ---------------------------------------------------------------------------


class DockerProviderConfig(BaseModel, ProviderConfig):
    """Configuration for local Docker execution (no cloud, no cost)."""

    runtime: str = "nvidia"
    network: str = "host"
    max_concurrent: int = 2
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"
    # Base Docker image override (if not specified, uses spec.hardware.image)
    image_override: str = ""

    def validate(self) -> None:
        # Docker provider has no mandatory credentials
        pass


# ---------------------------------------------------------------------------
# fal.ai Serverless (stub)
# ---------------------------------------------------------------------------


class FalProviderConfig(BaseModel, ProviderConfig):
    """Configuration for fal.ai serverless GPU functions."""

    api_key: str = ""
    machine_type: str = "GPU_A100"
    max_concurrent: int = 10
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("FalProviderConfig requires a non-empty api_key")


# ---------------------------------------------------------------------------
# Config resolution helper
# ---------------------------------------------------------------------------

# Maps provider name → config class for factory construction.
PROVIDER_CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    "vast": VastProviderConfig,
    "hf": HFProviderConfig,
    "docker": DockerProviderConfig,
    "fal": FalProviderConfig,
}


def resolve_provider_config(
    provider_name: str,
    raw: dict | None = None,
) -> ProviderConfig:
    """Instantiate a typed ProviderConfig from a raw dict.

    Args:
        provider_name: Registry key (e.g. ``"vast"``).
        raw: Dict of config values (from YAML ``provider_config:`` block).

    Returns:
        Validated ProviderConfig instance.

    Raises:
        ValueError: Unknown provider or validation failure.
    """
    if provider_name not in PROVIDER_CONFIG_CLASSES:
        available = ", ".join(sorted(PROVIDER_CONFIG_CLASSES.keys()))
        raise ValueError(
            f"No config class for provider '{provider_name}'. "
            f"Available: {available}"
        )

    cls = PROVIDER_CONFIG_CLASSES[provider_name]
    instance = cls.model_validate(raw or {})
    instance.validate()  # type: ignore[attr-defined]
    return instance  # type: ignore[return-value]
