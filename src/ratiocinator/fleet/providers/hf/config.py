"""HuggingFace Jobs provider configuration."""

from __future__ import annotations

from pydantic import BaseModel

from ratiocinator.fleet.provider import ProviderConfig


class HFProviderConfig(BaseModel, ProviderConfig):
    """Runtime configuration for the HuggingFace Jobs provider.

    Credentials:
        token: HuggingFace API token (from env ``HF_TOKEN`` or config).

    HF-specific:
        namespace: HF username or org for job ownership.
        bucket_prefix: Prefix for output bucket names.
        flavor: Hardware flavor string (e.g. ``"a100-large"``).
        max_timeout: Maximum job duration (e.g. ``"4h"``, ``"2d"``).

    Output:
        results_path: JSON file for persisting experiment results.
        log_dir: Directory for per-arm logs.
    """

    token: str
    namespace: str = ""
    bucket_prefix: str = ""
    flavor: str = ""
    max_timeout: str = "4h"
    results_path: str = ".ratiocinator/results/experiments.json"
    log_dir: str = ".ratiocinator/results"

    def validate(self) -> None:
        if not self.token:
            raise ValueError("HFProviderConfig requires a non-empty token")
