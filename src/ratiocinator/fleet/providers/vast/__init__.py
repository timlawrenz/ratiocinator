"""Vast.ai compute provider — SSH-based GPU instance execution."""

from ratiocinator.fleet.providers.vast.config import VastProviderConfig
from ratiocinator.fleet.providers.vast.executor import VastProvider

__all__ = ["VastProvider", "VastProviderConfig"]
