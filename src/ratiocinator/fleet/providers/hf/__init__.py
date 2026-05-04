"""HuggingFace Jobs compute provider — API-based container execution."""

from ratiocinator.fleet.providers.hf.config import HFProviderConfig
from ratiocinator.fleet.providers.hf.executor import HFProvider

__all__ = ["HFProvider", "HFProviderConfig"]
