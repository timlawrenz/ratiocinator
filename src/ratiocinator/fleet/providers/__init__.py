"""Auto-discovery for compute providers.

Importing this package triggers registration of all built-in providers.
Third-party providers can register themselves by calling
``register_provider()`` from ``ratiocinator.fleet.provider`` at import time.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Import each provider sub-package to trigger @register_provider decoration.
# Use try/except so missing optional deps don't break the whole package.

try:
    from ratiocinator.fleet.providers import vast  # noqa: F401
except ImportError:  # pragma: no cover
    _logger.debug("vast provider unavailable (missing deps)")

try:
    from ratiocinator.fleet.providers import hf  # noqa: F401
except ImportError:  # pragma: no cover
    _logger.debug("hf provider unavailable (missing deps)")

# Docker has no external deps — always available.
from ratiocinator.fleet.providers import docker  # noqa: F401, E402
