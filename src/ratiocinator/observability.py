"""Sentry observability: logging, tracing, and metrics.

Initialises the Sentry SDK once and exposes helpers for creating
spans and recording custom metrics across the codebase.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SENTRY_DSN = (
    "https://aa13b4120c65783055a01e636a708bd7"
    "@o213028.ingest.us.sentry.io/4511191538139136"
)

_initialised = False


def init_sentry(
    *,
    dsn: str | None = None,
    environment: str | None = None,
    release: str | None = None,
    traces_sample_rate: float = 1.0,
    profiles_sample_rate: float = 0.1,
) -> bool:
    """Initialise the Sentry SDK (idempotent — safe to call multiple times).

    Returns True if Sentry was (or already had been) initialised.
    """
    global _initialised
    if _initialised:
        return True

    try:
        import sentry_sdk
    except ImportError:
        logger.debug("sentry-sdk not installed — observability disabled")
        return False

    resolved_dsn = dsn or os.environ.get("SENTRY_DSN") or _SENTRY_DSN
    resolved_env = environment or os.environ.get("SENTRY_ENVIRONMENT", "development")
    resolved_release = release or _get_release()

    sentry_sdk.init(
        dsn=resolved_dsn,
        environment=resolved_env,
        release=resolved_release,
        enable_logs=True,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        send_default_pii=False,
        enable_tracing=True,
    )

    _initialised = True
    logger.debug("Sentry initialised (env=%s, release=%s)", resolved_env, resolved_release)
    return True


def _get_release() -> str | None:
    """Derive a release string from git or package metadata."""
    # 1. Env override
    if rel := os.environ.get("SENTRY_RELEASE"):
        return rel

    # 2. Git short SHA
    try:
        import subprocess

        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"ratiocinator@{sha}" if sha else None
    except Exception:
        pass

    # 3. Package version
    try:
        from importlib.metadata import version

        return f"ratiocinator@{version('ratiocinator')}"
    except Exception:
        return None
