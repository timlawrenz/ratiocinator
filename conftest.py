"""Test-suite guards.

The suite deliberately exercises failure paths (budget exceeded, instance
boot timeouts, refused SSH). Those must never reach a real Sentry project,
so the DSN is forced empty for the whole session regardless of whatever is
set in the ambient environment.
"""

import os

os.environ["SENTRY_DSN"] = ""
os.environ["SENTRY_ENVIRONMENT"] = "test"
