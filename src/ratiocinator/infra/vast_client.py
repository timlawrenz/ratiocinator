"""Async Vast.ai API client.

Modeled after patterns in timlawrenz/stratum-api — uses httpx async client
with config-driven endpoint management and classified error handling.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"


class InstanceStatus(Enum):
    CREATING = "creating"
    LOADING = "loading"
    RUNNING = "running"
    EXITED = "exited"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class InstanceInfo:
    """Represents a Vast.ai instance."""

    instance_id: int
    status: InstanceStatus
    gpu_name: str = ""
    gpu_ram_gb: float = 0.0
    dph_total: float = 0.0
    ssh_host: str = ""
    ssh_port: int = 0
    actual_status: str = ""
    label: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class VastError(Exception):
    """Vast.ai API error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VastClient:
    """Async HTTP client for the Vast.ai API."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=VAST_API_BASE,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=True,
        )
        # Per-client invoice cache, keyed by (sdate, edate).  A fleet
        # run reconciles cost for every arm in ``finally``; without
        # caching that's an N+1 fetch of the full invoice list.  The
        # cache also de-duplicates concurrent requests via the lock so
        # parallel arm cleanups make at most one HTTP call per window.
        self._invoice_cache: dict[
            tuple[float | None, float | None], list[dict[str, Any]]
        ] = {}
        self._invoice_lock = asyncio.Lock()
        # Remember which (status_code) we've already logged with a
        # traceback so per-arm calls don't spam stderr with stack
        # traces for the same expected auth failure.
        self._invoice_warned_status: set[int | None] = set()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> VastClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def search_offers(
        self,
        gpu_name: str | None = None,
        min_ram_gb: float = 0,
        max_dph: float = 1.0,
        num_gpus: int = 1,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search available GPU offers.

        Args:
            gpu_name: Filter by GPU name (e.g., "RTX 4090").
            min_ram_gb: Minimum GPU VRAM in GB.
            max_dph: Maximum dollars per hour.
            num_gpus: Number of GPUs required.
            limit: Max results to return.
        """
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "num_gpus": {"eq": num_gpus},
            "dph_total": {"lte": max_dph},
            "gpu_ram": {"gte": min_ram_gb * 1024},
            "order": [["dph_total", "asc"]],
            "limit": limit,
            "type": "on-demand",
        }
        if gpu_name:
            query["gpu_name"] = {"eq": gpu_name}

        resp = await self._request("GET", "/bundles", params={"q": _json.dumps(query)})
        return resp.get("offers", [])

    async def create_instance(
        self,
        offer_id: int,
        image: str,
        onstart: str = "",
        env: dict[str, str] | None = None,
        disk_gb: float = 20.0,
        label: str = "",
    ) -> int:
        """Create a new instance from an offer.

        Returns the instance ID.
        """
        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": label,
            "onstart": onstart,
            "runtype": "ssh",
        }
        if env:
            body["env"] = env

        resp = await self._request("PUT", f"/asks/{offer_id}/", json=body)
        instance_id = resp.get("new_contract")
        if not instance_id:
            raise VastError(f"No instance ID in response: {resp}")
        logger.info("Created instance %s from offer %s", instance_id, offer_id)
        return instance_id

    async def get_instance(self, instance_id: int) -> InstanceInfo:
        """Get current status of an instance."""
        resp = await self._request("GET", f"/instances/{instance_id}/")
        # Single-instance endpoint wraps data in {"instances": {...}}
        data = resp.get("instances", resp)
        return self._parse_instance(data)

    async def list_instances(self) -> list[InstanceInfo]:
        """List all active instances."""
        resp = await self._request("GET", "/instances/")
        instances = resp.get("instances", [])
        # list endpoint returns a list; single-instance returns a dict
        if isinstance(instances, dict):
            return [self._parse_instance(instances)]
        return [self._parse_instance(i) for i in instances]

    async def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance."""
        await self._request("DELETE", f"/instances/{instance_id}/")
        logger.info("Destroyed instance %s", instance_id)

    async def request_logs(self, instance_id: int) -> str | None:
        """Request instance logs and return the download URL.

        Vast.ai uploads logs to S3 asynchronously — poll the URL
        after a few seconds.
        """
        resp = await self._request(
            "PUT", f"/instances/request_logs/{instance_id}/", json={}
        )
        return resp.get("result_url")

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        """List SSH keys registered on the account."""
        resp = await self._request("GET", "/ssh/")
        # API returns a list directly or wrapped in a dict
        if isinstance(resp, list):
            return resp
        return resp.get("ssh_keys", [])

    async def add_ssh_key(self, public_key: str) -> dict[str, Any]:
        """Register an SSH public key on the account."""
        resp = await self._request("POST", "/ssh/", json={"ssh_key": public_key})
        logger.info("Registered SSH key on Vast.ai account")
        return resp

    async def get_spending(self) -> dict[str, Any]:
        """Get current spending information."""
        return await self._request("GET", "/users/current/")

    async def get_invoices(
        self,
        start_date: float | None = None,
        end_date: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch invoice/charge entries for the account.

        Args:
            start_date: Optional UNIX timestamp lower bound.
            end_date: Optional UNIX timestamp upper bound.

        Returns:
            List of charge entries.  Each entry typically contains an
            ``amount`` (negative for charges, positive for credits) and
            an ``instance_id`` (when the charge is tied to a specific
            instance).  Returns an empty list if the API is unavailable
            or returns an unexpected payload — callers should treat this
            as "actual cost unknown" and fall back to estimation.
        """
        params: dict[str, Any] = {}
        if start_date is not None:
            params["sdate"] = start_date
        if end_date is not None:
            params["edate"] = end_date

        cache_key = (start_date, end_date)
        # Fast path: already cached.
        cached = self._invoice_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._invoice_lock:
            # Re-check after acquiring lock — another concurrent caller
            # may have already populated the cache for this window.
            cached = self._invoice_cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                resp = await self._request(
                    "GET", "/users/current/invoices/", params=params or None,
                )
            except VastError as e:
                # Auth/permission errors (401/403) are common when the
                # billing scope isn't enabled on the API key.  Log them
                # quietly without a traceback the first time, and
                # silently after that.  Reserve ``exc_info`` for
                # unexpected failures (5xx, network).
                expected = e.status_code in (401, 403, 404)
                if expected:
                    if e.status_code not in self._invoice_warned_status:
                        logger.warning(
                            "Vast.ai invoice API unavailable (HTTP %s) — "
                            "actual cost unknown, falling back to estimate",
                            e.status_code,
                        )
                        self._invoice_warned_status.add(e.status_code)
                else:
                    logger.warning(
                        "Vast.ai invoice API error — actual cost unknown",
                        exc_info=True,
                    )
                # Cache the empty result so we don't keep retrying.
                self._invoice_cache[cache_key] = []
                return []

            # API may return either a list directly or a dict wrapping
            # the entries under one of several documented keys
            # ("invoices", "charges", "items").  Newer Vast.ai builds
            # use "items" — older docs reference the other two.
            entries: list[dict[str, Any]] = []
            if isinstance(resp, list):
                entries = resp
            elif isinstance(resp, dict):
                for key in ("invoices", "charges", "items"):
                    value = resp.get(key)
                    if isinstance(value, list):
                        entries = value
                        break

            self._invoice_cache[cache_key] = entries
            return entries

    def clear_invoice_cache(self) -> None:
        """Drop any cached invoice responses.

        Call between distinct runs (or after charges are expected to
        have settled) to force a fresh fetch on the next
        :meth:`get_invoices` / :meth:`get_instance_cost` call.
        """
        self._invoice_cache.clear()
        self._invoice_warned_status.clear()

    async def get_instance_cost(
        self,
        instance_id: int,
        start_date: float | None = None,
        end_date: float | None = None,
    ) -> float | None:
        """Return the actual billed cost for an instance, or ``None``.

        Sums the absolute value of charge entries from
        :meth:`get_invoices` that reference ``instance_id``.  Returns
        ``None`` if the billing API is unavailable or no entries
        reference the instance — callers should fall back to estimated
        cost in that case.
        """
        invoices = await self.get_invoices(
            start_date=start_date, end_date=end_date,
        )
        if not invoices:
            return None

        # Vast.ai documents ``amount`` as negative for charges and
        # positive for credits/refunds.  Flip the sign so charges add
        # positive cost while refunds subtract.  Clamp the final total
        # at zero — a net-credit instance shouldn't reduce overall
        # tracked spend below what other instances consumed.
        net = 0.0
        matched = False
        for entry in invoices:
            if not isinstance(entry, dict):
                continue
            entry_iid = entry.get("instance_id")
            if entry_iid is None:
                continue
            try:
                if int(entry_iid) != int(instance_id):
                    continue
            except (TypeError, ValueError):
                continue
            amount = entry.get("amount", entry.get("total"))
            if amount is None:
                continue
            try:
                amt_f = float(amount)
            except (TypeError, ValueError):
                continue
            net += -amt_f
            matched = True

        if not matched:
            return None
        return max(0.0, net)

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an API request with error handling."""
        span_ctx = (
            sentry_sdk.start_span(op="http.client", name=f"vast.ai {method} {path}")
            if sentry_sdk
            else None
        )
        if span_ctx:
            span_ctx.__enter__()
            span_ctx.set_data("http.method", method)
            span_ctx.set_data("url", f"{VAST_API_BASE}{path}")

        try:
            resp = await self._client.request(method, path, **kwargs)
            resp.raise_for_status()
            if span_ctx:
                span_ctx.set_data("http.status_code", resp.status_code)
                span_ctx.__exit__(None, None, None)
            return resp.json()
        except httpx.HTTPStatusError as e:
            if span_ctx:
                span_ctx.set_data("http.status_code", e.response.status_code)
                span_ctx.set_status("internal_error")
                span_ctx.__exit__(None, None, None)
            raise VastError(
                f"Vast.ai API error: {e.response.status_code} {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            if span_ctx:
                span_ctx.set_status("internal_error")
                span_ctx.__exit__(None, None, None)
            raise VastError(f"Vast.ai request failed: {e}") from e

    def _parse_instance(self, data: dict[str, Any]) -> InstanceInfo:
        raw_status = data.get("actual_status", "unknown")
        status_map = {
            "created": InstanceStatus.CREATING,
            "loading": InstanceStatus.LOADING,
            "running": InstanceStatus.RUNNING,
            "exited": InstanceStatus.EXITED,
            "error": InstanceStatus.ERROR,
        }
        return InstanceInfo(
            instance_id=data.get("id", 0),
            status=status_map.get(raw_status, InstanceStatus.UNKNOWN),
            gpu_name=data.get("gpu_name", ""),
            gpu_ram_gb=data.get("gpu_ram", 0) / 1024,
            dph_total=data.get("dph_total", 0),
            ssh_host=data.get("ssh_host", ""),
            ssh_port=data.get("ssh_port", 0),
            actual_status=raw_status,
            label=data.get("label", ""),
            raw=data,
        )


# --- Retry utilities (following stratum-api patterns) ---

MAX_RETRIES = 3
BASE_DELAY_S = 2.0
MAX_DELAY_S = 120.0


def compute_backoff(retry_count: int) -> float:
    """Exponential backoff with jitter."""
    delay = min(BASE_DELAY_S * (2**retry_count), MAX_DELAY_S)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter
