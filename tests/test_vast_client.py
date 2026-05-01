"""Tests for the Vast.ai client."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ratiocinator.infra.vast_client import (
    InstanceStatus,
    VastClient,
    VastError,
    compute_backoff,
)


class TestVastClientParsing:
    def test_parse_running_instance(self):
        client = VastClient.__new__(VastClient)
        data = {
            "id": 12345,
            "actual_status": "running",
            "gpu_name": "RTX 4090",
            "gpu_ram": 24576,
            "dph_total": 0.45,
            "ssh_host": "ssh5.vast.ai",
            "ssh_port": 22222,
            "label": "ratiocinator-exp-1",
        }
        info = client._parse_instance(data)
        assert info.instance_id == 12345
        assert info.status == InstanceStatus.RUNNING
        assert info.gpu_name == "RTX 4090"
        assert info.gpu_ram_gb == 24.0
        assert info.dph_total == 0.45
        assert info.label == "ratiocinator-exp-1"

    def test_parse_unknown_status(self):
        client = VastClient.__new__(VastClient)
        data = {"id": 1, "actual_status": "weird"}
        info = client._parse_instance(data)
        assert info.status == InstanceStatus.UNKNOWN

    def test_parse_missing_fields(self):
        client = VastClient.__new__(VastClient)
        data = {}
        info = client._parse_instance(data)
        assert info.instance_id == 0
        assert info.gpu_ram_gb == 0.0


class TestVastError:
    def test_error_with_status_code(self):
        err = VastError("test error", status_code=403)
        assert err.status_code == 403
        assert "test error" in str(err)

    def test_error_without_status_code(self):
        err = VastError("network failure")
        assert err.status_code is None


class TestBackoff:
    def test_backoff_increases(self):
        b0 = compute_backoff(0)
        b1 = compute_backoff(1)
        b2 = compute_backoff(2)
        # Base values (before jitter): 2, 4, 8
        assert b0 < 3.0  # 2 + up to 0.5 jitter
        assert b1 < 6.0  # 4 + up to 1.0 jitter
        assert b2 < 12.0  # 8 + up to 2.0 jitter

    def test_backoff_capped(self):
        b = compute_backoff(100)
        assert b <= 150.0  # 120 + 25% jitter


class TestBillingAPI:
    """Tests for the billing/invoice API methods used by cost tracking."""

    @staticmethod
    def _make_client() -> VastClient:
        """Build a VastClient with cache attrs but no real HTTP client."""
        import asyncio as _asyncio
        client = VastClient.__new__(VastClient)
        client._invoice_cache = {}
        client._invoice_lock = _asyncio.Lock()
        client._invoice_warned_status = set()
        return client

    @pytest.mark.asyncio
    async def test_get_invoices_returns_list(self):
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[{"instance_id": 1, "amount": -0.50}]
        )
        invoices = await client.get_invoices()
        assert invoices == [{"instance_id": 1, "amount": -0.50}]

    @pytest.mark.asyncio
    async def test_get_invoices_unwraps_dict(self):
        client = self._make_client()
        client._request = AsyncMock(
            return_value={"invoices": [{"instance_id": 2, "amount": -0.25}]}
        )
        invoices = await client.get_invoices()
        assert invoices == [{"instance_id": 2, "amount": -0.25}]

    @pytest.mark.asyncio
    async def test_get_invoices_handles_api_error(self):
        client = self._make_client()
        client._request = AsyncMock(side_effect=VastError("boom", status_code=500))
        invoices = await client.get_invoices()
        assert invoices == []

    @pytest.mark.asyncio
    async def test_get_invoices_caches_response(self):
        """Repeated calls hit the API at most once per (sdate, edate) window."""
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[{"instance_id": 1, "amount": -0.10}]
        )
        await client.get_invoices()
        await client.get_invoices()
        await client.get_invoices()
        assert client._request.call_count == 1
        # clear_invoice_cache forces a refetch
        client.clear_invoice_cache()
        await client.get_invoices()
        assert client._request.call_count == 2

    @pytest.mark.asyncio
    async def test_get_invoices_quiet_on_auth_error(self, caplog):
        """401/403 errors should not log a stack trace, and only warn once."""
        import logging
        client = self._make_client()
        client._request = AsyncMock(side_effect=VastError("forbidden", status_code=403))
        with caplog.at_level(logging.WARNING):
            await client.get_invoices()
            # Force a second call by clearing the cache
            client._invoice_cache.clear()
            await client.get_invoices()
        # Stack trace would include the word "Traceback"; we shouldn't see it.
        assert "Traceback" not in caplog.text
        # Only one warning should have been emitted for the 403.
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_get_instance_cost_sums_matching_charges(self):
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[
                {"instance_id": 100, "amount": -0.40},
                {"instance_id": 100, "amount": -0.10},
                {"instance_id": 200, "amount": -1.00},
                {"instance_id": 100, "amount": "-0.05"},  # string amount
                {"amount": -1.00},  # no instance_id
            ]
        )
        cost = await client.get_instance_cost(100)
        assert cost == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_get_instance_cost_no_match_returns_none(self):
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[{"instance_id": 999, "amount": -1.00}]
        )
        assert await client.get_instance_cost(123) is None

    @pytest.mark.asyncio
    async def test_get_instance_cost_api_unavailable_returns_none(self):
        client = self._make_client()
        client._request = AsyncMock(side_effect=VastError("forbidden", status_code=403))
        assert await client.get_instance_cost(123) is None

    @pytest.mark.asyncio
    async def test_get_instance_cost_charges_negative_credits_positive(self):
        """Charges are negative; credits/refunds are positive — sign matters."""
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[
                {"instance_id": 100, "amount": -1.00},  # charge
                {"instance_id": 100, "amount": 0.20},  # refund
            ]
        )
        cost = await client.get_instance_cost(100)
        # 1.00 charged, 0.20 refunded → net 0.80
        assert cost == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_get_instance_cost_clamped_at_zero_for_net_credit(self):
        client = self._make_client()
        client._request = AsyncMock(
            return_value=[
                {"instance_id": 100, "amount": -0.30},
                {"instance_id": 100, "amount": 1.00},  # large refund
            ]
        )
        cost = await client.get_instance_cost(100)
        assert cost == 0.0

