"""Tests for the safety controller."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from ratiocinator.config import SafetyConfig
from ratiocinator.infra.safety import SafetyController
from ratiocinator.infra.vast_client import VastClient


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=VastClient)
    client.destroy_instance = AsyncMock()
    return client


@pytest.fixture
def controller(mock_client):
    config = SafetyConfig(max_dollars_per_run=5.0, instance_ttl_seconds=60)
    return SafetyController(config, mock_client)


class TestTracking:
    def test_track_and_untrack(self, controller):
        controller.track(123, dph=0.5)
        assert controller.active_count == 1
        controller.untrack(123)
        assert controller.active_count == 0

    def test_untrack_nonexistent(self, controller):
        controller.untrack(999)  # Should not raise


class TestBudget:
    def test_can_launch_within_budget(self, controller):
        assert controller.can_launch(dph=0.5) is True

    def test_can_launch_over_budget(self, controller):
        controller._total_spent = 4.9
        assert controller.can_launch(dph=0.5) is False

    def test_estimate_spend(self, controller):
        controller.track(1, dph=1.0)
        # Manipulate creation time to simulate 1 hour ago
        controller._tracked[1].created_at = time.time() - 3600
        spend = controller.estimate_spend()
        assert 0.9 < spend < 1.1  # ~$1 for 1 hour at $1/hr

    def test_budget_remaining(self, controller):
        assert controller.budget_remaining == 5.0
        controller._total_spent = 3.0
        assert controller.budget_remaining == 2.0


class TestTTL:
    def test_get_expired_none(self, controller):
        controller.track(1, dph=0.5)
        assert controller.get_expired() == []

    def test_get_expired(self, controller):
        controller.track(1, dph=0.5)
        controller._tracked[1].created_at = time.time() - 120  # 2 min, TTL=60s
        assert controller.get_expired() == [1]

    @pytest.mark.asyncio
    async def test_enforce_ttl(self, controller, mock_client):
        controller.track(1, dph=0.5)
        controller._tracked[1].created_at = time.time() - 120
        destroyed = await controller.enforce_ttl()
        assert destroyed == [1]
        mock_client.destroy_instance.assert_called_once_with(1)
        assert controller.active_count == 0


class TestEnforceBudget:
    @pytest.mark.asyncio
    async def test_enforce_budget_under(self, controller):
        controller.track(1, dph=0.1)
        destroyed = await controller.enforce_budget()
        assert destroyed == []

    @pytest.mark.asyncio
    async def test_enforce_budget_over(self, controller, mock_client):
        controller._total_spent = 6.0  # Over $5 budget
        controller.track(1, dph=0.5)
        controller.track(2, dph=0.5)
        destroyed = await controller.enforce_budget()
        assert set(destroyed) == {1, 2}
        assert controller.active_count == 0


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_all(self, controller, mock_client):
        controller.track(1)
        controller.track(2)
        controller.track(3)
        destroyed = await controller.cleanup_all()
        assert len(destroyed) == 3
        assert controller.active_count == 0


class TestActualCost:
    def test_set_actual_cost_overrides_estimate(self, controller):
        controller.track(1, dph=10.0)
        # Pretend the instance has been running for an hour at $10/hr → estimate $10
        controller._tracked[1].created_at = time.time() - 3600
        # But Vast.ai billed only $4
        controller.set_actual_cost(1, 4.0)
        spend = controller.estimate_spend()
        assert spend == pytest.approx(4.0)

    def test_set_actual_cost_unknown_instance_banks_total(self, controller):
        # Instance was untracked before billing reported in
        controller.set_actual_cost(999, 1.25)
        assert controller._total_spent == pytest.approx(1.25)
        # Negative actuals are ignored — defensive against API quirks
        controller.set_actual_cost(998, -5.0)
        assert controller._total_spent == pytest.approx(1.25)

    def test_actual_cost_drives_budget_enforcement(self, controller, mock_client):
        # estimated dph would put us within budget, but actual cost is over
        controller.track(1, dph=0.10)
        controller.set_actual_cost(1, 6.0)  # Over $5 budget
        assert controller.can_launch(dph=0.0) is False

