"""Tests for the Vast.ai GPU runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ratiocinator.config import Config
from ratiocinator.infra.vast_client import InstanceInfo, InstanceStatus
from ratiocinator.infra.vast_runner import VastRunner


@pytest.fixture
def config():
    c = Config()
    c.vast.api_key = "test-key"
    c.vast.max_dph = 0.50
    return c


@pytest.fixture
def runner(config):
    return VastRunner(config)


class TestVastRunnerInit:
    def test_default_ssh_key(self, runner):
        assert runner.ssh_key == Path.home() / ".ssh" / "id_rsa"

    def test_custom_ssh_key(self, config):
        r = VastRunner(config, ssh_key=Path("/custom/key"))
        assert r.ssh_key == Path("/custom/key")

    def test_no_api_key_raises(self):
        c = Config()
        c.vast.api_key = ""
        r = VastRunner(c)
        with pytest.raises(RuntimeError, match="VAST_API_KEY"):
            r._get_client()


class TestVastRunnerNoOffers:
    @pytest.mark.asyncio
    async def test_no_offers_returns_error(self, runner):
        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[])
        runner._client = mock_client

        result = await runner.run_async("pytorch:latest", "python train.py")
        assert result.exit_code == 1
        assert "No Vast.ai offers" in result.stderr


class TestVastRunnerBudgetCheck:
    @pytest.mark.asyncio
    async def test_over_budget_returns_error(self, runner):
        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.50},
        ])
        runner._client = mock_client

        mock_safety = MagicMock()
        mock_safety.can_launch.return_value = False
        mock_safety.estimate_spend.return_value = 15.0
        runner._safety = mock_safety

        result = await runner.run_async("pytorch:latest", "python train.py")
        assert result.exit_code == 1
        assert "Budget exceeded" in result.stderr


class TestVastRunnerBootTimeout:
    @pytest.mark.asyncio
    async def test_boot_timeout(self, runner):
        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.10},
        ])
        mock_client.create_instance = AsyncMock(return_value=999)
        mock_client.destroy_instance = AsyncMock()

        # Always return "creating" status
        creating_info = InstanceInfo(
            instance_id=999,
            status=InstanceStatus.CREATING,
        )
        mock_client.get_instance = AsyncMock(return_value=creating_info)
        runner._client = mock_client

        mock_safety = MagicMock()
        mock_safety.can_launch.return_value = True
        runner._safety = mock_safety

        # Patch BOOT_TIMEOUT_S to be very short for testing
        with patch("ratiocinator.infra.vast_runner.BOOT_TIMEOUT_S", 0.1), \
             patch("ratiocinator.infra.vast_runner.BOOT_POLL_INTERVAL_S", 0.05):
            result = await runner.run_async("pytorch:latest", "python train.py")

        assert result.exit_code == 1
        assert "failed" in result.stderr.lower()
        # Instance should be destroyed in finally block (once per retry attempt)
        mock_client.destroy_instance.assert_called_with(999)


class TestVastRunnerCleanup:
    @pytest.mark.asyncio
    async def test_instance_destroyed_on_error(self, runner):
        """Ensure instance is always destroyed, even if SSH fails."""
        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.10},
        ])
        mock_client.create_instance = AsyncMock(return_value=888)
        mock_client.destroy_instance = AsyncMock()

        running_info = InstanceInfo(
            instance_id=888,
            status=InstanceStatus.RUNNING,
            ssh_host="ssh.vast.ai",
            ssh_port=22222,
        )
        mock_client.get_instance = AsyncMock(return_value=running_info)
        runner._client = mock_client

        mock_safety = MagicMock()
        mock_safety.can_launch.return_value = True
        runner._safety = mock_safety

        # Mock RemoteExecutor to simulate SSH ready but rsync failure
        with patch("ratiocinator.infra.vast_runner.RemoteExecutor") as mock_remote_cls:
            mock_remote_inst = mock_remote_cls.return_value
            mock_remote_inst.wait_for_ssh = AsyncMock(return_value=True)
            mock_rsync_result = MagicMock()
            mock_rsync_result.success = False
            mock_remote_inst.rsync_to = AsyncMock(return_value=mock_rsync_result)

            result = await runner.run_async(
                "pytorch:latest", "python train.py",
                repo_path=Path("/tmp/test"),
            )

        assert result.exit_code == 1
        mock_client.destroy_instance.assert_called_with(888)
