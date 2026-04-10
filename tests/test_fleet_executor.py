"""Tests for FleetExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.fleet.data import NullProvisioner
from ratiocinator.fleet.executor import (
    FleetConfig,
    FleetExecutor,
    _parse_remote_traceback,
)
from ratiocinator.fleet.results import ResultStore
from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    HardwareSpec,
    MetricsSpec,
    RepoSpec,
)
from ratiocinator.infra.vast_client import InstanceInfo, InstanceStatus


@pytest.fixture
def spec():
    return ExperimentSpec(
        name="test-experiment",
        hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
        repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
        arms=[
            ArmSpec(name="baseline", command="python train.py"),
            ArmSpec(name="optimized", command="python train.py --fast"),
        ],
        metrics=MetricsSpec(
            protocol="json_line",
            json_prefix="METRICS:",
        ),
    )


@pytest.fixture
def fleet_config(tmp_path):
    return FleetConfig(
        api_key="test-key",
        ssh_key="/tmp/test_key",
        results_path=str(tmp_path / "results.json"),
    )


class TestParseRemoteTraceback:
    def test_parse_traceback(self):
        stderr = '''Traceback (most recent call last):
  File "train.py", line 42, in main
    model.forward(x)
  File "model.py", line 100, in forward
    return self.net(x)
RuntimeError: CUDA out of memory'''
        frames, exc_type, exc_value = _parse_remote_traceback(stderr)
        assert len(frames) == 2
        assert frames[0]["filename"] == "train.py"
        assert frames[0]["lineno"] == 42
        assert frames[1]["function"] == "forward"
        assert exc_type == "RuntimeError"
        assert "CUDA out of memory" in exc_value

    def test_no_traceback(self):
        frames, exc_type, _exc_value = _parse_remote_traceback("some error")
        assert frames == []
        assert exc_type == "RemoteTrainingError"

    def test_empty_stderr(self):
        frames, _exc_type, _exc_value = _parse_remote_traceback("")
        assert frames == []


class TestFleetExecutor:
    @pytest.mark.asyncio
    async def test_no_offers(self, spec, fleet_config, tmp_path):
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client):
            results = await executor.run()

        assert results == []

    @pytest.mark.asyncio
    async def test_dry_run(self, spec, fleet_config, tmp_path):
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40, "pcie_bw": 25, "cpu_ram": 128000},
            {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.45, "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client):
            results = await executor.run(dry_run=True)

        assert results == []

    @pytest.mark.asyncio
    async def test_arm_selection(self, spec, fleet_config, tmp_path):
        """Selecting specific arm indices works."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40, "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=999)
        # Return boot failure so the arm fails early (we're testing selection logic)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=999, status=InstanceStatus.ERROR,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[1])

        # Should only run arm index 1 (optimized), not both arms
        assert len(results) == 1
        assert results[0].arm_name == "optimized"

    @pytest.mark.asyncio
    async def test_boot_failure_reports_error(self, spec, fleet_config, tmp_path):
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40, "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=888)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=888, status=InstanceStatus.ERROR,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        # Override budget to have very short boot timeout
        spec.budget.boot_timeout_s = 1

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert len(results) == 1
        assert not results[0].success
        assert "boot" in results[0].error.lower()
        mock_client.destroy_instance.assert_called_with(888)
