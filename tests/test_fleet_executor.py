"""Tests for FleetExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ratiocinator.fleet.data import NullProvisioner
from ratiocinator.fleet.executor import (
    FleetConfig,
    FleetExecutor,
    _parse_remote_traceback,
    _report_remote_crash,
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


class TestReportRemoteCrash:
    def test_adds_breadcrumb_on_crash(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.fleet.executor.sentry_sdk", mock_sdk), \
             patch("ratiocinator.fleet.executor.fleet_breadcrumb") as mock_bc:
            _report_remote_crash(
                experiment="test-exp",
                arm_name="baseline",
                exit_code=1,
                stderr_text="RuntimeError: CUDA out of memory",
                stdout_tail="step 100",
                gpu_info="RTX 4090",
                instance_id=12345,
            )

        mock_bc.assert_called_once()
        call_kwargs = mock_bc.call_args[1]
        assert call_kwargs["category"] == "fleet.crash"
        assert call_kwargs["level"] == "error"

    def test_attaches_stderr_and_stdout(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.fleet.executor.sentry_sdk", mock_sdk):
            _report_remote_crash(
                experiment="test-exp",
                arm_name="baseline",
                exit_code=1,
                stderr_text="RuntimeError: CUDA out of memory",
                stdout_tail="training output...",
                gpu_info="RTX 4090",
                instance_id=12345,
            )

        event = mock_sdk.capture_event.call_args[0][0]
        assert "attachments" in event
        filenames = [a["filename"] for a in event["attachments"]]
        assert "baseline_stderr.txt" in filenames
        assert "baseline_stdout.txt" in filenames

    def test_no_attachments_when_empty(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.fleet.executor.sentry_sdk", mock_sdk):
            _report_remote_crash(
                experiment="test-exp",
                arm_name="baseline",
                exit_code=1,
                stderr_text="",
                stdout_tail="",
                gpu_info="",
                instance_id=12345,
            )

        event = mock_sdk.capture_event.call_args[0][0]
        assert "attachments" not in event

    def test_noop_without_sentry(self):
        with patch("ratiocinator.fleet.executor.sentry_sdk", None):
            # Should not raise
            _report_remote_crash(
                experiment="test-exp",
                arm_name="baseline",
                exit_code=1,
                stderr_text="error",
                stdout_tail="",
                gpu_info="",
                instance_id=None,
            )


class TestWriteArmLog:
    def test_writes_log_file(self, spec, fleet_config, tmp_path):
        fleet_config.log_dir = str(tmp_path / "logs")
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        log_path = executor._write_arm_log(
            "baseline", "stdout content\n", "stderr content\n",
        )

        assert log_path is not None
        assert log_path.exists()
        content = log_path.read_text()
        assert "=== STDOUT ===" in content
        assert "stdout content" in content
        assert "=== STDERR ===" in content
        assert "stderr content" in content

    def test_creates_experiment_subdirectory(self, spec, fleet_config, tmp_path):
        fleet_config.log_dir = str(tmp_path / "logs")
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        log_path = executor._write_arm_log("arm1", "output", "")

        assert log_path is not None
        assert log_path.parent.name == "test-experiment"

    def test_empty_output_skips_sections(self, spec, fleet_config, tmp_path):
        fleet_config.log_dir = str(tmp_path / "logs")
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        log_path = executor._write_arm_log("arm1", "stdout only\n", "")

        content = log_path.read_text()
        assert "=== STDOUT ===" in content
        assert "=== STDERR ===" not in content
