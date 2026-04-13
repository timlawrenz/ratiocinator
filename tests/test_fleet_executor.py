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
    PreflightSpec,
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

        # Attachments are added via sentry_sdk.add_attachment(), not event dict
        assert mock_sdk.add_attachment.call_count == 2
        filenames = [
            call.kwargs["filename"]
            for call in mock_sdk.add_attachment.call_args_list
        ]
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

        mock_sdk.add_attachment.assert_not_called()

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


class TestPreflight:
    """Tests for preflight validation in _run_arm."""

    @pytest.fixture
    def preflight_spec(self):
        return ExperimentSpec(
            name="preflight-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            preflight=PreflightSpec(
                command="python train.py --epochs 1 --batch_size 2",
                timeout_s=60,
                check_metrics=False,
            ),
        )

    @pytest.fixture
    def preflight_check_metrics_spec(self):
        return ExperimentSpec(
            name="preflight-metrics-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            preflight=PreflightSpec(
                command="python train.py --epochs 1",
                timeout_s=30,
                check_metrics=True,
            ),
        )

    def _make_remote_result(self, exit_code=0, stdout="", stderr=""):
        """Create a mock RemoteResult."""
        result = MagicMock()
        result.exit_code = exit_code
        result.stdout = stdout
        result.stderr = stderr
        result.duration_seconds = 1.0
        result.success = exit_code == 0
        return result

    @pytest.mark.asyncio
    async def test_preflight_failure_skips_training(
        self, preflight_spec, fleet_config, tmp_path,
    ):
        """When preflight exits non-zero, the arm fails immediately."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            preflight_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=100)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=100,
            status=InstanceStatus.RUNNING,
            ssh_host="1.2.3.4",
            ssh_port=22,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        # Mock RemoteExecutor
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        # hwinfo → clone → deps → data provision → preflight (fails)
        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        preflight_result = self._make_remote_result(
            exit_code=1,
            stderr="ImportError: No module named 'some_module'",
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, preflight_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert len(results) == 1
        assert not results[0].success
        assert results[0].exit_code == 1
        assert "Preflight failed" in results[0].error
        # Training should NOT have been called (only 3 remote.run calls above)
        assert mock_remote.run.call_count == 3

    @pytest.mark.asyncio
    async def test_preflight_success_continues_training(
        self, preflight_spec, fleet_config, tmp_path,
    ):
        """When preflight passes, training proceeds normally."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            preflight_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=200)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=200,
            status=InstanceStatus.RUNNING,
            ssh_host="1.2.3.4",
            ssh_port=22,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        preflight_result = self._make_remote_result(stdout="preflight ok")
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.5}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, preflight_result, train_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert len(results) == 1
        assert results[0].success
        assert results[0].metrics.get("loss") == 0.5
        # 4 calls: hwinfo, clone, preflight, train
        assert mock_remote.run.call_count == 4

    @pytest.mark.asyncio
    async def test_preflight_check_metrics_missing(
        self, preflight_check_metrics_spec, fleet_config, tmp_path,
    ):
        """When check_metrics=True and no metrics in preflight output, arm fails."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            preflight_check_metrics_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=300)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=300,
            status=InstanceStatus.RUNNING,
            ssh_host="1.2.3.4",
            ssh_port=22,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        # Preflight succeeds but produces no METRICS: output
        preflight_result = self._make_remote_result(
            stdout="Training started... done.",
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, preflight_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert len(results) == 1
        assert not results[0].success
        assert "no metrics" in results[0].error.lower()

    @pytest.mark.asyncio
    async def test_preflight_check_metrics_present(
        self, preflight_check_metrics_spec, fleet_config, tmp_path,
    ):
        """When check_metrics=True and metrics appear, training proceeds."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            preflight_check_metrics_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=400)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=400,
            status=InstanceStatus.RUNNING,
            ssh_host="1.2.3.4",
            ssh_port=22,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        # Preflight succeeds WITH metrics
        preflight_result = self._make_remote_result(
            stdout='Training...\nMETRICS:{"loss": 2.5}\nDone',
        )
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.3}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, preflight_result, train_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert len(results) == 1
        assert results[0].success
        assert results[0].metrics.get("loss") == 0.3
        # 4 calls: hwinfo, clone, preflight, train
        assert mock_remote.run.call_count == 4
