"""Tests for FleetExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ratiocinator.fleet.data import NullProvisioner
from ratiocinator.fleet.executor import (
    FleetConfig,
    FleetExecutor,
    _build_env_prefix,
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
    ValidationSpec,
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
        log_dir=str(tmp_path / "logs"),
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


class TestBuildEnvPrefix:
    """Tests for _build_env_prefix shell-safe env var helper."""

    def test_none_returns_empty(self):
        assert _build_env_prefix(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _build_env_prefix({}) == ""

    def test_simple_vars(self):
        result = _build_env_prefix({"FOO": "bar", "BAZ": "123"})
        assert "FOO=" in result
        assert "BAZ=" in result

    def test_values_are_shell_quoted(self):
        result = _build_env_prefix({"MSG": "hello world; rm -rf /"})
        # shlex.quote wraps in single quotes
        assert "MSG='hello world; rm -rf /'" in result

    def test_rejects_invalid_key_with_semicolon(self):
        with pytest.raises(ValueError, match="Invalid env var name"):
            _build_env_prefix({"FOO; rm -rf /": "val"})

    def test_rejects_key_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid env var name"):
            _build_env_prefix({"1BAD": "val"})

    def test_rejects_key_with_spaces(self):
        with pytest.raises(ValueError, match="Invalid env var name"):
            _build_env_prefix({"BAD KEY": "val"})

    def test_accepts_underscored_key(self):
        result = _build_env_prefix({"_MY_VAR_2": "ok"})
        assert "_MY_VAR_2=" in result


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
            self._make_remote_result(),  # architecture cat
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
        # 5 calls: hwinfo, clone, preflight, train, arch cat
        assert mock_remote.run.call_count == 5

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
            self._make_remote_result(),  # architecture cat
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
        # 5 calls: hwinfo, clone, preflight, train, arch cat
        assert mock_remote.run.call_count == 5

    @pytest.mark.asyncio
    async def test_batch_size_injected_into_preflight_and_training(
        self, fleet_config, tmp_path,
    ):
        """BATCH_SIZE env var appears in preflight and training remote commands."""
        spec = ExperimentSpec(
            name="batch-preflight-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50, batch_size=32),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            preflight=PreflightSpec(
                command="python train.py --epochs 1",
                timeout_s=60,
                check_metrics=False,
            ),
        )
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=500)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=500,
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
        preflight_result = self._make_remote_result(stdout="ok")
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

        assert results[0].success
        # Verify BATCH_SIZE appears in the preflight command (call index 2)
        preflight_cmd = mock_remote.run.call_args_list[2][0][0]
        assert "BATCH_SIZE=32" in preflight_cmd
        # Verify BATCH_SIZE appears in the training command (call index 3)
        train_cmd = mock_remote.run.call_args_list[3][0][0]
        assert "BATCH_SIZE=32" in train_cmd


class TestValidation:
    """Tests for post-training validation in _run_arm."""

    @pytest.fixture
    def validation_spec(self):
        return ExperimentSpec(
            name="validation-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py --output /workspace/output",
                timeout_s=120,
            ),
        )

    @pytest.fixture
    def validation_required_spec(self):
        return ExperimentSpec(
            name="validation-required-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=60,
                required_metrics=["real_validity_pct"],
            ),
        )

    @pytest.fixture
    def validation_prefix_spec(self):
        return ExperimentSpec(
            name="validation-prefix-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=60,
                prefix="val_",
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

    def _make_mock_client(self, instance_id=100):
        mock_client = AsyncMock()
        mock_client.search_offers = AsyncMock(return_value=[
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.40,
             "pcie_bw": 25, "cpu_ram": 128000},
        ])
        mock_client.create_instance = AsyncMock(return_value=instance_id)
        mock_client.get_instance = AsyncMock(return_value=InstanceInfo(
            instance_id=instance_id,
            status=InstanceStatus.RUNNING,
            ssh_host="1.2.3.4",
            ssh_port=22,
        ))
        mock_client.destroy_instance = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_validation_failure_marks_arm_failed(
        self, validation_spec, fleet_config, tmp_path,
    ):
        """When validation exits non-zero, the arm is marked failed."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(500)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.5}',
        )
        validation_result = self._make_remote_result(
            exit_code=1,
            stderr="Error: 0/100 files parsed successfully",
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
            self._make_remote_result(),  # architecture cat (still attempted)
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
        assert "Validation failed" in results[0].error
        # 5 calls: hwinfo, clone, train, validation, arch cat
        assert mock_remote.run.call_count == 5
        # Instance should still be cleaned up
        mock_client.destroy_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_success_merges_metrics(
        self, validation_spec, fleet_config, tmp_path,
    ):
        """When validation passes, its metrics merge into the result."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(600)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"heuristic_validity": 99.5}',
        )
        validation_result = self._make_remote_result(
            stdout='Validating...\nMETRICS:{"real_validity_pct": 0.0, "parse_errors": 47}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
            self._make_remote_result(),  # architecture cat
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
        # Training metric preserved
        assert results[0].metrics["heuristic_validity"] == 99.5
        # Validation metrics merged in
        assert results[0].metrics["real_validity_pct"] == 0.0
        assert results[0].metrics["parse_errors"] == 47
        # 5 calls: hwinfo, clone, train, validation, arch cat
        assert mock_remote.run.call_count == 5

    @pytest.mark.asyncio
    async def test_validation_overrides_training_metric(
        self, validation_spec, fleet_config, tmp_path,
    ):
        """Validation metrics override same-named training metrics."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(700)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"validity": 99.5}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"validity": 0.0}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
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
        # Validation overwrites heuristic
        assert results[0].metrics["validity"] == 0.0

    @pytest.mark.asyncio
    async def test_validation_required_metrics_missing(
        self, validation_required_spec, fleet_config, tmp_path,
    ):
        """When required_metrics are not in output, the arm fails."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_required_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(800)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.5}',
        )
        # Validation succeeds but doesn't output the required metric
        validation_result = self._make_remote_result(
            stdout='METRICS:{"other_metric": 42}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
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
        assert results[0].exit_code == -1
        assert "real_validity_pct" in results[0].error

    @pytest.mark.asyncio
    async def test_validation_required_metrics_present(
        self, validation_required_spec, fleet_config, tmp_path,
    ):
        """When required_metrics are present, the arm succeeds."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_required_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(900)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.3}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"real_validity_pct": 12.5}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
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
        assert results[0].metrics["real_validity_pct"] == 12.5
        assert results[0].metrics["loss"] == 0.3

    @pytest.mark.asyncio
    async def test_validation_prefix_namespaces_metrics(
        self, validation_prefix_spec, fleet_config, tmp_path,
    ):
        """When prefix is set, validation metrics are namespaced."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_prefix_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(1000)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"validity": 99.5}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"validity": 0.0, "parse_errors": 47}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
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
        # Training metric untouched (no prefix collision)
        assert results[0].metrics["validity"] == 99.5
        # Validation metrics namespaced
        assert results[0].metrics["val_validity"] == 0.0
        assert results[0].metrics["val_parse_errors"] == 47

    @pytest.mark.asyncio
    async def test_validation_skipped_on_training_failure(
        self, validation_spec, fleet_config, tmp_path,
    ):
        """Validation is NOT run if training fails."""
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            validation_spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(1100)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            exit_code=1,
            stderr="RuntimeError: CUDA OOM",
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result,
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
        # Only 3 calls: hwinfo, clone, train — no validation
        assert mock_remote.run.call_count == 3

    @pytest.mark.asyncio
    async def test_validation_required_with_prefix_uses_raw_keys(
        self, fleet_config, tmp_path,
    ):
        """required_metrics are checked against raw validation output keys,
        not the prefixed storage keys.  Users write the names their
        validation script actually emits."""
        spec = ExperimentSpec(
            name="prefix-required-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=60,
                prefix="val_",
                # Raw name as emitted by the script, not the prefixed key
                required_metrics=["real_validity_pct"],
            ),
        )
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(1200)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.3}',
        )
        # Validation outputs real_validity_pct → stored as val_real_validity_pct
        validation_result = self._make_remote_result(
            stdout='METRICS:{"real_validity_pct": 12.5}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
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
        assert results[0].metrics["val_real_validity_pct"] == 12.5

    @pytest.mark.asyncio
    async def test_validation_required_not_satisfied_by_training_metric(
        self, fleet_config, tmp_path,
    ):
        """A training metric of the same name must NOT satisfy required_metrics."""
        spec = ExperimentSpec(
            name="collision-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=60,
                required_metrics=["accuracy"],
            ),
        )
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(1300)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        # Training outputs "accuracy" — but validation does NOT
        train_result = self._make_remote_result(
            stdout='METRICS:{"accuracy": 95.0, "loss": 0.1}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"other_metric": 42}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        # Must fail — "accuracy" came from training, not validation
        assert len(results) == 1
        assert not results[0].success
        assert results[0].exit_code == -1
        assert "accuracy" in results[0].error

    @pytest.mark.asyncio
    async def test_validation_output_persisted_to_log(
        self, validation_spec, fleet_config, tmp_path,
    ):
        """Validation stdout/stderr are written to a separate log file."""
        store = ResultStore(tmp_path / "results.json")
        fleet_config_with_log = FleetConfig(
            api_key="test-key",
            ssh_key="/tmp/ssh",
            log_dir=str(tmp_path / "logs"),
        )
        executor = FleetExecutor(
            validation_spec, fleet_config_with_log,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(1400)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.3}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"validity": 100}\nAll checks passed',
            stderr='some warning',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            await executor.run(arm_indices=[0])

        log_path = tmp_path / "logs" / "validation-test" / "baseline.validation.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "All checks passed" in content
        assert "some warning" in content

    @pytest.mark.asyncio
    async def test_batch_size_injected_into_validation_command(
        self, fleet_config, tmp_path,
    ):
        """BATCH_SIZE env var appears in validation remote command."""
        spec = ExperimentSpec(
            name="batch-validation-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50, batch_size=16),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=120,
            ),
        )
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
            provisioner=NullProvisioner(),
            result_store=store,
        )

        mock_client = self._make_mock_client(900)
        mock_remote = AsyncMock()
        mock_remote.wait_for_ssh = AsyncMock(return_value=True)

        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.5}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"accuracy": 0.9}',
        )

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, validation_result,
        ])

        mock_provisioner = AsyncMock()
        mock_provisioner.provision = AsyncMock(return_value=(True, None))
        executor.provisioner = mock_provisioner

        with patch("ratiocinator.fleet.executor.VastClient", return_value=mock_client), \
             patch("ratiocinator.fleet.executor.RemoteExecutor", return_value=mock_remote), \
             patch("ratiocinator.fleet.executor.BOOT_POLL_INTERVAL_S", 0.01):
            results = await executor.run(arm_indices=[0])

        assert results[0].success
        # Verify BATCH_SIZE in training command (call index 2)
        train_cmd = mock_remote.run.call_args_list[2][0][0]
        assert "BATCH_SIZE=16" in train_cmd
        # Verify BATCH_SIZE in validation command (call index 3)
        val_cmd = mock_remote.run.call_args_list[3][0][0]
        assert "BATCH_SIZE=16" in val_cmd


class TestApplyDedup:
    def _make_spec(self):
        return ExperimentSpec(
            name="dup-exp",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[
                ArmSpec(name="a", command="python train.py", env={"LR": "0.1"}),
                ArmSpec(
                    name="b-dup",
                    description="dup of a",
                    command="python train.py",
                    env={"LR": "0.1"},
                ),
                ArmSpec(name="c-unique", command="python train.py", env={"LR": "0.2"}),
            ],
            metrics=MetricsSpec(),
        )

    def test_no_skip_keeps_all(self, fleet_config, caplog):
        import logging
        spec = self._make_spec()
        ex = FleetExecutor(spec, fleet_config, provisioner=NullProvisioner())
        pairs = [(i, a) for i, a in enumerate(spec.arms)]
        with caplog.at_level(logging.WARNING):
            kept = ex._apply_dedup(pairs, skip_duplicates=False)
        assert len(kept) == 3  # All retained
        assert any("Duplicate arm configs detected" in m for m in caplog.messages)

    def test_skip_drops_duplicates(self, fleet_config, caplog):
        import logging
        spec = self._make_spec()
        ex = FleetExecutor(spec, fleet_config, provisioner=NullProvisioner())
        pairs = [(i, a) for i, a in enumerate(spec.arms)]
        with caplog.at_level(logging.WARNING):
            kept = ex._apply_dedup(pairs, skip_duplicates=True)
        names = [a.name for _, a in kept]
        assert names == ["a", "c-unique"]
        assert any("Skipping duplicate arms" in m for m in caplog.messages)

    def test_no_duplicates_no_warning(self, fleet_config, caplog):
        import logging
        spec = ExperimentSpec(
            name="ok",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[
                ArmSpec(name="a", command="python train.py --lr 0.1"),
                ArmSpec(name="b", command="python train.py --lr 0.2"),
            ],
            metrics=MetricsSpec(),
        )
        ex = FleetExecutor(spec, fleet_config, provisioner=NullProvisioner())
        pairs = [(i, a) for i, a in enumerate(spec.arms)]
        with caplog.at_level(logging.WARNING):
            kept = ex._apply_dedup(pairs, skip_duplicates=True)
        assert len(kept) == 2
        assert not any("Duplicate" in m for m in caplog.messages)


class TestPrintCostSummary:
    def _result(self, **kw):
        from ratiocinator.fleet.results import ArmResult
        defaults = {"experiment": "exp", "arm_name": "arm"}
        defaults.update(kw)
        return ArmResult(**defaults)

    def test_summary_with_actual_cost(self, capsys):
        from ratiocinator.fleet.executor import print_cost_summary

        results = [
            self._result(
                arm_name="a", instance_dph=0.50, boot_time_s=180.0,
                estimated_cost=0.10, actual_cost=0.12,
            ),
            self._result(
                arm_name="b", instance_dph=0.50, boot_time_s=180.0,
                estimated_cost=0.10, actual_cost=0.13,
            ),
        ]
        print_cost_summary(results, budget=10.00)
        out = capsys.readouterr().out
        assert "Cost Summary:" in out
        assert "Estimated: $0.20" in out
        assert "Actual:    $0.25" in out
        assert "+25%" in out
        assert "Budget:    $10.00" in out
        assert "Boot overhead:" in out

    def test_summary_falls_back_to_estimate(self, capsys):
        from ratiocinator.fleet.executor import print_cost_summary

        results = [
            self._result(
                arm_name="a", instance_dph=0.50,
                estimated_cost=0.10, actual_cost=None,
            ),
        ]
        print_cost_summary(results, budget=1.0)
        out = capsys.readouterr().out
        assert "no billing data available" in out
        assert "Estimated: $0.10" in out

    def test_summary_partial_actual_coverage(self, capsys):
        from ratiocinator.fleet.executor import print_cost_summary

        results = [
            self._result(
                arm_name="a", instance_dph=0.50,
                estimated_cost=0.10, actual_cost=0.12,
            ),
            self._result(
                arm_name="b", instance_dph=0.50,
                estimated_cost=0.10, actual_cost=None,  # billing missing
            ),
        ]
        print_cost_summary(results, budget=1.0)
        out = capsys.readouterr().out
        assert "1/2 arms" in out
        assert "$0.12" in out  # actual known
        assert "$0.10 estimated for the rest" in out

    def test_summary_empty_results_is_noop(self, capsys):
        from ratiocinator.fleet.executor import print_cost_summary

        print_cost_summary([], budget=10.0)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Architecture env var injection
# ---------------------------------------------------------------------------


class TestArchitectureEnvVar:
    """Tests that RATIOCINATOR_ARCHITECTURE_PATH is injected and retrieved."""

    def _make_remote_result(self, exit_code=0, stdout="", stderr=""):
        result = MagicMock()
        result.exit_code = exit_code
        result.stdout = stdout
        result.stderr = stderr
        result.duration_seconds = 1.0
        result.success = exit_code == 0
        return result

    @pytest.mark.asyncio
    async def test_architecture_env_var_in_training_command(
        self, spec, fleet_config, tmp_path,
    ):
        """_run_arm injects RATIOCINATOR_ARCHITECTURE_PATH into the training command."""
        from ratiocinator.fleet.executor import ARCHITECTURE_ENV_VAR

        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
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

        arch_json = '{"hidden_size": 768, "repa_proj": true}'
        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.3}',
        )
        arch_cat_result = self._make_remote_result(stdout=arch_json)

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result, arch_cat_result,
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

        # Verify the training command includes the architecture env var
        train_call = mock_remote.run.call_args_list[2]
        train_cmd = train_call.args[0] if train_call.args else train_call[0][0]
        assert ARCHITECTURE_ENV_VAR in train_cmd
        assert "resolved_architecture.json" in train_cmd

        # Verify the architecture file was retrieved and persisted locally
        log_dir = tmp_path / "logs" / "test-experiment"
        arch_file = log_dir / "baseline.resolved_architecture.json"
        assert arch_file.exists()
        assert arch_file.read_text() == arch_json

    @pytest.mark.asyncio
    async def test_architecture_env_var_in_validation_command(
        self, fleet_config, tmp_path,
    ):
        """_run_arm injects RATIOCINATOR_ARCHITECTURE_PATH into validation."""
        from ratiocinator.fleet.executor import ARCHITECTURE_ENV_VAR

        spec = ExperimentSpec(
            name="arch-val-test",
            hardware=HardwareSpec(gpu="RTX 4090", max_dph=0.50),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            metrics=MetricsSpec(protocol="json_line", json_prefix="METRICS:"),
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=120,
            ),
        )
        store = ResultStore(tmp_path / "results.json")
        executor = FleetExecutor(
            spec, fleet_config,
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

        arch_json = '{"layers": 12}'
        hw_result = self._make_remote_result(stdout="RTX 4090, 24GB")
        clone_result = self._make_remote_result()
        train_result = self._make_remote_result(
            stdout='METRICS:{"loss": 0.2}',
        )
        validation_result = self._make_remote_result(
            stdout='METRICS:{"accuracy": 0.95}',
        )
        arch_cat_result = self._make_remote_result(stdout=arch_json)

        mock_remote.run = AsyncMock(side_effect=[
            hw_result, clone_result, train_result,
            validation_result, arch_cat_result,
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

        # Verify RATIOCINATOR_ARCHITECTURE_PATH in validation command
        val_call = mock_remote.run.call_args_list[3]
        val_cmd = val_call.args[0] if val_call.args else val_call[0][0]
        assert ARCHITECTURE_ENV_VAR in val_cmd
        assert "resolved_architecture.json" in val_cmd
