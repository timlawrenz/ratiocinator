"""Tests for the HuggingFace Jobs fleet executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.fleet.hf_executor import (
    HFFleetConfig,
    HFFleetExecutor,
)
from ratiocinator.fleet.spec import (
    ArmSpec,
    BudgetSpec,
    DataSpec,
    DepsSpec,
    ExperimentSpec,
    HardwareSpec,
    RepoSpec,
)
from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_config(tmp_path):
    return HFFleetConfig(
        token="hf_test_token",
        namespace="test-ns",
        bucket_prefix="test-ns",
        results_path=str(tmp_path / "results.json"),
        log_dir=str(tmp_path / "logs"),
    )


@pytest.fixture
def basic_spec():
    return ExperimentSpec(
        name="test-experiment",
        hardware=HardwareSpec(
            gpu="A100",
            hf_flavor="a100-large",
            image="pytorch/pytorch:2.7.0",
        ),
        repo=RepoSpec(
            url="https://github.com/test/repo.git",
            branch="main",
        ),
        arms=[
            ArmSpec(name="baseline", command="python train.py --lr 0.001"),
            ArmSpec(
                name="optimized",
                command="python train.py --lr 0.0001",
                env={"CUSTOM_VAR": "42"},
            ),
        ],
        data=DataSpec(
            source="hf-dataset",
            hf_source="test-ns/my-data",
            hf_mount_path="/data",
        ),
        deps=DepsSpec(
            pre_install=["pip install torch"],
            requirements="requirements.txt",
        ),
        budget=BudgetSpec(
            max_dollars=10.0,
            train_timeout_s=1800,
        ),
    )


@pytest.fixture
def spec_no_hf_flavor():
    return ExperimentSpec(
        name="no-flavor",
        hardware=HardwareSpec(gpu="RTX 4090"),
        repo=RepoSpec(url="https://github.com/test/repo.git"),
        arms=[ArmSpec(name="arm-1", command="python train.py")],
    )


# ---------------------------------------------------------------------------
# HFFleetConfig
# ---------------------------------------------------------------------------


class TestHFFleetConfig:
    def test_defaults(self):
        cfg = HFFleetConfig(token="hf_test")
        assert cfg.token == "hf_test"
        assert cfg.namespace == ""
        assert cfg.max_timeout == "4h"

    def test_custom(self, tmp_path):
        cfg = HFFleetConfig(
            token="hf_tok",
            namespace="my-org",
            bucket_prefix="my-org/ratiocinator",
            results_path=str(tmp_path / "r.json"),
        )
        assert cfg.namespace == "my-org"


# ---------------------------------------------------------------------------
# Executor init
# ---------------------------------------------------------------------------


class TestHFFleetExecutorInit:
    def test_creates_output_bucket_name(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        assert "test-experiment" in executor._output_bucket
        assert "test-ns" in executor._output_bucket

    def test_requires_namespace_or_prefix(self, basic_spec):
        cfg = HFFleetConfig(token="hf_tok")
        with pytest.raises(ValueError, match="bucket_prefix or namespace"):
            HFFleetExecutor(basic_spec, cfg)


# ---------------------------------------------------------------------------
# Wrapper script generation
# ---------------------------------------------------------------------------


class TestWrapperScriptGeneration:
    def test_basic_script(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        arm = basic_spec.arms[0]
        script = executor._build_wrapper_script(arm)

        assert "#!/bin/bash" in script
        assert "set -euo pipefail" in script
        assert "PYTHONUNBUFFERED=1" in script
        assert "git clone" in script
        assert "pip install torch" in script
        assert "requirements.txt" in script
        assert "python train.py --lr 0.001" in script

    def test_env_vars_in_script(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        arm = basic_spec.arms[1]  # optimized arm has CUSTOM_VAR
        script = executor._build_wrapper_script(arm)

        assert "CUSTOM_VAR" in script
        assert "42" in script

    def test_preflight_in_script(self, basic_spec, hf_config):
        from ratiocinator.fleet.spec import PreflightSpec

        basic_spec.preflight = PreflightSpec(
            command="python train.py --epochs 1",
            timeout_s=60,
        )
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        assert "Preflight" in script
        assert "python train.py --epochs 1" in script

    def test_validation_in_script(self, basic_spec, hf_config):
        from ratiocinator.fleet.spec import ValidationSpec

        basic_spec.validation = ValidationSpec(
            command="python validate.py",
            timeout_s=120,
        )
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        assert "Validation" in script
        assert "python validate.py" in script
        assert "TRAIN_EXIT" in script

    def test_no_env_section_when_empty(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        arm = basic_spec.arms[0]  # baseline — no env
        script = executor._build_wrapper_script(arm)

        assert "Arm-specific environment" not in script

    def test_preemption_detection_in_script(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        # Should include nonce-based preemption detection
        assert "PREEMPTION DETECTED" in script
        assert "RATIOCINATOR_RESUME_CHECKPOINT" in script
        assert "checkpoint_" in script
        assert "_job_started" in script
        assert "RUN_NONCE" in script
        # Uses state.json with run_nonce for orchestrator signaling
        assert "preemption_detected" in script
        assert "run_nonce" in script
        # Uses version sort for correct checkpoint ordering
        assert "sort -V" in script

    def test_preemption_nonce_avoids_false_positive(self, basic_spec, hf_config):
        """On a fresh run (sentinel absent or has wrong nonce), checkpoints
        are NOT searched — the nonce is written fresh instead."""
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        # Nonce must be stored in _run_nonces
        assert basic_spec.arms[0].name in executor._run_nonces
        nonce = executor._run_nonces[basic_spec.arms[0].name]
        assert len(nonce) == 32  # UUID4 hex

        # Sentinel is written with nonce content (not just touch)
        assert "echo \"$RUN_NONCE\" >" in script
        # Checkpoint find is gated behind nonce-matching sentinel check
        lines = script.splitlines()
        sentinel_check = next(
            i for i, line in enumerate(lines)
            if "_job_started" in line and "if" in line
        )
        find_line = next(
            i for i, line in enumerate(lines) if "RESUME_CKPT" in line
        )
        # find must come after the sentinel check
        assert find_line > sentinel_check

    def test_batch_size_from_hardware_in_script(self, basic_spec, hf_config):
        basic_spec.hardware.batch_size = 64
        executor = HFFleetExecutor(basic_spec, hf_config)
        arm = basic_spec.arms[0]  # baseline — no per-arm override
        script = executor._build_wrapper_script(arm)

        assert "export BATCH_SIZE=64" in script

    def test_batch_size_per_arm_override_in_script(self, basic_spec, hf_config):
        basic_spec.hardware.batch_size = 64
        basic_spec.arms[0].batch_size = 16
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        assert "export BATCH_SIZE=16" in script
        assert "BATCH_SIZE=64" not in script

    def test_explicit_env_batch_size_not_overridden_in_script(
        self, basic_spec, hf_config,
    ):
        basic_spec.hardware.batch_size = 64
        basic_spec.arms[1].env["BATCH_SIZE"] = "256"
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[1])

        assert "export BATCH_SIZE=256" in script
        assert "BATCH_SIZE=64" not in script

    def test_hf_data_source_pins_huggingface_hub(self, basic_spec, hf_config):
        """hf-dataset/hf-bucket specs inject huggingface_hub pin after requirements."""
        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        assert "huggingface_hub>=1.9.0" in script
        # Pin must appear AFTER requirements install to prevent downgrade
        req_pos = script.find("requirements.txt")
        pin_pos = script.find("huggingface_hub>=1.9.0")
        assert req_pos < pin_pos

    def test_non_hf_data_source_no_pin(self, hf_config):
        """Non-HF data sources should not inject the huggingface_hub pin."""
        spec = ExperimentSpec(
            name="s3-experiment",
            hardware=HardwareSpec(
                gpu="A100", hf_flavor="a100-large", image="pytorch/pytorch:2.7.0",
            ),
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            data=DataSpec(source="s3-presigned"),
            budget=BudgetSpec(max_dollars=5.0, train_timeout_s=1800),
        )
        executor = HFFleetExecutor(spec, hf_config)
        script = executor._build_wrapper_script(spec.arms[0])

        assert "huggingface_hub>=1.9.0" not in script


# ---------------------------------------------------------------------------
# Volume building
# ---------------------------------------------------------------------------


class TestBuildVolumes:
    def test_includes_script_data_output(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        volumes = executor._build_volumes(basic_spec.arms[0])

        assert len(volumes) == 3
        types = [v["type"] for v in volumes]
        assert "bucket" in types
        assert "dataset" in types

        mount_paths = [v["mount_path"] for v in volumes]
        assert "/input" in mount_paths
        assert "/data" in mount_paths
        assert "/output" in mount_paths

    def test_no_data_volume_when_none(self, hf_config):
        spec = ExperimentSpec(
            name="no-data",
            hardware=HardwareSpec(hf_flavor="l4"),
            repo=RepoSpec(url="https://github.com/t/r.git"),
            arms=[ArmSpec(name="a", command="echo hi")],
            data=DataSpec(source="none"),
        )
        executor = HFFleetExecutor(spec, hf_config)
        volumes = executor._build_volumes(spec.arms[0])

        # Script + output, no data
        assert len(volumes) == 2


# ---------------------------------------------------------------------------
# hf_flavor validation
# ---------------------------------------------------------------------------


class TestHFFlavorRequired:
    async def test_run_without_hf_flavor_raises(self, spec_no_hf_flavor, hf_config):
        executor = HFFleetExecutor(spec_no_hf_flavor, hf_config)
        with pytest.raises(ValueError, match="hf_flavor is required"):
            await executor.run()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    async def test_dry_run_returns_empty(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        with patch("builtins.print"):
            results = await executor.run(dry_run=True)
        assert results == []


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPollJob:
    async def test_polls_until_terminal(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()

        # Return RUNNING twice, then COMPLETED
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j1", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j1", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j1", stage=HFJobStage.COMPLETED),
        ])
        # No heartbeat file yet — bucket reads return None
        client.download_from_bucket = AsyncMock(return_value=None)

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0), \
             patch("ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0):
            stage = await executor._poll_job(client, "j1", "test-arm")

        assert stage == HFJobStage.COMPLETED
        assert client.get_job.call_count == 3

    async def test_handles_poll_error(self, basic_spec, hf_config):
        from ratiocinator.infra.hf_client import HFClientError

        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()

        # Error, then success
        client.get_job = AsyncMock(side_effect=[
            HFClientError("network error"),
            HFJobInfo(job_id="j2", stage=HFJobStage.FAILED),
        ])
        client.download_from_bucket = AsyncMock(return_value=None)

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0), \
             patch("ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0):
            stage = await executor._poll_job(client, "j2", "arm-x")

        assert stage == HFJobStage.FAILED


# ---------------------------------------------------------------------------
# Full run with mocked client
# ---------------------------------------------------------------------------


class TestRunArm:
    async def test_successful_arm(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)

        mock_client = AsyncMock()
        mock_client.create_bucket = AsyncMock()
        mock_client.upload_to_bucket = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-success")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-success", stage=HFJobStage.COMPLETED),
        )
        mock_client.get_job_logs = AsyncMock(
            return_value='Training...\nMETRICS:{"loss": 0.33, "accuracy": 0.95}\nDone.\n',
        )

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            result = await executor._run_arm(mock_client, 0, basic_spec.arms[0], 0)

        assert result.exit_code == 0
        assert result.metrics["loss"] == 0.33
        assert result.metrics["accuracy"] == 0.95

    async def test_failed_arm(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)

        mock_client = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-fail")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-fail", stage=HFJobStage.FAILED),
        )
        mock_client.get_job_logs = AsyncMock(
            return_value="Traceback (most recent call last):\n  ImportError: no module",
        )

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            result = await executor._run_arm(mock_client, 0, basic_spec.arms[0], 0)

        assert result.exit_code == 1
        assert "ImportError" in result.error

    async def test_cancelled_arm(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)

        mock_client = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-cancel")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-cancel", stage=HFJobStage.CANCELLED),
        )
        mock_client.get_job_logs = AsyncMock(return_value="")

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            result = await executor._run_arm(mock_client, 0, basic_spec.arms[0], 0)

        assert result.exit_code == -2
        assert "cancelled" in result.error.lower()

    async def test_run_job_receives_batch_size_env(self, hf_config):
        """run_job() is called with BATCH_SIZE from hardware spec in env."""
        spec = ExperimentSpec(
            name="batch-job-test",
            hardware=HardwareSpec(
                gpu="A100", hf_flavor="a100-large",
                image="pytorch/pytorch:2.7.0", batch_size=64,
            ),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(name="baseline", command="python train.py")],
            data=DataSpec(
                source="hf-dataset",
                hf_source="test-ns/data",
                hf_mount_path="/data",
            ),
            budget=BudgetSpec(max_dollars=5.0, train_timeout_s=1800),
        )
        executor = HFFleetExecutor(spec, hf_config)

        mock_client = AsyncMock()
        mock_client.create_bucket = AsyncMock()
        mock_client.upload_to_bucket = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-batch")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-batch", stage=HFJobStage.COMPLETED),
        )
        mock_client.get_job_logs = AsyncMock(
            return_value='METRICS:{"loss": 0.1}\n',
        )

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            await executor._run_arm(mock_client, 0, spec.arms[0], 0)

        # Verify run_job was called with env containing BATCH_SIZE
        call_kwargs = mock_client.run_job.call_args[1]
        assert call_kwargs["env"]["BATCH_SIZE"] == "64"

    async def test_run_job_explicit_env_batch_size_wins(self, hf_config):
        """Explicit arm.env['BATCH_SIZE'] takes precedence over hardware spec."""
        spec = ExperimentSpec(
            name="batch-explicit-test",
            hardware=HardwareSpec(
                gpu="A100", hf_flavor="a100-large",
                image="pytorch/pytorch:2.7.0", batch_size=64,
            ),
            repo=RepoSpec(url="https://github.com/test/repo.git", branch="main"),
            arms=[ArmSpec(
                name="custom",
                command="python train.py",
                env={"BATCH_SIZE": "256"},
            )],
            data=DataSpec(
                source="hf-dataset",
                hf_source="test-ns/data",
                hf_mount_path="/data",
            ),
            budget=BudgetSpec(max_dollars=5.0, train_timeout_s=1800),
        )
        executor = HFFleetExecutor(spec, hf_config)

        mock_client = AsyncMock()
        mock_client.create_bucket = AsyncMock()
        mock_client.upload_to_bucket = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-explicit")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-explicit", stage=HFJobStage.COMPLETED),
        )
        mock_client.get_job_logs = AsyncMock(
            return_value='METRICS:{"loss": 0.2}\n',
        )

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            await executor._run_arm(mock_client, 0, spec.arms[0], 0)

        # Explicit env should win
        call_kwargs = mock_client.run_job.call_args[1]
        assert call_kwargs["env"]["BATCH_SIZE"] == "256"


# ---------------------------------------------------------------------------
# Log saving
# ---------------------------------------------------------------------------


class TestSaveArmLog:
    def test_saves_log_file(self, basic_spec, hf_config, tmp_path):
        hf_config.log_dir = str(tmp_path / "logs")
        executor = HFFleetExecutor(basic_spec, hf_config)
        executor._save_arm_log("baseline", "some log content")

        log_file = tmp_path / "logs" / "test-experiment_baseline_hf.log"
        assert log_file.exists()
        assert log_file.read_text() == "some log content"


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------


class TestExtractError:
    def test_extracts_last_lines(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        logs = "\n".join([f"line {i}" for i in range(20)])
        error = executor._extract_error(logs)

        assert "line 19" in error
        assert "line 10" in error
        assert "line 9" not in error

    def test_empty_logs(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        error = executor._extract_error("")
        assert "No logs" in error


# ---------------------------------------------------------------------------
# Heartbeat / state.json bucket-backed monitoring
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_wrapper_script_exports_state_path(self, basic_spec, hf_config):
        from ratiocinator.fleet.hf_executor import STATE_ENV_VAR

        executor = HFFleetExecutor(basic_spec, hf_config)
        script = executor._build_wrapper_script(basic_spec.arms[0])

        # Env var advertises the heartbeat path
        assert STATE_ENV_VAR in script
        assert "test-experiment/baseline/state.json" in script
        # Parent dir is created so the training script can write
        assert "mkdir -p" in script and "/output/test-experiment/baseline" in script

    def test_arm_state_remote_path(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        assert executor._arm_state_remote_path("baseline") == (
            "test-experiment/baseline/state.json"
        )

    async def test_read_heartbeat_returns_parsed_json(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        client.download_from_bucket = AsyncMock(
            return_value='{"step": 100, "loss": 0.5, "eta_seconds": 600}',
        )

        state = await executor._read_heartbeat(client, "baseline")
        assert state == {"step": 100, "loss": 0.5, "eta_seconds": 600}
        client.download_from_bucket.assert_awaited_once_with(
            executor._output_bucket, "test-experiment/baseline/state.json",
        )

    async def test_read_heartbeat_missing_returns_none(self, basic_spec, hf_config):
        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        client.download_from_bucket = AsyncMock(return_value=None)

        assert await executor._read_heartbeat(client, "baseline") is None

    async def test_read_heartbeat_invalid_json_returns_none(
        self, basic_spec, hf_config,
    ):
        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        client.download_from_bucket = AsyncMock(return_value="not json {{{")

        assert await executor._read_heartbeat(client, "baseline") is None

    async def test_read_heartbeat_non_object_returns_none(
        self, basic_spec, hf_config,
    ):
        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        # Valid JSON but not an object — callers expect dict semantics
        client.download_from_bucket = AsyncMock(return_value="[1, 2, 3]")

        assert await executor._read_heartbeat(client, "baseline") is None

    async def test_read_heartbeat_swallows_client_error(
        self, basic_spec, hf_config,
    ):
        from ratiocinator.infra.hf_client import HFClientError

        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        client.download_from_bucket = AsyncMock(
            side_effect=HFClientError("transient"),
        )

        # Heartbeats are advisory — never fatal
        assert await executor._read_heartbeat(client, "baseline") is None

    async def test_metric_fallback_to_heartbeat_when_logs_empty(
        self, basic_spec, hf_config,
    ):
        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)

        mock_client = AsyncMock()
        mock_client.run_job = AsyncMock(return_value="job-ok")
        mock_client.get_job = AsyncMock(
            return_value=HFJobInfo(job_id="job-ok", stage=HFJobStage.COMPLETED),
        )
        # Logs API returned nothing useful (the very failure mode this issue addresses)
        mock_client.get_job_logs = AsyncMock(return_value="")
        mock_client.download_from_bucket = AsyncMock(
            return_value='{"step": 5000, "loss": 0.12, "final": true}',
        )

        with patch("ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0):
            result = await executor._run_arm(mock_client, 0, basic_spec.arms[0], 0)

        assert result.exit_code == 0
        # Heartbeat metrics promoted to result.metrics
        assert result.metrics["step"] == 5000
        assert result.metrics["loss"] == 0.12
        assert result.metrics["final"] is True

    async def test_poll_emits_heartbeat_breadcrumb(self, basic_spec, hf_config):
        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.COMPLETED),
        ])
        client.download_from_bucket = AsyncMock(
            return_value='{"step": 10, "loss": 1.0}',
        )

        with patch(
            "ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.fleet_breadcrumb",
        ) as breadcrumb:
            stage = await executor._poll_job(client, "j", "baseline")

        assert stage == HFJobStage.COMPLETED
        # At least one breadcrumb category should be the heartbeat one
        categories = [c.kwargs.get("category") for c in breadcrumb.call_args_list]
        assert "fleet.hf.heartbeat" in categories

    async def test_poll_detects_preemption_on_step_regression(
        self, basic_spec, hf_config,
    ):
        """When the heartbeat step goes backwards after being established
        (i.e. after forward progress), the executor should log a preemption
        warning.  Establishing requires at least two increasing reads."""
        import json

        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        # Four polls: step 100 (first), step 200 (establishes baseline via
        # forward progress), step 50 (regression → preemption!), then done.
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.COMPLETED),
        ])
        client.download_from_bucket = AsyncMock(side_effect=[
            json.dumps({"step": 100, "loss": 0.5}),
            json.dumps({"step": 200, "loss": 0.4}),   # forward progress
            json.dumps({"step": 50, "loss": 0.8}),    # step went backwards
        ])

        with patch(
            "ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.fleet_breadcrumb",
        ) as breadcrumb, patch(
            "ratiocinator.fleet.hf_executor.fleet_metric",
        ) as metric:
            stage = await executor._poll_job(client, "j", "baseline")

        assert stage == HFJobStage.COMPLETED
        # Should have emitted preemption breadcrumb
        categories = [c.kwargs.get("category") for c in breadcrumb.call_args_list]
        assert "fleet.hf.preemption" in categories
        # Should have emitted preemption metric
        metric_names = [c.args[0] for c in metric.call_args_list]
        assert "fleet.arm.preemption" in metric_names

    async def test_poll_no_false_preemption_from_stale_state(
        self, basic_spec, hf_config,
    ):
        """A fresh run that reads stale state.json (from a previous completed
        run with step=5000) then writes step=1 should NOT trigger a preemption
        alert.  The first heartbeat read (stale step=5000) does NOT mark
        heartbeat_established — only a step *increase* does — so when step=1
        arrives it's just a step change, not a regression."""
        import json

        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)
        client = AsyncMock()
        # Keep job RUNNING through enough polls to read both heartbeats,
        # then transition to COMPLETED.
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.COMPLETED),
        ])
        client.download_from_bucket = AsyncMock(side_effect=[
            json.dumps({"step": 5000, "loss": 0.1}),  # stale from old run
            json.dumps({"step": 1, "loss": 2.0}),      # new job starts
        ])

        with patch(
            "ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.fleet_breadcrumb",
        ) as breadcrumb, patch(
            "ratiocinator.fleet.hf_executor.fleet_metric",
        ) as metric:
            stage = await executor._poll_job(client, "j", "baseline")

        assert stage == HFJobStage.COMPLETED
        # NO preemption breadcrumb should have been emitted
        categories = [c.kwargs.get("category") for c in breadcrumb.call_args_list]
        assert "fleet.hf.preemption" not in categories
        # NO preemption metric
        metric_names = [c.args[0] for c in metric.call_args_list]
        assert "fleet.arm.preemption" not in metric_names

    async def test_poll_detects_preemption_from_state_json_flag(
        self, basic_spec, hf_config,
    ):
        """When the wrapper script writes preemption_detected into state.json
        with the correct run_nonce, the poll loop should surface it."""
        import json

        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)
        # Simulate the nonce that would be generated in _build_wrapper_script
        executor._run_nonces["baseline"] = "test-nonce-123"
        client = AsyncMock()
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.COMPLETED),
        ])
        # Wrapper script writes preemption info with matching nonce
        client.download_from_bucket = AsyncMock(side_effect=[
            json.dumps({
                "preemption_detected": True,
                "resume_checkpoint": "/output/exp/arm/checkpoint_500.pt",
                "run_nonce": "test-nonce-123",
            }),
            json.dumps({"step": 501, "loss": 0.5}),
        ])

        with patch(
            "ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.fleet_breadcrumb",
        ) as breadcrumb, patch(
            "ratiocinator.fleet.hf_executor.fleet_metric",
        ) as metric:
            stage = await executor._poll_job(client, "j", "baseline")

        assert stage == HFJobStage.COMPLETED
        # Preemption detected from state.json flag
        categories = [c.kwargs.get("category") for c in breadcrumb.call_args_list]
        assert "fleet.hf.preemption" in categories
        metric_names = [c.args[0] for c in metric.call_args_list]
        assert "fleet.arm.preemption" in metric_names

    async def test_poll_ignores_stale_preemption_flag(
        self, basic_spec, hf_config,
    ):
        """A stale preemption_detected payload with wrong/missing run_nonce
        should NOT trigger a preemption alert."""
        import json

        from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage

        executor = HFFleetExecutor(basic_spec, hf_config)
        executor._run_nonces["baseline"] = "current-nonce-xyz"
        client = AsyncMock()
        client.get_job = AsyncMock(side_effect=[
            HFJobInfo(job_id="j", stage=HFJobStage.RUNNING),
            HFJobInfo(job_id="j", stage=HFJobStage.COMPLETED),
        ])
        # Stale preemption payload from a PREVIOUS run (wrong nonce)
        client.download_from_bucket = AsyncMock(side_effect=[
            json.dumps({
                "preemption_detected": True,
                "resume_checkpoint": "/output/exp/arm/checkpoint_100.pt",
                "run_nonce": "old-nonce-abc",
            }),
            json.dumps({"step": 1, "loss": 2.0}),
        ])

        with patch(
            "ratiocinator.fleet.hf_executor.JOB_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.HEARTBEAT_POLL_INTERVAL_S", 0,
        ), patch(
            "ratiocinator.fleet.hf_executor.fleet_breadcrumb",
        ) as breadcrumb, patch(
            "ratiocinator.fleet.hf_executor.fleet_metric",
        ) as metric:
            stage = await executor._poll_job(client, "j", "baseline")

        assert stage == HFJobStage.COMPLETED
        categories = [c.kwargs.get("category") for c in breadcrumb.call_args_list]
        assert "fleet.hf.preemption" not in categories
        metric_names = [c.args[0] for c in metric.call_args_list]
        assert "fleet.arm.preemption" not in metric_names

