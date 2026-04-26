"""Tests for the HuggingFace Jobs + Buckets async client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ratiocinator.infra.hf_client import (
    HF_FLAVOR_PRICING,
    HFClient,
    HFClientError,
    HFJobInfo,
    HFJobStage,
)

# ---------------------------------------------------------------------------
# HFJobStage
# ---------------------------------------------------------------------------


class TestHFJobStage:
    def test_terminal_stages(self):
        assert HFJobStage.COMPLETED.is_terminal
        assert HFJobStage.ERROR.is_terminal
        assert HFJobStage.FAILED.is_terminal
        assert HFJobStage.CANCELLED.is_terminal
        assert HFJobStage.DELETED.is_terminal

    def test_non_terminal_stages(self):
        assert not HFJobStage.PENDING.is_terminal
        assert not HFJobStage.STARTING.is_terminal
        assert not HFJobStage.RUNNING.is_terminal
        assert not HFJobStage.UPDATING.is_terminal
        assert not HFJobStage.UNKNOWN.is_terminal


# ---------------------------------------------------------------------------
# HFJobInfo
# ---------------------------------------------------------------------------


class TestHFJobInfo:
    def test_defaults(self):
        info = HFJobInfo(job_id="job-123", stage=HFJobStage.RUNNING)
        assert info.job_id == "job-123"
        assert info.stage == HFJobStage.RUNNING
        assert info.flavor == ""
        assert info.labels == {}
        assert info.created_at is None

    def test_with_all_fields(self):
        from datetime import datetime

        now = datetime.now()
        info = HFJobInfo(
            job_id="job-456",
            stage=HFJobStage.COMPLETED,
            flavor="a100-large",
            image="pytorch/pytorch:2.7.0",
            created_at=now,
            namespace="my-org",
            labels={"experiment": "test"},
            status_message="done",
        )
        assert info.namespace == "my-org"
        assert info.labels["experiment"] == "test"


# ---------------------------------------------------------------------------
# HFClientError
# ---------------------------------------------------------------------------


class TestHFClientError:
    def test_message(self):
        err = HFClientError("something broke")
        assert str(err) == "something broke"

    def test_is_exception(self):
        assert issubclass(HFClientError, Exception)


# ---------------------------------------------------------------------------
# HFClient — mock the sync helpers, not huggingface_hub directly
# ---------------------------------------------------------------------------


class TestHFClient:
    @pytest.fixture
    def client(self):
        return HFClient(token="hf_test_token")

    async def test_context_manager(self, client):
        async with client as c:
            assert c is client
        assert client._api is None

    async def test_run_job(self, client):
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-abc123"
        mock_api.run_job.return_value = mock_job
        client._api = mock_api

        job_id = await client.run_job(
            image="pytorch/pytorch:2.7.0",
            command=["python", "train.py"],
            flavor="a100-large",
            timeout="2h",
            env={"LR": "0.001"},
        )

        assert job_id == "job-abc123"
        mock_api.run_job.assert_called_once()
        call_kwargs = mock_api.run_job.call_args.kwargs
        assert call_kwargs["image"] == "pytorch/pytorch:2.7.0"
        assert call_kwargs["flavor"] == "a100-large"
        assert call_kwargs["env"]["LR"] == "0.001"

    async def test_get_job(self, client):
        mock_api = MagicMock()
        mock_status = MagicMock()
        mock_status.stage = MagicMock(value="RUNNING")
        mock_status.message = "training in progress"
        mock_info = MagicMock()
        mock_info.id = "job-xyz"
        mock_info.status = mock_status
        mock_info.flavor = "a100-large"
        mock_info.image = "pytorch/pytorch:2.7.0"
        mock_info.created_at = None
        mock_info.owner = MagicMock(name="test-ns")
        mock_info.labels = {"arm": "baseline"}
        mock_api.inspect_job.return_value = mock_info
        client._api = mock_api

        info = await client.get_job("job-xyz")
        assert isinstance(info, HFJobInfo)
        assert info.job_id == "job-xyz"
        assert info.stage == HFJobStage.RUNNING
        assert info.flavor == "a100-large"

    async def test_get_job_logs(self, client):
        mock_api = MagicMock()
        mock_api.fetch_job_logs.return_value = iter(["METRICS:{\"loss\": 0.5}\n"])
        client._api = mock_api

        logs = await client.get_job_logs("job-log-test")
        assert "METRICS" in logs
        mock_api.fetch_job_logs.assert_called_once_with(job_id="job-log-test")

    async def test_cancel_job(self, client):
        mock_api = MagicMock()
        client._api = mock_api

        await client.cancel_job("job-cancel")
        mock_api.cancel_job.assert_called_once_with(job_id="job-cancel")

    async def test_list_jobs(self, client):
        mock_api = MagicMock()
        mock_status = MagicMock()
        mock_status.stage = MagicMock(value="COMPLETED")
        mock_status.message = ""
        mock_job = MagicMock()
        mock_job.id = "job-1"
        mock_job.status = mock_status
        mock_job.flavor = "l4"
        mock_job.image = "test:latest"
        mock_job.created_at = None
        mock_job.owner = None
        mock_job.labels = {}
        mock_api.list_jobs.return_value = [mock_job]
        client._api = mock_api

        jobs = await client.list_jobs(namespace="my-ns")
        assert len(jobs) == 1
        assert jobs[0].stage == HFJobStage.COMPLETED

    async def test_create_bucket(self, client):
        mock_api = MagicMock()
        client._api = mock_api

        await client.create_bucket("user/my-bucket", private=True)
        mock_api.create_bucket.assert_called_once_with(
            "user/my-bucket", private=True, exist_ok=True,
        )

    async def test_upload_to_bucket(self, client):
        mock_api = MagicMock()
        client._api = mock_api

        await client.upload_to_bucket(
            "user/my-bucket", "/tmp/test.sh", "experiment/arm/run.sh",
        )
        mock_api.batch_bucket_files.assert_called_once_with(
            "user/my-bucket",
            add=[("/tmp/test.sh", "experiment/arm/run.sh")],
        )

    async def test_api_error_wraps_in_hf_client_error(self, client):
        mock_api = MagicMock()
        mock_api.fetch_job_logs.side_effect = RuntimeError("network timeout")
        client._api = mock_api

        with pytest.raises(HFClientError, match="HF API call failed"):
            await client.get_job_logs("job-err")

    async def test_parse_job_unknown_stage(self, client):
        mock_info = MagicMock()
        mock_info.id = "job-unknown"
        mock_status = MagicMock()
        mock_status.stage = MagicMock(value="WEIRD_STATUS")
        mock_status.message = ""
        mock_info.status = mock_status
        mock_info.flavor = ""
        mock_info.image = ""
        mock_info.created_at = None
        mock_info.owner = None
        mock_info.labels = None

        result = client._parse_job(mock_info)
        assert result.stage == HFJobStage.UNKNOWN

    async def test_parse_job_no_status(self, client):
        mock_info = MagicMock()
        mock_info.id = "job-nostatus"
        mock_info.status = None
        mock_info.flavor = "l4"
        mock_info.image = ""
        mock_info.created_at = None
        mock_info.owner = None
        mock_info.labels = {}

        result = client._parse_job(mock_info)
        assert result.stage == HFJobStage.UNKNOWN
        assert result.flavor == "l4"


# ---------------------------------------------------------------------------
# Pricing data
# ---------------------------------------------------------------------------


class TestFlavorPricing:
    def test_known_flavors(self):
        assert HF_FLAVOR_PRICING["a100-large"] == 2.50
        assert HF_FLAVOR_PRICING["t4-small"] == 0.40
        assert HF_FLAVOR_PRICING["l4"] == 0.80

    def test_multi_gpu_flavors(self):
        assert HF_FLAVOR_PRICING["4xa100"] == 10.00
        assert HF_FLAVOR_PRICING["8xa100"] == 20.00
