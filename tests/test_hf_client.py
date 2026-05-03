"""Tests for the HuggingFace Jobs + Buckets async client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

    def test_rejects_old_huggingface_hub(self, client):
        """Version gate raises ImportError for huggingface_hub < 1.9.0."""
        mock_hf_module = MagicMock()
        mock_hf_module.__version__ = "1.8.3"
        mock_hf_module.HfApi = MagicMock()

        with (
            patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}),
            pytest.raises(ImportError, match=r"huggingface_hub>=1\.9\.0 is required"),
        ):
            client._get_api()

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

    async def test_download_from_bucket_returns_text(self, client, tmp_path):
        mock_api = MagicMock()
        client._api = mock_api

        # Simulate the SDK writing the requested local file
        def fake_download(bucket_id, files, raise_on_missing_files=False, token=None):
            assert bucket_id == "user/my-bucket"
            (remote, local) = files[0]
            assert remote == "exp/arm/state.json"
            from pathlib import Path
            Path(local).write_text('{"step": 42, "loss": 0.1}', encoding="utf-8")

        mock_api.download_bucket_files.side_effect = fake_download

        text = await client.download_from_bucket(
            "user/my-bucket", "exp/arm/state.json",
        )
        assert text == '{"step": 42, "loss": 0.1}'

    async def test_download_from_bucket_missing_returns_none(self, client):
        from huggingface_hub.errors import EntryNotFoundError

        mock_api = MagicMock()
        mock_api.download_bucket_files.side_effect = EntryNotFoundError(
            "404: state.json not found",
        )
        client._api = mock_api

        result = await client.download_from_bucket(
            "user/my-bucket", "exp/arm/state.json",
        )
        assert result is None

    async def test_download_from_bucket_other_error_wraps(self, client):
        mock_api = MagicMock()
        mock_api.download_bucket_files.side_effect = RuntimeError("auth failed")
        client._api = mock_api

        with pytest.raises(HFClientError, match="HF API call failed"):
            await client.download_from_bucket(
                "user/my-bucket", "exp/arm/state.json",
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

    async def test_find_job_by_labels_no_match(self, client):
        mock_api = MagicMock()
        mock_api.list_jobs.return_value = []
        client._api = mock_api

        result = await client.find_job_by_labels(
            {"experiment": "test", "arm": "baseline"},
        )
        assert result is None

    async def test_find_job_by_labels_prefers_active(self, client):
        from datetime import datetime

        mock_api = MagicMock()

        # Create a terminal (completed) job and an active (running) job
        def _make_mock_job(job_id, stage_val, labels, created_at=None):
            j = MagicMock()
            j.id = job_id
            status = MagicMock()
            status.stage = MagicMock(value=stage_val)
            status.message = ""
            j.status = status
            j.flavor = "l4"
            j.image = ""
            j.created_at = created_at
            j.owner = None
            j.labels = labels
            return j

        old_completed = _make_mock_job(
            "job-old", "COMPLETED",
            {"experiment": "test", "arm": "baseline"},
            created_at=datetime(2026, 1, 1),
        )
        new_running = _make_mock_job(
            "job-new", "RUNNING",
            {"experiment": "test", "arm": "baseline"},
            created_at=datetime(2026, 1, 2),
        )

        mock_api.list_jobs.return_value = [old_completed, new_running]
        client._api = mock_api

        result = await client.find_job_by_labels(
            {"experiment": "test", "arm": "baseline"},
        )
        assert result is not None
        assert result.job_id == "job-new"
        assert result.stage == HFJobStage.RUNNING

    async def test_find_job_by_labels_most_recent_terminal(self, client):
        from datetime import datetime

        mock_api = MagicMock()

        def _make_mock_job(job_id, stage_val, labels, created_at=None):
            j = MagicMock()
            j.id = job_id
            status = MagicMock()
            status.stage = MagicMock(value=stage_val)
            status.message = ""
            j.status = status
            j.flavor = "l4"
            j.image = ""
            j.created_at = created_at
            j.owner = None
            j.labels = labels
            return j

        old_error = _make_mock_job(
            "job-err-old", "ERROR",
            {"experiment": "test", "arm": "baseline"},
            created_at=datetime(2026, 1, 1),
        )
        new_completed = _make_mock_job(
            "job-done-new", "COMPLETED",
            {"experiment": "test", "arm": "baseline"},
            created_at=datetime(2026, 1, 5),
        )

        mock_api.list_jobs.return_value = [old_error, new_completed]
        client._api = mock_api

        result = await client.find_job_by_labels(
            {"experiment": "test", "arm": "baseline"},
        )
        assert result is not None
        assert result.job_id == "job-done-new"

    async def test_find_job_by_labels_partial_match_excluded(self, client):
        from datetime import datetime

        mock_api = MagicMock()

        def _make_mock_job(job_id, stage_val, labels, created_at=None):
            j = MagicMock()
            j.id = job_id
            status = MagicMock()
            status.stage = MagicMock(value=stage_val)
            status.message = ""
            j.status = status
            j.flavor = "l4"
            j.image = ""
            j.created_at = created_at
            j.owner = None
            j.labels = labels
            return j

        # Job matches experiment but wrong arm
        wrong_arm = _make_mock_job(
            "job-wrong", "RUNNING",
            {"experiment": "test", "arm": "other-arm"},
            created_at=datetime(2026, 1, 2),
        )

        mock_api.list_jobs.return_value = [wrong_arm]
        client._api = mock_api

        result = await client.find_job_by_labels(
            {"experiment": "test", "arm": "baseline"},
        )
        assert result is None


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


# ---------------------------------------------------------------------------
# download_artifact
# ---------------------------------------------------------------------------


class TestDownloadArtifact:
    @pytest.fixture
    def client(self):
        return HFClient(token="hf_test")

    async def test_successful_download(self, client, tmp_path):
        """Chunked download succeeds on first attempt."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"x" * 1024 * 100  # 100 KB
        dest = tmp_path / "model.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": len(data)}
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(data),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                chunk_size=1024,
            )

        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size == len(data)

    async def test_retries_on_failure(self, client, tmp_path):
        """Download retries on transient errors."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"checkpoint_data_here"
        dest = tmp_path / "model.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": len(data)}
        attempt = [0]

        def fake_open(*a, **kw):
            attempt[0] += 1
            if attempt[0] < 3:
                # First two attempts raise on read
                raise ConnectionError("connection reset")
            # Third attempt succeeds
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=BytesIO(data))
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        mock_fs.open = fake_open

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                retry_delay=0.01,
            )

        assert result == dest
        assert dest.read_bytes() == data

    async def test_fails_on_zero_bytes(self, client, tmp_path):
        """Download fails if file is 0 bytes after write."""
        from io import BytesIO
        from unittest.mock import patch

        dest = tmp_path / "model.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": 1024}  # Remote is non-empty
        # Return empty data — simulates silent connection drop
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(b""),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "huggingface_hub.HfFileSystem",
                return_value=mock_fs,
            ),
            pytest.raises(HFClientError, match="0 bytes"),
        ):
            await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                max_retries=1,
                retry_delay=0.01,
            )

    async def test_fails_on_size_mismatch(self, client, tmp_path):
        """Download fails if downloaded size doesn't match expected."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"short"
        dest = tmp_path / "model.pt"

        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(data),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "huggingface_hub.HfFileSystem",
                return_value=mock_fs,
            ),
            pytest.raises(HFClientError, match="Size mismatch"),
        ):
            await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                expected_size=999999,
                max_retries=1,
                retry_delay=0.01,
            )

    async def test_detects_truncated_download_via_remote_size(self, client, tmp_path):
        """Truncated download is caught even without expected_size."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"short"
        dest = tmp_path / "model.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": 999999}  # Remote is much larger
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(data),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "huggingface_hub.HfFileSystem",
                return_value=mock_fs,
            ),
            pytest.raises(HFClientError, match="Truncated download"),
        ):
            await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                max_retries=1,
                retry_delay=0.01,
            )

    async def test_creates_parent_directories(self, client, tmp_path):
        """Download creates parent directories if they don't exist."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"model_weights"
        dest = tmp_path / "deep" / "nested" / "dir" / "model.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": len(data)}
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(data),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await client.download_artifact(
                "user/bucket", "model.pt", dest,
            )

        assert result == dest
        assert dest.read_bytes() == data


class TestDownloadArtifactModuleFunction:
    async def test_module_function_delegates_to_client(self, tmp_path):
        """Module-level download_artifact creates a client and delegates."""
        from io import BytesIO
        from unittest.mock import patch

        from ratiocinator.infra.hf_client import (
            download_artifact,
        )

        data = b"checkpoint"
        dest = tmp_path / "out.pt"

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": len(data)}
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(data),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await download_artifact(
                "user/bucket", "model.pt", dest, token="hf_test",
            )

        assert result == dest
        assert dest.read_bytes() == data


class TestDownloadArtifactValidation:
    @pytest.fixture
    def client(self):
        return HFClient(token="hf_test")

    async def test_rejects_zero_max_retries(self, client, tmp_path):
        """max_retries=0 raises ValueError immediately."""
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            await client.download_artifact(
                "user/bucket", "model.pt", tmp_path / "out.pt",
                max_retries=0,
            )

    async def test_rejects_zero_chunk_size(self, client, tmp_path):
        """chunk_size<=0 raises ValueError immediately."""
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            await client.download_artifact(
                "user/bucket", "model.pt", tmp_path / "out.pt",
                chunk_size=0,
            )

    async def test_rejects_negative_retry_delay(self, client, tmp_path):
        """retry_delay<0 raises ValueError immediately."""
        with pytest.raises(ValueError, match="retry_delay must be >= 0"):
            await client.download_artifact(
                "user/bucket", "model.pt", tmp_path / "out.pt",
                retry_delay=-1.0,
            )

    async def test_expected_size_zero_allows_empty_file(self, client, tmp_path):
        """expected_size=0 permits legitimate empty artifacts."""
        from io import BytesIO
        from unittest.mock import patch

        dest = tmp_path / "empty.pt"

        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock(
            return_value=BytesIO(b""),
        )
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await client.download_artifact(
                "user/bucket", "empty.pt", dest,
                expected_size=0,
            )

        assert result == dest
        assert dest.read_bytes() == b""

    async def test_partial_write_cleanup_on_mid_read_failure(self, client, tmp_path):
        """Connection drop mid-read cleans up partial file and retries."""
        from io import BytesIO
        from unittest.mock import patch

        data = b"A" * 5000  # Full data
        dest = tmp_path / "model.pt"
        attempt = [0]

        class PartialThenSuccess:
            """First call reads some bytes then raises; second succeeds."""

            def __init__(self, succeed: bool):
                self._succeed = succeed
                self._buf = BytesIO(data)
                self._reads = 0

            def read(self, n):
                if not self._succeed:
                    self._reads += 1
                    if self._reads == 1:
                        # First chunk succeeds (partial write)
                        return b"X" * min(n, 1000)
                    # Second chunk — connection drops
                    raise ConnectionError("connection reset mid-transfer")
                return self._buf.read(n)

        def fake_open(*a, **kw):
            attempt[0] += 1
            succeed = attempt[0] >= 2
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=PartialThenSuccess(succeed))
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        mock_fs = MagicMock()
        mock_fs.info.return_value = {"size": len(data)}
        mock_fs.open = fake_open

        with patch(
            "huggingface_hub.HfFileSystem", return_value=mock_fs,
        ):
            result = await client.download_artifact(
                "user/bucket", "checkpoints/model.pt", dest,
                chunk_size=1000,
                retry_delay=0.01,
            )

        assert result == dest
        assert dest.read_bytes() == data
        # Verify no .tmp files remain (mkstemp creates dot-prefixed names)
        tmp_files = list(dest.parent.glob(".*.tmp"))
        assert tmp_files == []
