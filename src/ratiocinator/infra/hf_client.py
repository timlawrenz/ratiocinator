"""Async HuggingFace Jobs + Buckets client.

Wraps ``huggingface_hub.HfApi`` with async wrappers (``asyncio.to_thread``),
Sentry observability spans, and structured data models.  Mirrors the patterns
established by ``vast_client.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

try:
    from huggingface_hub.errors import EntryNotFoundError as _HFEntryNotFoundError
except ImportError:  # pragma: no cover - SDK not installed
    _HFEntryNotFoundError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Hourly price per flavor — used for budget estimation.
# Source: https://huggingface.co/docs/hub/jobs-pricing (April 2026)
HF_FLAVOR_PRICING: dict[str, float] = {
    "cpu-basic": 0.01,
    "cpu-upgrade": 0.03,
    "cpu-xl": 1.00,
    "cpu-performance": 1.90,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4": 0.80,
    "l4x1": 0.80,
    "l4x4": 3.80,
    "4xl4": 3.80,
    "l40s": 1.80,
    "l40sx1": 1.80,
    "l40sx4": 8.30,
    "4xl40s": 8.30,
    "l40sx8": 23.50,
    "8xl40s": 23.50,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "a10g-largex2": 3.00,
    "2xa10g-large": 3.00,
    "a10g-largex4": 5.00,
    "4xa10g-large": 5.00,
    "a100-large": 2.50,
    "4xa100": 10.00,
    "a100x4": 10.00,
    "8xa100": 20.00,
    "a100x8": 20.00,
    "h200": 5.00,
    "h200x2": 10.00,
    "h200x4": 20.00,
    "h200x8": 40.00,
}


class HFJobStage(Enum):
    """Terminal and non-terminal stages of an HF Job."""

    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    UPDATING = "UPDATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DELETED = "DELETED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in (
            HFJobStage.COMPLETED,
            HFJobStage.ERROR,
            HFJobStage.FAILED,
            HFJobStage.CANCELLED,
            HFJobStage.DELETED,
        )


@dataclass
class HFJobInfo:
    """Represents a HuggingFace Job."""

    job_id: str
    stage: HFJobStage
    flavor: str = ""
    image: str = ""
    created_at: datetime | None = None
    namespace: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    status_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class HFClientError(Exception):
    """HuggingFace client error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class HFClient:
    """Async wrapper around ``huggingface_hub.HfApi`` for Jobs and Buckets.

    All I/O runs through ``asyncio.to_thread`` so the sync HfApi calls
    don't block the event loop.

    Usage::

        async with HFClient(token="hf_...") as client:
            job_id = await client.run_job(
                image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
                command=["bash", "/input/run.sh"],
                flavor="a100-large",
            )
            info = await client.get_job(job_id)
    """

    def __init__(self, token: str = "") -> None:
        self.token = token
        self._api: Any = None  # Lazy init to avoid import at module level

    def _get_api(self) -> Any:
        if self._api is None:
            import huggingface_hub
            from huggingface_hub import HfApi

            _parts = huggingface_hub.__version__.split(".")[:2]
            _ver = tuple(int("".join(c for c in p if c.isdigit()) or "0") for p in _parts)
            if _ver < (1, 9):
                raise ImportError(
                    f"huggingface_hub>=1.9.0 is required for hf:// fsspec protocol "
                    f"support (found {huggingface_hub.__version__}). "
                    f"Upgrade with: pip install 'huggingface_hub>=1.9.0'"
                )

            # Pass token if explicitly provided; otherwise HfApi auto-discovers
            # from ~/.huggingface/token or HF_TOKEN env var.
            self._api = HfApi(token=self.token or None)
        return self._api

    async def close(self) -> None:
        self._api = None

    async def __aenter__(self) -> HFClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Jobs API
    # ------------------------------------------------------------------

    async def run_job(
        self,
        image: str,
        command: list[str],
        *,
        flavor: str = "a100-large",
        timeout: str = "2h",
        env: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        volumes: list[Any] | None = None,
        labels: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Submit a job and return its ID.

        Args:
            image: Docker image (from Docker Hub or HF Spaces).
            command: Command to run inside the container.
            flavor: Hardware flavor (e.g. ``"a100-large"``).
            timeout: Maximum duration (e.g. ``"2h"``, ``"30m"``).
            env: Environment variables passed to the container.
            secrets: Secret environment variables (fetched from HF account).
            volumes: List of ``huggingface_hub.Volume`` objects to mount.
            labels: Key-value labels for filtering/tracking.
            namespace: HF username or org to run under.

        Returns:
            Job ID string.
        """
        return await self._traced(
            "hf.jobs.run",
            f"run_job {flavor}",
            self._run_job_sync,
            image=image,
            command=command,
            flavor=flavor,
            timeout=timeout,
            env=env,
            secrets=secrets,
            volumes=volumes,
            labels=labels,
            namespace=namespace,
        )

    def _run_job_sync(
        self,
        *,
        image: str,
        command: list[str],
        flavor: str,
        timeout: str,
        env: dict[str, str] | None,
        secrets: dict[str, str] | None,
        volumes: list[Any] | None,
        labels: dict[str, str] | None,
        namespace: str | None,
    ) -> str:
        api = self._get_api()
        # Convert volume dicts to Volume objects if needed
        vol_objects = None
        if volumes:
            vol_objects = self._to_volume_objects(volumes)
        job_info = api.run_job(
            image=image,
            command=command,
            flavor=flavor,
            timeout=timeout,
            env=env or {},
            secrets=secrets or {},
            volumes=vol_objects,
            labels=labels,
            namespace=namespace,
        )
        return job_info.id

    @staticmethod
    def _to_volume_objects(volumes: list[Any]) -> list[Any]:
        """Convert volume dicts to huggingface_hub.Volume objects."""
        from huggingface_hub import Volume

        result = []
        for v in volumes:
            if isinstance(v, dict):
                result.append(Volume(
                    type=v.get("type", "bucket"),
                    source=v["source"],
                    mount_path=v.get("mountPath", v.get("mount_path", "/data")),
                    read_only=v.get("readOnly", v.get("read_only")),
                    revision=v.get("revision"),
                    path=v.get("path"),
                ))
            else:
                result.append(v)  # Already a Volume object
        return result

    async def get_job(self, job_id: str) -> HFJobInfo:
        """Get current status of a job."""
        return await self._traced(
            "hf.jobs.get",
            f"get_job {job_id}",
            self._get_job_sync,
            job_id=job_id,
        )

    def _get_job_sync(self, *, job_id: str) -> HFJobInfo:
        api = self._get_api()
        info = api.inspect_job(job_id=job_id)
        return self._parse_job(info)

    async def get_job_logs(self, job_id: str) -> str:
        """Fetch stdout/stderr logs for a completed or running job."""
        return await self._traced(
            "hf.jobs.logs",
            f"logs {job_id}",
            self._get_job_logs_sync,
            job_id=job_id,
        )

    def _get_job_logs_sync(self, *, job_id: str) -> str:
        api = self._get_api()
        # fetch_job_logs returns an iterable of string chunks (no newlines)
        return "\n".join(api.fetch_job_logs(job_id=job_id))

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        await self._traced(
            "hf.jobs.cancel",
            f"cancel {job_id}",
            self._cancel_job_sync,
            job_id=job_id,
        )

    def _cancel_job_sync(self, *, job_id: str) -> None:
        api = self._get_api()
        api.cancel_job(job_id=job_id)

    async def list_jobs(
        self,
        *,
        namespace: str | None = None,
    ) -> list[HFJobInfo]:
        """List jobs, optionally filtered by namespace."""
        return await self._traced(
            "hf.jobs.list",
            "list_jobs",
            self._list_jobs_sync,
            namespace=namespace,
        )

    def _list_jobs_sync(self, *, namespace: str | None) -> list[HFJobInfo]:
        api = self._get_api()
        jobs = api.list_jobs(namespace=namespace)
        return [self._parse_job(j) for j in jobs]

    async def find_job_by_labels(
        self,
        labels: dict[str, str],
        *,
        namespace: str | None = None,
    ) -> HFJobInfo | None:
        """Find the most recent job whose labels match all given key-value pairs.

        Prefers non-terminal (active) jobs.  If multiple active jobs match,
        returns the one created most recently.  If no active job matches,
        returns the most recently created terminal job.

        Returns ``None`` if no job matches the labels at all.
        """
        all_jobs = await self.list_jobs(namespace=namespace)
        matches = [
            j for j in all_jobs
            if all(j.labels.get(k) == v for k, v in labels.items())
        ]
        if not matches:
            return None

        def _sort_key(j: HFJobInfo) -> float:
            """Return a numeric timestamp for sorting; -inf if unknown."""
            if j.created_at is None:
                return float("-inf")
            return j.created_at.timestamp()

        # Separate active vs terminal
        active = [j for j in matches if not j.stage.is_terminal]
        if active:
            # Prefer most recently created active job
            active.sort(key=_sort_key, reverse=True)
            return active[0]

        # All matching jobs are terminal — return most recent
        matches.sort(key=_sort_key, reverse=True)
        return matches[0]

    # ------------------------------------------------------------------
    # Bucket API
    # ------------------------------------------------------------------

    async def create_bucket(
        self,
        name: str,
        *,
        private: bool = True,
    ) -> None:
        """Create an HF Bucket (idempotent with ``exist_ok=True``)."""
        await self._traced(
            "hf.bucket.create",
            f"create_bucket {name}",
            self._create_bucket_sync,
            name=name,
            private=private,
        )

    def _create_bucket_sync(self, *, name: str, private: bool) -> None:
        api = self._get_api()
        api.create_bucket(name, private=private, exist_ok=True)

    async def upload_to_bucket(
        self,
        bucket_name: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload a file or directory to an HF Bucket."""
        await self._traced(
            "hf.bucket.upload",
            f"upload {bucket_name}/{remote_path}",
            self._upload_to_bucket_sync,
            bucket_name=bucket_name,
            local_path=local_path,
            remote_path=remote_path,
        )

    def _upload_to_bucket_sync(
        self,
        *,
        bucket_name: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        api = self._get_api()
        api.batch_bucket_files(
            bucket_name,
            add=[(local_path, remote_path)],
        )

    async def download_from_bucket(
        self,
        bucket_name: str,
        remote_path: str,
    ) -> str | None:
        """Download a small text file from an HF Bucket.

        Returns the file's UTF-8 contents, or ``None`` if the file does
        not exist in the bucket (treated as a benign "not yet written"
        condition, e.g. heartbeat polling before the training script
        has produced its first ``state.json``).

        Other errors (auth, network, etc.) raise :class:`HFClientError`.
        """
        return await self._traced(
            "hf.bucket.download",
            f"download {bucket_name}/{remote_path}",
            self._download_from_bucket_sync,
            bucket_name=bucket_name,
            remote_path=remote_path,
        )

    def _download_from_bucket_sync(
        self,
        *,
        bucket_name: str,
        remote_path: str,
    ) -> str | None:
        api = self._get_api()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "downloaded"
            try:
                api.download_bucket_files(
                    bucket_name,
                    files=[(remote_path, str(local_path))],
                    raise_on_missing_files=True,
                )
            except Exception as exc:
                # Treat "file not found" as a benign None return — this
                # is the expected condition during heartbeat polling
                # before the training script writes its first state.json.
                if (
                    _HFEntryNotFoundError is not None
                    and isinstance(exc, _HFEntryNotFoundError)
                ):
                    return None
                raise
            if not local_path.exists():
                return None
            return local_path.read_text(encoding="utf-8", errors="replace")

    async def sync_to_bucket(
        self,
        local_dir: str,
        bucket_path: str,
    ) -> None:
        """Sync a local directory to an HF Bucket path.

        Args:
            local_dir: Local directory path.
            bucket_path: ``hf://buckets/user/bucket/prefix`` style path.
        """
        await self._traced(
            "hf.bucket.sync",
            f"sync_to {bucket_path}",
            self._sync_to_bucket_sync,
            local_dir=local_dir,
            bucket_path=bucket_path,
        )

    def _sync_to_bucket_sync(
        self,
        *,
        local_dir: str,
        bucket_path: str,
    ) -> None:
        api = self._get_api()
        api.sync_bucket(source=local_dir, dest=bucket_path)

    # ------------------------------------------------------------------
    # Large artifact download (chunked + retries)
    # ------------------------------------------------------------------

    async def download_artifact(
        self,
        bucket_name: str,
        remote_path: str,
        local_path: str | Path,
        *,
        expected_size: int | None = None,
        chunk_size: int = 10 * 1024 * 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> Path:
        """Download a large file from an HF Bucket with chunked I/O and retries.

        Designed for multi-GB artifacts (e.g. model checkpoints) where
        ``HfFileSystem.get()`` or simple downloads may silently drop
        connections, producing corrupted 0-byte files.

        Args:
            bucket_name: HF bucket identifier (e.g. ``"user/my-bucket"``).
            remote_path: Path within the bucket to download.
            local_path: Destination path on local filesystem.
            expected_size: If provided, validate downloaded size matches.
                Raises :class:`HFClientError` on mismatch.
            chunk_size: Read chunk size in bytes (default 10 MB).
            max_retries: Number of retry attempts on failure (default 3).
            retry_delay: Base delay between retries in seconds (doubles each
                attempt).

        Returns:
            Resolved :class:`~pathlib.Path` to the downloaded file.

        Raises:
            HFClientError: On download failure after retries, or size mismatch.
        """
        dest = Path(local_path)
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return await self._traced(
                    "hf.bucket.download_artifact",
                    f"download_artifact {bucket_name}/{remote_path}",
                    self._download_artifact_sync,
                    bucket_name=bucket_name,
                    remote_path=remote_path,
                    dest=str(dest),
                    expected_size=expected_size,
                    chunk_size=chunk_size,
                    attempt=attempt,
                    max_retries=max_retries,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))

        # All retries exhausted — last_exc is always set when we reach here
        assert last_exc is not None
        raise last_exc

    def _download_artifact_sync(
        self,
        *,
        bucket_name: str,
        remote_path: str,
        dest: str,
        expected_size: int | None,
        chunk_size: int,
        attempt: int,
        max_retries: int,
    ) -> Path:
        from huggingface_hub import HfFileSystem

        # Enforce version gate (huggingface_hub>=1.9.0) via shared initializer
        self._get_api()

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        hf_path = f"hf://buckets/{bucket_name}/{remote_path}"
        fs = HfFileSystem(token=self.token or None)

        # Use mkstemp in the destination directory for a truly unique temp
        # file — safe under concurrent asyncio tasks in the same process.
        fd, tmp_name = tempfile.mkstemp(
            suffix=".tmp", prefix=f".dl_{dest_path.stem}_", dir=dest_path.parent,
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            written = 0
            with fs.open(hf_path, "rb") as remote_f, open(tmp_path, "wb") as local_f:
                while True:
                    chunk = remote_f.read(chunk_size)
                    if not chunk:
                        break
                    local_f.write(chunk)
                    written += len(chunk)

            # Validate file size
            actual_size = tmp_path.stat().st_size
            if actual_size == 0 and expected_size != 0:
                msg = (
                    f"Downloaded file is 0 bytes "
                    f"(attempt {attempt}/{max_retries}): {hf_path}"
                )
                raise OSError(msg)

            if expected_size is not None and actual_size != expected_size:
                msg = (
                    f"Size mismatch for {hf_path}: "
                    f"expected {expected_size} bytes, got {actual_size}"
                )
                raise OSError(msg)

            # Success — atomically move to final destination
            os.replace(tmp_path, dest_path)
            logger.info(
                "Downloaded %s (%d bytes, attempt %d)",
                hf_path,
                actual_size,
                attempt,
            )
            return dest_path

        except Exception:
            # Clean up partial file
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _traced(
        self,
        op: str,
        description: str,
        sync_fn: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a sync function in a thread with Sentry span tracing."""
        span_ctx = (
            sentry_sdk.start_span(op=op, name=description)
            if sentry_sdk
            else None
        )
        if span_ctx:
            span_ctx.__enter__()

        try:
            result = await asyncio.to_thread(sync_fn, **kwargs)
            if span_ctx:
                span_ctx.__exit__(None, None, None)
            return result
        except Exception as exc:
            if span_ctx:
                span_ctx.set_status("internal_error")
                span_ctx.__exit__(None, None, None)
            raise HFClientError(f"HF API call failed ({op}): {exc}") from exc

    def _parse_job(self, info: Any) -> HFJobInfo:
        """Convert ``huggingface_hub.JobInfo`` to our dataclass."""
        stage_str = ""
        status_message = ""
        if hasattr(info, "status") and info.status is not None:
            if hasattr(info.status, "stage"):
                stage_val = info.status.stage
                stage_str = stage_val.value if hasattr(stage_val, "value") else str(stage_val)
            if hasattr(info.status, "message"):
                status_message = info.status.message or ""

        try:
            stage = HFJobStage(stage_str.upper())
        except (ValueError, KeyError):
            stage = HFJobStage.UNKNOWN

        labels: dict[str, str] = {}
        if hasattr(info, "labels") and info.labels:
            labels = dict(info.labels) if isinstance(info.labels, dict) else {}

        return HFJobInfo(
            job_id=info.id if hasattr(info, "id") else str(info),
            stage=stage,
            flavor=getattr(info, "flavor", "") or "",
            image=getattr(info, "image", "") or "",
            created_at=getattr(info, "created_at", None),
            namespace=(
                info.owner.name
                if hasattr(info, "owner") and info.owner
                else ""
            ),
            labels=labels,
            status_message=status_message,
            raw=info.__dict__ if hasattr(info, "__dict__") else {},
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


async def download_artifact(
    bucket_name: str,
    remote_path: str,
    local_path: str | Path,
    *,
    token: str = "",
    expected_size: int | None = None,
    chunk_size: int = 10 * 1024 * 1024,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Path:
    """Download a large artifact from an HF Bucket with chunked I/O and retries.

    Convenience wrapper around :meth:`HFClient.download_artifact` that
    manages its own client lifetime.  Designed for multi-GB files (e.g.
    model checkpoints) that are prone to silent connection drops when
    fetched via ``HfFileSystem.get()``.

    Args:
        bucket_name: HF bucket identifier (e.g. ``"user/my-bucket"``).
        remote_path: Path within the bucket to download.
        local_path: Destination path on local filesystem.
        token: HuggingFace API token (auto-discovered if empty).
        expected_size: If provided, validate downloaded size matches.
        chunk_size: Read chunk size in bytes (default 10 MB).
        max_retries: Number of retry attempts on failure (default 3).
        retry_delay: Base delay between retries in seconds.

    Returns:
        Resolved :class:`~pathlib.Path` to the downloaded file.

    Raises:
        HFClientError: On download failure after retries, or size mismatch.
    """
    async with HFClient(token=token) as client:
        return await client.download_artifact(
            bucket_name,
            remote_path,
            local_path,
            expected_size=expected_size,
            chunk_size=chunk_size,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
