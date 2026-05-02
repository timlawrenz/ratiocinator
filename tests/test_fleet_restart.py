"""Tests for ``ratiocinator.fleet.restart`` orphan-cleanup helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ratiocinator.fleet.restart import (
    cleanup_hf_orphans,
    cleanup_vast_orphans,
)
from ratiocinator.infra.hf_client import HFJobInfo, HFJobStage
from ratiocinator.infra.vast_client import InstanceInfo, InstanceStatus


def _hf_job(
    job_id: str,
    *,
    stage: HFJobStage,
    experiment: str = "exp1",
    arm: str = "baseline",
    extra_labels: dict[str, str] | None = None,
) -> HFJobInfo:
    labels = {"experiment": experiment, "arm": arm}
    if extra_labels:
        labels.update(extra_labels)
    return HFJobInfo(
        job_id=job_id,
        stage=stage,
        created_at=datetime.now(UTC),
        labels=labels,
    )


def _vast_instance(instance_id: int, label: str) -> InstanceInfo:
    return InstanceInfo(
        instance_id=instance_id,
        status=InstanceStatus.RUNNING,
        label=label,
    )


# ---------------------------------------------------------------------------
# HF cleanup
# ---------------------------------------------------------------------------


class TestCleanupHFOrphans:
    @pytest.mark.asyncio
    async def test_cancels_active_jobs_only(self):
        client = MagicMock()
        client.list_jobs = AsyncMock(return_value=[
            _hf_job("active-1", stage=HFJobStage.RUNNING),
            _hf_job("active-2", stage=HFJobStage.PENDING),
            _hf_job("done-1", stage=HFJobStage.COMPLETED),
            _hf_job("err-1", stage=HFJobStage.ERROR),
        ])
        client.cancel_job = AsyncMock()

        summary = await cleanup_hf_orphans(client, "exp1", "baseline")

        assert sorted(summary.cancelled_job_ids) == ["active-1", "active-2"]
        assert sorted(summary.skipped_terminal_job_ids) == ["done-1", "err-1"]
        assert client.cancel_job.await_count == 2
        cancel_args = {c.args[0] for c in client.cancel_job.await_args_list}
        assert cancel_args == {"active-1", "active-2"}

    @pytest.mark.asyncio
    async def test_filters_by_label_pair(self):
        client = MagicMock()
        client.list_jobs = AsyncMock(return_value=[
            _hf_job("match", stage=HFJobStage.RUNNING),
            _hf_job("wrong-arm", stage=HFJobStage.RUNNING, arm="other"),
            _hf_job("wrong-exp", stage=HFJobStage.RUNNING, experiment="otherexp"),
        ])
        client.cancel_job = AsyncMock()

        summary = await cleanup_hf_orphans(client, "exp1", "baseline")

        assert summary.cancelled_job_ids == ["match"]
        client.cancel_job.assert_awaited_once_with("match")

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(self):
        client = MagicMock()
        client.list_jobs = AsyncMock(return_value=[])
        client.cancel_job = AsyncMock()

        summary = await cleanup_hf_orphans(client, "exp1", "baseline")

        assert summary.cancelled_job_ids == []
        assert summary.skipped_terminal_job_ids == []
        client.cancel_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_propagates_namespace(self):
        client = MagicMock()
        client.list_jobs = AsyncMock(return_value=[])
        client.cancel_job = AsyncMock()

        await cleanup_hf_orphans(client, "exp1", "arm-a", namespace="my-org")

        client.list_jobs.assert_awaited_once_with(namespace="my-org")

    @pytest.mark.asyncio
    async def test_continues_on_cancel_failure(self):
        client = MagicMock()
        client.list_jobs = AsyncMock(return_value=[
            _hf_job("a", stage=HFJobStage.RUNNING),
            _hf_job("b", stage=HFJobStage.RUNNING),
        ])
        client.cancel_job = AsyncMock(side_effect=[RuntimeError("boom"), None])

        summary = await cleanup_hf_orphans(client, "exp1", "baseline")

        # Only the second cancel succeeded — the first failure must not
        # stop us from cleaning up subsequent jobs.
        assert summary.cancelled_job_ids == ["b"]
        assert client.cancel_job.await_count == 2

    @pytest.mark.asyncio
    async def test_uses_prefetched_jobs_without_api_call(self):
        client = MagicMock()
        client.list_jobs = AsyncMock()
        client.cancel_job = AsyncMock()
        prefetched = [
            _hf_job("a", stage=HFJobStage.RUNNING),
            _hf_job("b", stage=HFJobStage.RUNNING, arm="other"),
        ]

        summary = await cleanup_hf_orphans(
            client, "exp1", "baseline", jobs=prefetched,
        )

        client.list_jobs.assert_not_awaited()
        assert summary.cancelled_job_ids == ["a"]


# ---------------------------------------------------------------------------
# Vast cleanup
# ---------------------------------------------------------------------------


class TestCleanupVastOrphans:
    @pytest.mark.asyncio
    async def test_destroys_matching_instances(self):
        client = MagicMock()
        client.list_instances = AsyncMock(return_value=[
            _vast_instance(101, "exp1-baseline"),
            _vast_instance(102, "exp1-baseline"),
            _vast_instance(103, "exp1-other"),
            _vast_instance(104, "otherexp-baseline"),
        ])
        client.destroy_instance = AsyncMock()

        summary = await cleanup_vast_orphans(client, "exp1", "baseline")

        assert sorted(summary.destroyed_instance_ids) == [101, 102]
        destroyed = {c.args[0] for c in client.destroy_instance.await_args_list}
        assert destroyed == {101, 102}

    @pytest.mark.asyncio
    async def test_no_match(self):
        client = MagicMock()
        client.list_instances = AsyncMock(return_value=[
            _vast_instance(1, "unrelated"),
        ])
        client.destroy_instance = AsyncMock()

        summary = await cleanup_vast_orphans(client, "exp1", "baseline")

        assert summary.destroyed_instance_ids == []
        client.destroy_instance.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_continues_on_destroy_failure(self):
        client = MagicMock()
        client.list_instances = AsyncMock(return_value=[
            _vast_instance(1, "exp1-baseline"),
            _vast_instance(2, "exp1-baseline"),
        ])
        client.destroy_instance = AsyncMock(side_effect=[RuntimeError("nope"), None])

        summary = await cleanup_vast_orphans(client, "exp1", "baseline")

        assert summary.destroyed_instance_ids == [2]
        assert client.destroy_instance.await_count == 2

    @pytest.mark.asyncio
    async def test_uses_prefetched_instances_without_api_call(self):
        client = MagicMock()
        client.list_instances = AsyncMock()
        client.destroy_instance = AsyncMock()
        prefetched = [
            _vast_instance(10, "exp1-baseline"),
            _vast_instance(11, "exp1-other"),
        ]

        summary = await cleanup_vast_orphans(
            client, "exp1", "baseline", instances=prefetched,
        )

        client.list_instances.assert_not_awaited()
        assert summary.destroyed_instance_ids == [10]
