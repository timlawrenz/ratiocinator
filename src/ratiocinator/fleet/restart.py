"""Orphan job/instance cleanup for ``ratiocinator fleet restart``.

In practice, the dangling-lock problem comes from non-terminal HF Jobs
(``PENDING`` / ``STARTING`` / ``RUNNING`` / ``UPDATING``) that still own
their bucket FUSE mount, and from Vast.ai instances that were never
destroyed.  Terminal HF stages (``COMPLETED`` / ``ERROR`` / ``FAILED``
/ ``CANCELLED`` / ``DELETED``) have already released their locks
server-side and do not need cancelling — they are reported for
visibility only.

This module provides small, side-effect-only helpers that locate and
free the active orphans for a given experiment / arm pair.  The CLI's
``fleet restart`` command composes them with the regular ``fleet run``
launch path so that a stuck arm can be reissued without any custom
agent scripting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ratiocinator.infra.hf_client import HFJobInfo

if TYPE_CHECKING:
    from ratiocinator.infra.hf_client import HFClient
    from ratiocinator.infra.vast_client import InstanceInfo, VastClient

logger = logging.getLogger(__name__)


@dataclass
class OrphanCleanupResult:
    """Summary of an orphan-cleanup pass for a single arm."""

    arm: str
    cancelled_job_ids: list[str]
    destroyed_instance_ids: list[int]
    skipped_terminal_job_ids: list[str]


async def cleanup_hf_orphans(
    client: HFClient,
    experiment: str,
    arm_name: str,
    *,
    namespace: str | None = None,
    jobs: list[HFJobInfo] | None = None,
) -> OrphanCleanupResult:
    """Cancel any non-terminal HF Jobs matching ``(experiment, arm_name)``.

    HF Jobs are tagged at submission time with ``experiment`` and ``arm``
    labels (see ``HFFleetExecutor._run_arm``).  Any active (non-terminal)
    job carrying that label pair is presumed to hold the bucket lock for
    this arm and is cancelled.  Terminal jobs (``COMPLETED`` / ``ERROR``
    / ``FAILED`` / ``CANCELLED`` / ``DELETED``) have already released
    their locks server-side, so they are reported via
    ``skipped_terminal_job_ids`` for visibility but not cancelled.

    Pass ``jobs=`` with a pre-fetched ``list_jobs()`` result to skip the
    namespace-wide API scan — useful when cleaning up several arms at
    once to avoid one full ``list_jobs`` call per arm.
    """
    labels = {"experiment": experiment, "arm": arm_name}

    # Pull every job in the namespace and filter locally — ``find_job_by_labels``
    # only returns one match, but a stuck arm can have several stale jobs.
    if jobs is None:
        jobs = await client.list_jobs(namespace=namespace)
    matches = [
        j for j in jobs
        if all(j.labels.get(k) == v for k, v in labels.items())
    ]

    cancelled: list[str] = []
    skipped: list[str] = []
    for job in matches:
        if job.stage.is_terminal:
            skipped.append(job.job_id)
            continue
        try:
            await client.cancel_job(job.job_id)
            cancelled.append(job.job_id)
            logger.info(
                "Cancelled orphan HF Job %s (stage=%s) for arm %s",
                job.job_id, job.stage.value, arm_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to cancel HF Job %s for arm %s: %s",
                job.job_id, arm_name, exc,
            )

    return OrphanCleanupResult(
        arm=arm_name,
        cancelled_job_ids=cancelled,
        destroyed_instance_ids=[],
        skipped_terminal_job_ids=skipped,
    )


async def cleanup_vast_orphans(
    client: VastClient,
    experiment: str,
    arm_name: str,
    *,
    instances: list[InstanceInfo] | None = None,
) -> OrphanCleanupResult:
    """Destroy any Vast.ai instances labelled ``{experiment}-{arm_name}``.

    ``FleetExecutor`` labels every instance with ``f"{experiment}-{arm}"``
    when it provisions them, so the orphan check is a simple list+filter.

    Pass ``instances=`` with a pre-fetched ``list_instances()`` result to
    skip the API call — useful when cleaning up several arms at once to
    avoid one full ``list_instances`` call per arm.
    """
    target_label = f"{experiment}-{arm_name}"
    if instances is None:
        instances = await client.list_instances()
    matches = [i for i in instances if i.label == target_label]

    destroyed: list[int] = []
    for inst in matches:
        try:
            await client.destroy_instance(inst.instance_id)
            destroyed.append(inst.instance_id)
            logger.info(
                "Destroyed orphan Vast instance %s (label=%s)",
                inst.instance_id, inst.label,
            )
        except Exception as exc:
            logger.warning(
                "Failed to destroy Vast instance %s: %s",
                inst.instance_id, exc,
            )

    return OrphanCleanupResult(
        arm=arm_name,
        cancelled_job_ids=[],
        destroyed_instance_ids=destroyed,
        skipped_terminal_job_ids=[],
    )
