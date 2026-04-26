"""HuggingFace data provisioning via Buckets and volume mounts.

Instead of SSH-based data transfer (rsync / SCP / wget), HF Jobs uses
**volume mounts** — the orchestrator tells the HF Jobs API to mount a
dataset, bucket, or model repo directly into the container at a given
path.  No download step is needed inside the container.

This module also provides utilities for uploading experiment scripts
to HF Buckets so they can be mounted into job containers.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_data_volume(
    source_type: str,
    hf_source: str,
    mount_path: str = "/data",
) -> dict[str, Any]:
    """Build a volume dict for HF Jobs from a DataSpec.

    Args:
        source_type: ``"hf-dataset"`` or ``"hf-bucket"``.
        hf_source: HF repo ID or bucket name (e.g. ``"user/my-data"``).
        mount_path: Path inside the container.

    Returns:
        Dict suitable for constructing a ``huggingface_hub.Volume``.
    """
    type_map = {
        "hf-dataset": "dataset",
        "hf-bucket": "bucket",
    }
    vol_type = type_map.get(source_type, "dataset")
    return {
        "type": vol_type,
        "source": hf_source,
        "mount_path": mount_path,
    }


def build_output_volume(
    bucket_name: str,
    mount_path: str = "/output",
) -> dict[str, Any]:
    """Build a writable bucket volume for collecting job artifacts.

    Args:
        bucket_name: HF bucket name (e.g. ``"user/experiment-artifacts"``).
        mount_path: Path inside the container.
    """
    return {
        "type": "bucket",
        "source": bucket_name,
        "mount_path": mount_path,
    }


def build_script_volume(
    bucket_name: str,
    mount_path: str = "/input",
) -> dict[str, Any]:
    """Build a read-only bucket volume for the per-arm wrapper script.

    The script is uploaded to the bucket beforehand and mounted here.
    """
    return {
        "type": "bucket",
        "source": bucket_name,
        "mount_path": mount_path,
    }


async def upload_arm_script(
    client: Any,
    bucket_name: str,
    arm_name: str,
    script_content: str,
    experiment_name: str,
) -> str:
    """Upload a per-arm wrapper script to an HF Bucket.

    Uses a unique path ``<experiment>/<arm>/run.sh`` to prevent race
    conditions between parallel arm uploads.

    Args:
        client: ``HFClient`` instance.
        bucket_name: Bucket to upload to.
        arm_name: Arm name (used in path).
        script_content: Bash script content.
        experiment_name: Experiment name (used in path).

    Returns:
        Remote path within the bucket.
    """
    remote_path = f"{experiment_name}/{arm_name}/run.sh"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False,
    ) as f:
        f.write(script_content)
        local_path = f.name

    try:
        await client.upload_to_bucket(bucket_name, local_path, remote_path)
        logger.info(
            "Uploaded arm script to %s/%s", bucket_name, remote_path,
        )
    finally:
        Path(local_path).unlink(missing_ok=True)

    return remote_path


def volumes_from_dicts(volume_dicts: list[dict[str, Any]]) -> list[Any]:
    """Convert volume dicts to ``huggingface_hub.Volume`` objects.

    Lazy-imports ``huggingface_hub`` so callers without it installed
    don't crash at import time.
    """
    from huggingface_hub import Volume

    return [
        Volume(
            type=v["type"],
            source=v["source"],
            mount_path=v["mount_path"],
        )
        for v in volume_dicts
    ]
