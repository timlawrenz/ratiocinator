"""Data provisioning for remote instances.

Provides `DataProvisioner` implementations that stage training data
onto ephemeral GPU instances via presigned S3 URLs, rsync, or local paths.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from ratiocinator.infra.remote import RemoteExecutor

logger = logging.getLogger(__name__)


class DataProvisioner(ABC):
    """Base class for data provisioning strategies."""

    @abstractmethod
    async def provision(
        self,
        remote: RemoteExecutor,
        target_dir: str,
        *,
        timeout: int = 7200,
    ) -> tuple[bool, str]:
        """Stage data on the remote instance.

        Args:
            remote: SSH executor for the target instance.
            target_dir: Remote directory where data should be placed.
            timeout: Maximum time in seconds for the operation.

        Returns:
            (success, error_message) tuple.
        """
        ...


class S3PresignedProvisioner(DataProvisioner):
    """Downloads data shards from presigned S3 URLs in parallel."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls

    @classmethod
    def from_file(cls, urls_file: str | Path) -> S3PresignedProvisioner:
        """Load presigned URLs from a file (one per line)."""
        text = Path(urls_file).read_text()
        urls = [line.strip() for line in text.splitlines() if line.strip()]
        return cls(urls)

    async def provision(
        self,
        remote: RemoteExecutor,
        target_dir: str,
        *,
        timeout: int = 7200,
    ) -> tuple[bool, str]:
        if not self.urls:
            return False, "No presigned URLs provided"

        await remote.run(f"mkdir -p {target_dir}", timeout=15)

        script_lines = ["#!/bin/bash", "set -e"]
        seen_dirs: set[str] = set()

        for url in self.urls:
            # Preserve bucket_* subdirectory structure
            url_path = url.split("?")[0]
            parts = url_path.split("/")
            bucket_idx = next(
                (i for i, p in enumerate(parts) if p.startswith("bucket_")), None
            )
            if bucket_idx is not None:
                rel_path = "/".join(parts[bucket_idx:])
                bucket_dir = parts[bucket_idx]
            else:
                rel_path = parts[-1]
                bucket_dir = None

            if bucket_dir and bucket_dir not in seen_dirs:
                script_lines.append(f"mkdir -p {target_dir}/{bucket_dir}")
                seen_dirs.add(bucket_dir)

            script_lines.append(
                f"wget -q -O '{target_dir}/{rel_path}' '{url}' &"
            )

        script_lines.append("wait")
        script_content = "\n".join(script_lines) + "\n"

        result = await remote.write_remote_script(
            script_content, "/tmp/download_shards.sh"
        )
        if not result.success:
            return False, f"Failed to upload download script: {result.stderr}"

        result = await remote.run(
            "/tmp/download_shards.sh",
            timeout=timeout,
            span_op="data.download",
        )
        if not result.success:
            return False, f"Download failed (exit {result.exit_code}): {result.stderr[:500]}"

        logger.info("Downloaded %d shard(s) to %s", len(self.urls), target_dir)
        return True, ""


class RsyncProvisioner(DataProvisioner):
    """Syncs data from a remote host via rsync."""

    def __init__(
        self,
        server: str,
        *,
        port: int = 22,
        max_shards: int | None = None,
    ) -> None:
        """
        Args:
            server: rsync source in user@host:/path format.
            port: SSH port for the data server.
            max_shards: Maximum number of .tar shards to sync.
        """
        self.server = server
        self.port = port
        self.max_shards = max_shards

    async def provision(
        self,
        remote: RemoteExecutor,
        target_dir: str,
        *,
        timeout: int = 7200,
    ) -> tuple[bool, str]:
        await remote.run(f"mkdir -p {target_dir}", timeout=15)

        # Transfer SSH key for data server access
        await remote.run("mkdir -p /root/.ssh", timeout=15)
        await remote.scp_to(remote.ssh_key, "/root/.ssh/data_key")
        await remote.run("chmod 600 /root/.ssh/data_key", timeout=15)

        ssh_remote = (
            f"ssh -p {self.port} -i /root/.ssh/data_key "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        )
        ds_host, ds_path = self.server.split(":", 1)

        # List available shards
        head_cmd = f" | head -{self.max_shards}" if self.max_shards else ""
        list_cmd = (
            f'{ssh_remote} {ds_host} "ls {ds_path}/*.tar 2>/dev/null{head_cmd}"'
        )
        result = await remote.run(list_cmd, timeout=60)
        if not result.success or not result.stdout.strip():
            return False, f"Failed to list shards: {result.stderr[:300]}"

        shard_files = [
            os.path.basename(s.strip())
            for s in result.stdout.strip().splitlines() if s.strip()
        ]

        logger.info(
            "Syncing %d shard(s) from %s", len(shard_files), self.server
        )

        include_args = " ".join(f"--include='{f}'" for f in shard_files)
        rsync_cmd = (
            f'rsync -xahP --inplace {include_args} --exclude="*" '
            f'-e "{ssh_remote}" {self.server}/ {target_dir}/'
        )
        result = await remote.run(rsync_cmd, timeout=timeout, span_op="data.rsync")
        if not result.success:
            return False, f"rsync failed (exit {result.exit_code}): {result.stderr[:500]}"

        logger.info("Synced %d shard(s) to %s", len(shard_files), target_dir)
        return True, ""


class LocalProvisioner(DataProvisioner):
    """Provisions data by rsync from local filesystem to remote instance."""

    def __init__(self, local_path: str | Path) -> None:
        self.local_path = Path(local_path)

    async def provision(
        self,
        remote: RemoteExecutor,
        target_dir: str,
        *,
        timeout: int = 7200,
    ) -> tuple[bool, str]:
        if not self.local_path.exists():
            return False, f"Local data path does not exist: {self.local_path}"

        result = await remote.rsync_to(
            self.local_path, target_dir, timeout=timeout, delete=False,
        )
        if not result.success:
            return False, f"rsync failed: {result.stderr[:500]}"

        logger.info("Transferred local data %s → %s", self.local_path, target_dir)
        return True, ""


class NullProvisioner(DataProvisioner):
    """No-op provisioner for experiments that don't need external data."""

    async def provision(
        self,
        remote: RemoteExecutor,
        target_dir: str,
        *,
        timeout: int = 7200,
    ) -> tuple[bool, str]:
        return True, ""


def create_provisioner(
    source: str,
    *,
    urls_file: str = "",
    urls: list[str] | None = None,
    rsync_server: str = "",
    rsync_port: int = 22,
    max_shards: int | None = None,
    local_path: str = "",
) -> DataProvisioner:
    """Factory function to create the appropriate DataProvisioner.

    Args:
        source: One of "s3-presigned", "rsync", "local", "none".
    """
    if source == "s3-presigned":
        if urls:
            return S3PresignedProvisioner(urls)
        if urls_file:
            return S3PresignedProvisioner.from_file(urls_file)
        raise ValueError("s3-presigned requires urls or urls_file")
    elif source == "rsync":
        if not rsync_server:
            raise ValueError("rsync requires rsync_server")
        return RsyncProvisioner(rsync_server, port=rsync_port, max_shards=max_shards)
    elif source == "local":
        if not local_path:
            raise ValueError("local requires local_path")
        return LocalProvisioner(local_path)
    elif source == "none":
        return NullProvisioner()
    else:
        raise ValueError(f"Unknown data source: {source}")
