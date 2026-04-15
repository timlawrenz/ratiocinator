"""Tests for DataProvisioner implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ratiocinator.fleet.data import (
    LocalProvisioner,
    NullProvisioner,
    RsyncProvisioner,
    S3PresignedProvisioner,
    create_provisioner,
)
from ratiocinator.infra.remote import RemoteExecutor, RemoteResult


@pytest.fixture
def mock_remote():
    """A RemoteExecutor with all methods mocked."""
    remote = RemoteExecutor("test.host", 22222, "/tmp/key")
    remote.run = AsyncMock(return_value=RemoteResult(exit_code=0, stdout="", stderr=""))
    remote.scp_to = AsyncMock(return_value=RemoteResult(exit_code=0, stdout="", stderr=""))
    remote.write_remote_script = AsyncMock(
        return_value=RemoteResult(exit_code=0, stdout="", stderr="")
    )
    return remote


class TestNullProvisioner:
    @pytest.mark.asyncio
    async def test_always_succeeds(self, mock_remote):
        prov = NullProvisioner()
        ok, err = await prov.provision(mock_remote, "/workspace/data")
        assert ok is True
        assert err == ""


class TestS3PresignedProvisioner:
    @pytest.mark.asyncio
    async def test_empty_urls(self, mock_remote):
        prov = S3PresignedProvisioner([])
        ok, err = await prov.provision(mock_remote, "/workspace/data")
        assert ok is False
        assert "No presigned URLs" in err

    @pytest.mark.asyncio
    async def test_download_success(self, mock_remote):
        urls = [
            "https://s3.example.com/bucket_1024x1024/shard-00000.tar?AWSAccessKeyId=XXX",
            "https://s3.example.com/bucket_1024x1024/shard-00001.tar?AWSAccessKeyId=XXX",
        ]
        prov = S3PresignedProvisioner(urls)
        ok, err = await prov.provision(mock_remote, "/workspace/data")
        assert ok is True
        assert err == ""
        # Should have called write_remote_script for the download script
        mock_remote.write_remote_script.assert_called_once()
        # And run to execute it
        assert mock_remote.run.call_count >= 2  # mkdir + execute script

    @pytest.mark.asyncio
    async def test_download_failure(self, mock_remote):
        mock_remote.run = AsyncMock(side_effect=[
            RemoteResult(exit_code=0, stdout="", stderr=""),  # mkdir
            RemoteResult(exit_code=1, stdout="", stderr="wget: error"),  # download
        ])
        prov = S3PresignedProvisioner(["https://s3.example.com/shard.tar"])
        ok, err = await prov.provision(mock_remote, "/data")
        assert ok is False
        assert "Download failed" in err

    def test_from_file(self, tmp_path):
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://s3.example.com/a.tar\nhttps://s3.example.com/b.tar\n")
        prov = S3PresignedProvisioner.from_file(urls_file)
        assert len(prov.urls) == 2


class TestRsyncProvisioner:
    @pytest.mark.asyncio
    async def test_rsync_success(self, mock_remote):
        mock_remote.run = AsyncMock(side_effect=[
            RemoteResult(exit_code=0, stdout="", stderr=""),  # mkdir data
            RemoteResult(exit_code=0, stdout="", stderr=""),  # mkdir .ssh
            RemoteResult(exit_code=0, stdout="", stderr=""),  # chmod
            RemoteResult(exit_code=0, stdout="shard-0.tar\nshard-1.tar\n", stderr=""),  # ls
            RemoteResult(exit_code=0, stdout="", stderr=""),  # rsync
        ])
        prov = RsyncProvisioner("root@data-host:/data/shards", port=25706, max_shards=3)
        ok, _err = await prov.provision(mock_remote, "/workspace/data")
        assert ok is True

    @pytest.mark.asyncio
    async def test_rsync_no_shards_falls_back_to_directory(self, mock_remote):
        mock_remote.run = AsyncMock(side_effect=[
            RemoteResult(exit_code=0, stdout="", stderr=""),  # mkdir data
            RemoteResult(exit_code=0, stdout="", stderr=""),  # mkdir .ssh
            RemoteResult(exit_code=0, stdout="", stderr=""),  # chmod
            RemoteResult(exit_code=1, stdout="", stderr="No such file"),  # ls fails
            RemoteResult(exit_code=0, stdout="", stderr=""),  # directory rsync
        ])
        prov = RsyncProvisioner("root@host:/data")
        ok, _err = await prov.provision(mock_remote, "/data")
        assert ok is True
        rsync_call = mock_remote.run.call_args_list[4]
        assert "--include" not in rsync_call.args[0]
        assert "--exclude" not in rsync_call.args[0]


class TestLocalProvisioner:
    @pytest.mark.asyncio
    async def test_local_path_missing(self, mock_remote):
        prov = LocalProvisioner("/nonexistent/path")
        ok, err = await prov.provision(mock_remote, "/data")
        assert ok is False
        assert "does not exist" in err


class TestCreateProvisioner:
    def test_s3_presigned(self):
        prov = create_provisioner("s3-presigned", urls=["https://example.com/a.tar"])
        assert isinstance(prov, S3PresignedProvisioner)

    def test_rsync(self):
        prov = create_provisioner("rsync", rsync_server="root@host:/data")
        assert isinstance(prov, RsyncProvisioner)

    def test_local(self, tmp_path):
        prov = create_provisioner("local", local_path=str(tmp_path))
        assert isinstance(prov, LocalProvisioner)

    def test_none(self):
        prov = create_provisioner("none")
        assert isinstance(prov, NullProvisioner)

    def test_unknown_source(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            create_provisioner("ftp")

    def test_s3_missing_urls(self):
        with pytest.raises(ValueError, match="urls or urls_file"):
            create_provisioner("s3-presigned")

    def test_rsync_missing_server(self):
        with pytest.raises(ValueError, match="rsync_server"):
            create_provisioner("rsync")
