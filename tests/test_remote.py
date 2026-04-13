"""Tests for RemoteExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ratiocinator.infra.remote import RemoteExecutor, RemoteResult


@pytest.fixture
def remote():
    return RemoteExecutor("test.host", 22222, "/tmp/test_key", label="test-instance")


class TestRemoteResult:
    def test_success_property(self):
        assert RemoteResult(exit_code=0, stdout="ok", stderr="").success is True
        assert RemoteResult(exit_code=1, stdout="", stderr="err").success is False

    def test_timeout_code(self):
        r = RemoteResult(exit_code=124, stdout="", stderr="Timed out")
        assert not r.success


class TestRemoteExecutorInit:
    def test_basic_init(self, remote):
        assert remote.host == "test.host"
        assert remote.port == 22222
        assert remote.ssh_key == "/tmp/test_key"
        assert remote.label == "test-instance"

    def test_default_label(self):
        r = RemoteExecutor("host.ai", 12345, "/key")
        assert r.label == "host.ai:12345"


class TestRemoteExecutorRun:
    @pytest.mark.asyncio
    async def test_successful_command(self, remote):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await remote.run("echo hello", timeout=10)

        assert result.success
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_failed_command(self, remote):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error\n"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await remote.run("false", timeout=10)

        assert not result.success
        assert result.exit_code == 1
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self, remote):
        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await remote.run("sleep 100", timeout=1)

        assert result.exit_code == 124
        assert "Timed out" in result.stderr
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception(self, remote):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("SSH not found"),
        ):
            result = await remote.run("echo test", timeout=10)

        assert result.exit_code == 1
        assert "SSH not found" in result.stderr


class TestRemoteExecutorWaitForSSH:
    @pytest.mark.asyncio
    async def test_ssh_ready_first_try(self, remote):
        with patch.object(
            remote, "run",
            new_callable=AsyncMock,
            return_value=RemoteResult(exit_code=0, stdout="ok", stderr=""),
        ):
            ready = await remote.wait_for_ssh(retries=3, interval=0.01)

        assert ready is True

    @pytest.mark.asyncio
    async def test_ssh_ready_after_retries(self, remote):
        results = [
            RemoteResult(exit_code=255, stdout="", stderr="Connection refused"),
            RemoteResult(exit_code=255, stdout="", stderr="Connection refused"),
            RemoteResult(exit_code=0, stdout="ok", stderr=""),
        ]
        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            r = results[min(call_count, len(results) - 1)]
            call_count += 1
            return r

        with patch.object(remote, "run", side_effect=mock_run):
            ready = await remote.wait_for_ssh(retries=5, interval=0.01)

        assert ready is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_ssh_never_ready(self, remote):
        with patch.object(
            remote, "run",
            new_callable=AsyncMock,
            return_value=RemoteResult(exit_code=255, stdout="", stderr="refused"),
        ):
            ready = await remote.wait_for_ssh(retries=2, interval=0.01)

        assert ready is False

    @pytest.mark.asyncio
    async def test_ssh_ready_adds_breadcrumb(self, remote):
        with patch.object(
            remote, "run",
            new_callable=AsyncMock,
            return_value=RemoteResult(exit_code=0, stdout="ok", stderr=""),
        ), patch("ratiocinator.infra.remote.fleet_breadcrumb") as mock_bc:
            ready = await remote.wait_for_ssh(retries=3, interval=0.01)

        assert ready is True
        mock_bc.assert_called_once()
        call_kwargs = mock_bc.call_args[1]
        assert call_kwargs["category"] == "remote.ssh"
        assert "host" in call_kwargs["data"]

    @pytest.mark.asyncio
    async def test_ssh_failure_adds_error_breadcrumb(self, remote):
        with patch.object(
            remote, "run",
            new_callable=AsyncMock,
            return_value=RemoteResult(exit_code=255, stdout="", stderr="refused"),
        ), patch("ratiocinator.infra.remote.fleet_breadcrumb") as mock_bc:
            ready = await remote.wait_for_ssh(retries=2, interval=0.01)

        assert ready is False
        mock_bc.assert_called_once()
        call_kwargs = mock_bc.call_args[1]
        assert call_kwargs["category"] == "remote.ssh"
        assert call_kwargs["level"] == "error"


class TestRemoteExecutorSCP:
    @pytest.mark.asyncio
    async def test_scp_to(self, remote):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await remote.scp_to("/tmp/local.txt", "/remote/file.txt")

        assert result.success


class TestRemoteExecutorWriteScript:
    @pytest.mark.asyncio
    async def test_write_remote_script(self, remote):
        scp_result = RemoteResult(exit_code=0, stdout="", stderr="")
        chmod_result = RemoteResult(exit_code=0, stdout="", stderr="")

        async def mock_scp_to(*args, **kwargs):
            return scp_result

        async def mock_run(*args, **kwargs):
            return chmod_result

        with patch.object(remote, "scp_to", side_effect=mock_scp_to), \
             patch.object(remote, "run", side_effect=mock_run):
            result = await remote.write_remote_script("#!/bin/bash\necho hi")

        assert result.success
