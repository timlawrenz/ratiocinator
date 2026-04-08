"""Tests for the Vast.ai client."""

from __future__ import annotations

from ratiocinator.infra.vast_client import (
    InstanceStatus,
    VastClient,
    VastError,
    compute_backoff,
)


class TestVastClientParsing:
    def test_parse_running_instance(self):
        client = VastClient.__new__(VastClient)
        data = {
            "id": 12345,
            "actual_status": "running",
            "gpu_name": "RTX 4090",
            "gpu_ram": 24576,
            "dph_total": 0.45,
            "ssh_host": "ssh5.vast.ai",
            "ssh_port": 22222,
            "label": "ratiocinator-exp-1",
        }
        info = client._parse_instance(data)
        assert info.instance_id == 12345
        assert info.status == InstanceStatus.RUNNING
        assert info.gpu_name == "RTX 4090"
        assert info.gpu_ram_gb == 24.0
        assert info.dph_total == 0.45
        assert info.label == "ratiocinator-exp-1"

    def test_parse_unknown_status(self):
        client = VastClient.__new__(VastClient)
        data = {"id": 1, "actual_status": "weird"}
        info = client._parse_instance(data)
        assert info.status == InstanceStatus.UNKNOWN

    def test_parse_missing_fields(self):
        client = VastClient.__new__(VastClient)
        data = {}
        info = client._parse_instance(data)
        assert info.instance_id == 0
        assert info.gpu_ram_gb == 0.0


class TestVastError:
    def test_error_with_status_code(self):
        err = VastError("test error", status_code=403)
        assert err.status_code == 403
        assert "test error" in str(err)

    def test_error_without_status_code(self):
        err = VastError("network failure")
        assert err.status_code is None


class TestBackoff:
    def test_backoff_increases(self):
        b0 = compute_backoff(0)
        b1 = compute_backoff(1)
        b2 = compute_backoff(2)
        # Base values (before jitter): 2, 4, 8
        assert b0 < 3.0  # 2 + up to 0.5 jitter
        assert b1 < 6.0  # 4 + up to 1.0 jitter
        assert b2 < 12.0  # 8 + up to 2.0 jitter

    def test_backoff_capped(self):
        b = compute_backoff(100)
        assert b <= 150.0  # 120 + 25% jitter
