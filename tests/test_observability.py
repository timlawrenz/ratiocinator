"""Tests for observability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ratiocinator.observability import fleet_breadcrumb, fleet_metric


class TestFleetBreadcrumb:
    def test_adds_breadcrumb_when_sentry_available(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_breadcrumb(
                "provisioned instance",
                category="fleet.provision",
                data={"arm": "baseline", "instance_id": 123},
            )

        mock_sdk.add_breadcrumb.assert_called_once_with(
            message="provisioned instance",
            category="fleet.provision",
            level="info",
            data={"arm": "baseline", "instance_id": 123},
        )

    def test_noop_when_sentry_unavailable(self):
        with patch("ratiocinator.observability.sentry_sdk", None):
            # Should not raise
            fleet_breadcrumb("test message", category="fleet.test")

    def test_error_level(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_breadcrumb("boot failed", level="error")

        call_kwargs = mock_sdk.add_breadcrumb.call_args[1]
        assert call_kwargs["level"] == "error"

    def test_default_empty_data(self):
        mock_sdk = MagicMock()
        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_breadcrumb("simple message")

        call_kwargs = mock_sdk.add_breadcrumb.call_args[1]
        assert call_kwargs["data"] == {}


class TestFleetMetric:
    def test_emits_distribution_when_sentry_available(self):
        mock_sdk = MagicMock()
        mock_metrics = MagicMock()
        mock_sdk.metrics = mock_metrics

        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_metric(
                "fleet.arm.duration", 120.5,
                unit="second",
                tags={"experiment": "test", "arm": "baseline"},
            )

        mock_metrics.distribution.assert_called_once_with(
            "fleet.arm.duration",
            120.5,
            unit="second",
            attributes={"experiment": "test", "arm": "baseline"},
        )

    def test_noop_when_sentry_unavailable(self):
        with patch("ratiocinator.observability.sentry_sdk", None):
            # Should not raise
            fleet_metric("fleet.arm.duration", 120.5)

    def test_noop_when_metrics_missing(self):
        mock_sdk = MagicMock(spec=[])  # no metrics attribute
        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            # Should not raise
            fleet_metric("fleet.arm.duration", 120.5)

    def test_default_empty_tags(self):
        mock_sdk = MagicMock()
        mock_metrics = MagicMock()
        mock_sdk.metrics = mock_metrics

        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_metric("fleet.arm.cost", 0.05)

        call_kwargs = mock_metrics.distribution.call_args[1]
        assert call_kwargs["attributes"] == {}

    def test_empty_unit_passes_none(self):
        mock_sdk = MagicMock()
        mock_metrics = MagicMock()
        mock_sdk.metrics = mock_metrics

        with patch("ratiocinator.observability.sentry_sdk", mock_sdk):
            fleet_metric("fleet.arm.exit_code", 0.0)

        call_kwargs = mock_metrics.distribution.call_args[1]
        assert call_kwargs["unit"] is None
