"""Tests for the compute provider interface, registry, and configs."""

from __future__ import annotations

import pytest

import ratiocinator.fleet.providers  # noqa: F401  — triggers auto-registration
from ratiocinator.fleet.provider import (
    ComputeProvider,
    ProviderCapability,
    ProviderMeta,
    available_provider_names,
    get_provider_class,
    list_providers,
    register_provider,
)
from ratiocinator.fleet.provider_config import (
    DockerProviderConfig,
    HFProviderConfig,
    VastProviderConfig,
    resolve_provider_config,
)
from ratiocinator.fleet.results import ArmResult
from ratiocinator.fleet.spec import ExperimentSpec

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_vast_registered(self):
        cls = get_provider_class("vast")
        assert cls is not None
        assert issubclass(cls, ComputeProvider)

    def test_hf_registered(self):
        cls = get_provider_class("hf")
        assert cls is not None
        assert issubclass(cls, ComputeProvider)

    def test_docker_registered(self):
        cls = get_provider_class("docker")
        assert cls is not None
        assert issubclass(cls, ComputeProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown compute provider"):
            get_provider_class("nonexistent-cloud")

    def test_available_provider_names(self):
        names = available_provider_names()
        assert "vast" in names
        assert "hf" in names
        assert "docker" in names

    def test_list_providers_returns_meta(self):
        providers = list_providers()
        assert "vast" in providers
        assert providers["vast"].display_name == "Vast.ai"
        assert "hf" in providers
        assert providers["hf"].display_name == "HuggingFace Jobs"
        assert "docker" in providers
        assert providers["docker"].display_name == "Local Docker"

    def test_register_custom_provider(self):
        """Test that @register_provider works for new providers."""

        @register_provider("test-custom")
        class TestCustomProvider(ComputeProvider):
            _META = ProviderMeta(
                name="test-custom",
                display_name="Test Custom",
                capabilities=ProviderCapability.NONE,
            )

            def __init__(self):
                pass

            @property
            def meta(self) -> ProviderMeta:
                return self._META

            async def _run_arm(self, spec, arm_idx, arm, launch_order):
                return ArmResult(
                    experiment="test",
                    arm_name="arm-0",
                    description="test arm",
                )

        cls = get_provider_class("test-custom")
        assert cls is TestCustomProvider


# ---------------------------------------------------------------------------
# Provider meta tests
# ---------------------------------------------------------------------------


class TestProviderMeta:
    def test_vast_meta(self):
        cls = get_provider_class("vast")
        config = VastProviderConfig(api_key="test", ssh_key="/tmp/key")
        provider = cls(config)
        meta = provider.meta
        assert meta.name == "vast"
        assert meta.requires_ssh_key is True
        assert ProviderCapability.SSH_ACCESS in meta.capabilities

    def test_hf_meta(self):
        cls = get_provider_class("hf")
        config = HFProviderConfig(token="test-token")
        provider = cls(config)
        meta = provider.meta
        assert meta.name == "hf"
        assert meta.requires_ssh_key is False
        assert ProviderCapability.PREEMPTION_RECOVERY in meta.capabilities

    def test_docker_meta(self):
        cls = get_provider_class("docker")
        config = DockerProviderConfig()
        provider = cls(config)
        meta = provider.meta
        assert meta.name == "docker"
        assert meta.requires_api_token is False


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestProviderConfigs:
    def test_vast_config_valid(self):
        config = VastProviderConfig(api_key="key123", ssh_key="/path/to/key")
        config.validate()  # Should not raise

    def test_vast_config_missing_api_key(self):
        config = VastProviderConfig(api_key="", ssh_key="/path/to/key")
        with pytest.raises(ValueError, match="api_key"):
            config.validate()

    def test_vast_config_missing_ssh_key(self):
        config = VastProviderConfig(api_key="key123", ssh_key="")
        with pytest.raises(ValueError, match="ssh_key"):
            config.validate()

    def test_hf_config_valid(self):
        config = HFProviderConfig(token="hf_abc123")
        config.validate()  # Should not raise

    def test_hf_config_missing_token(self):
        config = HFProviderConfig(token="")
        with pytest.raises(ValueError, match="token"):
            config.validate()

    def test_docker_config_always_valid(self):
        config = DockerProviderConfig()
        config.validate()  # Should not raise (no required creds)

    def test_resolve_provider_config_vast(self):
        config = resolve_provider_config("vast", {
            "api_key": "test-key",
            "ssh_key": "/tmp/key",
        })
        assert isinstance(config, VastProviderConfig)

    def test_resolve_provider_config_hf(self):
        config = resolve_provider_config("hf", {"token": "hf_test"})
        assert isinstance(config, HFProviderConfig)

    def test_resolve_provider_config_docker(self):
        config = resolve_provider_config("docker", {})
        assert isinstance(config, DockerProviderConfig)

    def test_resolve_unknown_provider(self):
        with pytest.raises(ValueError, match="No config class"):
            resolve_provider_config("imaginary", {})


# ---------------------------------------------------------------------------
# Provider capability flags
# ---------------------------------------------------------------------------


class TestProviderCapabilities:
    def test_flag_combination(self):
        caps = ProviderCapability.SSH_ACCESS | ProviderCapability.COST_TRACKING
        assert ProviderCapability.SSH_ACCESS in caps
        assert ProviderCapability.COST_TRACKING in caps
        assert ProviderCapability.VOLUME_MOUNTS not in caps

    def test_none_flag(self):
        assert ProviderCapability(0) == ProviderCapability.NONE


# ---------------------------------------------------------------------------
# Template method (base class run logic) tests
# ---------------------------------------------------------------------------


class TestTemplateMethod:
    @pytest.fixture
    def minimal_spec(self, tmp_path):
        """Build a minimal ExperimentSpec for testing."""
        import yaml

        spec_data = {
            "name": "test-template",
            "hardware": {"image": "python:3.11", "gpu": "RTX 4090"},
            "repo": {
                "url": "https://github.com/test/repo.git",
                "branch": "main",
            },
            "arms": [
                {"name": "arm-a", "command": "echo hello"},
                {"name": "arm-b", "command": "echo world"},
            ],
        }
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(spec_data))
        return ExperimentSpec.from_yaml(spec_file)

    async def test_dry_run_returns_empty(self, minimal_spec, tmp_path):
        """Base class dry_run should return [] without calling _run_arm."""
        config = DockerProviderConfig(
            results_path=str(tmp_path / "results.json"),
        )
        provider = get_provider_class("docker")(config)
        results = await provider.run(minimal_spec, dry_run=True)
        assert results == []

    async def test_run_dispatches_all_arms(self, minimal_spec, tmp_path, monkeypatch):
        """Base class run() dispatches to _run_arm for each arm."""
        config = DockerProviderConfig(
            results_path=str(tmp_path / "results.json"),
        )
        provider = get_provider_class("docker")(config)

        called_arms = []

        async def mock_run_arm(spec, arm_idx, arm, launch_order):
            called_arms.append(arm.name)
            return ArmResult(
                experiment=spec.name,
                arm_name=arm.name,
                description=arm.description,
            )

        monkeypatch.setattr(provider, "_run_arm", mock_run_arm)
        results = await provider.run(minimal_spec)
        assert len(results) == 2
        assert set(called_arms) == {"arm-a", "arm-b"}

    async def test_run_with_arm_indices(self, minimal_spec, tmp_path, monkeypatch):
        """Only selected arm indices are dispatched."""
        config = DockerProviderConfig(
            results_path=str(tmp_path / "results.json"),
        )
        provider = get_provider_class("docker")(config)

        called_arms = []

        async def mock_run_arm(spec, arm_idx, arm, launch_order):
            called_arms.append(arm.name)
            return ArmResult(
                experiment=spec.name,
                arm_name=arm.name,
                description=arm.description,
            )

        monkeypatch.setattr(provider, "_run_arm", mock_run_arm)
        results = await provider.run(minimal_spec, arm_indices=[1])
        assert len(results) == 1
        assert called_arms == ["arm-b"]


# ---------------------------------------------------------------------------
# ExperimentSpec provider_config field
# ---------------------------------------------------------------------------


class TestSpecProviderConfig:
    def test_spec_accepts_provider_config(self, tmp_path):
        """ExperimentSpec supports an optional provider_config dict."""
        import yaml

        spec_data = {
            "name": "test-config",
            "hardware": {"image": "python:3.11", "gpu": "RTX 4090"},
            "repo": {
                "url": "https://github.com/test/repo.git",
                "branch": "main",
            },
            "arms": [{"name": "arm-0", "command": "echo hi"}],
            "provider": "docker",
            "provider_config": {
                "runtime": "cpu",
                "max_concurrent": 4,
            },
        }
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(spec_data))
        spec = ExperimentSpec.from_yaml(spec_file)
        assert spec.provider == "docker"
        assert spec.provider_config == {"runtime": "cpu", "max_concurrent": 4}

    def test_spec_provider_config_defaults_none(self, tmp_path):
        """provider_config defaults to None when not specified."""
        import yaml

        spec_data = {
            "name": "test-default",
            "hardware": {"image": "python:3.11", "gpu": "RTX 4090"},
            "repo": {
                "url": "https://github.com/test/repo.git",
                "branch": "main",
            },
            "arms": [{"name": "arm-0", "command": "echo hi"}],
        }
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(spec_data))
        spec = ExperimentSpec.from_yaml(spec_file)
        assert spec.provider_config is None
