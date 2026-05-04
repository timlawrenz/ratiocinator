"""Tests for HF provider integration in config and coordinator."""

from __future__ import annotations

from ratiocinator.config import Config, HFConfig, load_config


class TestHFConfig:
    def test_defaults(self):
        cfg = HFConfig()
        assert cfg.token == ""
        assert cfg.default_flavor == "a100-large"
        assert cfg.namespace == ""
        assert cfg.max_timeout == "4h"

    def test_config_has_hf_field(self):
        cfg = Config()
        assert isinstance(cfg.hf, HFConfig)
        assert cfg.hf.default_image == "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"


class TestHFConfigEnvVars:
    def test_hf_token_from_env(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "hf_test_123")

        cfg = load_config()
        assert cfg.hf.token == "hf_test_123"

    def test_hf_namespace_from_env(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_NAMESPACE", "my-org")

        cfg = load_config()
        assert cfg.hf.namespace == "my-org"


class TestResearchSpecProvider:
    def test_default_provider_is_vast(self):
        from ratiocinator.fleet.spec import RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchSpec

        spec = ResearchSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/t/r.git"),
        )
        assert spec.provider == "vast"

    def test_hf_provider(self):
        from ratiocinator.fleet.spec import RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchSpec

        spec = ResearchSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/t/r.git"),
            provider="hf",
        )
        assert spec.provider == "hf"


class TestCoordinatorProvider:
    def test_default_provider_is_vast(self):
        from ratiocinator.fleet.spec import RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchCoordinator, ResearchSpec

        cfg = Config()
        spec = ResearchSpec(
            name="test", repo=RepoSpec(url="https://github.com/t/r.git"),
        )
        coord = ResearchCoordinator(cfg, spec)
        assert coord.provider == "vast"

    def test_hf_provider(self):
        from ratiocinator.fleet.spec import RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchCoordinator, ResearchSpec

        cfg = Config()
        spec = ResearchSpec(
            name="test", repo=RepoSpec(url="https://github.com/t/r.git"),
        )
        coord = ResearchCoordinator(cfg, spec, provider="hf")
        assert coord.provider == "hf"

    def test_create_executor_hf(self):
        from ratiocinator.fleet.providers.hf.executor import HFProvider
        from ratiocinator.fleet.spec import ExperimentSpec, HardwareSpec, RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchCoordinator, ResearchSpec

        cfg = Config()
        cfg.hf.token = "hf_test"
        cfg.hf.namespace = "test-ns"
        spec = ResearchSpec(
            name="test", repo=RepoSpec(url="https://github.com/t/r.git"),
        )
        coord = ResearchCoordinator(cfg, spec, provider="hf")

        exp_spec = ExperimentSpec(
            name="test",
            hardware=HardwareSpec(hf_flavor="a100-large"),
            repo=RepoSpec(url="https://github.com/t/r.git"),
            arms=[],
        )
        executor = coord._create_executor(exp_spec, "/tmp/results.json", None)
        assert isinstance(executor, HFProvider)

    def test_create_executor_vast(self):
        from ratiocinator.fleet.providers.vast.executor import VastProvider
        from ratiocinator.fleet.spec import ExperimentSpec, HardwareSpec, RepoSpec
        from ratiocinator.orchestration.coordinator import ResearchCoordinator, ResearchSpec

        cfg = Config()
        cfg.vast.api_key = "vast_test_key"
        spec = ResearchSpec(
            name="test", repo=RepoSpec(url="https://github.com/t/r.git"),
        )
        coord = ResearchCoordinator(cfg, spec, provider="vast")

        exp_spec = ExperimentSpec(
            name="test",
            hardware=HardwareSpec(),
            repo=RepoSpec(url="https://github.com/t/r.git"),
            arms=[],
        )
        executor = coord._create_executor(exp_spec, "/tmp/results.json", None)
        assert isinstance(executor, VastProvider)


class TestExperimentSpecProvider:
    def test_default_provider_is_vast(self):
        from ratiocinator.fleet.spec import ExperimentSpec, HardwareSpec, RepoSpec

        spec = ExperimentSpec(
            name="test",
            hardware=HardwareSpec(),
            repo=RepoSpec(url="https://github.com/t/r.git"),
            arms=[],
        )
        assert spec.provider == "vast"

    def test_hf_provider(self):
        from ratiocinator.fleet.spec import ExperimentSpec, HardwareSpec, RepoSpec

        spec = ExperimentSpec(
            name="test",
            hardware=HardwareSpec(hf_flavor="l4"),
            repo=RepoSpec(url="https://github.com/t/r.git"),
            arms=[],
            provider="hf",
        )
        assert spec.provider == "hf"
        assert spec.hardware.hf_flavor == "l4"
