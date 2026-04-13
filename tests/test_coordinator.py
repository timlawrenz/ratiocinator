"""Tests for orchestration.coordinator — env var propagation and spec translation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from ratiocinator.config import Config
from ratiocinator.fleet.spec import ArmSpec
from ratiocinator.orchestration.coordinator import (
    ConfigDuplicateError,
    ResearchCoordinator,
    ResearchSpec,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def research_spec():
    """Minimal ResearchSpec for testing."""
    return ResearchSpec(
        name="gnn-study",
        description="GNN architecture ablation",
        repo={"url": "https://github.com/user/repo.git", "branch": "main"},
        base_command="python train.py",
        num_arms=3,
        iterations=1,
        score_key="mae",
        maximize=False,
    )


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def coordinator(config, research_spec):
    """Coordinator with mocked LLM."""
    llm = MagicMock()
    llm.complete_json = AsyncMock()
    return ResearchCoordinator(config, research_spec, llm=llm)


# ---------------------------------------------------------------------------
# _translate_to_spec tests — the critical fix
# ---------------------------------------------------------------------------


class TestTranslateToSpec:
    """Verify that _translate_to_spec correctly populates arm.env."""

    def test_env_populated_with_config_overrides(self, coordinator):
        """Each arm's config_overrides must appear in arm.env."""
        llm_arms = [
            {
                "name": "sage_64",
                "description": "SAGE with hidden_dim=64",
                "config_overrides": {
                    "CONV_TYPE": "SAGE",
                    "HIDDEN_DIM": "64",
                    "NUM_LAYERS": "3",
                },
            },
            {
                "name": "gcn_128",
                "description": "GCN with hidden_dim=128",
                "config_overrides": {
                    "CONV_TYPE": "GCN",
                    "HIDDEN_DIM": "128",
                    "NUM_LAYERS": "4",
                },
            },
        ]

        spec = coordinator._translate_to_spec(llm_arms, iteration=0)

        assert len(spec.arms) == 2
        assert spec.arms[0].env == {
            "CONV_TYPE": "SAGE",
            "HIDDEN_DIM": "64",
            "NUM_LAYERS": "3",
        }
        assert spec.arms[1].env == {
            "CONV_TYPE": "GCN",
            "HIDDEN_DIM": "128",
            "NUM_LAYERS": "4",
        }

    def test_env_not_empty(self, coordinator):
        """Regression: env must never be empty when overrides are provided."""
        llm_arms = [
            {
                "name": "arm_a",
                "config_overrides": {"LR": "0.001"},
            },
            {
                "name": "arm_b",
                "config_overrides": {"LR": "0.01"},
            },
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        for arm in spec.arms:
            assert arm.env, f"arm {arm.name} has empty env"

    def test_numeric_values_become_strings(self, coordinator):
        """Env vars must be strings — numeric overrides are stringified."""
        llm_arms = [
            {
                "name": "arm_a",
                "config_overrides": {"HIDDEN_DIM": 64, "DROPOUT": 0.5},
            },
            {
                "name": "arm_b",
                "config_overrides": {"HIDDEN_DIM": 128, "DROPOUT": 0.3},
            },
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        assert spec.arms[0].env["HIDDEN_DIM"] == "64"
        assert spec.arms[0].env["DROPOUT"] == "0.5"

    def test_empty_config_overrides_allowed_for_baseline(self, coordinator):
        """An arm with no overrides (baseline) is valid alongside diverse arms."""
        llm_arms = [
            {
                "name": "baseline",
                "description": "Default config",
                "config_overrides": {},
            },
            {
                "name": "variant",
                "description": "Modified config",
                "config_overrides": {"CONV_TYPE": "GAT"},
            },
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        assert spec.arms[0].env == {}
        assert spec.arms[1].env == {"CONV_TYPE": "GAT"}

    def test_spec_name_includes_iteration(self, coordinator):
        """Spec name includes iteration number for tracking."""
        llm_arms = [
            {"name": "a", "config_overrides": {"X": "1"}},
            {"name": "b", "config_overrides": {"X": "2"}},
        ]

        spec = coordinator._translate_to_spec(llm_arms, iteration=2)

        assert spec.name == "gnn-study-iter2"

    def test_missing_config_overrides_key(self, coordinator):
        """Arms without config_overrides key get empty env (defaults)."""
        llm_arms = [
            {"name": "a"},
            {"name": "b", "config_overrides": {"X": "1"}},
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        assert spec.arms[0].env == {}
        assert spec.arms[1].env == {"X": "1"}

    def test_command_propagated_from_research_spec(self, coordinator):
        """All arms inherit the base_command from the research spec."""
        llm_arms = [
            {"name": "a", "config_overrides": {"X": "1"}},
            {"name": "b", "config_overrides": {"X": "2"}},
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        for arm in spec.arms:
            assert arm.command == "python train.py"

    def test_hardware_data_deps_propagated(self, coordinator):
        """Fleet infrastructure fields are copied from research spec."""
        llm_arms = [
            {"name": "a", "config_overrides": {"X": "1"}},
            {"name": "b", "config_overrides": {"X": "2"}},
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        assert spec.repo.url == "https://github.com/user/repo.git"
        assert spec.repo.branch == "main"

    def test_arm_descriptions_propagated(self, coordinator):
        """Arm descriptions from LLM are forwarded to the spec."""
        llm_arms = [
            {
                "name": "sage",
                "description": "GraphSAGE baseline",
                "config_overrides": {"CONV": "SAGE"},
            },
            {
                "name": "gat",
                "description": "GAT with attention",
                "config_overrides": {"CONV": "GAT"},
            },
        ]

        spec = coordinator._translate_to_spec(llm_arms)

        assert spec.arms[0].description == "GraphSAGE baseline"
        assert spec.arms[1].description == "GAT with attention"


# ---------------------------------------------------------------------------
# _assert_arms_differ tests
# ---------------------------------------------------------------------------


class TestAssertArmsDiffer:
    """Verify the duplicate config detection guard."""

    def test_identical_envs_raises(self):
        """All arms having the same env must raise ConfigDuplicateError."""
        arms = [
            ArmSpec(name="a", command="train", env={"X": "1"}),
            ArmSpec(name="b", command="train", env={"X": "1"}),
            ArmSpec(name="c", command="train", env={"X": "1"}),
        ]

        with pytest.raises(ConfigDuplicateError, match="identical env"):
            ResearchCoordinator._assert_arms_differ(arms)

    def test_all_empty_envs_raises(self):
        """All arms with empty env must raise (no differentiation)."""
        arms = [
            ArmSpec(name="a", command="train", env={}),
            ArmSpec(name="b", command="train", env={}),
        ]

        with pytest.raises(ConfigDuplicateError):
            ResearchCoordinator._assert_arms_differ(arms)

    def test_diverse_envs_passes(self):
        """Arms with different envs should not raise."""
        arms = [
            ArmSpec(name="a", command="train", env={"X": "1"}),
            ArmSpec(name="b", command="train", env={"X": "2"}),
        ]

        # Should not raise
        ResearchCoordinator._assert_arms_differ(arms)

    def test_single_arm_passes(self):
        """A single arm is always valid (no comparison needed)."""
        arms = [ArmSpec(name="a", command="train", env={"X": "1"})]
        ResearchCoordinator._assert_arms_differ(arms)

    def test_partially_different_passes(self):
        """At least two distinct configs is enough (baseline + variant)."""
        arms = [
            ArmSpec(name="a", command="train", env={}),
            ArmSpec(name="b", command="train", env={"X": "1"}),
        ]

        # Should not raise — baseline + variant is valid
        ResearchCoordinator._assert_arms_differ(arms)


# ---------------------------------------------------------------------------
# ResearchSpec loading
# ---------------------------------------------------------------------------


class TestResearchSpec:
    def test_from_yaml(self, tmp_path):
        spec_data = {
            "name": "test-research",
            "description": "Test research spec",
            "repo": {"url": "https://github.com/user/repo.git"},
            "base_command": "python train.py --config {config}",
            "num_arms": 4,
            "iterations": 2,
            "score_key": "val_loss",
        }
        spec_file = tmp_path / "research.yaml"
        spec_file.write_text(yaml.dump(spec_data))

        spec = ResearchSpec.from_yaml(spec_file)

        assert spec.name == "test-research"
        assert spec.num_arms == 4
        assert spec.iterations == 2
        assert spec.score_key == "val_loss"
        assert spec.repo.url == "https://github.com/user/repo.git"


# ---------------------------------------------------------------------------
# propose_arms tests
# ---------------------------------------------------------------------------


class TestProposeArms:
    async def test_propose_returns_llm_arms(self, coordinator):
        """propose_arms should return the arms from LLM response."""
        coordinator.llm.complete_json.return_value = {
            "reasoning": "Testing diverse GNN architectures",
            "arms": [
                {
                    "name": "sage_64",
                    "description": "SAGE 64",
                    "config_overrides": {"CONV_TYPE": "SAGE"},
                },
                {
                    "name": "gcn_128",
                    "description": "GCN 128",
                    "config_overrides": {"CONV_TYPE": "GCN"},
                },
            ],
        }

        arms = await coordinator.propose_arms()

        assert len(arms) == 2
        assert arms[0]["name"] == "sage_64"
        assert arms[0]["config_overrides"]["CONV_TYPE"] == "SAGE"

    async def test_propose_with_prior_results(self, coordinator):
        """Prior results should be included in the LLM prompt."""
        coordinator.llm.complete_json.return_value = {
            "arms": [
                {"name": "arm_a", "config_overrides": {"X": "1"}},
                {"name": "arm_b", "config_overrides": {"X": "2"}},
            ],
        }

        prior = [{"arm": "old_arm", "metrics": {"mae": 0.5}}]
        await coordinator.propose_arms(prior_results=prior)

        # Verify prior results were passed to LLM
        call_args = coordinator.llm.complete_json.call_args
        prompt = call_args[0][0]
        assert "Prior results" in prompt
        assert "old_arm" in prompt

    async def test_propose_empty_arms_raises(self, coordinator):
        """Empty arms from LLM should raise ValueError."""
        coordinator.llm.complete_json.return_value = {"arms": []}

        with pytest.raises(ValueError, match="no experiment arms"):
            await coordinator.propose_arms()


# ---------------------------------------------------------------------------
# End-to-end: propose → translate → verify env propagation
# ---------------------------------------------------------------------------


class TestEndToEndEnvPropagation:
    """Integration test for the full ideation → spec pipeline."""

    async def test_llm_overrides_reach_arm_env(self, coordinator):
        """Config overrides from LLM must end up in ArmSpec.env."""
        # Simulate LLM proposing GNN variants
        coordinator.llm.complete_json.return_value = {
            "reasoning": "Testing GNN architectures",
            "arms": [
                {
                    "name": "sage_64_3",
                    "description": "SAGE with 64 hidden dim, 3 layers",
                    "config_overrides": {
                        "CONV_TYPE": "SAGE",
                        "HIDDEN_DIM": "64",
                        "NUM_LAYERS": "3",
                    },
                },
                {
                    "name": "gcn_128_4",
                    "description": "GCN with 128 hidden dim, 4 layers",
                    "config_overrides": {
                        "CONV_TYPE": "GCN",
                        "HIDDEN_DIM": "128",
                        "NUM_LAYERS": "4",
                    },
                },
                {
                    "name": "gat_64_2",
                    "description": "GAT with 64 hidden dim, 2 layers",
                    "config_overrides": {
                        "CONV_TYPE": "GAT",
                        "HIDDEN_DIM": "64",
                        "NUM_LAYERS": "2",
                    },
                },
            ],
        }

        # Step 1: propose
        llm_arms = await coordinator.propose_arms()

        # Step 2: translate
        spec = coordinator._translate_to_spec(llm_arms, iteration=0)

        # Step 3: verify env propagation (the bug fix)
        assert len(spec.arms) == 3

        envs = [arm.env for arm in spec.arms]
        assert envs[0]["CONV_TYPE"] == "SAGE"
        assert envs[1]["CONV_TYPE"] == "GCN"
        assert envs[2]["CONV_TYPE"] == "GAT"

        # Verify all arms are distinct
        env_strs = [json.dumps(e, sort_keys=True) for e in envs]
        assert len(set(env_strs)) == 3, "All arms should have distinct envs"

    async def test_gnn_ruby_regression(self, coordinator):
        """Regression test for the GNN Ruby study bug.

        Previously, all 18 runs would get identical SAGE/64/3 config
        because env was always empty. This test verifies that the
        diverse configs from the LLM actually reach the arm specs.
        """
        coordinator.llm.complete_json.return_value = {
            "arms": [
                {
                    "name": "sage",
                    "config_overrides": {"CONV_TYPE": "SAGE", "HIDDEN_DIM": "64"},
                },
                {
                    "name": "gcn",
                    "config_overrides": {"CONV_TYPE": "GCN", "HIDDEN_DIM": "128"},
                },
                {
                    "name": "gat",
                    "config_overrides": {"CONV_TYPE": "GAT", "HIDDEN_DIM": "256"},
                },
                {
                    "name": "gin",
                    "config_overrides": {"CONV_TYPE": "GIN", "HIDDEN_DIM": "64"},
                },
                {
                    "name": "graphconv",
                    "config_overrides": {
                        "CONV_TYPE": "GraphConv",
                        "HIDDEN_DIM": "128",
                    },
                },
                {
                    "name": "sage_deep",
                    "config_overrides": {
                        "CONV_TYPE": "SAGE",
                        "HIDDEN_DIM": "64",
                        "NUM_LAYERS": "6",
                    },
                },
            ],
        }

        llm_arms = await coordinator.propose_arms()
        spec = coordinator._translate_to_spec(llm_arms, iteration=0)

        # The critical assertion: each arm must have its own distinct env
        conv_types = {arm.env.get("CONV_TYPE") for arm in spec.arms}
        assert len(conv_types) >= 4, (
            f"Expected diverse CONV_TYPE values, got {conv_types}"
        )

        # No arm should have an empty env
        for arm in spec.arms:
            assert arm.env, f"arm {arm.name} has empty env — regression!"


# ---------------------------------------------------------------------------
# Base config loading
# ---------------------------------------------------------------------------


class TestBaseConfigLoading:
    def test_loads_base_config(self, tmp_path, config):
        base_config = {"conv_type": "SAGE", "hidden_dim": 64}
        config_file = tmp_path / "base.yaml"
        config_file.write_text(yaml.dump(base_config))

        spec = ResearchSpec(
            name="test",
            repo={"url": "https://github.com/user/repo.git"},
            base_config_path=str(config_file),
        )

        coord = ResearchCoordinator(config, spec)

        assert coord._base_config == {"conv_type": "SAGE", "hidden_dim": 64}

    def test_missing_base_config_warns(self, config, caplog):
        spec = ResearchSpec(
            name="test",
            repo={"url": "https://github.com/user/repo.git"},
            base_config_path="/nonexistent/config.yaml",
        )

        coord = ResearchCoordinator(config, spec)

        assert coord._base_config == {}
        assert "not found" in caplog.text

    def test_no_base_config_path(self, config):
        spec = ResearchSpec(
            name="test",
            repo={"url": "https://github.com/user/repo.git"},
        )

        coord = ResearchCoordinator(config, spec)

        assert coord._base_config == {}
