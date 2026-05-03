"""Tests for ExperimentSpec and metric parsing."""

from __future__ import annotations

import tempfile

import pytest
from pydantic import ValidationError

from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    MetricsSpec,
    PreflightSpec,
    RepoSpec,
    ValidationSpec,
    parse_metrics,
    parse_metrics_block,
    parse_metrics_json_line,
)


@pytest.fixture
def minimal_spec():
    return ExperimentSpec(
        name="test-experiment",
        repo=RepoSpec(url="https://github.com/test/repo.git"),
        arms=[
            ArmSpec(name="baseline", command="python train.py"),
            ArmSpec(name="optimized", command="python train.py --fast", description="+Optimize"),
        ],
    )


class TestExperimentSpec:
    def test_minimal_construction(self, minimal_spec):
        assert minimal_spec.name == "test-experiment"
        assert len(minimal_spec.arms) == 2
        assert minimal_spec.hardware.gpu == "RTX 4090"

    def test_defaults(self, minimal_spec):
        assert minimal_spec.hardware.max_dph == 0.50
        assert minimal_spec.budget.max_dollars == 10.0
        assert minimal_spec.data.source == "none"
        assert minimal_spec.metrics.protocol == "json_line"
        assert minimal_spec.preflight is None
        assert minimal_spec.validation is None

    def test_get_arm(self, minimal_spec):
        arm = minimal_spec.get_arm("baseline")
        assert arm is not None
        assert arm.command == "python train.py"
        assert minimal_spec.get_arm("nonexistent") is None

    def test_get_arms_by_index(self, minimal_spec):
        arms = minimal_spec.get_arms_by_index([1])
        assert len(arms) == 1
        assert arms[0].name == "optimized"

    def test_get_arms_by_index_out_of_range(self, minimal_spec):
        arms = minimal_spec.get_arms_by_index([0, 99])
        assert len(arms) == 1

    def test_resolve_command(self, minimal_spec):
        arm = ArmSpec(
            name="test",
            config="configs/test.yaml",
            command="bash run.sh {config}",
        )
        resolved = minimal_spec.resolve_command(arm)
        assert resolved == "bash run.sh configs/test.yaml"


class TestPreflightSpec:
    def test_construction(self):
        pf = PreflightSpec(command="python train.py --epochs 1")
        assert pf.command == "python train.py --epochs 1"
        assert pf.timeout_s == 60
        assert pf.check_metrics is False

    def test_custom_values(self):
        pf = PreflightSpec(command="python test.py", timeout_s=120, check_metrics=True)
        assert pf.timeout_s == 120
        assert pf.check_metrics is True

    def test_spec_with_preflight(self):
        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
            preflight=PreflightSpec(
                command="python train.py --epochs 1 --batch_size 2",
                timeout_s=90,
                check_metrics=True,
            ),
        )
        assert spec.preflight is not None
        assert spec.preflight.command == "python train.py --epochs 1 --batch_size 2"
        assert spec.preflight.timeout_s == 90
        assert spec.preflight.check_metrics is True

    def test_yaml_round_trip_with_preflight(self):
        spec = ExperimentSpec(
            name="preflight-test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
            preflight=PreflightSpec(
                command="python train.py --epochs 1",
                timeout_s=30,
                check_metrics=True,
            ),
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            spec.to_yaml(f.name)
            loaded = ExperimentSpec.from_yaml(f.name)

        assert loaded.preflight is not None
        assert loaded.preflight.command == "python train.py --epochs 1"
        assert loaded.preflight.timeout_s == 30
        assert loaded.preflight.check_metrics is True

    def test_yaml_without_preflight(self):
        yaml_content = """\
name: no-preflight
repo:
  url: https://github.com/test/repo.git
arms:
  - name: arm1
    command: python train.py
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)
        assert spec.preflight is None

    def test_yaml_with_preflight(self):
        yaml_content = """\
name: with-preflight
repo:
  url: https://github.com/test/repo.git
arms:
  - name: arm1
    command: python train.py
preflight:
  command: "python train.py --epochs 1 --batch_size 2 --max_steps 5"
  timeout_s: 60
  check_metrics: true
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)
        assert spec.preflight is not None
        assert spec.preflight.check_metrics is True
        assert spec.preflight.timeout_s == 60


class TestValidationSpec:
    def test_construction(self):
        val = ValidationSpec(command="python validate.py")
        assert val.command == "python validate.py"
        assert val.timeout_s == 120
        assert val.required_metrics == []
        assert val.prefix == ""

    def test_custom_values(self):
        val = ValidationSpec(
            command="python validate.py --output /workspace/output",
            timeout_s=300,
            required_metrics=["real_validity_pct", "ast_match_pct"],
            prefix="val_",
        )
        assert val.timeout_s == 300
        assert len(val.required_metrics) == 2
        assert val.prefix == "val_"

    def test_spec_with_validation(self):
        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
            validation=ValidationSpec(
                command="python validate.py",
                timeout_s=90,
                required_metrics=["real_validity_pct"],
            ),
        )
        assert spec.validation is not None
        assert spec.validation.command == "python validate.py"
        assert spec.validation.timeout_s == 90
        assert spec.validation.required_metrics == ["real_validity_pct"]

    def test_yaml_round_trip_with_validation(self):
        spec = ExperimentSpec(
            name="validation-test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
            validation=ValidationSpec(
                command="python validate.py --strict",
                timeout_s=180,
                required_metrics=["syntax_valid_pct"],
                prefix="val_",
            ),
        )
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False,
        ) as f:
            spec.to_yaml(f.name)
            loaded = ExperimentSpec.from_yaml(f.name)

        assert loaded.validation is not None
        assert loaded.validation.command == "python validate.py --strict"
        assert loaded.validation.timeout_s == 180
        assert loaded.validation.required_metrics == ["syntax_valid_pct"]
        assert loaded.validation.prefix == "val_"

    def test_yaml_without_validation(self):
        yaml_content = """\
name: no-validation
repo:
  url: https://github.com/test/repo.git
arms:
  - name: arm1
    command: python train.py
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)
        assert spec.validation is None

    def test_yaml_with_validation(self):
        yaml_content = """\
name: with-validation
repo:
  url: https://github.com/test/repo.git
arms:
  - name: arm1
    command: python train.py
validation:
  command: "ruby -c generated/*.rb | python count_valid.py"
  timeout_s: 60
  required_metrics:
    - real_validity_pct
    - parse_error_count
  prefix: "val_"
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)
        assert spec.validation is not None
        assert spec.validation.required_metrics == [
            "real_validity_pct", "parse_error_count",
        ]
        assert spec.validation.prefix == "val_"


class TestExperimentSpecYAML:
    def test_round_trip(self, minimal_spec):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            minimal_spec.to_yaml(f.name)
            loaded = ExperimentSpec.from_yaml(f.name)

        assert loaded.name == minimal_spec.name
        assert len(loaded.arms) == len(minimal_spec.arms)
        assert loaded.arms[0].name == "baseline"

    def test_load_full_spec(self):
        yaml_content = """\
name: throughput-ablation
description: Test throughput optimizations on 4090
hardware:
  gpu: RTX 4090
  num_gpus: 1
  max_dph: 0.45
  image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
data:
  source: s3-presigned
  urls_file: data-urls.txt
  target: /workspace/data/shards
repo:
  url: https://github.com/user/repo.git
  branch: experiment/test
deps:
  pre_install:
    - pip uninstall torch -y
    - pip install torch --index-url https://download.pytorch.org/whl/cu130
  requirements: requirements.txt
  exclude_from_requirements:
    - torch
    - torchvision
  verify: "python -c \\"import torch; assert hasattr(torch.optim, 'Muon')\\""
arms:
  - name: baseline
    command: bash run.sh {config}
    config: configs/baseline.yaml
  - name: optimized
    command: bash run.sh {config}
    config: configs/optimized.yaml
    description: +AllOptimizations
    env:
      CUDA_LAUNCH_BLOCKING: "0"
metrics:
  protocol: block
  start_marker: "--- RESULTS ---"
  end_marker: "--- END RESULTS ---"
budget:
  max_dollars: 5.0
  train_timeout_s: 900
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)

        assert spec.name == "throughput-ablation"
        assert spec.hardware.gpu == "RTX 4090"
        assert spec.hardware.max_dph == 0.45
        assert spec.data.source == "s3-presigned"
        assert spec.data.urls_file == "data-urls.txt"
        assert spec.repo.branch == "experiment/test"
        assert len(spec.deps.pre_install) == 2
        assert "torch" in spec.deps.exclude_from_requirements
        assert len(spec.arms) == 2
        assert spec.arms[1].env == {"CUDA_LAUNCH_BLOCKING": "0"}
        assert spec.metrics.protocol == "block"
        assert spec.budget.max_dollars == 5.0


class TestMetricsBlock:
    def test_parse_key_value(self):
        stdout = """\
Some output
--- RESULTS ---
avg_iter_per_sec: 3.528
peak_vram_gb: 5.49
final_loss: 0.3255
total_steps: 300
--- END RESULTS ---
More output
"""
        spec = MetricsSpec(protocol="block")
        result = parse_metrics_block(stdout, spec)
        assert result["avg_iter_per_sec"] == pytest.approx(3.528)
        assert result["peak_vram_gb"] == pytest.approx(5.49)
        assert result["total_steps"] == 300.0

    def test_no_block(self):
        result = parse_metrics_block("no markers here", MetricsSpec())
        assert result == {}

    def test_string_values(self):
        stdout = """\
--- RESULTS ---
status: completed
score: 0.95
--- END RESULTS ---
"""
        result = parse_metrics_block(stdout, MetricsSpec())
        assert result["status"] == "completed"
        assert result["score"] == pytest.approx(0.95)


class TestMetricsJsonLine:
    def test_parse_json(self):
        stdout = 'Starting...\nMETRICS:{"loss": 0.5, "it_per_sec": 3.2}\nDone\n'
        spec = MetricsSpec(protocol="json_line")
        result = parse_metrics_json_line(stdout, spec)
        assert result["loss"] == pytest.approx(0.5)
        assert result["it_per_sec"] == pytest.approx(3.2)

    def test_last_line_wins(self):
        stdout = 'METRICS:{"step": 1}\nMETRICS:{"step": 100}\n'
        result = parse_metrics_json_line(stdout, MetricsSpec())
        assert result["step"] == 100

    def test_no_metrics(self):
        result = parse_metrics_json_line("no metrics here", MetricsSpec())
        assert result == {}

    def test_invalid_json(self):
        result = parse_metrics_json_line("METRICS:not-json", MetricsSpec())
        assert result == {}


class TestParseMetricsDispatch:
    def test_block_protocol(self):
        stdout = "--- RESULTS ---\nval: 42\n--- END RESULTS ---"
        spec = MetricsSpec(protocol="block")
        result = parse_metrics(stdout, spec)
        assert result["val"] == 42.0

    def test_json_protocol(self):
        stdout = 'METRICS:{"val": 42}'
        spec = MetricsSpec(protocol="json_line")
        result = parse_metrics(stdout, spec)
        assert result["val"] == 42


class TestArmConfigHash:
    def test_hash_is_deterministic(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a1 = ArmSpec(name="a", command="python train.py", env={"LR": "0.1"})
        a2 = ArmSpec(
            name="other-name",
            description="prose",
            command="python train.py",
            env={"LR": "0.1"},
        )
        # Same command + env → same hash even though name/description differ.
        assert arm_config_hash(a1) == arm_config_hash(a2)

    def test_hash_differs_on_command(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a1 = ArmSpec(name="a", command="python train.py")
        a2 = ArmSpec(name="b", command="python train.py --fast")
        assert arm_config_hash(a1) != arm_config_hash(a2)

    def test_hash_differs_on_env(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a1 = ArmSpec(name="a", command="python train.py", env={"LR": "0.1"})
        a2 = ArmSpec(name="b", command="python train.py", env={"LR": "0.2"})
        assert arm_config_hash(a1) != arm_config_hash(a2)

    def test_hash_is_order_independent_for_env(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a1 = ArmSpec(name="a", command="python train.py", env={"A": "1", "B": "2"})
        a2 = ArmSpec(name="b", command="python train.py", env={"B": "2", "A": "1"})
        assert arm_config_hash(a1) == arm_config_hash(a2)

    def test_hash_length(self):
        from ratiocinator.fleet.spec import arm_config_hash

        h = arm_config_hash(ArmSpec(name="a", command="python train.py"))
        assert len(h) == 12


class TestFindDuplicateArms:
    def test_no_duplicates(self):
        from ratiocinator.fleet.spec import find_duplicate_arms

        arms = [
            ArmSpec(name="a", command="python train.py --lr 0.1"),
            ArmSpec(name="b", command="python train.py --lr 0.2"),
        ]
        assert find_duplicate_arms(arms) == {}

    def test_duplicates_grouped_by_hash(self):
        from ratiocinator.fleet.spec import arm_config_hash, find_duplicate_arms

        arms = [
            ArmSpec(name="a", command="python train.py", env={"X": "1"}),
            ArmSpec(
                name="b-prose",
                description="different prose",
                command="python train.py",
                env={"X": "1"},
            ),
            ArmSpec(name="c", command="python train.py", env={"X": "2"}),
        ]
        dups = find_duplicate_arms(arms)
        assert len(dups) == 1
        h = arm_config_hash(arms[0])
        assert sorted(dups[h]) == ["a", "b-prose"]


class TestArmConfigHashIncludesConfig:
    def test_template_command_distinct_configs(self):
        from ratiocinator.fleet.spec import arm_config_hash

        # Same template command + env, different `config` values used by
        # `{config}` placeholder substitution → must hash differently.
        a = ArmSpec(name="a", command="python train.py {config}", config="baseline.yaml")
        b = ArmSpec(name="b", command="python train.py {config}", config="optimized.yaml")
        assert arm_config_hash(a) != arm_config_hash(b)

    def test_template_command_same_config(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a = ArmSpec(name="a", command="python train.py {config}", config="x.yaml")
        b = ArmSpec(
            name="b-other",
            description="prose",
            command="python train.py {config}",
            config="x.yaml",
        )
        assert arm_config_hash(a) == arm_config_hash(b)

    def test_resolved_command_collides_with_hardcoded(self):
        """Arms that resolve to the same effective command should collide."""
        from ratiocinator.fleet.spec import arm_config_hash

        a = ArmSpec(name="a", command="python train.py baseline.yaml")
        b = ArmSpec(
            name="b", command="python train.py {config}", config="baseline.yaml"
        )
        assert arm_config_hash(a) == arm_config_hash(b)

    def test_name_placeholder_does_not_affect_hash(self):
        from ratiocinator.fleet.spec import arm_config_hash

        a = ArmSpec(name="run-A", command="python train.py --tag {name}")
        b = ArmSpec(name="run-B", command="python train.py --tag {name}")
        assert arm_config_hash(a) == arm_config_hash(b)


class TestBatchSize:
    """Tests for batch_size on HardwareSpec and ArmSpec."""

    def test_hardware_batch_size_default_none(self):
        from ratiocinator.fleet.spec import HardwareSpec

        hw = HardwareSpec()
        assert hw.batch_size is None

    def test_arm_batch_size_default_none(self):
        arm = ArmSpec(name="test", command="python train.py")
        assert arm.batch_size is None

    def test_resolve_batch_size_from_hardware(self):
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(name="arm1", command="python train.py")],
        )
        assert spec.resolve_batch_size(spec.arms[0]) == 64

    def test_resolve_batch_size_arm_overrides_hardware(self):
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(name="arm1", command="python train.py", batch_size=16)],
        )
        assert spec.resolve_batch_size(spec.arms[0]) == 16

    def test_resolve_batch_size_none_when_unset(self):
        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
        )
        assert spec.resolve_batch_size(spec.arms[0]) is None

    def test_arm_config_hash_differs_on_batch_size(self):
        from ratiocinator.fleet.spec import HardwareSpec, arm_config_hash

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[
                ArmSpec(name="a", command="python train.py"),
                ArmSpec(name="b", command="python train.py", batch_size=16),
            ],
        )
        h_a = arm_config_hash(spec.arms[0], spec)
        h_b = arm_config_hash(spec.arms[1], spec)
        assert h_a != h_b

    def test_yaml_round_trip_with_batch_size(self):
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="batch-test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[
                ArmSpec(name="wide", command="python train.py", batch_size=16),
                ArmSpec(name="narrow", command="python train.py", batch_size=128),
            ],
        )
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            spec.to_yaml(f.name)
            loaded = ExperimentSpec.from_yaml(f.name)

        assert loaded.hardware.batch_size == 64
        assert loaded.arms[0].batch_size == 16
        assert loaded.arms[1].batch_size == 128

    def test_yaml_batch_size_parsing(self):
        yaml_content = """\
name: batch-yaml
repo:
  url: https://github.com/test/repo.git
hardware:
  gpu: "RTX 4090"
  batch_size: 64
arms:
  - name: wide-model
    command: python train.py
    batch_size: 16
  - name: narrow-model
    command: python train.py
    batch_size: 128
"""
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            spec = ExperimentSpec.from_yaml(f.name)

        assert spec.hardware.batch_size == 64
        assert spec.arms[0].batch_size == 16
        assert spec.arms[1].batch_size == 128
        assert spec.resolve_batch_size(spec.arms[0]) == 16
        assert spec.resolve_batch_size(spec.arms[1]) == 128

    def test_hardware_batch_size_rejects_zero(self):
        from ratiocinator.fleet.spec import HardwareSpec

        with pytest.raises(ValidationError):
            HardwareSpec(batch_size=0)

    def test_hardware_batch_size_rejects_negative(self):
        from ratiocinator.fleet.spec import HardwareSpec

        with pytest.raises(ValidationError):
            HardwareSpec(batch_size=-1)

    def test_arm_batch_size_rejects_zero(self):
        with pytest.raises(ValidationError):
            ArmSpec(name="bad", command="train.py", batch_size=0)

    def test_arm_batch_size_rejects_negative(self):
        with pytest.raises(ValidationError):
            ArmSpec(name="bad", command="train.py", batch_size=-4)


class TestBatchSizeEnvInjection:
    """Tests for BATCH_SIZE env var injection in executor."""

    def test_arm_env_with_batch_size_from_hardware(self):
        from ratiocinator.fleet.executor import _arm_env_with_batch_size
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(name="arm1", command="python train.py")],
        )
        env = _arm_env_with_batch_size(spec.arms[0], spec)
        assert env["BATCH_SIZE"] == "64"

    def test_arm_env_with_batch_size_per_arm_override(self):
        from ratiocinator.fleet.executor import _arm_env_with_batch_size
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(name="arm1", command="python train.py", batch_size=16)],
        )
        env = _arm_env_with_batch_size(spec.arms[0], spec)
        assert env["BATCH_SIZE"] == "16"

    def test_arm_env_no_batch_size_when_unset(self):
        from ratiocinator.fleet.executor import _arm_env_with_batch_size

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            arms=[ArmSpec(name="arm1", command="python train.py")],
        )
        env = _arm_env_with_batch_size(spec.arms[0], spec)
        assert "BATCH_SIZE" not in env

    def test_explicit_env_batch_size_not_overridden(self):
        from ratiocinator.fleet.executor import _arm_env_with_batch_size
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(
                name="arm1",
                command="python train.py",
                env={"BATCH_SIZE": "256"},
            )],
        )
        env = _arm_env_with_batch_size(spec.arms[0], spec)
        assert env["BATCH_SIZE"] == "256"

    def test_arm_env_preserves_other_vars(self):
        from ratiocinator.fleet.executor import _arm_env_with_batch_size
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[ArmSpec(
                name="arm1",
                command="python train.py",
                env={"LR": "0.001"},
            )],
        )
        env = _arm_env_with_batch_size(spec.arms[0], spec)
        assert env["LR"] == "0.001"
        assert env["BATCH_SIZE"] == "64"

    def test_hash_same_when_env_batch_size_matches_spec(self):
        """Arms with explicit env BATCH_SIZE matching spec resolve same hash."""
        from ratiocinator.fleet.spec import HardwareSpec, arm_config_hash

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[
                ArmSpec(name="a", command="python train.py"),
                ArmSpec(
                    name="b",
                    command="python train.py",
                    env={"BATCH_SIZE": "64"},
                ),
            ],
        )
        # Both arms effectively run with BATCH_SIZE=64, so they should
        # hash identically for duplicate detection purposes.
        h_a = arm_config_hash(spec.arms[0], spec)
        h_b = arm_config_hash(spec.arms[1], spec)
        assert h_a == h_b

    def test_hash_differs_when_env_batch_size_overrides_spec(self):
        """arm.env['BATCH_SIZE'] overriding spec batch_size changes the hash."""
        from ratiocinator.fleet.spec import HardwareSpec, arm_config_hash

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[
                ArmSpec(name="a", command="python train.py"),
                ArmSpec(
                    name="b",
                    command="python train.py",
                    env={"BATCH_SIZE": "128"},
                ),
            ],
        )
        h_a = arm_config_hash(spec.arms[0], spec)
        h_b = arm_config_hash(spec.arms[1], spec)
        assert h_a != h_b

    def test_resolve_arm_env_same_as_executor_helper(self):
        """resolve_arm_env and _arm_env_with_batch_size produce identical results."""
        from ratiocinator.fleet.executor import _arm_env_with_batch_size
        from ratiocinator.fleet.spec import HardwareSpec

        spec = ExperimentSpec(
            name="test",
            repo=RepoSpec(url="https://github.com/test/repo.git"),
            hardware=HardwareSpec(batch_size=64),
            arms=[
                ArmSpec(name="a", command="python train.py"),
                ArmSpec(name="b", command="python train.py", batch_size=16),
                ArmSpec(
                    name="c",
                    command="python train.py",
                    env={"BATCH_SIZE": "256"},
                ),
            ],
        )
        for arm in spec.arms:
            assert spec.resolve_arm_env(arm) == _arm_env_with_batch_size(arm, spec)


class TestDetectGitContext:
    """Tests for detect_git_context() auto-detection."""

    def test_detect_from_current_repo(self):
        """Smoke test: detect_git_context returns a valid RepoSpec in this repo."""
        from ratiocinator.fleet.spec import detect_git_context

        result = detect_git_context()
        # We're running inside the ratiocinator repo clone, so this should work
        assert result is not None
        assert "ratiocinator" in result.url
        assert result.branch  # non-empty
        assert len(result.commit) == 40  # full SHA

    def test_detect_from_non_git_dir(self, tmp_path):
        """Returns None when cwd is not a git repo."""
        from ratiocinator.fleet.spec import detect_git_context

        result = detect_git_context(cwd=tmp_path)
        assert result is None

    def test_from_yaml_auto_detects_repo(self, tmp_path):
        """from_yaml auto-fills repo when omitted in YAML."""
        from unittest.mock import patch

        yaml_content = """\
name: auto-detect-test
arms:
  - name: baseline
    command: python train.py
"""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml_content)

        fake_repo = RepoSpec(
            url="https://github.com/user/repo.git",
            branch="feature/test",
            commit="a" * 40,
        )
        with patch(
            "ratiocinator.fleet.spec.detect_git_context", return_value=fake_repo
        ):
            spec = ExperimentSpec.from_yaml(spec_file)

        assert spec.repo is not None
        assert spec.repo.url == "https://github.com/user/repo.git"
        assert spec.repo.branch == "feature/test"
        assert spec.repo.commit == "a" * 40

    def test_from_yaml_raises_when_no_repo_and_no_git(self, tmp_path):
        """from_yaml raises ValueError when repo is missing and git fails."""
        from unittest.mock import patch

        yaml_content = """\
name: no-repo-test
arms:
  - name: baseline
    command: python train.py
"""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml_content)

        with patch(
            "ratiocinator.fleet.spec.detect_git_context", return_value=None
        ), pytest.raises(ValueError, match="No 'repo' block in spec"):
            ExperimentSpec.from_yaml(spec_file)

    def test_from_yaml_explicit_repo_not_overridden(self, tmp_path):
        """from_yaml does not call detect when repo is explicit."""
        from unittest.mock import patch

        yaml_content = """\
name: explicit-repo-test
repo:
  url: https://github.com/explicit/repo.git
  branch: main
arms:
  - name: baseline
    command: python train.py
"""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml_content)

        with patch(
            "ratiocinator.fleet.spec.detect_git_context"
        ) as mock_detect:
            spec = ExperimentSpec.from_yaml(spec_file)

        mock_detect.assert_not_called()
        assert spec.repo.url == "https://github.com/explicit/repo.git"
