"""Tests for ExperimentSpec and metric parsing."""

from __future__ import annotations

import tempfile

import pytest

from ratiocinator.fleet.spec import (
    ArmSpec,
    ExperimentSpec,
    MetricsSpec,
    RepoSpec,
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
