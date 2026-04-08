"""Tests for the experiment loop."""

from __future__ import annotations

import pytest

from ratiocinator.experiment import ExperimentLoop


@pytest.fixture
def toy_repo(tmp_path):
    """Create a minimal training script."""
    train_py = tmp_path / "train.py"
    train_py.write_text(
        'import os\n'
        'LR = 0.01\n'
        'print("training...")\n'
        'print("METRICS:" + \'{"loss": 0.5, "accuracy": 0.8}\')\n'
    )
    return tmp_path


def test_read_source(toy_repo):
    loop = ExperimentLoop()
    sources = loop._read_source(toy_repo)
    assert "train.py" in sources
    assert "LR = 0.01" in sources["train.py"]


def test_apply_diff(toy_repo):
    loop = ExperimentLoop()
    proposal = {
        "filename": "train.py",
        "original": "LR = 0.01",
        "replacement": "LR = 0.001",
    }
    diff = loop._apply_diff(toy_repo, proposal)
    assert diff["replacement"] == "LR = 0.001"
    content = (toy_repo / "train.py").read_text()
    assert "LR = 0.001" in content


def test_extract_metrics():
    loop = ExperimentLoop()
    stdout = 'training...\nMETRICS:{"loss": 0.5, "accuracy": 0.8}\n'
    metrics = loop._extract_metrics(stdout)
    assert metrics == {"loss": 0.5, "accuracy": 0.8}


def test_extract_metrics_no_metrics_line():
    loop = ExperimentLoop()
    stdout = "training...\ndone\n"
    metrics = loop._extract_metrics(stdout)
    assert metrics == {}
