"""Tests for the Best-First Tree Search."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ratiocinator.config import Config, SearchConfig
from ratiocinator.sandbox.runner import RunResult
from ratiocinator.search.bfts import BestFirstSearch, BudgetExhaustedError


@pytest.fixture
def toy_repo(tmp_path):
    """Create a minimal training script."""
    train_py = tmp_path / "repo" / "train.py"
    train_py.parent.mkdir()
    train_py.write_text(
        'LR = 0.01\n'
        'print("training")\n'
        'print("METRICS:" + \'{"train_loss": 0.5}\')\n'
    )
    return train_py.parent


@pytest.fixture
def config(tmp_path):
    return Config(
        work_dir=tmp_path / ".ratiocinator",
        search=SearchConfig(
            max_depth=2, max_nodes=10, branching_factor=2, max_wall_clock_seconds=60
        ),
    )


def _mock_run_result(metrics: dict | None = None) -> RunResult:
    m = metrics or {"train_loss": 0.3}
    import json
    stdout = f'training\nMETRICS:{json.dumps(m)}\n'
    return RunResult(exit_code=0, stdout=stdout, stderr="", duration_seconds=1.0)


def _mock_failed_result() -> RunResult:
    return RunResult(exit_code=1, stdout="", stderr="RuntimeError: OOM", duration_seconds=0.5)


class TestBudgetChecks:
    def test_node_budget(self, config, toy_repo):
        config.search.max_nodes = 2
        bfts = BestFirstSearch(config, toy_repo)
        # Manually add nodes to hit budget
        bfts.tree.add_root("r1")
        bfts.tree.add_root("r2")
        with pytest.raises(BudgetExhaustedError, match="Node limit"):
            bfts._check_budgets(time.monotonic())

    def test_wall_clock_budget(self, config, toy_repo):
        config.search.max_wall_clock_seconds = 1
        bfts = BestFirstSearch(config, toy_repo)
        # Simulate time elapsed
        with pytest.raises(BudgetExhaustedError, match="Wall clock"):
            bfts._check_budgets(time.monotonic() - 10)


class TestDiffApplication:
    def test_apply_diff(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        diff = {"filename": "train.py", "original": "LR = 0.01", "replacement": "LR = 0.001"}
        bfts._apply_diff(toy_repo, diff)
        content = (toy_repo / "train.py").read_text()
        assert "LR = 0.001" in content

    def test_apply_diff_missing_file(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        diff = {"filename": "nonexistent.py", "original": "x", "replacement": "y"}
        bfts._apply_diff(toy_repo, diff)  # Should not raise

    def test_apply_ancestor_diffs(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        root = bfts.tree.add_root("baseline")
        child = bfts.tree.add_child(
            root.id, "change lr",
            {"filename": "train.py", "original": "LR = 0.01", "replacement": "LR = 0.001"},
        )

        import shutil
        import tempfile
        work = Path(tempfile.mkdtemp())
        try:
            ws = work / "ws"
            shutil.copytree(toy_repo, ws)
            bfts._apply_ancestor_diffs(child, ws)
            content = (ws / "train.py").read_text()
            assert "LR = 0.001" in content
        finally:
            shutil.rmtree(work)


class TestMetricsExtraction:
    def test_extracts_metrics(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        metrics = bfts._extract_metrics('stuff\nMETRICS:{"train_loss": 0.3}\nmore stuff')
        assert metrics == {"train_loss": 0.3}

    def test_returns_empty_on_no_metrics(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        assert bfts._extract_metrics("no metrics here") == {}
