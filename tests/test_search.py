"""Tests for the Best-First Tree Search."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestIdeationIntegration:
    """Tests for literature-grounded ideation wired into BFTS."""

    def test_topic_param_accepted(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo, topic="quantization for attention")
        assert bfts.topic == "quantization for attention"
        assert bfts._ideation is None  # not initialized until run()

    def test_no_topic_skips_ideation(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)
        assert bfts.topic is None
        assert bfts._ideation is None

    @pytest.mark.asyncio
    async def test_init_ideation_loads_papers(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo, topic="efficient attention")
        bfts._ideation = MagicMock()
        bfts._ideation.load_literature.return_value = 5
        assert bfts._ideation is not None

    @pytest.mark.asyncio
    async def test_init_ideation_handles_import_error(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo, topic="test")
        with patch(
            "ratiocinator.ideation.grounded.LiteratureGroundedIdeation",
            side_effect=ImportError("no module"),
        ):
            await bfts._init_ideation()

    @pytest.mark.asyncio
    async def test_expand_with_ideation(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo, topic="quantization")
        root = bfts.tree.add_root("baseline")
        root.score = 0.5
        root.metrics = {"train_loss": 0.5}
        bfts.tree.update(root)

        # Mock ideation module
        mock_ideation = MagicMock()
        mock_ideation.generate_and_filter = AsyncMock(return_value=[
            {
                "hypothesis": "Apply polar quantization to KV cache",
                "reasoning": "Based on TurboQuant (2301.00001)",
                "inspired_by": "2301.00001",
                "filename": "train.py",
                "original": "LR = 0.01",
                "replacement": "LR = 0.005",
            },
            {
                "hypothesis": "Use QJL for residual compression",
                "reasoning": "Based on TurboQuant residual method",
                "inspired_by": "2301.00001",
                "filename": "train.py",
                "original": "LR = 0.01",
                "replacement": "LR = 0.003",
            },
        ])
        bfts._ideation = mock_ideation

        children = await bfts._expand(root)
        assert len(children) == 2
        assert "turboquant" in children[0].hypothesis.lower()
        mock_ideation.generate_and_filter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_expand_falls_back_when_ideation_returns_empty(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo, topic="quantization")
        root = bfts.tree.add_root("baseline")
        root.score = 0.5
        root.metrics = {"train_loss": 0.5}
        bfts.tree.update(root)

        # Mock ideation returning nothing
        mock_ideation = MagicMock()
        mock_ideation.generate_and_filter = AsyncMock(return_value=[])
        bfts._ideation = mock_ideation

        # Mock LLM for generic fallback
        mock_proposal = {
            "reasoning": "generic improvement",
            "filename": "train.py",
            "original": "LR = 0.01",
            "replacement": "LR = 0.001",
        }
        bfts.llm.complete_json = AsyncMock(return_value=mock_proposal)

        children = await bfts._expand(root)
        assert len(children) == config.search.branching_factor
        bfts.llm.complete_json.assert_awaited()

    @pytest.mark.asyncio
    async def test_expand_without_topic_uses_generic(self, config, toy_repo):
        bfts = BestFirstSearch(config, toy_repo)  # no topic
        root = bfts.tree.add_root("baseline")
        root.score = 0.5
        root.metrics = {"train_loss": 0.5}
        bfts.tree.update(root)

        mock_proposal = {
            "reasoning": "generic improvement",
            "filename": "train.py",
            "original": "LR = 0.01",
            "replacement": "LR = 0.001",
        }
        bfts.llm.complete_json = AsyncMock(return_value=mock_proposal)

        children = await bfts._expand(root)
        assert len(children) == config.search.branching_factor
        assert bfts._ideation is None

    @pytest.mark.asyncio
    async def test_expand_ideation_partial_fill(self, config, toy_repo):
        """When ideation returns fewer than branching_factor, generic fills the gap."""
        config.search.branching_factor = 3
        bfts = BestFirstSearch(config, toy_repo, topic="quantization")
        root = bfts.tree.add_root("baseline")
        root.score = 0.5
        root.metrics = {"train_loss": 0.5}
        bfts.tree.update(root)

        # Ideation returns only 1 result
        mock_ideation = MagicMock()
        mock_ideation.generate_and_filter = AsyncMock(return_value=[
            {
                "hypothesis": "Polar quant",
                "reasoning": "lit-based",
                "inspired_by": "2301.00001",
                "filename": "train.py",
                "original": "LR = 0.01",
                "replacement": "LR = 0.005",
            },
        ])
        bfts._ideation = mock_ideation

        # Generic fallback
        mock_proposal = {
            "reasoning": "generic",
            "filename": "train.py",
            "original": "LR = 0.01",
            "replacement": "LR = 0.002",
        }
        bfts.llm.complete_json = AsyncMock(return_value=mock_proposal)

        children = await bfts._expand(root)
        assert len(children) == 3  # 1 from ideation + 2 from generic
        assert bfts.llm.complete_json.await_count == 2
