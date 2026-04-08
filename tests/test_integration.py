"""Integration tests: validate the full search → plots → paper → review flow.

These tests use mocked LLM responses but exercise real components end-to-end.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.config import Config, LLMConfig, SandboxConfig, SearchConfig
from ratiocinator.llm.client import LLMClient, LLMResponse
from ratiocinator.sandbox.runner import RunResult
from ratiocinator.search.bfts import BestFirstSearch
from ratiocinator.search.tree import ExperimentTree, NodeStatus
from ratiocinator.synthesis.paper import PaperGenerator
from ratiocinator.synthesis.plotting import generate_plots
from ratiocinator.synthesis.reviewer import AutoReviewer


@pytest.fixture
def toy_repo(tmp_path):
    """Create a minimal trainable script that emits METRICS."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(
        'import random\n'
        'loss = random.uniform(0.5, 1.5)\n'
        'print(f"METRICS:{{\\"train_loss\\": {loss:.4f}}}")\n'
    )
    return repo


@pytest.fixture
def config(tmp_path):
    return Config(
        llm=LLMConfig(),
        sandbox=SandboxConfig(),
        search=SearchConfig(max_nodes=4, max_depth=2, branching_factor=1),
        work_dir=tmp_path / "work",
    )


@pytest.fixture
def populated_tree(tmp_path):
    """Build a realistic tree with baseline + 2 successful children + 1 failed."""
    db = tmp_path / "tree.db"
    tree = ExperimentTree(db)

    root = tree.add_root("baseline — unmodified code")
    root.status = NodeStatus.SUCCESS
    root.score = 1.09
    root.metrics = {"train_loss": 1.09, "val_loss": 1.77}
    tree.update(root)

    c1 = tree.add_child(root.id, "Add ReLU activation to hidden layer", {
        "filename": "train.py",
        "original": "nn.Linear(4, 4)",
        "replacement": "nn.Sequential(nn.Linear(4, 4), nn.ReLU())",
    })
    c1.status = NodeStatus.SUCCESS
    c1.score = 0.89
    c1.metrics = {"train_loss": 0.89, "val_loss": 1.47}
    tree.update(c1)

    c2 = tree.add_child(root.id, "Increase hidden size to 16", {
        "filename": "train.py",
        "original": "nn.Linear(4, 4)",
        "replacement": "nn.Linear(4, 16)",
    })
    c2.status = NodeStatus.SUCCESS
    c2.score = 0.95
    c2.metrics = {"train_loss": 0.95, "val_loss": 1.60}
    tree.update(c2)

    c3 = tree.add_child(c1.id, "Add batch normalization", {
        "filename": "train.py",
        "original": "nn.ReLU()",
        "replacement": "nn.Sequential(nn.ReLU(), nn.BatchNorm1d(4))",
    })
    c3.status = NodeStatus.FAILED
    c3.stderr = "RuntimeError: expected 2D input"
    tree.update(c3)

    yield tree
    tree.close()


class TestFullSynthesisPipeline:
    """Test the complete flow: tree → plots → paper → review."""

    @pytest.mark.asyncio
    async def test_plots_paper_review_pipeline(self, populated_tree, tmp_path):
        """End-to-end: generate plots, write paper, review & revise."""
        llm = LLMClient(LLMConfig())

        # --- Phase 1: Plots ---
        plot_dir = tmp_path / "plots"
        plots = generate_plots(populated_tree, plot_dir, score_key="train_loss")
        assert len(plots) == 3
        assert all(p.exists() for p in plots)

        # --- Phase 2: Paper ---
        section_resp = LLMResponse(
            content="We present an automated study of code optimization.",
            model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        gen = PaperGenerator(llm)
        paper_path = tmp_path / "paper.md"

        with patch.object(llm, "complete", new_callable=AsyncMock, return_value=section_resp):
            paper = await gen.generate(
                populated_tree,
                title="Integration Test Paper",
                plot_paths=plots,
                output_path=paper_path,
            )

        assert paper_path.exists()
        assert "# Integration Test Paper" in paper
        assert "automated study" in paper
        # Verify plots are referenced
        assert "scores_comparison.png" in paper

        # --- Phase 3: Review (accept on first try) ---
        accept_result = {
            "scores": {
                "clarity": 2, "methodology": 2, "results": 2,
                "novelty": 1, "completeness": 1,
            },
            "total_score": 8,
            "issues": [],
            "suggestions": [],
            "verdict": "accept",
        }
        reviewer = AutoReviewer(llm, max_revisions=2)

        with patch.object(
            llm, "complete_json", new_callable=AsyncMock,
            return_value=accept_result,
        ):
            final_paper, reviews = await reviewer.review_and_revise(
                paper, min_score=6,
            )

        assert len(reviews) == 1
        assert reviews[0].total_score == 8
        assert reviews[0].verdict == "accept"
        assert final_paper == paper  # No revision needed

    @pytest.mark.asyncio
    async def test_review_revise_cycle(self, populated_tree, tmp_path):
        """Test that revision cycles work when paper scores too low."""
        llm = LLMClient(LLMConfig())

        low_review = {
            "scores": {
                "clarity": 1, "methodology": 1, "results": 1,
                "novelty": 0, "completeness": 0,
            },
            "total_score": 3,
            "issues": ["Missing ablation study", "No error bars"],
            "suggestions": ["Add table of results"],
            "verdict": "revise",
        }
        revision = {
            "section": "Results",
            "revised_content": "Improved results with detailed analysis.",
            "changes_made": "Added ablation study and error bars",
        }
        pass_review = {
            "scores": {
                "clarity": 2, "methodology": 2, "results": 2,
                "novelty": 1, "completeness": 1,
            },
            "total_score": 8,
            "issues": [],
            "suggestions": [],
            "verdict": "accept",
        }

        markdown = (
            "# Test Paper\n\n"
            "## 4. Results\n\nOriginal results.\n\n"
            "## 6. Conclusion\n\nDone.\n"
        )

        reviewer = AutoReviewer(llm, max_revisions=3)
        with patch.object(
            llm,
            "complete_json",
            new_callable=AsyncMock,
            side_effect=[low_review, revision, pass_review],
        ):
            final, reviews = await reviewer.review_and_revise(markdown, min_score=6)

        assert len(reviews) == 2
        assert reviews[0].total_score == 3
        assert reviews[1].total_score == 8
        assert "Improved results" in final


class TestSearchWithMockedSandbox:
    """Test tree search with a mocked sandbox runner."""

    @pytest.mark.asyncio
    async def test_search_builds_tree(self, config, toy_repo):
        """BFTS creates baseline + candidates and tracks them."""
        from ratiocinator.search.bfts import BudgetExhaustedError

        bfts = BestFirstSearch(
            config, toy_repo, score_key="train_loss", lower_is_better=True,
        )

        baseline_result = RunResult(
            exit_code=0,
            stdout='METRICS:{"train_loss": 1.09, "val_loss": 1.77}',
            stderr="",
        )
        improved_result = RunResult(
            exit_code=0,
            stdout='METRICS:{"train_loss": 0.85, "val_loss": 1.50}',
            stderr="",
        )

        proposal = {
            "reasoning": "Add ReLU activation",
            "filename": "train.py",
            "original": "loss = random.uniform(0.5, 1.5)",
            "replacement": "loss = random.uniform(0.3, 1.0)",
        }

        call_count = 0

        def mock_run(**kwargs):
            nonlocal call_count
            call_count += 1
            return baseline_result if call_count == 1 else improved_result

        bfts.sandbox.run = mock_run

        with patch.object(
            bfts.llm, "complete_json", new_callable=AsyncMock, return_value=proposal,
        ):
            try:
                summary = await bfts.run()
            except BudgetExhaustedError:
                summary = bfts.tree.summary()

        assert summary["total_nodes"] >= 2
        assert summary["best_score"] is not None
        best = bfts.tree.get_best_leaf(lower_is_better=True)
        assert best is not None
        assert best.score <= 1.09


class TestTreeAndPlottingIntegration:
    """Test tree persistence and plotting work together."""

    def test_tree_roundtrip_and_plot(self, tmp_path):
        """Tree persists to SQLite and plotting reads it back correctly."""
        db = tmp_path / "test.db"
        tree = ExperimentTree(db)

        root = tree.add_root("baseline")
        root.status = NodeStatus.SUCCESS
        root.score = 1.0
        root.metrics = {"train_loss": 1.0}
        tree.update(root)

        child = tree.add_child(root.id, "improvement", {
            "filename": "train.py", "original": "a", "replacement": "b",
        })
        child.status = NodeStatus.SUCCESS
        child.score = 0.8
        child.metrics = {"train_loss": 0.8}
        tree.update(child)

        # Verify persistence by reopening
        tree.close()
        tree2 = ExperimentTree(db)
        assert tree2.count() == 2
        summary = tree2.summary()
        assert summary["best_score"] == 0.8

        # Plot from reopened tree
        plot_dir = tmp_path / "plots"
        plots = generate_plots(tree2, plot_dir)
        assert len(plots) == 3
        for p in plots:
            assert p.exists()
            assert p.stat().st_size > 0

        tree2.close()


class TestResultJsonOutput:
    """Test that the synthesize pipeline produces valid result.json structure."""

    @pytest.mark.asyncio
    async def test_result_json_schema(self, populated_tree, tmp_path):
        """Verify the result.json has all expected fields."""
        llm = LLMClient(LLMConfig())
        output = tmp_path / "output"
        output.mkdir()

        # Generate plots
        plots = generate_plots(populated_tree, output / "plots")

        # Mock paper generation
        section_resp = LLMResponse(
            content="Test content.",
            model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        gen = PaperGenerator(llm)
        paper_path = output / "paper.md"

        with patch.object(
            llm, "complete", new_callable=AsyncMock, return_value=section_resp,
        ):
            paper = await gen.generate(
                populated_tree, "Test", plot_paths=plots,
                output_path=paper_path,
            )

        # Mock review
        accept = {
            "scores": {
                "clarity": 2, "methodology": 1, "results": 2,
                "novelty": 1, "completeness": 1,
            },
            "total_score": 7,
            "issues": [],
            "suggestions": [],
            "verdict": "accept",
        }
        reviewer = AutoReviewer(llm, max_revisions=1)
        with patch.object(llm, "complete_json", new_callable=AsyncMock, return_value=accept):
            _final_paper, reviews = await reviewer.review_and_revise(
                paper, min_score=6,
            )

        # Build result.json same way cli.py does
        summary = populated_tree.summary()
        result = {
            "tree_summary": summary,
            "plots": [str(p) for p in plots],
            "paper": str(paper_path),
            "paper_final": str(output / "paper_final.md"),
            "review_scores": [r.total_score for r in reviews],
            "final_verdict": reviews[-1].verdict,
        }

        result_path = output / "result.json"
        result_path.write_text(json.dumps(result, indent=2))

        # Validate
        loaded = json.loads(result_path.read_text())
        assert "tree_summary" in loaded
        assert loaded["tree_summary"]["total_nodes"] == 4
        assert loaded["tree_summary"]["best_score"] == 0.89
        assert len(loaded["plots"]) == 3
        assert loaded["final_verdict"] == "accept"
        assert loaded["review_scores"] == [7]
