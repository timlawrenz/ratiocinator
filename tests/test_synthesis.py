"""Tests for the paper generator and reviewer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.config import LLMConfig
from ratiocinator.llm.client import LLMClient, LLMResponse
from ratiocinator.search.tree import ExperimentTree, NodeStatus
from ratiocinator.synthesis.paper import PaperGenerator, _clean_section, _latex_escape
from ratiocinator.synthesis.reviewer import AutoReviewer


@pytest.fixture
def llm_client():
    return LLMClient(LLMConfig())


@pytest.fixture
def tree_with_results(tmp_path):
    db = tmp_path / "test.db"
    tree = ExperimentTree(db)
    root = tree.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    root.metrics = {"train_loss": 0.5}
    tree.update(root)
    yield tree
    tree.close()


class TestLatexEscape:
    def test_escapes_special_chars(self):
        assert _latex_escape("A & B") == r"A \& B"
        assert _latex_escape("100%") == r"100\%"
        assert _latex_escape("$x$") == r"\$x\$"

    def test_plain_text_unchanged(self):
        assert _latex_escape("hello world") == "hello world"


class TestCleanSection:
    def test_strips_documentclass(self):
        text = r"\documentclass{article}" + "\nHello world."
        assert _clean_section(text) == "Hello world."

    def test_strips_documentclass_with_options(self):
        text = r"\documentclass[11pt]{article}" + "\nHello."
        assert _clean_section(text) == "Hello."

    def test_strips_usepackage(self):
        text = r"\usepackage{amsmath}" + "\n" + r"\usepackage[utf8]{inputenc}" + "\nContent."
        assert _clean_section(text) == "Content."

    def test_strips_document_env(self):
        text = r"\begin{document}" + "\nContent here.\n" + r"\end{document}"
        assert _clean_section(text) == "Content here."

    def test_strips_abstract_env(self):
        text = r"\begin{abstract}" + "\nAbstract text.\n" + r"\end{abstract}"
        assert _clean_section(text) == "Abstract text."

    def test_strips_section_headers(self):
        text = r"\section{Introduction}" + "\nParagraph text."
        assert _clean_section(text) == "Paragraph text."

    def test_strips_full_preamble(self):
        text = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{amsmath}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{abstract}" + "\n"
            "We present a study.\n"
            r"\end{abstract}" + "\n"
            r"\end{document}"
        )
        assert _clean_section(text) == "We present a study."

    def test_preserves_normal_latex(self):
        text = r"We use $\alpha = 0.01$ and the loss $\mathcal{L}$."
        assert _clean_section(text) == text

    def test_collapses_blank_lines(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        assert _clean_section(text) == "First paragraph.\n\nSecond paragraph."


class TestPaperGenerator:
    @pytest.mark.asyncio
    async def test_generate_produces_latex(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)

        mock_resp = LLMResponse(
            content="Some section content.",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        with patch.object(
            llm_client, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            latex = await gen.generate(
                tree_with_results, "Test Paper"
            )

        assert r"\documentclass" in latex
        assert r"\title{Test Paper}" in latex
        assert "Some section content." in latex

    @pytest.mark.asyncio
    async def test_generate_writes_file(
        self, llm_client, tree_with_results, tmp_path
    ):
        gen = PaperGenerator(llm_client)
        output = tmp_path / "paper.tex"

        mock_resp = LLMResponse(
            content="Content.",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        with patch.object(
            llm_client, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            await gen.generate(tree_with_results, "Test", output_path=output)

        assert output.exists()
        assert r"\documentclass" in output.read_text()

    def test_build_figures(self, llm_client):
        gen = PaperGenerator(llm_client)
        paths = [Path("plot1.png"), Path("plot2.png")]
        figs = gen._build_figures(paths)
        assert r"\includegraphics" in figs
        assert "plot1.png" in figs
        assert "plot2.png" in figs

    def test_build_figures_empty(self, llm_client):
        gen = PaperGenerator(llm_client)
        assert gen._build_figures(None) == ""


class TestAutoReviewer:
    @pytest.mark.asyncio
    async def test_review(self, llm_client):
        reviewer = AutoReviewer(llm_client)
        mock_result = {
            "scores": {
                "clarity": 2,
                "methodology": 1,
                "results": 2,
                "novelty": 1,
                "completeness": 1,
            },
            "total_score": 7,
            "issues": ["Missing ablation study"],
            "suggestions": ["Add more plots"],
            "verdict": "accept",
        }
        with patch.object(
            llm_client,
            "complete_json",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await reviewer.review(r"\documentclass{article}...")

        assert result.total_score == 7
        assert result.verdict == "accept"
        assert len(result.issues) == 1

    @pytest.mark.asyncio
    async def test_review_and_revise_accepts(self, llm_client):
        reviewer = AutoReviewer(llm_client, max_revisions=3)

        mock_result = {
            "scores": {
                "clarity": 2, "methodology": 2, "results": 2,
                "novelty": 1, "completeness": 1,
            },
            "total_score": 8,
            "issues": [],
            "suggestions": [],
            "verdict": "accept",
        }
        with patch.object(
            llm_client,
            "complete_json",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            _, reviews = await reviewer.review_and_revise(
                r"\documentclass{article}...", min_score=6
            )

        assert len(reviews) == 1
        assert reviews[0].verdict == "accept"
