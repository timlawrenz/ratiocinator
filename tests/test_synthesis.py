"""Tests for the paper generator and reviewer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.config import LLMConfig
from ratiocinator.llm.client import LLMClient, LLMResponse
from ratiocinator.search.tree import ExperimentTree, NodeStatus
from ratiocinator.synthesis.paper import (
    PaperGenerator,
    SectionResult,
    _clean_markdown,
    validate_section,
)
from ratiocinator.synthesis.reviewer import AutoReviewer, SectionReviewResult


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


class TestCleanMarkdown:
    def test_strips_markdown_fences(self):
        text = "```markdown\nHello world.\n```"
        assert _clean_markdown(text) == "Hello world."

    def test_strips_section_headings(self):
        text = "## Introduction\nParagraph text."
        assert _clean_markdown(text) == "Paragraph text."

    def test_strips_numbered_headings(self):
        text = "## 1. Introduction\nParagraph text."
        # Numbered headings without matching pattern are preserved
        assert "Paragraph text." in _clean_markdown(text)

    def test_preserves_normal_markdown(self):
        text = "We use $\\alpha = 0.01$ and **bold** emphasis."
        assert _clean_markdown(text) == text

    def test_collapses_blank_lines(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        assert _clean_markdown(text) == "First paragraph.\n\nSecond paragraph."

    def test_strips_abstract_heading(self):
        text = "## Abstract\nWe present a study."
        assert _clean_markdown(text) == "We present a study."

    def test_strips_latex_documentclass(self):
        text = "\\documentclass{article}\n\\begin{document}\nContent here.\n\\end{document}"
        result = _clean_markdown(text)
        assert "\\documentclass" not in result
        assert "\\begin{document}" not in result
        assert "\\end{document}" not in result
        assert "Content here." in result

    def test_strips_latex_with_options(self):
        text = "\\documentclass[12pt]{article}\n\\usepackage[utf8]{inputenc}\nContent."
        result = _clean_markdown(text)
        assert "\\documentclass" not in result
        assert "\\usepackage" not in result
        assert "Content." in result

    def test_strips_latex_preamble_commands(self):
        text = "\\title{My Paper}\n\\author{Alice}\n\\date{2024}\n\\maketitle\nBody text."
        result = _clean_markdown(text)
        assert "\\title" not in result
        assert "\\author" not in result
        assert "\\maketitle" not in result
        assert "Body text." in result

    def test_strips_latex_code_fence(self):
        text = "```latex\nHello world.\n```"
        assert _clean_markdown(text) == "Hello world."


class TestValidateSection:
    def test_empty_content(self):
        issues = validate_section("")
        assert "empty" in issues[0].lower()

    def test_clean_content_no_issues(self):
        issues = validate_section("This is a well-formed section about experiments.")
        assert issues == []

    def test_detects_documentclass(self):
        issues = validate_section("\\documentclass{article}\nSome content.")
        assert any("\\documentclass" in i for i in issues)

    def test_detects_begin_document(self):
        issues = validate_section("\\begin{document}\nContent.\\end{document}")
        assert any("\\begin{document}" in i for i in issues)

    def test_detects_excessive_length(self):
        long_content = " ".join(["word"] * 6000)
        issues = validate_section(long_content)
        assert any("5000 words" in i for i in issues)


class TestSectionResult:
    def test_default_fields(self):
        sr = SectionResult(section_title="intro", content="Hello.")
        assert sr.tables == []
        assert sr.figures_referenced == []
        assert sr.citations_used == []
        assert sr.validation_issues == []


class TestPaperGenerator:
    @pytest.mark.asyncio
    async def test_generate_produces_markdown(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)

        mock_resp = LLMResponse(
            content="Some section content.",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        with patch.object(
            llm_client, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            paper = await gen.generate(
                tree_with_results, "Test Paper"
            )

        assert "# Test Paper" in paper
        assert "## Abstract" in paper
        assert "## 2. Related Work" in paper
        assert "Some section content." in paper

    @pytest.mark.asyncio
    async def test_generate_writes_file(
        self, llm_client, tree_with_results, tmp_path
    ):
        gen = PaperGenerator(llm_client)
        output = tmp_path / "paper.md"

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
        assert "# Test" in output.read_text()

    @pytest.mark.asyncio
    async def test_generate_with_literature_context(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)

        literature = [
            {
                "arxiv_id": "2301.00001",
                "title": "Efficient Attention",
                "abstract": "We propose efficient attention.",
                "authors": ["Alice", "Bob"],
                "published": "2023-01-01",
            },
        ]

        mock_resp = LLMResponse(
            content="Section with citations.",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        with patch.object(
            llm_client, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            paper = await gen.generate(
                tree_with_results, "Test Paper",
                literature_context=literature,
            )

        assert "## References" in paper
        assert "2301.00001" in paper
        assert "Efficient Attention" in paper

    @pytest.mark.asyncio
    async def test_generate_structured(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)

        mock_json_result = {
            "section_title": "abstract",
            "content": "Structured section content.",
            "tables": [],
            "figures_referenced": [],
            "citations_used": ["kipf2017gcn"],
        }
        with patch.object(
            llm_client, "complete_json", new_callable=AsyncMock, return_value=mock_json_result
        ):
            paper = await gen.generate(
                tree_with_results, "Test Paper", structured=True
            )

        assert "# Test Paper" in paper
        assert "Structured section content." in paper

    @pytest.mark.asyncio
    async def test_generate_section_structured(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)

        mock_json_result = {
            "section_title": "Results",
            "content": "We evaluated five architectures.",
            "tables": ["| Model | Accuracy |"],
            "figures_referenced": ["fig:comparison"],
            "citations_used": ["kipf2017gcn"],
        }
        with patch.object(
            llm_client, "complete_json", new_callable=AsyncMock, return_value=mock_json_result
        ):
            result = await gen.generate_section_structured(
                "results", "experiment context here"
            )

        assert isinstance(result, SectionResult)
        assert result.content == "We evaluated five architectures."
        assert result.figures_referenced == ["fig:comparison"]
        assert result.citations_used == ["kipf2017gcn"]

    @pytest.mark.asyncio
    async def test_generate_section_structured_fallback(self, llm_client, tree_with_results):
        """When JSON parsing fails, falls back to plain generation."""
        gen = PaperGenerator(llm_client)

        mock_resp = LLMResponse(
            content="Fallback content.",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        with patch.object(
            llm_client, "complete_json", new_callable=AsyncMock,
            side_effect=ValueError("Bad JSON"),
        ), patch.object(
            llm_client, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            result = await gen.generate_section_structured(
                "introduction", "context"
            )

        assert result.content == "Fallback content."
        assert result.section_title == "introduction"

    def test_build_figures(self, llm_client):
        gen = PaperGenerator(llm_client)
        paths = [Path("plot1.png"), Path("plot2.png")]
        figs = gen._build_figures(paths)
        assert "![Figure 1" in figs
        assert "plot1.png" in figs
        assert "plot2.png" in figs

    def test_build_figures_empty(self, llm_client):
        gen = PaperGenerator(llm_client)
        assert gen._build_figures(None) == ""

    def test_build_results_table(self, llm_client, tree_with_results):
        gen = PaperGenerator(llm_client)
        nodes = [n for n in tree_with_results.all_nodes()
                 if n.status == NodeStatus.SUCCESS]
        table = gen._build_results_table(nodes)
        assert "| Experiment" in table
        assert "train_loss" in table
        assert "---" in table

    def test_build_results_table_empty(self, llm_client):
        gen = PaperGenerator(llm_client)
        result = gen._build_results_table([])
        assert "No successful" in result

    def test_build_references_with_papers(self, llm_client):
        gen = PaperGenerator(llm_client)
        lit = [
            {
                "arxiv_id": "2301.00001",
                "title": "Paper One",
                "authors": ["Alice", "Bob", "Charlie", "Dave"],
                "published": "2023-01-01",
            },
        ]
        refs = gen._build_references(lit)
        assert "2301.00001" in refs
        assert "Paper One" in refs
        assert "et al." in refs
        assert "arxiv.org" in refs

    def test_build_references_empty(self, llm_client):
        gen = PaperGenerator(llm_client)
        refs = gen._build_references(None)
        assert "No literature references" in refs


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
            result = await reviewer.review("# Test Paper\n\nContent here.")

        assert result.total_score == 7
        assert result.verdict == "accept"
        assert len(result.issues) == 1

    @pytest.mark.asyncio
    async def test_review_section(self, llm_client):
        reviewer = AutoReviewer(llm_client)
        mock_result = {
            "scores": {"clarity": 2, "accuracy": 1, "completeness": 2},
            "total_score": 5,
            "issues": ["Missing confidence intervals"],
            "suggestions": ["Add error bars"],
            "verdict": "accept",
        }
        with patch.object(
            llm_client,
            "complete_json",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await reviewer.review_section(
                "Results", "We tested 5 models. Model A scored 0.95."
            )

        assert isinstance(result, SectionReviewResult)
        assert result.section_name == "Results"
        assert result.total_score == 5
        assert result.verdict == "accept"

    @pytest.mark.asyncio
    async def test_review_section_with_context(self, llm_client):
        reviewer = AutoReviewer(llm_client)
        mock_result = {
            "scores": {"clarity": 2, "accuracy": 2, "completeness": 2},
            "total_score": 6,
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
            result = await reviewer.review_section(
                "Methodology",
                "We use BFTS with 10 expansions.",
                paper_context="Abstract: We explore automated optimization.",
            )

        assert result.total_score == 6

    @pytest.mark.asyncio
    async def test_review_all_sections(self, llm_client):
        reviewer = AutoReviewer(llm_client)
        mock_result = {
            "scores": {"clarity": 2, "accuracy": 1, "completeness": 1},
            "total_score": 4,
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
            results = await reviewer.review_all_sections({
                "Introduction": "Intro content.",
                "Methodology": "Method content.",
                "Results": "Results content.",
            })

        assert len(results) == 3
        assert all(isinstance(r, SectionReviewResult) for r in results)
        assert results[0].section_name == "Introduction"

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
                "# Paper\n\n## 1. Introduction\n\nContent.", min_score=6
            )

        assert len(reviews) == 1
        assert reviews[0].verdict == "accept"

    def test_replace_section(self):
        paper = (
            "# Title\n\n"
            "## 1. Introduction\n\nOld intro.\n\n"
            "## 2. Related Work\n\nOld related.\n\n"
            "## 3. Methodology\n\nOld method.\n"
        )
        result = AutoReviewer._replace_section(paper, "Introduction", "New intro content.")
        assert "New intro content." in result
        assert "Old intro." not in result
        assert "Old related." in result

    def test_replace_section_no_match(self):
        paper = "# Title\n\n## 1. Introduction\n\nContent.\n"
        result = AutoReviewer._replace_section(paper, "Nonexistent", "New content.")
        assert result == paper
