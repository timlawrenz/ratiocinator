"""Tests for literature-grounded ideation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.config import Config
from ratiocinator.ideation.grounded import LiteratureGroundedIdeation
from ratiocinator.ideation.retriever import Paper


def _sample_papers() -> list[Paper]:
    return [
        Paper(
            arxiv_id="2301.00001",
            title="Efficient Attention via Low-Rank Decomposition",
            abstract="We propose a low-rank attention mechanism.",
            authors=["Alice"],
            published="2023-01-01T00:00:00",
        ),
        Paper(
            arxiv_id="2301.00002",
            title="GaLore: Memory-Efficient Training",
            abstract="Gradient low-rank projection for memory efficiency.",
            authors=["Bob"],
            published="2023-06-01T00:00:00",
        ),
    ]


@pytest.fixture
def ideation(tmp_path):
    config = Config(work_dir=tmp_path / ".ratiocinator")
    return LiteratureGroundedIdeation(config)


def test_load_cached_empty(ideation):
    assert ideation.load_cached() == 0


@pytest.mark.asyncio
async def test_generate_hypotheses_no_papers(ideation):
    results = await ideation.generate_hypotheses(
        "attention", "code", {"loss": 0.5}, n=2
    )
    assert results == []


@pytest.mark.asyncio
async def test_generate_hypotheses_with_papers(ideation):
    ideation._papers = _sample_papers()
    from ratiocinator.ideation.retriever import SemanticSearch

    ideation._search = SemanticSearch(ideation._papers)

    mock_result = {
        "hypothesis": "Apply low-rank projection to attention heads",
        "reasoning": "Based on paper 2301.00001",
        "inspired_by": "2301.00001",
        "filename": "model.py",
        "original": "attn = ...",
        "replacement": "attn = low_rank(...)",
    }

    with patch.object(
        ideation.llm, "complete_json", new_callable=AsyncMock, return_value=mock_result
    ):
        results = await ideation.generate_hypotheses(
            "efficient attention", "code", {"loss": 0.5}, n=2
        )

    assert len(results) == 2
    assert results[0]["hypothesis"] == "Apply low-rank projection to attention heads"
    assert "cited_papers" in results[0]
    assert len(results[0]["cited_papers"]) > 0


@pytest.mark.asyncio
async def test_generate_and_filter(ideation):
    ideation._papers = _sample_papers()
    from ratiocinator.ideation.retriever import SemanticSearch

    ideation._search = SemanticSearch(ideation._papers)

    hypothesis_result = {
        "hypothesis": "Novel approach X",
        "reasoning": "Based on paper",
        "inspired_by": "2301.00001",
        "filename": "model.py",
        "original": "x",
        "replacement": "y",
    }
    novelty_result = {
        "is_novel": True,
        "similarity_score": 0.2,
        "closest_paper_id": None,
        "reasoning": "Novel",
        "refined_hypothesis": None,
    }

    with patch.object(
        ideation.llm,
        "complete_json",
        new_callable=AsyncMock,
        side_effect=[hypothesis_result, novelty_result],
    ):
        results = await ideation.generate_and_filter(
            "attention", "code", {"loss": 0.5}, n=1
        )

    assert len(results) == 1
    assert results[0]["novelty"]["is_novel"] is True
