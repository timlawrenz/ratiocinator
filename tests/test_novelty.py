"""Tests for the novelty filter."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ratiocinator.config import LLMConfig
from ratiocinator.ideation.novelty import NoveltyFilter
from ratiocinator.ideation.retriever import Paper
from ratiocinator.llm.client import LLMClient


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
def llm_client():
    return LLMClient(LLMConfig())


@pytest.mark.asyncio
async def test_check_returns_assessment(llm_client):
    nf = NoveltyFilter(llm_client, _sample_papers())

    mock_result = {
        "is_novel": True,
        "similarity_score": 0.3,
        "closest_paper_id": "2301.00001",
        "reasoning": "Different approach",
        "refined_hypothesis": None,
    }

    with patch.object(
        llm_client, "complete_json", new_callable=AsyncMock, return_value=mock_result
    ):
        result = await nf.check("Apply sparse attention to DiT blocks")

    assert result["is_novel"] is True
    assert "similar_papers" in result
    assert len(result["similar_papers"]) > 0


@pytest.mark.asyncio
async def test_filter_hypotheses(llm_client):
    nf = NoveltyFilter(llm_client, _sample_papers())

    mock_result = {
        "is_novel": True,
        "similarity_score": 0.2,
        "closest_paper_id": None,
        "reasoning": "Novel",
        "refined_hypothesis": None,
    }

    with patch.object(
        llm_client, "complete_json", new_callable=AsyncMock, return_value=mock_result
    ):
        results = await nf.filter_hypotheses(["hyp1", "hyp2"])

    assert len(results) == 2
    assert results[0]["hypothesis"] == "hyp1"
    assert results[1]["hypothesis"] == "hyp2"
