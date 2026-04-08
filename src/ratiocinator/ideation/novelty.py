"""Novelty filter: checks proposed hypotheses against existing literature."""

from __future__ import annotations

import logging
from typing import Any

from ratiocinator.ideation.retriever import Paper, SemanticSearch
from ratiocinator.llm.client import LLMClient

logger = logging.getLogger(__name__)

NOVELTY_SYSTEM = """\
You are a research novelty assessor. Given a proposed hypothesis and a list of \
existing papers, determine if the hypothesis is novel.

Respond with JSON containing:
- "is_novel": true/false
- "similarity_score": 0.0 to 1.0 (how similar to existing work)
- "closest_paper_id": arXiv ID of the most similar paper (or null)
- "reasoning": brief explanation
- "refined_hypothesis": if not novel, suggest a novel variation (or null)
"""


class NoveltyFilter:
    """Checks proposed hypotheses against retrieved literature."""

    def __init__(
        self,
        llm: LLMClient,
        papers: list[Paper],
        similarity_threshold: float = 0.85,
    ) -> None:
        self.llm = llm
        self.papers = papers
        self.similarity_threshold = similarity_threshold
        self.search = SemanticSearch(papers)

    async def check(self, hypothesis: str) -> dict[str, Any]:
        """Check a hypothesis for novelty against the literature.

        Returns a dict with novelty assessment, closest papers, and
        optionally a refined hypothesis.
        """
        # Find closest papers
        similar = self.search.search(hypothesis, top_k=5)

        papers_context = "\n\n".join(
            f"**[{p.arxiv_id}] {p.title}**\n{p.abstract[:500]}"
            for p, _score in similar
        )

        prompt = (
            f"## Proposed hypothesis\n{hypothesis}\n\n"
            f"## Most similar existing papers\n{papers_context}"
        )

        result = await self.llm.complete_json(
            prompt, system=NOVELTY_SYSTEM, task="generalist"
        )

        result["similar_papers"] = [
            {"arxiv_id": p.arxiv_id, "title": p.title, "score": score}
            for p, score in similar
        ]

        return result

    async def filter_hypotheses(
        self, hypotheses: list[str]
    ) -> list[dict[str, Any]]:
        """Check multiple hypotheses and return novelty assessments."""
        results = []
        for h in hypotheses:
            assessment = await self.check(h)
            results.append({"hypothesis": h, **assessment})
        return results
