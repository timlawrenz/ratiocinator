"""Integration: literature-grounded ideation feeding into tree search expansion."""

from __future__ import annotations

import logging
from typing import Any

from ratiocinator.config import Config
from ratiocinator.ideation.novelty import NoveltyFilter
from ratiocinator.ideation.retriever import ArxivRetriever, SemanticSearch
from ratiocinator.llm.client import LLMClient

logger = logging.getLogger(__name__)

IDEATION_SYSTEM = """\
You are an AI research assistant. Given a research topic, recent relevant papers, \
and current experimental results, propose a novel code modification hypothesis.

Ground your proposal in the literature — cite which paper inspired it and explain \
how your variation differs from existing work.

Respond with JSON containing:
- "hypothesis": a clear statement of the proposed change
- "reasoning": why this should work, grounded in the cited literature
- "inspired_by": arXiv ID of the paper that inspired this
- "filename": the file to modify
- "original": the exact code snippet to replace
- "replacement": the new code snippet
"""


class LiteratureGroundedIdeation:
    """Generates hypotheses grounded in recent literature for tree search expansion."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        cache_dir = config.work_dir / "arxiv_cache"
        self.retriever = ArxivRetriever(cache_dir)
        self._papers: list = []
        self._search: SemanticSearch | None = None

    def load_literature(self, topic: str, max_papers: int = 20) -> int:
        """Retrieve papers on a topic and prepare for search.

        Returns the number of papers retrieved.
        """
        self._papers = self.retriever.search(topic, max_results=max_papers)
        self._search = SemanticSearch(self._papers)
        return len(self._papers)

    def load_cached(self) -> int:
        """Load previously cached papers."""
        self._papers = self.retriever.get_cached()
        if self._papers:
            self._search = SemanticSearch(self._papers)
        return len(self._papers)

    async def generate_hypotheses(
        self,
        topic: str,
        current_code: str,
        current_metrics: dict[str, Any],
        n: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate literature-grounded hypotheses for code modifications.

        Args:
            topic: Research topic for context.
            current_code: The current source code.
            current_metrics: Current experimental metrics.
            n: Number of hypotheses to generate.

        Returns:
            List of hypothesis dicts, each with literature grounding.
        """
        if not self._search or not self._papers:
            logger.warning("No papers loaded, generating without literature grounding")
            return []

        relevant = self._search.search(topic, top_k=5)
        papers_context = "\n\n".join(
            f"**[{p.arxiv_id}] {p.title}** (score: {score:.2f})\n{p.abstract[:400]}"
            for p, score in relevant
        )

        hypotheses = []
        for i in range(n):
            prompt = (
                f"## Research topic\n{topic}\n\n"
                f"## Relevant papers\n{papers_context}\n\n"
                f"## Current code\n```python\n{current_code[:3000]}\n```\n\n"
                f"## Current metrics\n{current_metrics}\n\n"
                f"## Request\nGenerate hypothesis {i + 1} of {n}. "
                f"Make it distinct from previous proposals."
            )

            try:
                result = await self.llm.complete_json(
                    prompt, system=IDEATION_SYSTEM, task="coding"
                )
                result["cited_papers"] = [
                    {"arxiv_id": p.arxiv_id, "title": p.title, "relevance": score}
                    for p, score in relevant
                ]
                hypotheses.append(result)
            except Exception:
                logger.exception("Failed to generate hypothesis %d", i)

        return hypotheses

    async def generate_and_filter(
        self,
        topic: str,
        current_code: str,
        current_metrics: dict[str, Any],
        n: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate hypotheses and filter for novelty."""
        hypotheses = await self.generate_hypotheses(
            topic, current_code, current_metrics, n=n
        )

        if not hypotheses:
            return []

        nf = NoveltyFilter(self.llm, self._papers)
        filtered = []
        for h in hypotheses:
            assessment = await nf.check(h.get("hypothesis", ""))
            h["novelty"] = assessment
            if assessment.get("is_novel", True):
                filtered.append(h)
            else:
                refined = assessment.get("refined_hypothesis")
                if refined:
                    h["hypothesis"] = refined
                    h["was_refined"] = True
                    filtered.append(h)

        return filtered
