"""Automated paper reviewer: LLM-as-a-judge with revision cycles."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ratiocinator.llm.client import LLMClient

logger = logging.getLogger(__name__)

REVIEW_SYSTEM = """\
You are a rigorous scientific paper reviewer. Evaluate the paper on this rubric:

1. Clarity (0-2): Is the writing clear and well-structured?
2. Methodology (0-2): Is the experimental approach sound and well-described?
3. Results (0-2): Are claims supported by data? Are comparisons fair?
4. Novelty (0-2): Does the work present new insights or approaches?
5. Completeness (0-2): Are there missing ablations, plots, or analyses?

Respond with JSON containing:
- "scores": {"clarity": N, "methodology": N, "results": N, "novelty": N, "completeness": N}
- "total_score": sum of all scores (0-10)
- "issues": list of specific issues to fix
- "suggestions": list of improvement suggestions
- "verdict": "accept" if total >= 6, else "revise"
"""

REVISION_SYSTEM = """\
You are a scientific paper writer revising a Markdown draft based on reviewer \
feedback. Make targeted improvements to address the specific issues raised. \
Return the revised content for the section that needs the most improvement.

Respond with JSON containing:
- "section": which section was revised (e.g., "Results", "Methodology")
- "revised_content": the new Markdown content for that section
- "changes_made": brief description of what was changed
"""


@dataclass
class ReviewResult:
    """Result of an automated review."""

    scores: dict[str, int]
    total_score: int
    issues: list[str]
    suggestions: list[str]
    verdict: str


class AutoReviewer:
    """Automated paper reviewer with revision cycles."""

    def __init__(self, llm: LLMClient, max_revisions: int = 3) -> None:
        self.llm = llm
        self.max_revisions = max_revisions

    async def review(self, paper: str) -> ReviewResult:
        """Review a Markdown paper and return structured feedback."""
        prompt = f"## Paper to review\n\n{paper[:8000]}"
        result = await self.llm.complete_json(
            prompt, system=REVIEW_SYSTEM, task="generalist"
        )

        return ReviewResult(
            scores=result.get("scores", {}),
            total_score=result.get("total_score", 0),
            issues=result.get("issues", []),
            suggestions=result.get("suggestions", []),
            verdict=result.get("verdict", "revise"),
        )

    async def review_and_revise(
        self,
        paper: str,
        min_score: int = 6,
    ) -> tuple[str, list[ReviewResult]]:
        """Review the paper and iteratively revise until it passes.

        Returns (final_paper, list_of_review_results).
        """
        reviews = []
        current = paper

        for cycle in range(self.max_revisions + 1):
            review = await self.review(current)
            reviews.append(review)
            logger.info(
                "Review cycle %d: score=%d verdict=%s",
                cycle,
                review.total_score,
                review.verdict,
            )

            if review.total_score >= min_score:
                logger.info("Paper accepted with score %d", review.total_score)
                break

            if cycle >= self.max_revisions:
                logger.warning(
                    "Max revisions reached, final score: %d", review.total_score
                )
                break

            # Attempt revision
            current = await self._revise(current, review)

        return current, reviews

    async def _revise(self, paper: str, review: ReviewResult) -> str:
        """Apply one revision based on review feedback."""
        issues_text = "\n".join(f"- {i}" for i in review.issues)
        suggestions_text = "\n".join(f"- {s}" for s in review.suggestions)

        prompt = (
            f"## Current paper\n\n{paper[:6000]}\n\n"
            f"## Review scores: {review.scores}\n\n"
            f"## Issues to fix\n{issues_text}\n\n"
            f"## Suggestions\n{suggestions_text}"
        )

        try:
            result = await self.llm.complete_json(
                prompt, system=REVISION_SYSTEM, task="generalist"
            )
            section = result.get("section", "")
            revised = result.get("revised_content", "")
            changes = result.get("changes_made", "")

            if section and revised:
                paper = self._replace_section(paper, section, revised)
                logger.info("Revised section '%s': %s", section, changes)

        except Exception:
            logger.exception("Revision failed")

        return paper

    @staticmethod
    def _replace_section(paper: str, section_name: str, new_content: str) -> str:
        """Replace a Markdown section's content by heading match."""
        import re

        # Match ## N. Section Name or ## Section Name
        pattern = re.compile(
            rf"(##\s+(?:\d+\.\s+)?{re.escape(section_name)}\s*\n)"
            r"(.*?)"
            r"(?=\n##\s|\n---\s*$|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(paper)
        if match:
            paper = paper[:match.start(2)] + "\n" + new_content + "\n\n" + paper[match.end(2):]
        return paper
