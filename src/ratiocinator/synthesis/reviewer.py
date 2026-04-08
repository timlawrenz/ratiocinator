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
You are a scientific paper writer revising a draft based on reviewer feedback. \
Make targeted improvements to address the specific issues raised. Return the \
revised LaTeX content for the section that needs the most improvement.

Respond with JSON containing:
- "section": which section was revised (e.g., "results", "methodology")
- "revised_content": the new LaTeX content for that section
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

    async def review(self, latex: str) -> ReviewResult:
        """Review a LaTeX paper and return structured feedback."""
        prompt = f"## Paper to review\n```latex\n{latex[:8000]}\n```"
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
        latex: str,
        min_score: int = 6,
    ) -> tuple[str, list[ReviewResult]]:
        """Review the paper and iteratively revise until it passes.

        Returns (final_latex, list_of_review_results).
        """
        reviews = []
        current = latex

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

    async def _revise(self, latex: str, review: ReviewResult) -> str:
        """Apply one revision based on review feedback."""
        issues_text = "\n".join(f"- {i}" for i in review.issues)
        suggestions_text = "\n".join(f"- {s}" for s in review.suggestions)

        prompt = (
            f"## Current paper\n```latex\n{latex[:6000]}\n```\n\n"
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
                marker = f"\\section{{{section.title()}}}"
                if marker in latex:
                    # Find the section and replace its content
                    start = latex.index(marker) + len(marker)
                    next_section = latex.find("\\section{", start)
                    if next_section == -1:
                        next_section = latex.find("\\end{document}", start)
                    if next_section > start:
                        latex = latex[:start] + "\n" + revised + "\n\n" + latex[next_section:]
                        logger.info("Revised section '%s': %s", section, changes)

        except Exception:
            logger.exception("Revision failed")

        return latex
