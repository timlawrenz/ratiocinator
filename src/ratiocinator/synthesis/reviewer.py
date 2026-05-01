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

SECTION_REVIEW_SYSTEM = """\
You are a rigorous scientific paper reviewer. Review ONLY the following section \
of a research paper. Evaluate it on:

1. Clarity (0-2): Is the writing clear and well-structured?
2. Accuracy (0-2): Are claims supported? Is content technically correct?
3. Completeness (0-2): Is anything missing for this section?

Respond with JSON containing:
- "scores": {{"clarity": N, "accuracy": N, "completeness": N}}
- "total_score": sum of all scores (0-6)
- "issues": list of specific issues to fix in THIS section
- "suggestions": list of improvement suggestions
- "verdict": "accept" if total >= 4, else "revise"
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


@dataclass
class SectionReviewResult:
    """Result of reviewing a single section."""

    section_name: str
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

    async def review_section(
        self,
        section_name: str,
        section_content: str,
        *,
        paper_context: str | None = None,
    ) -> SectionReviewResult:
        """Review a single section of a paper.

        Args:
            section_name: Name of the section being reviewed.
            section_content: The content of this specific section.
            paper_context: Optional context from other sections.

        Returns:
            SectionReviewResult with scores and feedback for this section.
        """
        context_block = ""
        if paper_context:
            context_block = f"\n\n## Paper context (other sections summary)\n{paper_context[:2000]}"

        prompt = (
            f"## Section: {section_name}{context_block}\n\n"
            f"## Content to review\n\n{section_content[:4000]}"
        )
        result = await self.llm.complete_json(
            prompt, system=SECTION_REVIEW_SYSTEM, task="generalist"
        )

        return SectionReviewResult(
            section_name=section_name,
            scores=result.get("scores", {}),
            total_score=result.get("total_score", 0),
            issues=result.get("issues", []),
            suggestions=result.get("suggestions", []),
            verdict=result.get("verdict", "revise"),
        )

    async def review_all_sections(
        self,
        sections: dict[str, str],
        *,
        min_score: int = 4,
    ) -> list[SectionReviewResult]:
        """Review each section of a paper independently.

        Args:
            sections: Dict mapping section names to their content.
            min_score: Minimum acceptable score per section (0-6).

        Returns:
            List of SectionReviewResult for each section.
        """
        results = []
        # Build brief context from all sections
        context = "\n".join(
            f"- {name}: {content[:100]}..." for name, content in sections.items()
        )

        for name, content in sections.items():
            result = await self.review_section(
                name, content, paper_context=context
            )
            results.append(result)
            logger.info(
                "Section '%s' review: score=%d/%d verdict=%s",
                name, result.total_score, 6, result.verdict,
            )

        return results

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
