"""Markdown paper generator: produces structured research reports from experiment results.

Generates professional academic-style Markdown papers with:
- Literature-grounded Related Work section (when ideation context is available)
- Structured results tables and hypothesis genealogy
- Embedded plot references
- Proper citation formatting
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ratiocinator.llm.client import LLMClient
from ratiocinator.search.tree import ExperimentTree, NodeStatus, TreeNode

logger = logging.getLogger(__name__)

SECTION_SYSTEM = """\
You are a scientific paper writer. Write the {section} section of a research \
paper about automated code optimization experiments. Be concise, technical, \
and precise.

CRITICAL FORMATTING RULES:
- Output ONLY the body text for this section in Markdown.
- Do NOT include the section heading (e.g., no "## Introduction") — the template adds it.
- Use inline math with $...$ and display math with $$...$$ when needed.
- Use Markdown emphasis (*italic*, **bold**) appropriately.
- Write in third person, academic tone.
- Be substantive — avoid filler phrases and empty generalities.
"""

PAPER_TEMPLATE = """# {title}

**Ratiocinator Auto-Researcher** | {date}

---

## Abstract

{abstract}

---

## 1. Introduction

{introduction}

## 2. Related Work

{related_work}

## 3. Methodology

{methodology}

## 4. Results

{results_table}

{results}

{figures}

## 5. Discussion

{discussion}

## 6. Conclusion

{conclusion}

---

{references}
"""


class PaperGenerator:
    """Generates a Markdown paper from experiment tree results."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(
        self,
        tree: ExperimentTree,
        title: str,
        plot_paths: list[Path] | None = None,
        output_path: Path | None = None,
        literature_context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a complete Markdown paper.

        Args:
            tree: The experiment tree with results.
            title: Paper title.
            plot_paths: Paths to generated plot images.
            output_path: If provided, write .md file here.
            literature_context: List of cited paper dicts from ideation
                (each with arxiv_id, title, abstract, relevance).

        Returns:
            The Markdown source as a string.
        """
        summary = tree.summary()
        nodes = tree.all_nodes()
        successful = [n for n in nodes if n.status == NodeStatus.SUCCESS]

        best = tree.get_best_leaf()
        context = self._build_context(summary, successful, best, literature_context)

        abstract = await self._generate_section("abstract", context)
        introduction = await self._generate_section("introduction", context)
        related_work = await self._generate_related_work(context, literature_context)
        methodology = await self._generate_section("methodology", context)
        results = await self._generate_section("results analysis", context)
        discussion = await self._generate_section("discussion", context)
        conclusion = await self._generate_section("conclusion", context)

        results_table = self._build_results_table(successful)
        figures = self._build_figures(plot_paths)
        references = self._build_references(literature_context)

        from datetime import UTC, datetime

        paper = PAPER_TEMPLATE.format(
            title=title,
            date=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            abstract=abstract,
            introduction=introduction,
            related_work=related_work,
            methodology=methodology,
            results_table=results_table,
            results=results,
            figures=figures,
            discussion=discussion,
            conclusion=conclusion,
            references=references,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(paper)
            logger.info("Paper written to %s", output_path)

        return paper

    async def _generate_section(self, section: str, context: str) -> str:
        system = SECTION_SYSTEM.format(section=section)
        resp = await self.llm.complete(
            f"## Context\n{context}\n\n## Write the {section}",
            system=system,
            task="generalist",
        )
        return _clean_markdown(resp.content)

    async def _generate_related_work(
        self,
        context: str,
        literature_context: list[dict[str, Any]] | None,
    ) -> str:
        if not literature_context:
            return await self._generate_section("related work", context)

        papers_text = "\n".join(
            f"- **[{p.get('arxiv_id', '?')}]** {p.get('title', 'Untitled')}: "
            f"{p.get('abstract', '')[:200]}..."
            for p in literature_context[:10]
        )

        system = (
            "You are a scientific paper writer. Write the Related Work section. "
            "Cite the provided papers using their arXiv IDs in brackets, e.g. [2301.00001]. "
            "Group related papers thematically. Explain how each relates to the current work. "
            "Output ONLY the body text in Markdown — no section heading."
        )
        resp = await self.llm.complete(
            f"## Experiment context\n{context}\n\n"
            f"## Papers to cite\n{papers_text}\n\n"
            f"## Write the related work section, citing these papers.",
            system=system,
            task="generalist",
        )
        return _clean_markdown(resp.content)

    def _build_context(
        self,
        summary: dict[str, Any],
        successful: list[TreeNode],
        best: TreeNode | None,
        literature_context: list[dict[str, Any]] | None = None,
    ) -> str:
        lines = [
            f"Total experiments: {summary['total_nodes']}",
            f"Max depth explored: {summary['max_depth']}",
            f"Status breakdown: {summary['by_status']}",
        ]
        if best:
            lines.append(f"Best score: {best.score}")
            lines.append(f"Best hypothesis: {best.hypothesis}")
            lines.append(f"Best metrics: {best.metrics}")

        lines.append("\n## Successful experiments:")
        for n in successful[:10]:
            lines.append(
                f"- [depth={n.depth}] {n.hypothesis[:80]} "
                f"| score={n.score} | metrics={n.metrics}"
            )

        if literature_context:
            lines.append("\n## Literature grounding:")
            lines.append(f"- {len(literature_context)} papers cited")
            for p in literature_context[:5]:
                lines.append(f"- [{p.get('arxiv_id', '?')}] {p.get('title', 'Untitled')}")

        return "\n".join(lines)

    def _build_results_table(self, successful: list[TreeNode]) -> str:
        """Build a Markdown table of experiment results."""
        if not successful:
            return "*No successful experiments.*"

        # Collect all metric keys across experiments
        all_keys: set[str] = set()
        for n in successful:
            if n.metrics:
                all_keys.update(n.metrics.keys())
        metric_keys = sorted(all_keys)

        if not metric_keys:
            return "*No metrics recorded.*"

        # Header
        header = "| Experiment | Depth | " + " | ".join(metric_keys) + " | Hypothesis |"
        separator = "|" + "|".join(["---"] * (len(metric_keys) + 3)) + "|"

        rows = []
        for n in successful[:15]:
            metrics_vals = [
                f"{n.metrics.get(k, '—'):.4f}" if isinstance(n.metrics.get(k), float)
                else str(n.metrics.get(k, "—"))
                for k in metric_keys
            ]
            hypothesis = n.hypothesis[:60].replace("|", "\\|")
            row = f"| {n.id[:8]} | {n.depth} | " + " | ".join(metrics_vals) + f" | {hypothesis} |"
            rows.append(row)

        return "\n".join([header, separator, *rows])

    def _build_figures(self, plot_paths: list[Path] | None) -> str:
        if not plot_paths:
            return ""
        parts = []
        for i, path in enumerate(plot_paths):
            parts.append(f"![Figure {i + 1}: Experiment results](plots/{path.name})")
        return "\n\n".join(parts)

    def _build_references(self, literature_context: list[dict[str, Any]] | None) -> str:
        """Build a References section from cited papers."""
        if not literature_context:
            return "## References\n\n*No literature references available.*"

        lines = ["## References", ""]
        for i, p in enumerate(literature_context, 1):
            arxiv_id = p.get("arxiv_id", "unknown")
            title = p.get("title", "Untitled")
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            published = p.get("published", "")[:4]  # year only
            lines.append(
                f"{i}. **[{arxiv_id}]** {author_str} ({published}). "
                f"*{title}*. arXiv:{arxiv_id}. "
                f"https://arxiv.org/abs/{arxiv_id}"
            )

        return "\n".join(lines)


def _clean_markdown(text: str) -> str:
    """Clean LLM-generated Markdown section content."""
    # Strip markdown code fences that LLMs sometimes wrap content in
    text = re.sub(r"^```(?:markdown)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    # Remove section headings (template provides these)
    text = re.sub(r"^#{1,3}\s+(?:Abstract|Introduction|Related Work|Methodology|"
                  r"Results|Discussion|Conclusion|References)\s*$",
                  "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
