"""Markdown paper generator: produces structured research reports from experiment results.

Generates professional academic-style Markdown papers with:
- Section-by-section generation with structured JSON output
- Literature-grounded Related Work section (when ideation context is available)
- Structured results tables and hypothesis genealogy
- Embedded plot references
- Proper citation formatting
- Per-section validation (LaTeX stripping, reference checks)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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

STRUCTURED_SECTION_SYSTEM = """\
You are a scientific paper writer. Write the {section} section of a research \
paper about automated code optimization experiments. Be concise, technical, \
and precise.

CRITICAL FORMATTING RULES:
- Output ONLY the body text for this section — no document preamble or postamble.
- Do NOT wrap in LaTeX document structure (no \\documentclass, \\begin{{document}}, etc.).
- Do NOT include the section heading — the template adds it.
- Write in third person, academic tone.
- Be substantive — avoid filler phrases and empty generalities.

Respond with valid JSON only. No markdown fences.
"""


@dataclass
class SectionResult:
    """Structured result from generating a single paper section."""

    section_title: str
    content: str
    tables: list[str] = field(default_factory=list)
    figures_referenced: list[str] = field(default_factory=list)
    citations_used: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)

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
    """Generates a Markdown paper from experiment tree results.

    Supports two generation modes:
    - Standard mode: generates each section via plain LLM calls (default).
    - Structured mode: generates each section via JSON-structured output,
      returning SectionResult objects with metadata for validation.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(
        self,
        tree: ExperimentTree,
        title: str,
        plot_paths: list[Path] | None = None,
        output_path: Path | None = None,
        literature_context: list[dict[str, Any]] | None = None,
        *,
        structured: bool = False,
    ) -> str:
        """Generate a complete Markdown paper.

        Args:
            tree: The experiment tree with results.
            title: Paper title.
            plot_paths: Paths to generated plot images.
            output_path: If provided, write .md file here.
            literature_context: List of cited paper dicts from ideation
                (each with arxiv_id, title, abstract, relevance).
            structured: If True, use structured JSON output per section.

        Returns:
            The Markdown source as a string.
        """
        summary = tree.summary()
        nodes = tree.all_nodes()
        successful = [n for n in nodes if n.status == NodeStatus.SUCCESS]

        best = tree.get_best_leaf()
        context = self._build_context(summary, successful, best, literature_context)

        if structured:
            sections = await self._generate_all_structured(
                context, literature_context, successful
            )
            abstract = sections["abstract"].content
            introduction = sections["introduction"].content
            related_work = sections["related work"].content
            methodology = sections["methodology"].content
            results = sections["results analysis"].content
            discussion = sections["discussion"].content
            conclusion = sections["conclusion"].content
        else:
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

    async def generate_section_structured(
        self,
        section: str,
        context: str,
        *,
        results_data: list[TreeNode] | None = None,
    ) -> SectionResult:
        """Generate a single section with structured JSON output.

        Returns a SectionResult with the section content and metadata
        (tables, figures_referenced, citations_used) for post-processing.
        """
        system = STRUCTURED_SECTION_SYSTEM.format(section=section)

        extra_context = ""
        if results_data:
            extra_context = "\n\n## Experiment results:\n"
            for n in results_data[:10]:
                extra_context += (
                    f"- {n.hypothesis[:80]} | score={n.score} | metrics={n.metrics}\n"
                )

        prompt = (
            f"## Context\n{context}{extra_context}\n\n"
            f"## Write the {section} section.\n"
            f"Return JSON with keys: section_title, content, tables, "
            f"figures_referenced, citations_used"
        )

        try:
            result = await self.llm.complete_json(prompt, system=system, task="generalist")
            content = _clean_markdown(str(result.get("content", "")))
            section_result = SectionResult(
                section_title=result.get("section_title", section),
                content=content,
                tables=result.get("tables", []),
                figures_referenced=result.get("figures_referenced", []),
                citations_used=result.get("citations_used", []),
            )
        except (ValueError, KeyError):
            logger.warning(
                "Structured generation failed for '%s', falling back to plain", section
            )
            content = await self._generate_section(section, context)
            section_result = SectionResult(section_title=section, content=content)

        # Validate the section
        section_result.validation_issues = validate_section(section_result.content)
        return section_result

    async def _generate_all_structured(
        self,
        context: str,
        literature_context: list[dict[str, Any]] | None,
        successful: list[TreeNode],
    ) -> dict[str, SectionResult]:
        """Generate all sections using structured JSON output."""
        section_names = [
            "abstract", "introduction", "related work",
            "methodology", "results analysis", "discussion", "conclusion",
        ]
        results: dict[str, SectionResult] = {}

        for section in section_names:
            results_data = successful if section == "results analysis" else None

            if section == "related work" and literature_context:
                # Use literature-aware generation for related work
                sr = await self._generate_related_work_structured(
                    context, literature_context
                )
            else:
                sr = await self.generate_section_structured(
                    section, context, results_data=results_data
                )
            results[section] = sr
            logger.info(
                "Generated section '%s' (%d chars, %d issues)",
                section, len(sr.content), len(sr.validation_issues),
            )

        return results

    async def _generate_related_work_structured(
        self,
        context: str,
        literature_context: list[dict[str, Any]],
    ) -> SectionResult:
        """Generate related work section with literature context via JSON output."""
        papers_text = "\n".join(
            f"- **[{p.get('arxiv_id', '?')}]** {p.get('title', 'Untitled')}: "
            f"{p.get('abstract', '')[:200]}..."
            for p in literature_context[:10]
        )

        system = (
            "You are a scientific paper writer. Write the Related Work section. "
            "Cite the provided papers using their arXiv IDs in brackets, e.g. [2301.00001]. "
            "Group related papers thematically. Explain how each relates to the current work. "
            "Output ONLY the body text — no section heading, no LaTeX document structure.\n"
            "Respond with valid JSON only. No markdown fences."
        )
        prompt = (
            f"## Experiment context\n{context}\n\n"
            f"## Papers to cite\n{papers_text}\n\n"
            f"## Write the related work section, citing these papers.\n"
            f"Return JSON with keys: section_title, content, tables, "
            f"figures_referenced, citations_used"
        )

        try:
            result = await self.llm.complete_json(prompt, system=system, task="generalist")
            content = _clean_markdown(str(result.get("content", "")))
            sr = SectionResult(
                section_title=result.get("section_title", "related work"),
                content=content,
                tables=result.get("tables", []),
                figures_referenced=result.get("figures_referenced", []),
                citations_used=result.get("citations_used", []),
            )
        except (ValueError, KeyError):
            logger.warning("Structured related work generation failed, using plain")
            content = await self._generate_related_work(context, literature_context)
            sr = SectionResult(section_title="related work", content=content)

        sr.validation_issues = validate_section(sr.content)
        return sr

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
    # Strip LaTeX document preamble/postamble (local LLMs often wrap in full docs)
    text = re.sub(r"\\documentclass(\[.*?\])?\{.*?\}", "", text)
    text = re.sub(r"\\usepackage(\[.*?\])?\{.*?\}", "", text)
    text = re.sub(r"\\(begin|end)\{document\}", "", text)
    text = re.sub(r"\\title\{.*?\}", "", text)
    text = re.sub(r"\\author\{.*?\}", "", text)
    text = re.sub(r"\\date\{.*?\}", "", text)
    text = re.sub(r"\\maketitle", "", text)
    # Strip markdown code fences that LLMs sometimes wrap content in
    text = re.sub(r"^```(?:markdown|latex)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    # Remove section headings (template provides these)
    text = re.sub(r"^#{1,3}\s+(?:Abstract|Introduction|Related Work|Methodology|"
                  r"Results|Discussion|Conclusion|References)\s*$",
                  "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_section(content: str) -> list[str]:
    """Validate a generated section and return a list of issues found.

    Checks for:
    - LaTeX document preamble/postamble remnants
    - Excessive length (potential context window overflow)
    - Empty content
    """
    issues: list[str] = []

    if not content or not content.strip():
        issues.append("Section content is empty")
        return issues

    # Check for LaTeX document structure remnants
    latex_patterns = [
        (r"\\documentclass", "Contains \\documentclass"),
        (r"\\begin\{document\}", "Contains \\begin{document}"),
        (r"\\end\{document\}", "Contains \\end{document}"),
        (r"\\usepackage", "Contains \\usepackage"),
    ]
    for pattern, message in latex_patterns:
        if re.search(pattern, content):
            issues.append(message)

    # Check for unreasonably long sections (>5000 words suggests overflow)
    word_count = len(content.split())
    if word_count > 5000:
        issues.append(f"Section exceeds 5000 words ({word_count} words)")

    return issues
