"""LaTeX paper generator: produces structured reports from experiment results."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ratiocinator.llm.client import LLMClient
from ratiocinator.search.tree import ExperimentTree, NodeStatus

logger = logging.getLogger(__name__)

SECTION_SYSTEM = """\
You are a scientific paper writer. Write the {section} section of a research \
paper about automated code optimization experiments. Be concise, technical, \
and precise.

CRITICAL FORMATTING RULES:
- Output ONLY the body text for this section — raw LaTeX paragraphs.
- Do NOT include \\documentclass, \\usepackage, \\begin{{document}}, \
\\end{{document}}, \\title, \\author, \\date, \\maketitle, or \\section commands.
- Do NOT wrap the output in \\begin{{abstract}}...\\end{{abstract}}.
- Just write the paragraph content. The template already provides the structure.
"""

LATEX_TEMPLATE = r"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage[margin=1in]{{geometry}}

\title{{{title}}}
\author{{Ratiocinator Auto-Researcher}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{Introduction}}
{introduction}

\section{{Methodology}}
{methodology}

\section{{Results}}
{results}

{figures}

\section{{Conclusion}}
{conclusion}

\end{{document}}
"""


class PaperGenerator:
    """Generates a LaTeX paper from experiment tree results."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(
        self,
        tree: ExperimentTree,
        title: str,
        plot_paths: list[Path] | None = None,
        output_path: Path | None = None,
    ) -> str:
        """Generate a complete LaTeX paper.

        Args:
            tree: The experiment tree with results.
            title: Paper title.
            plot_paths: Paths to generated plot images.
            output_path: If provided, write .tex file here.

        Returns:
            The LaTeX source as a string.
        """
        summary = tree.summary()
        nodes = tree.all_nodes()
        successful = [n for n in nodes if n.status == NodeStatus.SUCCESS]

        best = tree.get_best_leaf()
        context = self._build_context(summary, successful, best)

        abstract = await self._generate_section("abstract", context)
        introduction = await self._generate_section("introduction", context)
        methodology = await self._generate_section("methodology", context)
        results = await self._generate_section("results", context)
        conclusion = await self._generate_section("conclusion", context)

        figures = self._build_figures(plot_paths)

        latex = LATEX_TEMPLATE.format(
            title=_latex_escape(title),
            abstract=abstract,
            introduction=introduction,
            methodology=methodology,
            results=results,
            figures=figures,
            conclusion=conclusion,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(latex)
            logger.info("Paper written to %s", output_path)

        return latex

    async def _generate_section(self, section: str, context: str) -> str:
        system = SECTION_SYSTEM.format(section=section)
        resp = await self.llm.complete(
            f"## Context\n{context}\n\n## Write the {section}",
            system=system,
            task="generalist",
        )
        return _clean_section(resp.content)

    def _build_context(
        self,
        summary: dict[str, Any],
        successful: list,
        best: Any,
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

        return "\n".join(lines)

    def _build_figures(self, plot_paths: list[Path] | None) -> str:
        if not plot_paths:
            return ""
        parts = []
        for i, path in enumerate(plot_paths):
            parts.append(
                f"\\begin{{figure}}[h]\n"
                f"\\centering\n"
                f"\\includegraphics[width=0.8\\textwidth]{{{path.name}}}\n"
                f"\\caption{{Experiment results (Figure {i + 1})}}\n"
                f"\\end{{figure}}"
            )
        return "\n\n".join(parts)


def _clean_section(text: str) -> str:
    """Strip LaTeX preamble/document wrappers that LLMs incorrectly include."""
    # Remove \documentclass[...]{...} or \documentclass{...}
    text = re.sub(r"\\documentclass(\[[^\]]*\])?\{[^}]*\}", "", text)
    # Remove \usepackage[...]{...} or \usepackage{...}
    text = re.sub(r"\\usepackage(\[[^\]]*\])?\{[^}]*\}", "", text)
    # Remove \title{...}, \author{...}, \date{...}
    text = re.sub(r"\\(title|author|date)\{[^}]*\}", "", text)
    # Remove \maketitle
    text = re.sub(r"\\maketitle", "", text)
    # Remove \begin{document}, \end{document}
    text = re.sub(r"\\(begin|end)\{document\}", "", text)
    # Remove \begin{abstract}, \end{abstract} (template provides these)
    text = re.sub(r"\\(begin|end)\{abstract\}", "", text)
    # Remove \section{...} headers (template provides these)
    text = re.sub(r"\\section\*?\{[^}]*\}", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
