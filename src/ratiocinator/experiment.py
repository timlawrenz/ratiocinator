"""Single experiment loop: LLM proposes code changes, sandbox executes them."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ratiocinator.config import Config
from ratiocinator.llm.client import LLMClient
from ratiocinator.sandbox.runner import RunResult, SandboxRunner

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI research assistant that modifies Python training code to improve \
experimental metrics. Given the current source code and a task description, \
propose a concrete code modification.

Respond with JSON containing:
- "reasoning": brief explanation of why this change should help
- "filename": the file to modify
- "original": the exact code snippet to replace
- "replacement": the new code snippet
- "expected_effect": what metric improvement you expect
"""


@dataclass
class ExperimentResult:
    """Full result of one experiment iteration."""

    hypothesis: str
    diff: dict[str, str]
    run_result: RunResult
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "diff": self.diff,
            "exit_code": self.run_result.exit_code,
            "success": self.run_result.success,
            "stdout": self.run_result.stdout[-2000:],
            "stderr": self.run_result.stderr[-2000:],
            "metrics": self.metrics,
            "duration_seconds": self.run_result.duration_seconds,
        }


class ExperimentLoop:
    """Orchestrates a single experiment: propose → apply → run → report."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.llm = LLMClient(self.config.llm)
        self.sandbox = SandboxRunner(self.config.sandbox)

    async def run_experiment(
        self,
        repo_path: Path,
        task: str,
        *,
        image: str = "python:3.11-slim",
        train_command: str = "python train.py",
        steps: int = 500,
    ) -> ExperimentResult:
        """Execute one full experiment iteration.

        1. Read current source code
        2. Ask LLM for a code modification
        3. Apply the modification to a working copy
        4. Run training in sandbox
        5. Return results
        """
        source_files = self._read_source(repo_path)
        source_text = "\n\n".join(
            f"# --- {name} ---\n{content}" for name, content in source_files.items()
        )

        prompt = (
            f"## Task\n{task}\n\n"
            f"## Training steps\n{steps}\n\n"
            f"## Current source code\n```python\n{source_text}\n```"
        )

        proposal = await self.llm.complete_json(prompt, system=SYSTEM_PROMPT, task="coding")
        logger.info(
            "LLM proposed change to %s: %s", proposal.get("filename"), proposal.get("reasoning")
        )

        work_dir = Path(tempfile.mkdtemp(prefix="ratiocinator-"))
        try:
            shutil.copytree(repo_path, work_dir / "workspace", dirs_exist_ok=True)
            workspace = work_dir / "workspace"

            diff = self._apply_diff(workspace, proposal)

            env = {"TRAIN_STEPS": str(steps)}
            result = self.sandbox.run(
                image=image,
                command=train_command,
                repo_path=workspace,
                env=env,
            )

            metrics = self._extract_metrics(result.stdout)

            return ExperimentResult(
                hypothesis=proposal.get("reasoning", ""),
                diff=diff,
                run_result=result,
                metrics=metrics,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _read_source(self, repo_path: Path) -> dict[str, str]:
        """Read Python source files from the repo."""
        files = {}
        for py_file in sorted(repo_path.rglob("*.py")):
            rel = py_file.relative_to(repo_path)
            # Skip hidden dirs, __pycache__, etc.
            if any(part.startswith((".", "__")) for part in rel.parts):
                continue
            files[str(rel)] = py_file.read_text()
        return files

    def _apply_diff(self, workspace: Path, proposal: dict[str, Any]) -> dict[str, str]:
        """Apply the LLM's proposed code change."""
        filename = proposal.get("filename", "train.py")
        original = proposal.get("original", "")
        replacement = proposal.get("replacement", "")

        target = workspace / filename
        if not target.exists():
            logger.warning("Target file %s not found, skipping diff", filename)
            return {"error": f"File not found: {filename}"}

        content = target.read_text()
        if original and original in content:
            new_content = content.replace(original, replacement, 1)
            target.write_text(new_content)
            return {"filename": filename, "original": original, "replacement": replacement}

        logger.warning("Original snippet not found in %s, writing replacement as append", filename)
        return {"warning": "original snippet not found, change not applied"}

    def _extract_metrics(self, stdout: str) -> dict[str, Any]:
        """Extract metrics from training output.

        Looks for a JSON line starting with METRICS: in stdout.
        """
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("METRICS:"):
                try:
                    return json.loads(line[len("METRICS:"):])
                except json.JSONDecodeError:
                    continue
        return {}
