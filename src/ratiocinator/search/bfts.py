"""Best-First Tree Search over code modifications.

Orchestrates the expand → evaluate → select loop, with error recovery
and hard budget controls.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ratiocinator.config import Config
from ratiocinator.llm.client import LLMClient
from ratiocinator.sandbox.runner import SandboxRunner
from ratiocinator.search.tree import ExperimentTree, NodeStatus, TreeNode

logger = logging.getLogger(__name__)

EXPAND_SYSTEM = """\
You are an AI research assistant exploring code modifications to improve \
experimental metrics. Given the current best code and its metrics, propose \
a new modification.

You will receive the code with any previous modifications applied, along with
the current metrics. Propose a DIFFERENT change that could improve results.

Respond with JSON containing:
- "reasoning": brief explanation of why this change should help
- "filename": the file to modify
- "original": the exact code snippet to replace
- "replacement": the new code snippet
"""

RECOVERY_SYSTEM = """\
You are an AI research assistant. A previous code modification caused an error.
Analyze the error and propose a fix.

Respond with JSON containing:
- "reasoning": what went wrong and how this fix addresses it
- "filename": the file to modify
- "original": the exact code snippet to replace
- "replacement": the fixed code snippet
"""


class BudgetExhaustedError(Exception):
    """Raised when a search budget limit is reached."""


class BestFirstSearch:
    """Best-First Tree Search over code modifications."""

    def __init__(
        self,
        config: Config,
        repo_path: Path,
        *,
        image: str = "python:3.11-slim",
        train_command: str = "python train.py",
        steps: int = 500,
        score_key: str = "train_loss",
        lower_is_better: bool = True,
    ) -> None:
        self.config = config
        self.repo_path = repo_path
        self.image = image
        self.train_command = train_command
        self.steps = steps
        self.score_key = score_key
        self.lower_is_better = lower_is_better

        self.llm = LLMClient(config.llm)
        self.sandbox = SandboxRunner(config.sandbox)

        db_path = config.work_dir / "search.db"
        self.tree = ExperimentTree(db_path)

    async def run(self) -> dict[str, Any]:
        """Execute the full tree search loop.

        Returns a summary dict with the tree state and best result.
        """
        sc = self.config.search
        start_time = time.monotonic()

        # Create and evaluate root (baseline)
        root = self.tree.add_root("baseline — unmodified code")
        await self._evaluate_node(root, self.repo_path)

        logger.info("Baseline: score=%s metrics=%s", root.score, root.metrics)

        while True:
            self._check_budgets(start_time)

            # Select best expandable node
            node = self.tree.get_expandable(sc.max_depth, self.lower_is_better)
            if node is None:
                logger.info("No expandable nodes remaining")
                break

            # Expand: generate candidates
            candidates = await self._expand(node)

            for child_node in candidates:
                self._check_budgets(start_time)
                await self._evaluate_with_recovery(child_node)

            logger.info(
                "Tree: %d nodes, best_score=%s",
                self.tree.count(),
                self.tree.get_best_leaf(self.lower_is_better).score
                if self.tree.get_best_leaf(self.lower_is_better)
                else None,
            )

        return self.tree.summary()

    async def _expand(self, parent: TreeNode) -> list[TreeNode]:
        """Generate child candidates from a parent node."""
        source = self._read_modified_source(parent)
        children = []

        for i in range(self.config.search.branching_factor):
            prompt = (
                f"## Current best code (score: {parent.score})\n"
                f"```python\n{source}\n```\n\n"
                f"## Current metrics\n{json.dumps(parent.metrics, indent=2)}\n\n"
                f"## Optimization target\nMinimize `{self.score_key}`\n\n"
                f"## Candidate number {i + 1} of {self.config.search.branching_factor}\n"
                f"Propose a modification DIFFERENT from previous attempts."
            )

            try:
                proposal = await self.llm.complete_json(
                    prompt, system=EXPAND_SYSTEM, task="coding"
                )
                child = self.tree.add_child(
                    parent.id,
                    hypothesis=proposal.get("reasoning", f"candidate-{i}"),
                    diff={
                        "filename": proposal.get("filename", "train.py"),
                        "original": proposal.get("original", ""),
                        "replacement": proposal.get("replacement", ""),
                    },
                )
                children.append(child)
            except Exception:
                logger.exception("Failed to generate candidate %d", i)

        return children

    async def _evaluate_node(self, node: TreeNode, workspace: Path) -> None:
        """Run a node's experiment in the sandbox and record results."""
        node.status = NodeStatus.RUNNING
        self.tree.update(node)

        env = {"TRAIN_STEPS": str(self.steps)}
        result = self.sandbox.run(
            image=self.image,
            command=self.train_command,
            repo_path=workspace,
            env=env,
        )

        node.stdout = result.stdout
        node.stderr = result.stderr
        node.metrics = self._extract_metrics(result.stdout)

        if result.success:
            node.status = NodeStatus.SUCCESS
            node.score = node.metrics.get(self.score_key)
        else:
            node.status = NodeStatus.FAILED

        self.tree.update(node)

    async def _evaluate_with_recovery(self, node: TreeNode) -> None:
        """Evaluate a node, attempting error recovery on failure."""
        work_dir = Path(tempfile.mkdtemp(prefix="ratiocinator-"))
        try:
            workspace = work_dir / "workspace"
            shutil.copytree(self.repo_path, workspace)
            self._apply_ancestor_diffs(node, workspace)

            await self._evaluate_node(node, workspace)

            # Error recovery: if failed, ask LLM to fix it
            if node.status == NodeStatus.FAILED and node.stderr:
                logger.info("Attempting error recovery for node %s", node.id)
                recovery_node = await self._attempt_recovery(node, workspace)
                if recovery_node:
                    await self._evaluate_with_recovery(recovery_node)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _attempt_recovery(
        self, failed_node: TreeNode, workspace: Path
    ) -> TreeNode | None:
        """Ask the LLM to fix a failed experiment."""
        source = self._read_source(workspace)
        prompt = (
            f"## Code that caused the error\n```python\n{source}\n```\n\n"
            f"## Error output\n```\n{failed_node.stderr[-3000:]}\n```\n\n"
            f"## Stdout\n```\n{failed_node.stdout[-2000:]}\n```\n\n"
            f"Propose a fix for this error."
        )

        try:
            proposal = await self.llm.complete_json(
                prompt, system=RECOVERY_SYSTEM, task="coding"
            )
            return self.tree.add_child(
                failed_node.id,
                hypothesis=f"recovery: {proposal.get('reasoning', 'fix')}",
                diff={
                    "filename": proposal.get("filename", "train.py"),
                    "original": proposal.get("original", ""),
                    "replacement": proposal.get("replacement", ""),
                },
            )
        except Exception:
            logger.exception("Recovery failed for node %s", failed_node.id)
            return None

    def _apply_ancestor_diffs(self, node: TreeNode, workspace: Path) -> None:
        """Walk up the tree and apply all diffs from root to this node."""
        chain = []
        current: TreeNode | None = node
        while current and current.parent_id is not None:
            chain.append(current)
            current = self.tree.get(current.parent_id)
        chain.reverse()

        for ancestor in chain:
            self._apply_diff(workspace, ancestor.diff)

    def _apply_diff(self, workspace: Path, diff: dict[str, str]) -> None:
        """Apply a single diff to the workspace."""
        filename = diff.get("filename", "")
        original = diff.get("original", "")
        replacement = diff.get("replacement", "")

        if not filename or not original:
            return

        target = workspace / filename
        if not target.exists():
            return

        content = target.read_text()
        if original in content:
            target.write_text(content.replace(original, replacement, 1))

    def _check_budgets(self, start_time: float) -> None:
        """Enforce hard search budget limits."""
        sc = self.config.search

        elapsed = time.monotonic() - start_time
        if elapsed > sc.max_wall_clock_seconds:
            raise BudgetExhaustedError(
                f"Wall clock limit reached: {elapsed:.0f}s > {sc.max_wall_clock_seconds}s"
            )

        count = self.tree.count()
        if count >= sc.max_nodes:
            raise BudgetExhaustedError(f"Node limit reached: {count} >= {sc.max_nodes}")

    def _read_modified_source(self, node: TreeNode) -> str:
        """Reconstruct the source code as it would look after applying all diffs to this node."""
        work_dir = Path(tempfile.mkdtemp(prefix="ratiocinator-src-"))
        try:
            workspace = work_dir / "workspace"
            shutil.copytree(self.repo_path, workspace)

            # Apply all ancestor diffs including this node
            chain = []
            current: TreeNode | None = node
            while current and current.parent_id is not None:
                chain.append(current)
                current = self.tree.get(current.parent_id)
            chain.reverse()
            for ancestor in chain:
                self._apply_diff(workspace, ancestor.diff)

            return self._read_source(workspace)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _read_source(self, workspace: Path) -> str:
        """Read all Python files from a workspace."""
        files = {}
        for py_file in sorted(workspace.rglob("*.py")):
            rel = py_file.relative_to(workspace)
            if any(part.startswith((".", "__")) for part in rel.parts):
                continue
            files[str(rel)] = py_file.read_text()
        return "\n\n".join(f"# --- {name} ---\n{content}" for name, content in files.items())

    def _extract_metrics(self, stdout: str) -> dict[str, Any]:
        """Extract METRICS:{...} JSON from stdout."""
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("METRICS:"):
                try:
                    return json.loads(line[len("METRICS:"):])
                except json.JSONDecodeError:
                    continue
        return {}
