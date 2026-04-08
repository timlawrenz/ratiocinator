"""Parallel tree search: expands multiple nodes simultaneously via Vast.ai."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ratiocinator.config import Config
from ratiocinator.infra.safety import SafetyController
from ratiocinator.infra.vast_client import VastClient
from ratiocinator.infra.webhook import WebhookReceiver
from ratiocinator.llm.client import LLMClient
from ratiocinator.search.tree import ExperimentTree, NodeStatus, TreeNode

logger = logging.getLogger(__name__)


class ParallelSearch:
    """Parallel tree search that distributes experiments across Vast.ai instances.

    Wraps BestFirstSearch to expand multiple nodes simultaneously, each
    launching a separate Vast.ai instance. Throttled by max_concurrent.
    """

    def __init__(
        self,
        config: Config,
        vast_api_key: str,
        *,
        max_concurrent: int = 3,
        webhook_port: int = 0,
        gpu_name: str = "RTX_3060",
        max_gpu_price: float = 0.30,
    ) -> None:
        self.config = config
        self.max_concurrent = max_concurrent
        self.gpu_name = gpu_name
        self.max_gpu_price = max_gpu_price

        self.llm = LLMClient(config.llm)
        self.vast = VastClient(api_key=vast_api_key)
        self.webhook = WebhookReceiver(port=webhook_port)
        self.safety = SafetyController(
            vast=self.vast,
            max_budget=config.safety.max_dollars_per_run,
            instance_ttl=config.safety.instance_ttl_seconds,
        )

        db_path = config.work_dir / "parallel_search.db"
        self.tree = ExperimentTree(db_path)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def run(self, score_key: str = "train_loss") -> dict[str, Any]:
        """Execute parallel tree search.

        Returns a summary dict when complete.
        """
        self.webhook.start()
        start_time = time.monotonic()

        try:
            return await self._search_loop(score_key, start_time)
        finally:
            await self.safety.shutdown()
            self.webhook.stop()

    async def _search_loop(
        self, score_key: str, start_time: float
    ) -> dict[str, Any]:
        sc = self.config.search
        pending_tasks: list[asyncio.Task] = []

        while True:
            # Check budgets
            elapsed = time.monotonic() - start_time
            if elapsed > sc.max_wall_clock_seconds:
                logger.info("Wall clock limit reached")
                break

            if self.tree.count() >= sc.max_nodes:
                logger.info("Node limit reached")
                break

            # Find expandable nodes
            expandable = self._get_multiple_expandable(sc.max_depth)
            if not expandable and not pending_tasks:
                logger.info("No expandable nodes and no pending tasks")
                break

            # Launch experiments for expandable nodes (up to concurrency limit)
            for node in expandable:
                if self.tree.count() >= sc.max_nodes:
                    break
                task = asyncio.create_task(
                    self._expand_and_evaluate(node, score_key)
                )
                pending_tasks.append(task)

            # Wait for at least one task to complete
            if pending_tasks:
                done, pending = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = list(pending)

                for task in done:
                    try:
                        task.result()
                    except Exception:
                        logger.exception("Expansion task failed")

        # Wait for remaining tasks
        if pending_tasks:
            results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Task error: %s", r)

        return self.tree.summary()

    async def _expand_and_evaluate(
        self, parent: TreeNode, score_key: str
    ) -> None:
        """Expand a parent node and evaluate the child on a Vast.ai instance."""
        async with self._semaphore:
            logger.info(
                "Expanding node %s (depth=%d, score=%s)",
                parent.id,
                parent.depth,
                parent.score,
            )

            child = self.tree.add_child(
                parent.id,
                hypothesis=f"parallel-expansion from {parent.id[:6]}",
                diff={},
            )
            child.status = NodeStatus.RUNNING
            self.tree.update(child)

            try:
                # TODO: full integration pending
                # 1. Search for a cheap GPU offer
                # 2. Create a Vast.ai instance with bootstrap script
                # 3. Wait for webhook callback with metrics
                child.status = NodeStatus.PENDING
                self.tree.update(child)

                logger.info("Node %s queued for remote execution", child.id)

            except Exception:
                logger.exception("Failed to launch instance for node %s", child.id)
                child.status = NodeStatus.ERROR
                self.tree.update(child)

    def _get_multiple_expandable(self, max_depth: int) -> list[TreeNode]:
        """Get up to max_concurrent expandable nodes."""
        nodes = self.tree.all_nodes()
        expandable = [
            n
            for n in nodes
            if n.status == NodeStatus.SUCCESS
            and n.score is not None
            and n.depth < max_depth
        ]
        expandable.sort(key=lambda n: n.score)
        return expandable[: self.max_concurrent]
