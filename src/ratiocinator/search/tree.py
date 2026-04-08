"""Persistent tree data structure for experiment tracking.

Each node represents one experiment: a code modification, its execution result,
and metric scores. The tree is persisted to SQLite so runs survive crashes.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class TreeNode:
    """A single node in the experiment tree."""

    id: str
    parent_id: str | None
    hypothesis: str
    diff: dict[str, str]
    status: NodeStatus = NodeStatus.PENDING
    metrics: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    score: float | None = None
    depth: int = 0
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> tuple:
        return (
            self.id,
            self.parent_id,
            self.hypothesis,
            json.dumps(self.diff),
            self.status.value,
            json.dumps(self.metrics),
            self.stdout[-5000:],
            self.stderr[-5000:],
            self.score,
            self.depth,
            self.created_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> TreeNode:
        return cls(
            id=row["id"],
            parent_id=row["parent_id"],
            hypothesis=row["hypothesis"],
            diff=json.loads(row["diff"]),
            status=NodeStatus(row["status"]),
            metrics=json.loads(row["metrics"]),
            stdout=row["stdout"],
            stderr=row["stderr"],
            score=row["score"],
            depth=row["depth"],
            created_at=row["created_at"],
        )


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    hypothesis TEXT NOT NULL,
    diff TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    metrics TEXT NOT NULL DEFAULT '{}',
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    score REAL,
    depth INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES nodes(id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_nodes_score ON nodes(score);
"""


class ExperimentTree:
    """SQLite-backed experiment tree."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        """Delete all nodes, starting a fresh search."""
        self._conn.execute("DELETE FROM nodes")
        self._conn.commit()

    def add_root(self, hypothesis: str = "baseline") -> TreeNode:
        """Create the root node (no parent, no diff)."""
        node = TreeNode(
            id=_new_id(),
            parent_id=None,
            hypothesis=hypothesis,
            diff={},
            status=NodeStatus.PENDING,
            depth=0,
        )
        self._insert(node)
        return node

    def add_child(
        self,
        parent_id: str,
        hypothesis: str,
        diff: dict[str, str],
    ) -> TreeNode:
        """Add a child node to an existing parent."""
        parent = self.get(parent_id)
        if parent is None:
            raise ValueError(f"Parent node not found: {parent_id}")
        node = TreeNode(
            id=_new_id(),
            parent_id=parent_id,
            hypothesis=hypothesis,
            diff=diff,
            status=NodeStatus.PENDING,
            depth=parent.depth + 1,
        )
        self._insert(node)
        return node

    def get(self, node_id: str) -> TreeNode | None:
        """Retrieve a node by ID."""
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return TreeNode.from_row(row) if row else None

    def update(self, node: TreeNode) -> None:
        """Update a node's mutable fields."""
        self._conn.execute(
            """UPDATE nodes SET status=?, metrics=?, stdout=?, stderr=?, score=?
               WHERE id=?""",
            (
                node.status.value,
                json.dumps(node.metrics),
                node.stdout[-5000:],
                node.stderr[-5000:],
                node.score,
                node.id,
            ),
        )
        self._conn.commit()

    def get_children(self, parent_id: str) -> list[TreeNode]:
        """Get all children of a node."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()
        return [TreeNode.from_row(r) for r in rows]

    def get_best_leaf(self, lower_is_better: bool = True) -> TreeNode | None:
        """Get the node with the best score among completed nodes."""
        order = "ASC" if lower_is_better else "DESC"
        row = self._conn.execute(
            f"SELECT * FROM nodes WHERE status = 'success' AND score IS NOT NULL "
            f"ORDER BY score {order} LIMIT 1",
        ).fetchone()
        return TreeNode.from_row(row) if row else None

    def get_expandable(self, max_depth: int, lower_is_better: bool = True) -> TreeNode | None:
        """Get the best-scoring node that can still be expanded.

        A node is expandable if it succeeded and is below max_depth.
        """
        order = "ASC" if lower_is_better else "DESC"
        row = self._conn.execute(
            f"SELECT * FROM nodes WHERE status = 'success' AND score IS NOT NULL "
            f"AND depth < ? ORDER BY score {order} LIMIT 1",
            (max_depth,),
        ).fetchone()
        return TreeNode.from_row(row) if row else None

    def get_failed(self) -> list[TreeNode]:
        """Get all failed nodes (candidates for error recovery)."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE status IN ('failed', 'error') ORDER BY created_at",
        ).fetchall()
        return [TreeNode.from_row(r) for r in rows]

    def count(self) -> int:
        """Total number of nodes in the tree."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM nodes").fetchone()
        return row["cnt"]

    def all_nodes(self) -> list[TreeNode]:
        """Return all nodes ordered by depth then creation time."""
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY depth, created_at"
        ).fetchall()
        return [TreeNode.from_row(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        """Return a summary of the tree state."""
        nodes = self.all_nodes()
        by_status = {}
        for n in nodes:
            by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
        best = self.get_best_leaf()
        return {
            "total_nodes": len(nodes),
            "max_depth": max((n.depth for n in nodes), default=0),
            "by_status": by_status,
            "best_score": best.score if best else None,
            "best_node_id": best.id if best else None,
        }

    def _insert(self, node: TreeNode) -> None:
        self._conn.execute(
            """INSERT INTO nodes (id, parent_id, hypothesis, diff, status,
               metrics, stdout, stderr, score, depth, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            node.to_row(),
        )
        self._conn.commit()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]
