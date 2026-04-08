"""Tests for the experiment tree data structure."""

from __future__ import annotations

import pytest

from ratiocinator.search.tree import ExperimentTree, NodeStatus


@pytest.fixture
def tree(tmp_path):
    db_path = tmp_path / "test.db"
    t = ExperimentTree(db_path)
    yield t
    t.close()


def test_add_root(tree):
    root = tree.add_root("baseline")
    assert root.parent_id is None
    assert root.depth == 0
    assert root.status == NodeStatus.PENDING
    assert tree.count() == 1


def test_add_child(tree):
    root = tree.add_root("baseline")
    child = tree.add_child(
        root.id,
        "try lr=0.001",
        {"filename": "train.py", "original": "lr=0.01", "replacement": "lr=0.001"},
    )
    assert child.parent_id == root.id
    assert child.depth == 1
    assert tree.count() == 2


def test_add_child_invalid_parent(tree):
    with pytest.raises(ValueError, match="Parent node not found"):
        tree.add_child("nonexistent", "test", {})


def test_get_and_update(tree):
    root = tree.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    root.metrics = {"loss": 0.5}
    tree.update(root)

    fetched = tree.get(root.id)
    assert fetched.status == NodeStatus.SUCCESS
    assert fetched.score == 0.5
    assert fetched.metrics == {"loss": 0.5}


def test_get_children(tree):
    root = tree.add_root()
    c1 = tree.add_child(root.id, "child1", {"f": "a"})
    c2 = tree.add_child(root.id, "child2", {"f": "b"})
    children = tree.get_children(root.id)
    assert len(children) == 2
    assert {c.id for c in children} == {c1.id, c2.id}


def test_get_best_leaf(tree):
    root = tree.add_root()
    root.status = NodeStatus.SUCCESS
    root.score = 1.0
    tree.update(root)

    c1 = tree.add_child(root.id, "better", {})
    c1.status = NodeStatus.SUCCESS
    c1.score = 0.3
    tree.update(c1)

    c2 = tree.add_child(root.id, "worse", {})
    c2.status = NodeStatus.SUCCESS
    c2.score = 0.8
    tree.update(c2)

    best = tree.get_best_leaf(lower_is_better=True)
    assert best.id == c1.id

    best_max = tree.get_best_leaf(lower_is_better=False)
    assert best_max.id == root.id


def test_get_expandable_respects_depth(tree):
    root = tree.add_root()
    root.status = NodeStatus.SUCCESS
    root.score = 1.0
    tree.update(root)

    # Depth 0, max_depth=1 → expandable
    node = tree.get_expandable(max_depth=1)
    assert node is not None
    assert node.id == root.id

    # Depth 0, max_depth=0 → not expandable
    node = tree.get_expandable(max_depth=0)
    assert node is None


def test_get_failed(tree):
    root = tree.add_root()
    root.status = NodeStatus.SUCCESS
    root.score = 1.0
    tree.update(root)

    child = tree.add_child(root.id, "broken", {})
    child.status = NodeStatus.FAILED
    tree.update(child)

    failed = tree.get_failed()
    assert len(failed) == 1
    assert failed[0].id == child.id


def test_summary(tree):
    root = tree.add_root()
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    tree.update(root)

    child = tree.add_child(root.id, "test", {})
    child.status = NodeStatus.FAILED
    tree.update(child)

    summary = tree.summary()
    assert summary["total_nodes"] == 2
    assert summary["max_depth"] == 1
    assert summary["by_status"]["success"] == 1
    assert summary["by_status"]["failed"] == 1
    assert summary["best_score"] == 0.5


def test_persistence(tmp_path):
    db_path = tmp_path / "persist.db"

    tree1 = ExperimentTree(db_path)
    root = tree1.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.42
    tree1.update(root)
    root_id = root.id
    tree1.close()

    tree2 = ExperimentTree(db_path)
    fetched = tree2.get(root_id)
    assert fetched is not None
    assert fetched.score == 0.42
    tree2.close()
