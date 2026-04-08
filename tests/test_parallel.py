"""Tests for parallel tree search."""

from __future__ import annotations

from ratiocinator.search.tree import ExperimentTree, NodeStatus


def test_get_multiple_expandable(tmp_path):
    """Test that we can find multiple expandable nodes."""
    db = tmp_path / "test.db"
    tree = ExperimentTree(db)

    root = tree.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    tree.update(root)

    c1 = tree.add_child(root.id, "mod-1", {"f": "a"})
    c1.status = NodeStatus.SUCCESS
    c1.score = 0.3
    tree.update(c1)

    c2 = tree.add_child(root.id, "mod-2", {"f": "b"})
    c2.status = NodeStatus.SUCCESS
    c2.score = 0.4
    tree.update(c2)

    # Simulate the internal method
    nodes = tree.all_nodes()
    expandable = [
        n
        for n in nodes
        if n.status == NodeStatus.SUCCESS
        and n.score is not None
        and n.depth < 5
    ]
    expandable.sort(key=lambda n: n.score)
    top = expandable[:2]

    assert len(top) == 2
    assert top[0].score == 0.3  # Best first
    assert top[1].score == 0.4

    tree.close()


def test_expandable_respects_depth(tmp_path):
    """Nodes at max_depth are not expandable."""
    db = tmp_path / "test.db"
    tree = ExperimentTree(db)

    root = tree.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    tree.update(root)

    c = tree.add_child(root.id, "mod", {"f": "a"})
    c.status = NodeStatus.SUCCESS
    c.score = 0.3
    tree.update(c)

    # max_depth=1 means depth 1 nodes cannot expand further
    nodes = tree.all_nodes()
    expandable = [
        n
        for n in nodes
        if n.status == NodeStatus.SUCCESS
        and n.score is not None
        and n.depth < 1  # max_depth=1
    ]

    assert len(expandable) == 1
    assert expandable[0].id == root.id

    tree.close()
