"""Tests for the plotting agent."""

from __future__ import annotations

import pytest

from ratiocinator.search.tree import ExperimentTree, NodeStatus
from ratiocinator.synthesis.plotting import generate_plots


@pytest.fixture
def tree_with_results(tmp_path):
    db = tmp_path / "test.db"
    tree = ExperimentTree(db)

    root = tree.add_root("baseline")
    root.status = NodeStatus.SUCCESS
    root.score = 0.5
    root.metrics = {"train_loss": 0.5}
    tree.update(root)

    c1 = tree.add_child(root.id, "lr=0.001", {"f": "a"})
    c1.status = NodeStatus.SUCCESS
    c1.score = 0.3
    c1.metrics = {"train_loss": 0.3}
    tree.update(c1)

    c2 = tree.add_child(root.id, "lr=0.1", {"f": "b"})
    c2.status = NodeStatus.SUCCESS
    c2.score = 0.7
    c2.metrics = {"train_loss": 0.7}
    tree.update(c2)

    c3 = tree.add_child(c1.id, "lr=0.001 + warmup", {"f": "c"})
    c3.status = NodeStatus.SUCCESS
    c3.score = 0.2
    c3.metrics = {"train_loss": 0.2}
    tree.update(c3)

    yield tree
    tree.close()


def test_generate_plots(tree_with_results, tmp_path):
    output_dir = tmp_path / "plots"
    plots = generate_plots(tree_with_results, output_dir)
    assert len(plots) == 3
    assert all(p.exists() for p in plots)
    assert "scores_comparison.png" in str(plots[0])
    assert "depth_progression.png" in str(plots[1])
    assert "hypothesis_genealogy.png" in str(plots[2])


def test_generate_plots_empty_tree(tmp_path):
    db = tmp_path / "empty.db"
    tree = ExperimentTree(db)
    tree.add_root()  # No successful nodes
    output_dir = tmp_path / "plots"
    plots = generate_plots(tree, output_dir)
    assert plots == []
    tree.close()
