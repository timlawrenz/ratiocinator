"""Plotting agent: generates comparative charts from experiment tree metrics."""

from __future__ import annotations

import logging
from pathlib import Path

from ratiocinator.search.tree import ExperimentTree, NodeStatus

logger = logging.getLogger(__name__)


def generate_plots(
    tree: ExperimentTree,
    output_dir: Path,
    score_key: str = "train_loss",
) -> list[Path]:
    """Generate plots from experiment tree results.

    Creates:
    - A bar chart comparing metrics across successful experiments
    - A tree-depth progression chart showing score improvement over depth

    Returns list of generated plot file paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = [n for n in tree.all_nodes() if n.status == NodeStatus.SUCCESS and n.score is not None]

    if not nodes:
        logger.warning("No successful nodes to plot")
        return []

    plots = []

    # 1. Bar chart: score per experiment
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"d{n.depth}-{n.id[:6]}" for n in nodes]
    scores = [n.score for n in nodes]
    colors = ["#2ecc71" if n.depth == 0 else "#3498db" for n in nodes]
    ax.barh(labels, scores, color=colors)
    ax.set_xlabel(score_key)
    ax.set_title(f"Experiment Scores ({score_key})")
    ax.axvline(x=nodes[0].score, color="#e74c3c", linestyle="--", label="baseline")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "scores_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    # 2. Score by depth
    depth_scores: dict[int, list[float]] = {}
    for n in nodes:
        depth_scores.setdefault(n.depth, []).append(n.score)

    depths = sorted(depth_scores.keys())
    best_per_depth = [min(depth_scores[d]) for d in depths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, best_per_depth, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel(f"Best {score_key}")
    ax.set_title("Score Progression by Depth")
    ax.set_xticks(depths)
    plt.tight_layout()
    path = output_dir / "depth_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    logger.info("Generated %d plots in %s", len(plots), output_dir)
    return plots
