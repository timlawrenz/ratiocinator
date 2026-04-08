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
    - A hypothesis genealogy chart showing parent-child relationships

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

    # 3. Hypothesis genealogy tree
    all_nodes = [n for n in tree.all_nodes() if n.status == NodeStatus.SUCCESS]
    if len(all_nodes) > 1:
        path = _plot_genealogy(all_nodes, output_dir, score_key)
        if path:
            plots.append(path)

    logger.info("Generated %d plots in %s", len(plots), output_dir)
    return plots


def _plot_genealogy(
    nodes: list,
    output_dir: Path,
    score_key: str,
) -> Path | None:
    """Plot a hypothesis genealogy showing parent→child relationships with scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        node_map = {n.id: n for n in nodes}

        fig, ax = plt.subplots(figsize=(12, max(4, len(nodes) * 0.6)))

        # Position nodes: x = score, y = index (sorted by depth then score)
        sorted_nodes = sorted(nodes, key=lambda n: (n.depth, n.score or 0))
        y_positions = {n.id: i for i, n in enumerate(sorted_nodes)}

        for n in sorted_nodes:
            y = y_positions[n.id]
            score = n.score or 0
            color = "#2ecc71" if n.depth == 0 else "#3498db"
            ax.scatter(score, y, s=100, c=color, zorder=5, edgecolors="white", linewidth=1.5)

            label = n.hypothesis[:40] if n.hypothesis else n.id[:8]
            ax.annotate(label, (score, y), fontsize=7,
                        xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left")

            # Draw edge from parent
            if n.parent_id and n.parent_id in node_map:
                parent = node_map[n.parent_id]
                py = y_positions.get(parent.id)
                if py is not None:
                    ax.annotate(
                        "", xy=(score, y), xytext=(parent.score or 0, py),
                        arrowprops={"arrowstyle": "->", "color": "#bdc3c7",
                                    "lw": 1.2, "connectionstyle": "arc3,rad=0.1"},
                    )

        ax.set_xlabel(score_key)
        ax.set_ylabel("Experiment")
        ax.set_title("Hypothesis Genealogy")
        ax.set_yticks(range(len(sorted_nodes)))
        ax.set_yticklabels([f"d{n.depth}-{n.id[:6]}" for n in sorted_nodes], fontsize=7)
        plt.tight_layout()

        path = output_dir / "hypothesis_genealogy.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    except Exception:
        logger.exception("Failed to generate genealogy plot")
        return None
