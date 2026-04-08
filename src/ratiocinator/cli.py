"""CLI entry point for ratiocinator."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from ratiocinator.config import load_config


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def main(ctx: click.Context, config_path: Path | None, verbose: bool) -> None:
    """Ratiocinator — Distributed Scientific Discovery Pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.option("--task", required=True, help="Research task description")
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to target repo",
)
@click.option("--steps", default=500, help="Training steps per experiment")
@click.option("--image", default="python:3.11-slim", help="Docker image for sandbox")
@click.option("--command", default="python train.py", help="Training command to run")
@click.pass_context
def run(ctx: click.Context, task: str, repo: Path, steps: int, image: str, command: str) -> None:
    """Run a single experiment: propose a change, execute it, report metrics."""
    from ratiocinator.experiment import ExperimentLoop

    config = ctx.obj["config"]
    loop = ExperimentLoop(config)

    result = asyncio.run(
        loop.run_experiment(repo, task, image=image, train_command=command, steps=steps)
    )

    output = result.to_dict()
    click.echo(json.dumps(output, indent=2))

    if not result.run_result.success:
        sys.exit(1)


@main.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to target repo",
)
@click.option("--steps", default=500, help="Training steps per experiment")
@click.option("--image", default="python:3.11-slim", help="Docker image for sandbox")
@click.option("--command", default="python train.py", help="Training command")
@click.option("--score-key", default="train_loss", help="Metric key to optimize")
@click.option("--maximize", is_flag=True, help="Maximize score (default: minimize)")
@click.option("--local", is_flag=True, help="Use local subprocess instead of Docker")
@click.pass_context
def search(
    ctx: click.Context,
    repo: Path,
    steps: int,
    image: str,
    command: str,
    score_key: str,
    maximize: bool,
    local: bool,
) -> None:
    """Run Best-First Tree Search over code modifications."""
    from ratiocinator.search.bfts import BestFirstSearch, BudgetExhaustedError

    config = ctx.obj["config"]
    bfts = BestFirstSearch(
        config,
        repo,
        image=image,
        train_command=command,
        steps=steps,
        score_key=score_key,
        lower_is_better=not maximize,
    )

    if local:
        from ratiocinator.sandbox.runner import LocalRunner

        bfts.sandbox = LocalRunner(config.sandbox)

    try:
        summary = asyncio.run(bfts.run())
    except BudgetExhaustedError as e:
        click.echo(f"Search stopped: {e}", err=True)
        summary = bfts.tree.summary()

    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option("--model", default=None, help="Override model (e.g., ollama/llama3)")
@click.argument("prompt")
@click.pass_context
def ask(ctx: click.Context, model: str | None, prompt: str) -> None:
    """Send a one-off prompt to the configured LLM."""
    from ratiocinator.llm.client import LLMClient

    config = ctx.obj["config"]
    if model:
        config.llm.generalist.model = model
    client = LLMClient(config.llm)

    response = asyncio.run(client.complete(prompt))
    click.echo(response.content)


@main.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to target repo",
)
@click.option("--title", required=True, help="Paper title")
@click.option("--steps", default=500, help="Training steps per experiment")
@click.option("--command", default="python train.py", help="Training command")
@click.option("--score-key", default="train_loss", help="Metric key to optimize")
@click.option("--maximize", is_flag=True, help="Maximize score (default: minimize)")
@click.option("--local", is_flag=True, help="Use local subprocess instead of Docker")
@click.option("--image", default="python:3.11-slim", help="Docker image for sandbox")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for paper and plots (default: .ratiocinator/output)",
)
@click.pass_context
def synthesize(
    ctx: click.Context,
    repo: Path,
    title: str,
    steps: int,
    command: str,
    score_key: str,
    maximize: bool,
    local: bool,
    image: str,
    output_dir: Path | None,
) -> None:
    """Run full pipeline: search → plots → paper → review."""
    asyncio.run(_synthesize(ctx, repo, title, steps, command, score_key, maximize, local, image,
                            output_dir))


async def _synthesize(
    ctx: click.Context,
    repo: Path,
    title: str,
    steps: int,
    command: str,
    score_key: str,
    maximize: bool,
    local: bool,
    image: str,
    output_dir: Path | None,
) -> None:
    from ratiocinator.llm.client import LLMClient
    from ratiocinator.search.bfts import BestFirstSearch, BudgetExhaustedError
    from ratiocinator.synthesis.paper import PaperGenerator
    from ratiocinator.synthesis.plotting import generate_plots
    from ratiocinator.synthesis.reviewer import AutoReviewer

    config = ctx.obj["config"]
    output = output_dir or config.work_dir / "output"
    output.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Tree Search ---
    click.echo("=" * 60)
    click.echo("Phase 1: Tree Search")
    click.echo("=" * 60)

    bfts = BestFirstSearch(
        config,
        repo,
        image=image,
        train_command=command,
        steps=steps,
        score_key=score_key,
        lower_is_better=not maximize,
    )

    if local:
        from ratiocinator.sandbox.runner import LocalRunner

        bfts.sandbox = LocalRunner(config.sandbox)

    try:
        summary = await bfts.run()
    except BudgetExhaustedError as e:
        click.echo(f"  Search stopped: {e}", err=True)
        summary = bfts.tree.summary()

    click.echo(f"  Nodes: {summary['total_nodes']}, Best: {summary['best_score']}")

    # --- Phase 2: Plots ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Phase 2: Generating Plots")
    click.echo("=" * 60)

    plot_dir = output / "plots"
    plots = generate_plots(bfts.tree, plot_dir, score_key=score_key)
    click.echo(f"  Generated {len(plots)} plot(s)")

    # --- Phase 3: Paper ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Phase 3: Generating Paper")
    click.echo("=" * 60)

    llm = LLMClient(config.llm)
    generator = PaperGenerator(llm)
    paper_path = output / "paper.tex"
    latex = await generator.generate(
        bfts.tree,
        title=title,
        plot_paths=plots,
        output_path=paper_path,
    )
    click.echo(f"  Paper: {paper_path} ({len(latex)} chars)")

    # --- Phase 4: Review ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Phase 4: Automated Review")
    click.echo("=" * 60)

    reviewer = AutoReviewer(llm, max_revisions=2)
    final_latex, reviews = await reviewer.review_and_revise(latex, min_score=6)

    for i, review in enumerate(reviews):
        click.echo(f"  Review {i + 1}: score={review.total_score}/10 verdict={review.verdict}")
        if review.issues:
            for issue in review.issues[:3]:
                click.echo(f"    - {issue}")

    # Write final version
    final_path = output / "paper_final.tex"
    final_path.write_text(final_latex)

    # Write summary JSON
    result = {
        "tree_summary": summary,
        "plots": [str(p) for p in plots],
        "paper": str(paper_path),
        "paper_final": str(final_path),
        "review_scores": [r.total_score for r in reviews],
        "final_verdict": reviews[-1].verdict if reviews else "unknown",
    }
    summary_path = output / "result.json"
    summary_path.write_text(json.dumps(result, indent=2))

    click.echo()
    click.echo("=" * 60)
    click.echo("Complete!")
    click.echo("=" * 60)
    click.echo(f"  Output: {output}")
    click.echo(f"  Paper:  {final_path}")
    click.echo(f"  Score:  {reviews[-1].total_score}/10" if reviews else "  No reviews")


if __name__ == "__main__":
    main()
