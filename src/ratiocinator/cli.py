"""CLI entry point for ratiocinator."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from ratiocinator.config import load_config
from ratiocinator.observability import init_sentry


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
    init_sentry()
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
@click.option("--steps", default=10, help="Training steps per experiment")
@click.option("--image", default=None, help="Docker image (default: config or python:3.11-slim)")
@click.option("--command", default="python train.py", help="Training command")
@click.option("--score-key", default="train_loss", help="Metric key to optimize")
@click.option("--maximize", is_flag=True, help="Maximize score (default: minimize)")
@click.option("--local", is_flag=True, help="Use local subprocess instead of Docker")
@click.option("--vast", is_flag=True, help="Run experiments on Vast.ai GPU instances")
@click.option("--fresh", is_flag=True, help="Clear previous search results and start fresh")
@click.option("--topic", default=None, help="Research topic for literature-grounded ideation")
@click.pass_context
def search(
    ctx: click.Context,
    repo: Path,
    steps: int,
    image: str | None,
    command: str,
    score_key: str,
    maximize: bool,
    local: bool,
    vast: bool,
    fresh: bool,
    topic: str | None,
) -> None:
    """Run Best-First Tree Search over code modifications."""
    from ratiocinator.search.bfts import BestFirstSearch, BudgetExhaustedError

    config = ctx.obj["config"]

    # Resolve image: explicit > vast config > fallback
    if image is None:
        image = config.vast.default_image if vast else "python:3.11-slim"

    bfts = BestFirstSearch(
        config,
        repo,
        image=image,
        train_command=command,
        steps=steps,
        score_key=score_key,
        lower_is_better=not maximize,
        topic=topic,
    )

    if fresh:
        bfts.tree.reset()
        click.echo("Search database cleared.", err=True)

    if vast:
        from ratiocinator.infra.vast_runner import VastRunner

        bfts.sandbox = VastRunner(config)
    elif local:
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
@click.option("--steps", default=10, help="Training steps per experiment")
@click.option("--command", default="python train.py", help="Training command")
@click.option("--score-key", default="train_loss", help="Metric key to optimize")
@click.option("--maximize", is_flag=True, help="Maximize score (default: minimize)")
@click.option("--local", is_flag=True, help="Use local subprocess instead of Docker")
@click.option("--vast", is_flag=True, help="Run experiments on Vast.ai GPU instances")
@click.option("--fresh", is_flag=True, help="Clear previous search results and start fresh")
@click.option("--image", default=None, help="Docker image (default: config or python:3.11-slim)")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for paper and plots (default: .ratiocinator/output)",
)
@click.option("--publish-to", default=None, help="HuggingFace repo to publish results")
@click.option("--topic", default=None, help="Research topic for literature-grounded ideation")
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
    vast: bool,
    fresh: bool,
    image: str | None,
    output_dir: Path | None,
    publish_to: str | None,
    topic: str | None,
) -> None:
    """Run full pipeline: search → plots → paper → review [→ publish]."""
    asyncio.run(_synthesize(ctx, repo, title, steps, command, score_key, maximize, local, vast,
                            fresh, image, output_dir, publish_to, topic))


async def _synthesize(
    ctx: click.Context,
    repo: Path,
    title: str,
    steps: int,
    command: str,
    score_key: str,
    maximize: bool,
    local: bool,
    vast: bool,
    fresh: bool,
    image: str | None,
    output_dir: Path | None,
    publish_to: str | None,
    topic: str | None,
) -> None:
    from ratiocinator.llm.client import LLMClient
    from ratiocinator.search.bfts import BestFirstSearch, BudgetExhaustedError
    from ratiocinator.synthesis.paper import PaperGenerator
    from ratiocinator.synthesis.plotting import generate_plots
    from ratiocinator.synthesis.reviewer import AutoReviewer

    config = ctx.obj["config"]

    # Resolve image: explicit > vast config > fallback
    if image is None:
        image = config.vast.default_image if vast else "python:3.11-slim"

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
        topic=topic,
    )

    if fresh:
        bfts.tree.reset()
        click.echo("  Search database cleared.", err=True)

    if vast:
        from ratiocinator.infra.vast_runner import VastRunner

        bfts.sandbox = VastRunner(config)
    elif local:
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

    # Collect literature context from ideation if available
    literature_context = None
    if bfts._ideation and hasattr(bfts._ideation, "_papers"):
        literature_context = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract,
                "authors": p.authors if hasattr(p, "authors") else [],
                "published": p.published if hasattr(p, "published") else "",
            }
            for p in bfts._ideation._papers[:15]
        ]

    llm = LLMClient(config.llm)
    generator = PaperGenerator(llm)
    paper_path = output / "paper.md"
    paper = await generator.generate(
        bfts.tree,
        title=title,
        plot_paths=plots,
        output_path=paper_path,
        literature_context=literature_context,
    )
    click.echo(f"  Paper: {paper_path} ({len(paper)} chars)")

    # --- Phase 4: Review ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Phase 4: Automated Review")
    click.echo("=" * 60)

    reviewer = AutoReviewer(llm, max_revisions=2)
    final_paper, reviews = await reviewer.review_and_revise(paper, min_score=6)

    for i, review in enumerate(reviews):
        click.echo(f"  Review {i + 1}: score={review.total_score}/10 verdict={review.verdict}")
        if review.issues:
            for issue in review.issues[:3]:
                click.echo(f"    - {issue}")

    # Write final version
    final_path = output / "paper_final.md"
    final_path.write_text(final_paper)

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

    # --- Phase 5 (optional): Publish ---
    publish_url = None
    hf_repo = publish_to or config.publish.repo_id
    if hf_repo and config.publish.hf_token:
        click.echo()
        click.echo("=" * 60)
        click.echo("Phase 5: Publishing to HuggingFace Hub")
        click.echo("=" * 60)

        from ratiocinator.synthesis.publisher import ArtifactPublisher, get_git_hash

        artifacts: dict[str, Path] = {}
        for pattern in ["*.md", "*.json"]:
            for p in output.glob(pattern):
                artifacts[p.name] = p
        plot_dir = output / "plots"
        if plot_dir.exists():
            for p in plot_dir.glob("*.png"):
                artifacts[f"plots/{p.name}"] = p

        publisher = ArtifactPublisher(hf_repo, token=config.publish.hf_token)
        tags = {"git_hash": get_git_hash(repo), "title": title}
        publish_url = publisher.publish(artifacts, commit_message=f"Results: {title}", tags=tags)
        click.echo(f"  Published: {publish_url}")

    click.echo()
    click.echo("=" * 60)
    click.echo("Complete!")
    click.echo("=" * 60)
    click.echo(f"  Output: {output}")
    click.echo(f"  Paper:  {final_path}")
    click.echo(f"  Score:  {reviews[-1].total_score}/10" if reviews else "  No reviews")
    if publish_url:
        click.echo(f"  Published: {publish_url}")


@main.command()
@click.argument("spec_file", type=click.Path(exists=True, path_type=Path))
@click.option("--api-key", default=None, help="Vast.ai API key (or VAST_API_KEY env)")
@click.option("--ssh-key", default=str(Path.home() / ".ssh" / "id_rsa"))
@click.option("--results-file", default=None, help="Where to persist results")
@click.pass_context
def research(
    ctx: click.Context,
    spec_file: Path,
    api_key: str | None,
    ssh_key: str,
    results_file: str | None,
) -> None:
    """Run autonomous research: LLM proposes arms → fleet executes → iterate.

    Example:

        ratiocinator research specs/gnn_study.yaml
    """
    asyncio.run(_research(ctx, spec_file, api_key, ssh_key, results_file))


async def _research(
    ctx: click.Context,
    spec_file: Path,
    api_key: str | None,
    ssh_key: str,
    results_file: str | None,
) -> None:
    from ratiocinator.orchestration.coordinator import ResearchCoordinator, ResearchSpec

    config = ctx.obj["config"]
    resolved_api_key = api_key or config.vast.api_key
    if not resolved_api_key:
        click.echo(
            "Error: VAST_API_KEY not set. Set the environment variable, "
            "add to config.json, or use --api-key.",
            err=True,
        )
        sys.exit(1)

    config.vast.api_key = resolved_api_key

    research_spec = ResearchSpec.from_yaml(spec_file)
    click.echo(f"Research: {research_spec.name}")
    click.echo(f"  Description: {research_spec.description}")
    click.echo(f"  Iterations: {research_spec.iterations}")
    click.echo(f"  Arms per iteration: {research_spec.num_arms}")
    click.echo(f"  Score key: {research_spec.score_key}")

    coordinator = ResearchCoordinator(config, research_spec, ssh_key=ssh_key)
    results = await coordinator.run()

    click.echo(f"\nCompleted {len(results)} arm results across all iterations.")


@main.group()
def fleet() -> None:
    """Fleet orchestration: run parallel experiments on Vast.ai."""
    pass


main.add_command(fleet)


@fleet.command("run")
@click.argument("spec_file", type=click.Path(exists=True, path_type=Path))
@click.option("--arms", default=None, help="Comma-separated arm indices (e.g. '0,2,4')")
@click.option("--api-key", default=None, help="Vast.ai API key (or VAST_API_KEY env)")
@click.option("--ssh-key", default=str(Path.home() / ".ssh" / "id_rsa"))
@click.option(
    "--results-file", default=".ratiocinator/results/experiments.json",
    help="Where to persist results",
)
@click.option("--dry-run", is_flag=True, help="Show what would be launched without executing")
@click.option("--data-urls", default=None, help="Override data URLs file from spec")
@click.pass_context
def fleet_run(
    ctx: click.Context,
    spec_file: Path,
    arms: str | None,
    api_key: str | None,
    ssh_key: str,
    results_file: str,
    dry_run: bool,
    data_urls: str | None,
) -> None:
    """Run an experiment from a YAML spec file.

    Example:

        ratiocinator fleet run experiment.yaml --arms 0,2,4
    """
    asyncio.run(
        _fleet_run(ctx, spec_file, arms, api_key, ssh_key, results_file, dry_run, data_urls)
    )


async def _fleet_run(
    ctx: click.Context,
    spec_file: Path,
    arms: str | None,
    api_key: str | None,
    ssh_key: str,
    results_file: str,
    dry_run: bool,
    data_urls: str | None,
) -> None:
    from ratiocinator.fleet.executor import FleetConfig, FleetExecutor, print_results_table
    from ratiocinator.fleet.spec import ExperimentSpec

    config = ctx.obj["config"]
    resolved_api_key = api_key or config.vast.api_key
    if not resolved_api_key:
        click.echo("Error: VAST_API_KEY not set. Add to .env, config, or use --api-key.", err=True)
        sys.exit(1)

    spec = ExperimentSpec.from_yaml(spec_file)
    click.echo(f"Experiment: {spec.name}")
    click.echo(f"  Arms: {len(spec.arms)}")
    click.echo(f"  Hardware: {spec.hardware.gpu} x {spec.hardware.num_gpus}")
    click.echo(f"  Image: {spec.hardware.image}")

    # Override data URLs from CLI if provided
    if data_urls:
        spec.data.source = "s3-presigned"
        spec.data.urls_file = data_urls

    arm_indices = None
    if arms:
        arm_indices = [int(x.strip()) for x in arms.split(",")]
        click.echo(f"  Selected arms: {arm_indices}")

    fleet_config = FleetConfig(
        api_key=resolved_api_key,
        ssh_key=ssh_key,
        results_path=results_file,
    )

    executor = FleetExecutor(spec, fleet_config)
    results = await executor.run(arm_indices, dry_run=dry_run)

    if results:
        print_results_table(results)
        click.echo(f"\nResults saved to {results_file}")


@fleet.command("status")
@click.option("--results-file", default=".ratiocinator/results/experiments.json")
@click.option("--experiment", default=None, help="Filter to a specific experiment")
def fleet_status(results_file: str, experiment: str | None) -> None:
    """Show results from previous fleet runs."""
    from ratiocinator.fleet.results import ResultStore

    store = ResultStore(results_file)
    experiments = [experiment] if experiment else store.experiments

    if not experiments:
        click.echo("No experiments found.")
        return

    for exp in experiments:
        results = store.get_experiment(exp)
        click.echo(f"\n{exp} ({len(results)} arms)")
        click.echo("-" * 60)
        for r in results:
            status = "✓" if r.get("exit_code") == 0 else "✗"
            arm = r.get("arm_name", r.get("arm", "?"))
            metrics = r.get("metrics", {})
            metric_str = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:3])
            click.echo(f"  {status} {arm}: {metric_str or r.get('error', 'no data')[:60]}")


@main.command("vast-run")
@click.option(
    "--repo-url",
    required=True,
    help="Git repo URL (must be publicly cloneable or SSH-accessible from instance)",
)
@click.option("--branch", default="main", help="Git branch to clone")
@click.option("--steps", default=500, help="Training steps")
@click.option("--command", default="python train.py", help="Training command")
@click.option("--image", default=None, help="Docker image (default from config)")
@click.option("--max-dph", default=None, type=float, help="Max $/hr (default from config)")
@click.option("--no-install", is_flag=True, help="Skip pip install (for stdlib-only scripts)")
@click.option("--label", default="ratiocinator-run", help="Instance label")
@click.pass_context
def vast_run(
    ctx: click.Context,
    repo_url: str,
    branch: str,
    steps: int,
    command: str,
    image: str | None,
    max_dph: float | None,
    no_install: bool,
    label: str,
) -> None:
    """Run an experiment on a Vast.ai GPU instance."""
    asyncio.run(
        _vast_run(ctx, repo_url, branch, steps, command, image, max_dph, no_install, label)
    )


async def _vast_run(
    ctx: click.Context,
    repo_url: str,
    branch: str,
    steps: int,
    command: str,
    image: str | None,
    max_dph: float | None,
    no_install: bool,
    label: str,
) -> None:
    import time

    from ratiocinator.infra.bootstrap import generate_onstart
    from ratiocinator.infra.safety import SafetyController
    from ratiocinator.infra.vast_client import VastClient

    config = ctx.obj["config"]
    api_key = config.vast.api_key
    if not api_key:
        click.echo("Error: VAST_API_KEY not set. Add to .env or config file.", err=True)
        sys.exit(1)

    image = image or config.vast.default_image
    max_dph = max_dph or config.vast.max_dph

    async with VastClient(api_key) as client:
        safety = SafetyController(config.safety, client)

        # Find cheapest GPU
        click.echo("Searching for GPU offers...")
        offers = await client.search_offers(max_dph=max_dph, limit=5)
        if not offers:
            click.echo(f"No offers found under ${max_dph:.2f}/hr", err=True)
            sys.exit(1)

        offer = offers[0]
        gpu_name = offer.get("gpu_name", "unknown")
        dph = offer.get("dph_total", 0)
        click.echo(f"  Selected: {gpu_name} @ ${dph:.3f}/hr")

        # Generate bootstrap
        onstart = generate_onstart(
            repo_url=repo_url,
            branch=branch,
            train_command=command,
            steps=steps,
            install_deps=not no_install,
        )

        # Launch
        click.echo("Launching instance...")
        iid = await client.create_instance(
            offer_id=offer["id"],
            image=image,
            onstart=onstart,
            label=label,
            disk_gb=config.vast.disk_gb,
        )
        safety.track(iid, dph)
        click.echo(f"  Instance: {iid}")

        # Poll for completion
        click.echo("Waiting for instance to boot...")
        start = time.time()
        try:
            for _ in range(36):  # 6 min max
                try:
                    info = await client.get_instance(iid)
                    elapsed = time.time() - start
                    if info.status.value == "running":
                        ssh = f"{info.ssh_host}:{info.ssh_port}"
                        click.echo(f"  Running after {elapsed:.0f}s (SSH: {ssh})")
                        break
                except Exception:
                    pass
                await asyncio.sleep(10)
            else:
                click.echo("  Timeout waiting for boot", err=True)
                await client.destroy_instance(iid)
                safety.untrack(iid)
                sys.exit(1)

            # Wait for training
            click.echo("Waiting for training (polling logs)...")
            await asyncio.sleep(20)

            metrics_line = None
            for attempt in range(18):  # 3 min of polling
                try:
                    log_url = await client.request_logs(iid)
                    await asyncio.sleep(3)
                    import httpx

                    async with httpx.AsyncClient() as http:
                        resp = await http.get(log_url)
                        if "METRICS:" in resp.text:
                            for line in resp.text.splitlines():
                                if line.startswith("METRICS:"):
                                    metrics_line = line
                                if any(kw in line for kw in [
                                    "Ratiocinator", "METRICS", "Training", "train", "exit code"
                                ]):
                                    click.echo(f"  {line}")
                            break
                except Exception:
                    pass
                click.echo(f"  [{attempt + 1}] Training in progress...")
                await asyncio.sleep(10)

            # Results
            elapsed = time.time() - start
            spend = safety.estimate_spend()
            click.echo()
            click.echo(f"Duration: {elapsed:.0f}s | Est. cost: ${spend:.4f}")
            if metrics_line:
                click.echo(f"Result: {metrics_line}")
            else:
                click.echo("Warning: METRICS line not found in logs", err=True)

        finally:
            click.echo(f"Destroying instance {iid}...")
            await client.destroy_instance(iid)
            safety.untrack(iid)
            click.echo("Done.")


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing paper and artifacts",
)
@click.option("--repo-id", default=None, help="HuggingFace repo (user/dataset)")
@click.option("--message", default="Ratiocinator experiment results", help="Commit message")
@click.pass_context
def publish(ctx: click.Context, output_dir: Path, repo_id: str | None, message: str) -> None:
    """Publish experiment artifacts to HuggingFace Hub."""
    from ratiocinator.synthesis.publisher import ArtifactPublisher, get_git_hash

    config = ctx.obj["config"]
    token = config.publish.hf_token
    if not token:
        click.echo("Error: HF_TOKEN not set. Add to .env or config file.", err=True)
        sys.exit(1)

    repo_id = repo_id or config.publish.repo_id
    if not repo_id:
        click.echo("Error: No repo_id. Use --repo-id or set HF_REPO_ID.", err=True)
        sys.exit(1)

    # Collect artifacts
    artifacts: dict[str, Path] = {}
    for pattern in ["*.tex", "*.json", "plots/*.png", "plots/*.pdf"]:
        for p in output_dir.glob(pattern):
            artifacts[p.name if "/" not in pattern else f"plots/{p.name}"] = p

    if not artifacts:
        click.echo(f"No artifacts found in {output_dir}", err=True)
        sys.exit(1)

    click.echo(f"Publishing {len(artifacts)} artifact(s) to {repo_id}...")
    for name in artifacts:
        click.echo(f"  {name}")

    publisher = ArtifactPublisher(repo_id, token=token)
    tags = {"git_hash": get_git_hash(), "pipeline": "ratiocinator"}
    url = publisher.publish(artifacts, commit_message=message, tags=tags)
    click.echo(f"Published: {url}")


if __name__ == "__main__":
    main()
