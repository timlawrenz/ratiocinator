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


if __name__ == "__main__":
    main()
