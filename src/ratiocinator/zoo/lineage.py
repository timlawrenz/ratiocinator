"""Training data lineage tracking.

Records full provenance for every training run — which datasets
were used, their spacing distributions, sampling weights, and
all config needed for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from ratiocinator.zoo.models import (
    DatasetUsage,
    SpacingStats,
    TrainingLineage,
)

logger = logging.getLogger(__name__)


def compute_catalog_hash(catalog_dir: str | Path) -> str:
    """SHA-256 of all YAML files in the catalog directory.

    Captures the exact dataset definitions used at training time
    so that changes to the catalog can be detected later.
    """
    h = hashlib.sha256()
    catalog_path = Path(catalog_dir)
    if not catalog_path.is_dir():
        return "unknown"
    for yaml_file in sorted(catalog_path.glob("*.yaml")):
        h.update(yaml_file.read_bytes())
    return h.hexdigest()[:16]


def get_git_commit(repo_path: str | Path | None = None) -> str:
    """Return the current HEAD commit hash, or ``'unknown'``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record_lineage(
    *,
    model_name: str,
    architecture: str,
    modality: str,
    datasets: list[DatasetUsage],
    spacing_stats: SpacingStats,
    scale_aware: bool = False,
    training_config: dict | None = None,
    random_seed: int = 42,
    catalog_dir: str | Path | None = None,
    training_code_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> TrainingLineage:
    """Create and optionally save a training lineage record.

    Args:
        model_name: Name of the model being trained (e.g., "dinox-ct-vit-small-v1").
        architecture: Model architecture (e.g., "vit-small").
        modality: Training modality ("ct", "mri", "xray").
        datasets: List of DatasetUsage records.
        spacing_stats: Aggregate spacing statistics.
        scale_aware: Whether scale embedding was used.
        training_config: Dict of training hyperparameters.
        random_seed: Random seed used.
        catalog_dir: Path to dataset catalog (for hash).
        training_code_dir: Path to training code repo (for git hash).
        output_path: If provided, write lineage JSON to this path.

    Returns:
        The completed TrainingLineage record.
    """
    import ratiocinator

    lineage = TrainingLineage(
        model_name=model_name,
        architecture=architecture,
        modality=modality,
        datasets=datasets,
        total_slices=sum(d.slices_used for d in datasets),
        spacing_stats=spacing_stats,
        scale_aware=scale_aware,
        training_config=training_config or {},
        random_seed=random_seed,
        ratiocinator_version=getattr(ratiocinator, "__version__", "0.1.0"),
        training_code_commit=get_git_commit(training_code_dir),
        data_catalog_hash=(
            compute_catalog_hash(catalog_dir)
            if catalog_dir is not None
            else "unknown"
        ),
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(lineage.model_dump_json(indent=2))
        logger.info("Saved training lineage to %s", out)

    return lineage


def load_lineage(path: str | Path) -> TrainingLineage:
    """Load a training lineage record from JSON."""
    text = Path(path).read_text()
    data = json.loads(text)
    return TrainingLineage.model_validate(data)
