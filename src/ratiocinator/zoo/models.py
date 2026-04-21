"""Dataset registry models for the DINO-X model zoo.

Defines Pydantic models for cataloging medical imaging datasets,
per-slice metadata with physical spacing, and training lineage
for full provenance tracking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Dataset catalog entry — one per dataset, stored as YAML
# ---------------------------------------------------------------------------


class PreprocessingConfig(BaseModel):
    """How raw data has been converted to training-ready format."""

    format: Literal["png_16bit", "png_8bit", "npy", "nifti"] = "png_16bit"
    hu_shift: int = 32768
    scale: int = 10
    index_csv: str = ""


class DatasetEntry(BaseModel):
    """A single medical imaging dataset in the zoo catalog.

    Stored as individual YAML files in ``zoo/datasets/``.
    """

    name: str
    modality: Literal["ct", "mri", "xray"]
    organs: list[str]
    source_url: str = ""
    license: str = ""
    total_slices: int = 0
    total_series: int = 0
    pixel_spacing_range: tuple[float, float] = (0.0, 0.0)
    slice_thickness_range: tuple[float, float] = (0.0, 0.0)
    hu_range: tuple[int, int] = (-1024, 3071)
    annotations: list[str] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    citation: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-slice metadata — stored as Parquet for efficient querying
# ---------------------------------------------------------------------------


class SliceMetadata(BaseModel):
    """Physical metadata for a single image slice.

    These records are stored in Parquet files (one per dataset)
    and provide the spacing values fed to the ScaleEmbedding module.
    """

    dataset: str
    series_id: str
    slice_idx: int
    pixel_spacing_x: float
    pixel_spacing_y: float
    slice_thickness: float
    image_path: str
    organs_present: list[str] = Field(default_factory=list)
    patient_id: str | None = None
    study_date: str | None = None


# ---------------------------------------------------------------------------
# Dataset usage record — one per dataset in a training run
# ---------------------------------------------------------------------------


class DatasetUsage(BaseModel):
    """Records how a single dataset was used in a training run."""

    name: str
    slices_used: int
    weight: float
    pixel_spacing_min: float = 0.0
    pixel_spacing_max: float = 0.0
    slice_thickness_min: float = 0.0
    slice_thickness_max: float = 0.0


# ---------------------------------------------------------------------------
# Spacing statistics across the training corpus
# ---------------------------------------------------------------------------


class SpacingStats(BaseModel):
    """Aggregate spacing statistics for the training corpus."""

    pixel_spacing_x_min: float = 0.0
    pixel_spacing_x_max: float = 0.0
    pixel_spacing_x_mean: float = 0.0
    pixel_spacing_y_min: float = 0.0
    pixel_spacing_y_max: float = 0.0
    pixel_spacing_y_mean: float = 0.0
    slice_thickness_min: float = 0.0
    slice_thickness_max: float = 0.0
    slice_thickness_mean: float = 0.0


# ---------------------------------------------------------------------------
# Training lineage — full provenance record for a training run
# ---------------------------------------------------------------------------


class TrainingLineage(BaseModel):
    """Full provenance record for a model training run.

    Saved as ``lineage.json`` alongside the model checkpoint.
    Captures everything needed to reproduce or audit the training.
    """

    model_name: str
    architecture: str = "vit-small"
    modality: Literal["ct", "mri", "xray"] = "ct"
    datasets: list[DatasetUsage] = Field(default_factory=list)
    total_slices: int = 0
    spacing_stats: SpacingStats = Field(default_factory=SpacingStats)
    scale_aware: bool = False
    training_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    random_seed: int = 42
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    ratiocinator_version: str = ""
    training_code_commit: str = ""
    data_catalog_hash: str = ""

    def total_weight(self) -> float:
        """Sum of dataset weights (should be ~1.0)."""
        return sum(d.weight for d in self.datasets)
