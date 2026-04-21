"""Multi-dataset merging with weighted sampling.

Combines multiple ``DataManifest`` instances into a single training
set with configurable per-dataset weights, modality/organ filtering,
and spacing-stratified sampling.
"""

from __future__ import annotations

import logging
import random

from ratiocinator.zoo.manifest import DataManifest
from ratiocinator.zoo.models import DatasetUsage, SliceMetadata

logger = logging.getLogger(__name__)


class DatasetMerger:
    """Combines multiple dataset manifests for training.

    Example::

        merger = DatasetMerger()
        merger.add(lidc_manifest, weight=0.40)
        merger.add(ctorg_manifest, weight=0.25)
        merger.add(abdomen_manifest, weight=0.35)
        merged, usage = merger.build(seed=42, total_slices=500_000)
    """

    def __init__(self) -> None:
        self._sources: list[tuple[DataManifest, float]] = []

    def add(self, manifest: DataManifest, *, weight: float = 1.0) -> None:
        """Register a dataset manifest with a sampling weight.

        Weights are relative — they are normalized during ``build()``.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        self._sources.append((manifest, weight))

    def build(
        self,
        *,
        seed: int = 42,
        total_slices: int | None = None,
    ) -> tuple[DataManifest, list[DatasetUsage]]:
        """Merge all sources into a single manifest.

        Args:
            seed: Random seed for reproducible sampling.
            total_slices: If set, sample this many slices total
                (distributed according to weights). If ``None``,
                include all slices with epoch-level weighting metadata.

        Returns:
            Tuple of (merged manifest, list of DatasetUsage records).
        """
        if not self._sources:
            raise ValueError("No datasets added to merger")

        # Normalize weights
        total_weight = sum(w for _, w in self._sources)
        norm_weights = [(m, w / total_weight) for m, w in self._sources]

        rng = random.Random(seed)
        merged_records: list[SliceMetadata] = []
        usage_records: list[DatasetUsage] = []

        for manifest, weight in norm_weights:
            if total_slices is not None:
                # Sample a proportional number of slices
                n_target = max(1, int(total_slices * weight))
                n_actual = min(n_target, len(manifest))
                if n_actual < len(manifest):
                    selected = rng.sample(manifest.records, n_actual)
                else:
                    selected = list(manifest.records)
            else:
                # Include all slices
                selected = list(manifest.records)

            merged_records.extend(selected)

            # Compute usage stats
            stats = DataManifest(selected).spacing_stats()
            datasets = manifest.datasets()
            dataset_name = datasets[0] if len(datasets) == 1 else "+".join(datasets)

            usage_records.append(
                DatasetUsage(
                    name=dataset_name,
                    slices_used=len(selected),
                    weight=weight,
                    pixel_spacing_min=stats.pixel_spacing_x_min,
                    pixel_spacing_max=stats.pixel_spacing_x_max,
                    slice_thickness_min=stats.slice_thickness_min,
                    slice_thickness_max=stats.slice_thickness_max,
                )
            )

        # Shuffle the merged records
        rng.shuffle(merged_records)

        logger.info(
            "Merged %d datasets → %d slices (requested %s)",
            len(self._sources),
            len(merged_records),
            total_slices or "all",
        )

        return DataManifest(merged_records), usage_records
