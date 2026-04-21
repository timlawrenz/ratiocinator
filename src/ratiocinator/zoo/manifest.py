"""Per-slice metadata manifest backed by Parquet or in-memory lists.

Each dataset produces a Parquet manifest containing per-slice
physical spacing metadata. Manifests can be loaded, queried,
filtered, and combined for multi-dataset training.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from ratiocinator.zoo.models import SliceMetadata, SpacingStats

logger = logging.getLogger(__name__)


class DataManifest:
    """Per-slice metadata store for a single dataset.

    Supports loading from/saving to Parquet or constructing in memory.
    Provides filtering, sampling, and statistics.
    """

    def __init__(self, records: list[SliceMetadata] | None = None) -> None:
        self._records: list[SliceMetadata] = records or []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write manifest to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet manifests. "
                "Install with: pip install pyarrow"
            ) from e

        rows = [r.model_dump() for r in self._records]
        if not rows:
            logger.warning("Saving empty manifest to %s", path)
            # Create empty table with correct schema
            table = pa.table({
                "dataset": pa.array([], type=pa.string()),
                "series_id": pa.array([], type=pa.string()),
                "slice_idx": pa.array([], type=pa.int64()),
                "pixel_spacing_x": pa.array([], type=pa.float64()),
                "pixel_spacing_y": pa.array([], type=pa.float64()),
                "slice_thickness": pa.array([], type=pa.float64()),
                "image_path": pa.array([], type=pa.string()),
                "organs_present": pa.array([], type=pa.list_(pa.string())),
                "patient_id": pa.array([], type=pa.string()),
                "study_date": pa.array([], type=pa.string()),
            })
        else:
            table = pa.Table.from_pylist(rows)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        logger.info("Saved manifest with %d records to %s", len(rows), out)

    @classmethod
    def load(cls, path: str | Path) -> DataManifest:
        """Load manifest from a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet manifests. "
                "Install with: pip install pyarrow"
            ) from e

        table = pq.read_table(Path(path))
        rows = table.to_pylist()
        records = [SliceMetadata.model_validate(row) for row in rows]
        logger.info("Loaded manifest with %d records from %s", len(records), path)
        return cls(records)

    # ------------------------------------------------------------------
    # Query and filter
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[SliceMetadata]:
        """All slice metadata records."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def filter(
        self,
        *,
        dataset: str | None = None,
        organs: list[str] | None = None,
    ) -> DataManifest:
        """Return a new manifest with only matching records."""
        records = self._records

        if dataset is not None:
            records = [r for r in records if r.dataset == dataset]

        if organs is not None:
            organ_set = set(organs)
            records = [r for r in records if organ_set & set(r.organs_present)]

        return DataManifest(records)

    def sample(
        self,
        n: int,
        *,
        seed: int = 42,
        strategy: str = "uniform",
    ) -> DataManifest:
        """Sample *n* records from the manifest.

        Args:
            n: Number of records to sample.
            seed: Random seed for reproducibility.
            strategy: ``"uniform"`` for uniform random sampling,
                ``"spacing-stratified"`` for stratified by spacing bins.

        Returns:
            A new DataManifest with *n* sampled records.
        """
        import random

        rng = random.Random(seed)

        if strategy == "uniform":
            sampled = rng.sample(self._records, min(n, len(self._records)))
        elif strategy == "spacing-stratified":
            sampled = self._spacing_stratified_sample(n, rng)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return DataManifest(sampled)

    def spacing_stats(self) -> SpacingStats:
        """Compute aggregate spacing statistics."""
        if not self._records:
            return SpacingStats()

        px_x = [r.pixel_spacing_x for r in self._records]
        px_y = [r.pixel_spacing_y for r in self._records]
        st = [r.slice_thickness for r in self._records]

        return SpacingStats(
            pixel_spacing_x_min=min(px_x),
            pixel_spacing_x_max=max(px_x),
            pixel_spacing_x_mean=sum(px_x) / len(px_x),
            pixel_spacing_y_min=min(px_y),
            pixel_spacing_y_max=max(px_y),
            pixel_spacing_y_mean=sum(px_y) / len(px_y),
            slice_thickness_min=min(st),
            slice_thickness_max=max(st),
            slice_thickness_mean=sum(st) / len(st),
        )

    def datasets(self) -> list[str]:
        """Unique dataset names in this manifest."""
        return sorted({r.dataset for r in self._records})

    def add(self, record: SliceMetadata) -> None:
        """Append a single record."""
        self._records.append(record)

    def extend(self, records: list[SliceMetadata]) -> None:
        """Append multiple records."""
        self._records.extend(records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spacing_stratified_sample(
        self, n: int, rng: random.Random
    ) -> list[SliceMetadata]:
        """Stratify by pixel_spacing_x into 10 equal-width bins."""

        if not self._records:
            return []

        px_x = [r.pixel_spacing_x for r in self._records]
        min_px, max_px = min(px_x), max(px_x)

        if min_px == max_px:
            return rng.sample(self._records, min(n, len(self._records)))

        n_bins = 10
        bin_width = (max_px - min_px) / n_bins
        bins: list[list[SliceMetadata]] = [[] for _ in range(n_bins)]

        for record in self._records:
            idx = min(
                int((record.pixel_spacing_x - min_px) / bin_width),
                n_bins - 1,
            )
            bins[idx].append(record)

        # Sample proportionally from each bin
        non_empty = [b for b in bins if b]
        per_bin = max(1, n // len(non_empty))
        sampled: list[SliceMetadata] = []
        for b in non_empty:
            sampled.extend(rng.sample(b, min(per_bin, len(b))))

        # Top up if needed
        if len(sampled) < n:
            remaining = [r for r in self._records if r not in sampled]
            sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))

        return sampled[:n]
