"""Tests for training lineage tracking."""

from __future__ import annotations

import json

from ratiocinator.zoo.lineage import (
    compute_catalog_hash,
    load_lineage,
    record_lineage,
)
from ratiocinator.zoo.models import DatasetUsage, SpacingStats


class TestComputeCatalogHash:
    def test_hash_is_deterministic(self, tmp_path):
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "a.yaml").write_text("name: a\n")
        (d / "b.yaml").write_text("name: b\n")
        h1 = compute_catalog_hash(d)
        h2 = compute_catalog_hash(d)
        assert h1 == h2
        assert len(h1) == 16  # Truncated hex

    def test_hash_changes_on_content_change(self, tmp_path):
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "a.yaml").write_text("name: a\n")
        h1 = compute_catalog_hash(d)
        (d / "a.yaml").write_text("name: a_modified\n")
        h2 = compute_catalog_hash(d)
        assert h1 != h2

    def test_missing_dir_returns_unknown(self):
        assert compute_catalog_hash("/nonexistent/path") == "unknown"


class TestRecordLineage:
    def test_basic_record(self):
        usage = [DatasetUsage(name="test-ds", slices_used=1000, weight=1.0)]
        stats = SpacingStats(pixel_spacing_x_min=0.5, pixel_spacing_x_max=1.0)
        lineage = record_lineage(
            model_name="test-model",
            architecture="vit-small",
            modality="ct",
            datasets=usage,
            spacing_stats=stats,
            random_seed=42,
        )
        assert lineage.model_name == "test-model"
        assert lineage.total_slices == 1000
        assert lineage.random_seed == 42
        assert lineage.timestamp  # Auto-generated

    def test_save_to_file(self, tmp_path):
        usage = [DatasetUsage(name="ds", slices_used=500, weight=1.0)]
        stats = SpacingStats()
        output = tmp_path / "lineage.json"

        record_lineage(
            model_name="test",
            architecture="vit-small",
            modality="ct",
            datasets=usage,
            spacing_stats=stats,
            output_path=output,
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["model_name"] == "test"
        assert data["datasets"][0]["name"] == "ds"

    def test_load_lineage(self, tmp_path):
        usage = [DatasetUsage(name="ds", slices_used=500, weight=1.0)]
        stats = SpacingStats()
        output = tmp_path / "lineage.json"

        record_lineage(
            model_name="round-trip",
            architecture="vit-large",
            modality="mri",
            datasets=usage,
            spacing_stats=stats,
            output_path=output,
        )

        loaded = load_lineage(output)
        assert loaded.model_name == "round-trip"
        assert loaded.architecture == "vit-large"
        assert loaded.modality == "mri"

    def test_catalog_hash_included(self, tmp_path):
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        (catalog / "ds.yaml").write_text("name: ds\n")

        lineage = record_lineage(
            model_name="test",
            architecture="vit-small",
            modality="ct",
            datasets=[],
            spacing_stats=SpacingStats(),
            catalog_dir=catalog,
        )
        assert lineage.data_catalog_hash != "unknown"
        assert len(lineage.data_catalog_hash) == 16
