"""Tests for the zoo data registry models."""

from __future__ import annotations

import json

import pytest
import yaml

from ratiocinator.zoo.models import (
    DatasetEntry,
    DatasetUsage,
    PreprocessingConfig,
    SliceMetadata,
    SpacingStats,
    TrainingLineage,
)

# ── DatasetEntry ──────────────────────────────────────────────────────


class TestDatasetEntry:
    def test_minimal_construction(self):
        entry = DatasetEntry(name="test-dataset", modality="ct", organs=["lung"])
        assert entry.name == "test-dataset"
        assert entry.modality == "ct"
        assert entry.organs == ["lung"]
        assert entry.total_slices == 0

    def test_full_construction(self):
        entry = DatasetEntry(
            name="lidc-idri",
            modality="ct",
            organs=["lung"],
            source_url="https://example.com",
            license="CC-BY-3.0",
            total_slices=234943,
            total_series=981,
            pixel_spacing_range=(0.461, 0.977),
            slice_thickness_range=(0.625, 5.0),
            hu_range=(-1024, 3071),
            annotations=["nodule_bounding_box", "malignancy_score"],
            citation="Armato et al., 2011",
        )
        assert entry.total_slices == 234943
        assert entry.pixel_spacing_range == (0.461, 0.977)

    def test_yaml_round_trip(self, tmp_path):
        entry = DatasetEntry(
            name="lidc-idri",
            modality="ct",
            organs=["lung"],
            total_slices=234943,
            preprocessing=PreprocessingConfig(format="png_16bit", hu_shift=32768),
        )
        yaml_path = tmp_path / "entry.yaml"
        yaml_path.write_text(yaml.dump(entry.model_dump(mode="json")))

        loaded = DatasetEntry.model_validate(yaml.safe_load(yaml_path.read_text()))
        assert loaded.name == "lidc-idri"
        assert loaded.preprocessing.format == "png_16bit"
        assert loaded.total_slices == 234943

    def test_json_round_trip(self):
        entry = DatasetEntry(name="test", modality="mri", organs=["brain"])
        data = json.loads(entry.model_dump_json())
        restored = DatasetEntry.model_validate(data)
        assert restored.name == entry.name
        assert restored.modality == entry.modality

    def test_modality_validation(self):
        with pytest.raises(ValueError, match="Input should be 'ct', 'mri' or 'xray'"):
            DatasetEntry(name="bad", modality="ultrasound", organs=["heart"])


# ── SliceMetadata ─────────────────────────────────────────────────────


class TestSliceMetadata:
    def test_construction(self):
        meta = SliceMetadata(
            dataset="lidc-idri",
            series_id="LIDC-IDRI-0001",
            slice_idx=42,
            pixel_spacing_x=0.703,
            pixel_spacing_y=0.703,
            slice_thickness=1.25,
            image_path="data/processed/lidc-idri/LIDC-IDRI-0001/042.png",
        )
        assert meta.pixel_spacing_x == 0.703
        assert meta.slice_thickness == 1.25
        assert meta.patient_id is None

    def test_optional_fields(self):
        meta = SliceMetadata(
            dataset="test",
            series_id="S001",
            slice_idx=0,
            pixel_spacing_x=1.0,
            pixel_spacing_y=1.0,
            slice_thickness=1.0,
            image_path="test.png",
            organs_present=["lung", "heart"],
            patient_id="P001",
            study_date="2024-01-15",
        )
        assert meta.organs_present == ["lung", "heart"]
        assert meta.patient_id == "P001"

    def test_dict_round_trip(self):
        meta = SliceMetadata(
            dataset="test",
            series_id="S001",
            slice_idx=0,
            pixel_spacing_x=0.5,
            pixel_spacing_y=0.5,
            slice_thickness=2.0,
            image_path="test.png",
        )
        d = meta.model_dump()
        restored = SliceMetadata.model_validate(d)
        assert restored.pixel_spacing_x == 0.5


# ── TrainingLineage ───────────────────────────────────────────────────


class TestTrainingLineage:
    def test_minimal(self):
        lineage = TrainingLineage(model_name="dinox-ct-vit-small-v1")
        assert lineage.model_name == "dinox-ct-vit-small-v1"
        assert lineage.architecture == "vit-small"
        assert lineage.total_slices == 0
        assert lineage.timestamp  # Auto-generated

    def test_with_datasets(self):
        lineage = TrainingLineage(
            model_name="dinox-ct-vit-small-v1",
            datasets=[
                DatasetUsage(name="lidc-idri", slices_used=234943, weight=0.4),
                DatasetUsage(name="ct-org", slices_used=140000, weight=0.6),
            ],
            total_slices=374943,
            scale_aware=True,
        )
        assert len(lineage.datasets) == 2
        assert lineage.total_weight() == pytest.approx(1.0)
        assert lineage.scale_aware is True

    def test_json_round_trip(self, tmp_path):
        lineage = TrainingLineage(
            model_name="test",
            datasets=[
                DatasetUsage(name="ds1", slices_used=100, weight=1.0),
            ],
            spacing_stats=SpacingStats(
                pixel_spacing_x_min=0.5,
                pixel_spacing_x_max=1.0,
                pixel_spacing_x_mean=0.75,
            ),
            training_config={"lr": 2e-4, "batch_size": 64, "cm": 0.999},
        )
        json_path = tmp_path / "lineage.json"
        json_path.write_text(lineage.model_dump_json(indent=2))

        loaded = TrainingLineage.model_validate(
            json.loads(json_path.read_text())
        )
        assert loaded.model_name == "test"
        assert loaded.spacing_stats.pixel_spacing_x_min == 0.5
        assert loaded.training_config["lr"] == 2e-4
