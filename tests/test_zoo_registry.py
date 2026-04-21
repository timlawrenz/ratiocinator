"""Tests for the dataset registry."""

from __future__ import annotations

import pytest
import yaml

from ratiocinator.zoo.models import DatasetEntry
from ratiocinator.zoo.registry import DatasetRegistry


@pytest.fixture
def catalog_dir(tmp_path):
    """Create a temp catalog with three datasets."""
    d = tmp_path / "datasets"
    d.mkdir()

    (d / "lidc_idri.yaml").write_text(
        yaml.dump({
            "name": "lidc-idri",
            "modality": "ct",
            "organs": ["lung"],
            "license": "CC-BY-3.0",
            "total_slices": 234943,
            "total_series": 981,
            "pixel_spacing_range": [0.461, 0.977],
            "slice_thickness_range": [0.625, 5.0],
            "citation": "Armato et al., 2011",
        })
    )

    (d / "ct_org.yaml").write_text(
        yaml.dump({
            "name": "ct-org",
            "modality": "ct",
            "organs": ["liver", "lung", "bladder", "kidney", "bone"],
            "license": "CC-BY-4.0",
            "total_slices": 140000,
            "total_series": 100,
        })
    )

    (d / "brats.yaml").write_text(
        yaml.dump({
            "name": "brats",
            "modality": "mri",
            "organs": ["brain"],
            "license": "CC-BY-SA-4.0",
            "total_slices": 500000,
            "total_series": 2000,
        })
    )

    return d


class TestDatasetRegistry:
    def test_load(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        assert len(registry) == 3
        assert "lidc-idri" in registry
        assert "brats" in registry

    def test_get(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        entry = registry.get("lidc-idri")
        assert entry is not None
        assert entry.total_slices == 234943

    def test_get_missing(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        assert registry.get("nonexistent") is None

    def test_query_modality(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        ct = registry.query(modality="ct")
        assert len(ct) == 2
        mri = registry.query(modality="mri")
        assert len(mri) == 1
        assert mri[0].name == "brats"

    def test_query_organs(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        lung = registry.query(organs=["lung"])
        assert len(lung) == 2  # lidc-idri and ct-org both have lung

        brain = registry.query(organs=["brain"])
        assert len(brain) == 1

        kidney = registry.query(organs=["kidney"])
        assert len(kidney) == 1
        assert kidney[0].name == "ct-org"

    def test_query_license(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        cc = registry.query(license_prefix="CC")
        assert len(cc) == 3

        cc_by = registry.query(license_prefix="CC-BY-4")
        assert len(cc_by) == 1

    def test_query_combined(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        results = registry.query(modality="ct", organs=["lung"])
        assert len(results) == 2

        results = registry.query(modality="mri", organs=["lung"])
        assert len(results) == 0

    def test_names(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        assert registry.names == ["brats", "ct-org", "lidc-idri"]

    def test_list_all(self, catalog_dir):
        registry = DatasetRegistry(catalog_dir)
        assert len(registry.list_all()) == 3

    def test_register_programmatic(self):
        registry = DatasetRegistry()
        entry = DatasetEntry(name="custom", modality="xray", organs=["chest"])
        registry.register(entry)
        assert len(registry) == 1
        assert registry.get("custom") is not None

    def test_empty_catalog(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        registry = DatasetRegistry(d)
        assert len(registry) == 0

    def test_missing_catalog(self):
        with pytest.raises(FileNotFoundError):
            DatasetRegistry("/nonexistent/path")
