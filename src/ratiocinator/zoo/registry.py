"""YAML-backed dataset catalog for the model zoo.

Loads dataset entries from individual YAML files and supports
querying by modality, organ, and license.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ratiocinator.zoo.models import DatasetEntry

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Catalog of available medical imaging datasets.

    Reads ``DatasetEntry`` records from YAML files in a directory::

        registry = DatasetRegistry("zoo/datasets/")
        ct_datasets = registry.query(modality="ct")
        entry = registry.get("lidc-idri")
    """

    def __init__(self, catalog_dir: str | Path | None = None) -> None:
        self._entries: dict[str, DatasetEntry] = {}
        if catalog_dir is not None:
            self.load(catalog_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, catalog_dir: str | Path) -> int:
        """Load all YAML dataset entries from *catalog_dir*.

        Returns the number of entries loaded.
        """
        catalog_path = Path(catalog_dir)
        if not catalog_path.is_dir():
            raise FileNotFoundError(f"Catalog directory not found: {catalog_path}")

        count = 0
        for yaml_file in sorted(catalog_path.glob("*.yaml")):
            try:
                entry = self._load_entry(yaml_file)
                self._entries[entry.name] = entry
                count += 1
            except Exception:
                logger.exception("Failed to load dataset entry: %s", yaml_file)
        logger.info("Loaded %d dataset entries from %s", count, catalog_path)
        return count

    def register(self, entry: DatasetEntry) -> None:
        """Add a dataset entry programmatically."""
        self._entries[entry.name] = entry

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, name: str) -> DatasetEntry | None:
        """Look up a dataset by name."""
        return self._entries.get(name)

    def list_all(self) -> list[DatasetEntry]:
        """Return all registered datasets."""
        return list(self._entries.values())

    def query(
        self,
        *,
        modality: str | None = None,
        organs: list[str] | None = None,
        license_prefix: str | None = None,
    ) -> list[DatasetEntry]:
        """Filter datasets by modality, organs, or license.

        Args:
            modality: Filter to a specific modality (``ct``, ``mri``, ``xray``).
            organs: Filter to datasets containing *any* of these organs.
            license_prefix: Filter to licenses starting with this string
                (e.g., ``"CC"`` matches ``CC-BY-3.0``, ``CC-BY-4.0``, etc.).

        Returns:
            List of matching dataset entries.
        """
        results = list(self._entries.values())

        if modality is not None:
            results = [e for e in results if e.modality == modality]

        if organs is not None:
            organ_set = set(organs)
            results = [e for e in results if organ_set & set(e.organs)]

        if license_prefix is not None:
            results = [e for e in results if e.license.startswith(license_prefix)]

        return results

    @property
    def names(self) -> list[str]:
        """Sorted list of all dataset names."""
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_entry(path: Path) -> DatasetEntry:
        """Parse a single YAML file into a DatasetEntry."""
        data = yaml.safe_load(path.read_text())
        return DatasetEntry.model_validate(data)
