#!/usr/bin/env python3
"""MVP data preparation: build manifests and record lineage for two-organ training.

Creates DataManifest Parquet files from processed index CSVs,
merges them with weighted sampling, and records full training lineage.

Usage:
  python scripts/mvp_prepare_data.py \
    --lidc-index .../lidc-idri/_index/index_with_spacing.csv \
    --pancreas-index .../pancreas-ct/_index/index.csv \
    --out-dir .ratiocinator/mvp \
    --catalog-dir src/ratiocinator/zoo/datasets
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure the ratiocinator package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratiocinator.zoo.lineage import record_lineage
from ratiocinator.zoo.manifest import DataManifest
from ratiocinator.zoo.merge import DatasetMerger
from ratiocinator.zoo.models import SliceMetadata


def index_csv_to_manifest(
    csv_path: Path,
    dataset_name: str,
    organs: list[str],
) -> DataManifest:
    """Convert a processed index CSV into a DataManifest."""
    records: list[SliceMetadata] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SliceMetadata(
                    dataset=row.get("dataset", dataset_name),
                    series_id=row["series_dir"],
                    slice_idx=int(row["slice_index"]),
                    pixel_spacing_x=float(row.get("spacing_x", 1.0)),
                    pixel_spacing_y=float(row.get("spacing_y", 1.0)),
                    slice_thickness=float(row.get("spacing_z", 1.0)),
                    image_path=row["png_path"],
                    organs_present=organs,
                )
            )

    return DataManifest(records)


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP data preparation")
    ap.add_argument("--lidc-index", type=Path, required=True,
                    help="LIDC-IDRI index CSV (with spacing columns)")
    ap.add_argument("--pancreas-index", type=Path, required=True,
                    help="Pancreas-CT index CSV (with spacing columns)")
    ap.add_argument("--out-dir", type=Path, default=Path(".ratiocinator/mvp"),
                    help="Output directory for manifests and lineage")
    ap.add_argument("--catalog-dir", type=Path,
                    default=Path("src/ratiocinator/zoo/datasets"),
                    help="Dataset catalog directory for lineage hashing")
    ap.add_argument("--dino-x-dir", type=Path, default=None,
                    help="Path to DINO-X repo for git commit tracking")
    ap.add_argument("--lidc-weight", type=float, default=0.55,
                    help="Sampling weight for LIDC-IDRI (default: 0.55)")
    ap.add_argument("--pancreas-weight", type=float, default=0.45,
                    help="Sampling weight for Pancreas-CT (default: 0.45)")
    ap.add_argument("--total-slices", type=int, default=0,
                    help="Total slices for weighted sampling (0 = use all)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build per-dataset manifests ---
    print("Building LIDC-IDRI manifest ...")
    lidc = index_csv_to_manifest(args.lidc_index, "lidc-idri", ["lung"])
    print(f"  {len(lidc)} slices, spacing stats:")
    ls = lidc.spacing_stats()
    print(f"    px_x: {ls.pixel_spacing_x_min:.4f}-{ls.pixel_spacing_x_max:.4f}")
    print(f"    px_y: {ls.pixel_spacing_y_min:.4f}-{ls.pixel_spacing_y_max:.4f}")
    print(f"    st:   {ls.slice_thickness_min:.4f}-{ls.slice_thickness_max:.4f}")

    lidc.save(out_dir / "lidc_idri_manifest.parquet")

    print("\nBuilding Pancreas-CT manifest ...")
    pancreas = index_csv_to_manifest(
        args.pancreas_index, "pancreas-ct", ["pancreas", "abdomen"],
    )
    print(f"  {len(pancreas)} slices, spacing stats:")
    ps = pancreas.spacing_stats()
    print(f"    px_x: {ps.pixel_spacing_x_min:.4f}-{ps.pixel_spacing_x_max:.4f}")
    print(f"    px_y: {ps.pixel_spacing_y_min:.4f}-{ps.pixel_spacing_y_max:.4f}")
    print(f"    st:   {ps.slice_thickness_min:.4f}-{ps.slice_thickness_max:.4f}")

    pancreas.save(out_dir / "pancreas_ct_manifest.parquet")

    # --- Merge datasets ---
    print(f"\nMerging with weights: LIDC={args.lidc_weight}, Pancreas={args.pancreas_weight}")
    merger = DatasetMerger()
    merger.add(lidc, weight=args.lidc_weight)
    merger.add(pancreas, weight=args.pancreas_weight)
    merged, usage = merger.build(
        seed=42,
        total_slices=args.total_slices if args.total_slices > 0 else None,
    )

    merged.save(out_dir / "merged_manifest.parquet")
    print(f"Merged manifest: {len(merged)} slices")
    for u in usage:
        print(f"  {u.name}: {u.slices_used} slices (weight={u.weight:.2f})")

    # --- Record lineage ---
    print("\nRecording training lineage ...")
    lineage = record_lineage(
        model_name="dinox-ct-vit-small-mvp",
        architecture="vit-small",
        modality="ct",
        datasets=usage,
        spacing_stats=merged.spacing_stats(),
        scale_aware=True,
        training_config={
            "max_steps": 5000,
            "batch_size": 64,
            "lr": 2e-4,
            "warmup_steps": 500,
            "loss": "dino+gram(1.0)+koleo(0.1)",
            "center_momentum": 0.999,
            "augmentation": "spatial-only",
        },
        random_seed=42,
        catalog_dir=str(args.catalog_dir),
        training_code_dir=str(args.dino_x_dir) if args.dino_x_dir else None,
        output_path=out_dir / "lineage.json",
    )

    print("\nLineage recorded:")
    print(f"  Model: {lineage.model_name}")
    print(f"  Total slices: {lineage.total_slices}")
    print(f"  Scale aware: {lineage.scale_aware}")
    print(f"  Catalog hash: {lineage.data_catalog_hash}")
    print(f"  Code commit: {lineage.training_code_commit}")

    print(f"\nAll outputs in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
