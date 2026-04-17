#!/usr/bin/env python3
"""Extract LIDC-IDRI nodule malignancy labels from pylidc's built-in SQLite database.

Maps nodule Z-coordinates to preprocessed PNG slice indices using spatial clustering
to group annotations from different radiologists into physical nodules.

Requires: pylidc (pip install pylidc), which ships with a pre-built annotation DB.
No DICOM files needed.

Output: data/processed/lidc-idri/_labels/nodule_labels.csv
"""
from __future__ import annotations

import csv
import os
import sqlite3
from collections import defaultdict

import numpy as np


def main() -> None:
    db_path = "/opt/conda/lib/python3.11/site-packages/pylidc/pylidc.sqlite"
    index_csv = "data/processed/lidc-idri/_index/index.csv"
    out_path = "data/processed/lidc-idri/_labels/nodule_labels.csv"

    conn = sqlite3.connect(db_path)

    # Load all data upfront to avoid cursor interference
    series_info: dict[str, dict[int, str]] = {}
    with open(index_csv) as f:
        for row in csv.DictReader(f):
            uid = row["series_dir"].split("/")[-1]
            if uid not in series_info:
                series_info[uid] = {}
            series_info[uid][int(row["slice_index"])] = row["png_path"]

    scans = conn.execute("SELECT id, series_instance_uid, patient_id FROM scans").fetchall()
    scan_map = {s[0]: (s[1], s[2]) for s in scans}

    all_zvals = conn.execute("SELECT scan_id, val FROM zvals ORDER BY scan_id, val").fetchall()
    zvals_by_scan: dict[int, list[float]] = defaultdict(list)
    for scan_id, val in all_zvals:
        zvals_by_scan[scan_id].append(val)

    all_contours = conn.execute(
        "SELECT annotation_id, image_z_position FROM contours WHERE inclusion = 1"
    ).fetchall()
    contour_z_by_ann: dict[int, list[float]] = defaultdict(list)
    for ann_id, z_pos in all_contours:
        contour_z_by_ann[ann_id].append(z_pos)

    all_annotations = conn.execute(
        "SELECT id, scan_id, _nodule_id, malignancy, subtlety, texture FROM annotations"
    ).fetchall()
    conn.close()

    # Build annotation data with center Z
    anns_by_scan: dict[int, list[dict]] = defaultdict(list)
    for ann_id, scan_id, nod_id, mal, sub, tex in all_annotations:
        z_vals = contour_z_by_ann.get(ann_id)
        if not z_vals:
            continue
        anns_by_scan[scan_id].append({
            "ann_id": ann_id,
            "malignancy": mal,
            "center_z": float(np.median(z_vals)),
        })

    # Cluster into physical nodules (10mm threshold)
    label_rows = []
    for scan_id, ann_list in anns_by_scan.items():
        if scan_id not in scan_map:
            continue
        series_uid, patient_id = scan_map[scan_id]
        if series_uid not in series_info:
            continue
        zvals = zvals_by_scan.get(scan_id)
        if not zvals:
            continue
        zvals_arr = np.array(zvals)

        ann_list.sort(key=lambda x: x["center_z"])
        clusters: list[list[dict]] = []
        used: set[int] = set()
        for i, data_i in enumerate(ann_list):
            if i in used:
                continue
            cluster = [data_i]
            used.add(i)
            for j, data_j in enumerate(ann_list):
                if j in used:
                    continue
                if abs(data_j["center_z"] - data_i["center_z"]) <= 10.0:
                    cluster.append(data_j)
                    used.add(j)
            clusters.append(cluster)

        for nod_idx, cluster in enumerate(clusters):
            mals = [d["malignancy"] for d in cluster]
            mal_median = float(np.median(mals))
            center_z = float(np.median([d["center_z"] for d in cluster]))
            slice_idx = int(np.argmin(np.abs(zvals_arr - center_z)))

            if slice_idx not in series_info[series_uid]:
                continue

            label_rows.append({
                "png_path": series_info[series_uid][slice_idx],
                "series_uid": series_uid,
                "slice_index": slice_idx,
                "patient_id": patient_id,
                "nodule_id": f"{patient_id}_nod{nod_idx}",
                "num_annotators": len(cluster),
                "malignancy_scores": str(mals),
                "malignancy_median": round(mal_median, 2),
                "malignancy_binary": 1 if mal_median > 3 else 0,
                "center_z": round(center_z, 2),
            })

    label_rows.sort(key=lambda r: (r["patient_id"], r["nodule_id"]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
        writer.writeheader()
        writer.writerows(label_rows)

    n_mal = sum(1 for r in label_rows if r["malignancy_binary"] == 1)
    print(f"Extracted {len(label_rows)} nodules ({n_mal} malignant, {len(label_rows)-n_mal} benign)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
