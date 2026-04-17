#!/usr/bin/env python3
"""Phase 6: LIDC-IDRI malignancy linear probe.

Evaluates SSL checkpoint quality by training a frozen-backbone linear classifier
on nodule malignancy (binary: benign vs malignant).

Protocol:
  1. Load checkpoint, freeze backbone
  2. Extract CLS token features for all labeled nodule slices
  3. Patient-level train/val/test split (60/20/20)
  4. Train nn.Linear(dim, 1) with weighted BCE loss
  5. Sweep LR in [0.001, 0.01, 0.1, 1.0]
  6. Report AUC-ROC, AUC-PR, balanced accuracy, F1

Usage:
  python scripts/phase6_malignancy_linear_probe.py \
    --checkpoint data/runs/.../checkpoint_final_00100000.pth \
    --labels data/processed/lidc-idri/_labels/nodule_labels.csv \
    --index-csv data/processed/lidc-idri/_index/index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase5_big_run import DinoStudentTeacher, PatchViT


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_hu01(path: Path, level: float = 40.0, width: float = 400.0) -> np.ndarray:
    """Load a 16-bit HU PNG and apply standard mediastinal windowing."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    hu = (arr - 32768.0) * 0.1
    wmin = level - width / 2.0
    windowed = (hu - wmin) / max(width, 1.0)
    return np.clip(windowed, 0.0, 1.0)


class NoduleDataset(torch.utils.data.Dataset):
    """Dataset that loads labeled nodule slices with 3-slice context."""

    def __init__(
        self,
        label_rows: list[dict],
        series_map: dict[str, dict[int, Path]],
        img_size: int = 224,
        hu_level: float = 40.0,
        hu_width: float = 400.0,
    ):
        self.rows = label_rows
        self.series_map = series_map
        self.img_size = img_size
        self.hu_level = hu_level
        self.hu_width = hu_width

        self.transform = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        series_uid = row["series_uid"]
        z = int(row["slice_index"])
        label = int(row["malignancy_binary"])

        mp = self.series_map.get(series_uid, {})
        z_keys = sorted(mp.keys()) if mp else [z]
        z0, z1 = z_keys[0], z_keys[-1]

        def clamp(k: int) -> int:
            return max(z0, min(z1, k))

        paths = [
            mp.get(clamp(z - 1), Path(row["png_path"])),
            mp.get(clamp(z), Path(row["png_path"])),
            mp.get(clamp(z + 1), Path(row["png_path"])),
        ]

        slices = [
            _load_hu01(p, self.hu_level, self.hu_width) for p in paths
        ]
        x = np.stack(slices, axis=0)  # (3, H, W)
        x = self.transform(torch.from_numpy(x).contiguous())
        return x, label


def patient_split(
    rows: list[dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by patient_id to prevent data leakage."""
    rng = random.Random(seed)
    patients = sorted(set(r["patient_id"] for r in rows))
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_pats = set(patients[:n_train])
    val_pats = set(patients[n_train:n_train + n_val])
    test_pats = set(patients[n_train + n_val:])

    train = [r for r in rows if r["patient_id"] in train_pats]
    val = [r for r in rows if r["patient_id"] in val_pats]
    test = [r for r in rows if r["patient_id"] in test_pats]
    return train, val, test


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: NoduleDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract CLS token features for all samples."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    all_feats = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats = backbone(x)
        cls_token = feats[:, 0]  # (B, dim)
        all_feats.append(cls_token.cpu())
        all_labels.append(y)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def train_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    dim: int,
    lr: float,
    epochs: int = 100,
    pos_weight: float = 1.0,
    device: str = "cpu",
) -> tuple[nn.Linear, dict]:
    """Train a linear probe and return best model + metrics."""
    clf = nn.Linear(dim, 1).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    pw = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_f = train_feats.to(device)
    train_y = train_labels.float().to(device)
    val_f = val_feats.to(device)
    val_y = val_labels.float().to(device)

    best_auc = 0.0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        clf.train()
        logits = clf(train_f).squeeze(-1)
        loss = loss_fn(logits, train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Validate
        clf.eval()
        with torch.no_grad():
            val_logits = clf(val_f).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_true = val_y.cpu().numpy()

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(val_true, val_probs)
        except (ImportError, ValueError):
            auc = 0.0

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        clf.load_state_dict(best_state)

    return clf, {"best_val_auc": best_auc, "final_epoch": epoch + 1, "lr": lr}


def evaluate(
    clf: nn.Linear,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """Full evaluation: AUC-ROC, AUC-PR, balanced accuracy, F1."""
    clf.eval()
    with torch.no_grad():
        logits = clf(feats.to(device)).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        true = labels.numpy()

    metrics = {}
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            balanced_accuracy_score, f1_score, confusion_matrix,
        )
        metrics["auc_roc"] = float(roc_auc_score(true, probs))
        metrics["auc_pr"] = float(average_precision_score(true, probs))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(true, preds))
        metrics["f1"] = float(f1_score(true, preds))
        tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tn"] = int(tn)
    except Exception as e:
        print(f"Warning: sklearn evaluation failed: {e}")
        metrics["auc_roc"] = 0.0

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="LIDC-IDRI malignancy linear probe")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--index-csv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="AUC threshold for pass/fail")
    ap.add_argument("--exclude-indeterminate", action="store_true",
                    help="Exclude malignancy=3 nodules for cleaner labels")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    _seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    model_cfg = (cfg.get("model") or {})

    img_size = int(model_cfg.get("img_size", 224))
    patch = int(model_cfg.get("patch", 14))
    dim = int(model_cfg.get("dim", 384))
    depth = int(model_cfg.get("depth", 12))
    heads = int(model_cfg.get("heads", 6))
    mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
    out_dim = int(model_cfg.get("out_dim", 8192))

    print(f"Model: dim={dim}, depth={depth}, heads={heads}, patch={patch}, img={img_size}")

    vit = PatchViT(
        img_size=img_size, patch=patch, dim=dim, depth=depth,
        heads=heads, mlp_ratio=mlp_ratio, use_grad_checkpoint=False,
    )
    student = DinoStudentTeacher(vit, out_dim=out_dim)
    student.load_state_dict(payload["student"], strict=True)
    student.to(device)
    student.eval()
    backbone = student.backbone
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Load labels
    print(f"Loading labels: {args.labels}")
    with open(args.labels) as f:
        label_rows = list(csv.DictReader(f))
    print(f"  Total nodules: {len(label_rows)}")

    if args.exclude_indeterminate:
        label_rows = [r for r in label_rows if float(r["malignancy_median"]) != 3.0]
        print(f"  After excluding indeterminate (3.0): {len(label_rows)}")

    n_mal = sum(1 for r in label_rows if int(r["malignancy_binary"]) == 1)
    n_ben = len(label_rows) - n_mal
    print(f"  Malignant: {n_mal}, Benign: {n_ben}")

    # Build series map from index CSV
    print(f"Building series map from: {args.index_csv}")
    series_map: dict[str, dict[int, Path]] = {}
    with open(args.index_csv) as f:
        for row in csv.DictReader(f):
            uid = row["series_dir"].split("/")[-1]
            if uid not in series_map:
                series_map[uid] = {}
            series_map[uid][int(row["slice_index"])] = Path(row["png_path"])
    print(f"  {len(series_map)} series loaded")

    # Patient-level split
    train_rows, val_rows, test_rows = patient_split(label_rows, seed=args.seed)
    print(f"Split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Create datasets
    train_ds = NoduleDataset(train_rows, series_map, img_size=img_size)
    val_ds = NoduleDataset(val_rows, series_map, img_size=img_size)
    test_ds = NoduleDataset(test_rows, series_map, img_size=img_size)

    # Extract features
    print("Extracting features...")
    t0 = time.time()
    train_feats, train_labels = extract_features(backbone, train_ds, device, args.batch_size)
    val_feats, val_labels = extract_features(backbone, val_ds, device, args.batch_size)
    test_feats, test_labels = extract_features(backbone, test_ds, device, args.batch_size)
    feat_time = time.time() - t0
    print(f"  Features extracted in {feat_time:.1f}s")
    print(f"  Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")

    # Compute class weight for imbalanced data
    pos_weight = float(n_ben) / max(float(n_mal), 1.0)
    print(f"  pos_weight (benign/malignant ratio): {pos_weight:.2f}")

    # LR sweep
    lrs = [0.001, 0.01, 0.1, 1.0]
    best_val_auc = 0.0
    best_clf = None
    best_lr_info = None

    print(f"\nLR sweep: {lrs}")
    for lr in lrs:
        clf, info = train_probe(
            train_feats, train_labels, val_feats, val_labels,
            dim=dim, lr=lr, epochs=args.epochs,
            pos_weight=pos_weight, device=str(device),
        )
        print(f"  lr={lr}: val_auc={info['best_val_auc']:.4f} (epoch {info['final_epoch']})")
        if info["best_val_auc"] > best_val_auc:
            best_val_auc = info["best_val_auc"]
            best_clf = clf
            best_lr_info = info

    print(f"\nBest LR: {best_lr_info['lr']}, Val AUC: {best_val_auc:.4f}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(best_clf, test_feats, test_labels, device=str(device))

    auc = test_metrics.get("auc_roc", 0.0)
    passed = auc >= args.threshold

    print(f"\n{'='*60}")
    print(f"MALIGNANCY LINEAR PROBE RESULTS")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Nodules: {len(label_rows)} (train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)})")
    print(f"Best LR: {best_lr_info['lr']}")
    print(f"")
    print(f"Test AUC-ROC:     {test_metrics.get('auc_roc', 0):.4f}")
    print(f"Test AUC-PR:      {test_metrics.get('auc_pr', 0):.4f}")
    print(f"Balanced Acc:     {test_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"F1:               {test_metrics.get('f1', 0):.4f}")
    print(f"Sensitivity:      {test_metrics.get('sensitivity', 0):.4f}")
    print(f"Specificity:      {test_metrics.get('specificity', 0):.4f}")
    print(f"Confusion: TP={test_metrics.get('tp',0)} FP={test_metrics.get('fp',0)} "
          f"FN={test_metrics.get('fn',0)} TN={test_metrics.get('tn',0)}")
    print(f"")
    print(f"Threshold: {args.threshold}")
    print(f"PASSED: {passed}")
    print(f"{'='*60}")

    # Save results
    out_dir = args.checkpoint.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"malignancy_probe_{ts}.json"

    results = {
        "kind": "malignancy_linear_probe",
        "version": 1,
        "checkpoint": str(args.checkpoint),
        "labels_file": str(args.labels),
        "exclude_indeterminate": args.exclude_indeterminate,
        "num_nodules": len(label_rows),
        "num_malignant": n_mal,
        "num_benign": n_ben,
        "split": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "best_lr": best_lr_info["lr"],
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "passed": passed,
        "threshold": args.threshold,
        "seed": args.seed,
        "feature_dim": dim,
        "feature_extraction_time_s": feat_time,
        "device": str(device),
    }
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to: {out_path}")

    # Emit METRICS line for fleet compatibility
    m = {
        "auc_roc": round(test_metrics.get("auc_roc", 0), 4),
        "auc_pr": round(test_metrics.get("auc_pr", 0), 4),
        "balanced_accuracy": round(test_metrics.get("balanced_accuracy", 0), 4),
        "f1": round(test_metrics.get("f1", 0), 4),
        "passed": passed,
    }
    print(f"METRICS:{json.dumps(m)}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
