#!/usr/bin/env python3
"""
split_by_match.py

Enforces STRICT match-level train / val / test splitting.
Every frame from a given match_id is assigned entirely into one split.
NO MATCH IS EVER DIVIDED ACROSS SPLITS.

Inputs:
  - Staging images & labels (from merge_datasets.py)
  - data/manifest.csv

Outputs:
  - data/labeled/images/{train, val, test}
  - data/labeled/labels/{train, val, test}
  - Updated data/manifest.csv
"""

import os
import sys
import csv
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Strict match-level dataset splitting.")
    parser.add_argument("--staging-dir", type=str, default="data/staging", help="Staging folder containing merged data")
    parser.add_argument("--output-dir", type=str, default="data/labeled", help="Final YOLO dataset output folder")
    parser.add_argument("--manifest-path", type=str, default="data/manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible match splitting")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying to save disk space")
    return parser.parse_args()


def main():
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        print(f"[ERROR] Ratios must sum to 1.0. Current sum: {total_ratio}")
        sys.exit(1)

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        print("Run scripts/merge_datasets.py first!")
        sys.exit(1)

    staging_dir = Path(args.staging_dir)
    staging_images = staging_dir / "images"
    staging_labels = staging_dir / "labels"

    output_dir = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Read manifest
    rows = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("[!] Manifest is empty. No images to split.")
        sys.exit(1)

    # Group records by match_id
    matches = defaultdict(list)
    for row in rows:
        match_id = row.get("match_id", "unknown_match")
        matches[match_id].append(row)

    unique_match_ids = sorted(list(matches.keys()))
    print(f"[*] Found {len(unique_match_ids)} unique matches across {len(rows)} total images.")

    # Reproducible shuffle of matches
    random.seed(args.seed)
    random.shuffle(unique_match_ids)

    # Calculate match count per split with strict boundary guards
    num_matches = len(unique_match_ids)
    if num_matches == 1:
        train_matches = set(unique_match_ids)
        val_matches = set()
        test_matches = set()
    elif num_matches == 2:
        train_matches = {unique_match_ids[0]}
        val_matches = {unique_match_ids[1]}
        test_matches = set()
    else:
        n_train = max(1, min(num_matches - 2, int(round(num_matches * args.train_ratio))))
        remaining = num_matches - n_train
        n_val = max(1, min(remaining - 1, int(round(num_matches * args.val_ratio))))
        train_matches = set(unique_match_ids[:n_train])
        val_matches = set(unique_match_ids[n_train:n_train + n_val])
        test_matches = set(unique_match_ids[n_train + n_val:])

    print(f"[*] Match partition: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")

    # Move or copy files into respective split folders
    transfer_func = shutil.move if args.move else shutil.copy2
    action_verb = "Moving" if args.move else "Copying"

    split_counts = defaultdict(int)

    for row in rows:
        m_id = row["match_id"]
        if m_id in train_matches:
            target_split = "train"
        elif m_id in val_matches:
            target_split = "val"
        else:
            target_split = "test"

        row["split"] = target_split
        split_counts[target_split] += 1

        src_img = Path(row["image_path"])
        dst_img = output_dir / "images" / target_split / src_img.name

        src_lbl = staging_labels / (src_img.stem + ".txt")
        dst_lbl = output_dir / "labels" / target_split / (src_img.stem + ".txt")

        if src_img.exists():
            transfer_func(src_img, dst_img)
            # Update path in manifest to final destination
            row["image_path"] = str(dst_img)

        if src_lbl.exists():
            transfer_func(src_lbl, dst_lbl)

    # Re-save manifest with updated split column
    fieldnames = list(rows[0].keys())
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("==================================================")
    print(f"[+] Strict Match-Level Split Completed ({action_verb})!")
    print(f" Train images: {split_counts['train']}")
    print(f" Val images  : {split_counts['val']}")
    print(f" Test images : {split_counts['test']}")
    print(f" Updated manifest: {manifest_path}")
    print(" Match Isolation Guarantee: No match appears in multiple splits.")
    print("==================================================")


if __name__ == "__main__":
    main()
