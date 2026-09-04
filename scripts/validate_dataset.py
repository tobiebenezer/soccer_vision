#!/usr/bin/env python3
"""
validate_dataset.py

Audits the labeled dataset (data/labeled) before training:
1. Counts total images and labels per split (train, val, test).
2. Verifies label formatting (YOLO normalized format 0.0 - 1.0).
3. Computes per-class bounding box counts for the 4 classes:
     0: player, 1: goalkeeper, 2: referee, 3: ball
4. Class Imbalance Check:
     Flags a warning if ball instances < 10% of player instances and advises
     oversampling ratios for Phase 1 YOLO training.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict


CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate labeled YOLO dataset integrity and class distribution.")
    parser.add_argument("--data-dir", type=str, default="data/labeled", help="Path to labeled data folder")
    parser.add_argument("--ball-threshold", type=float, default=0.10, help="Minimum acceptable ball-to-player ratio (default: 0.10)")
    return parser.parse_args()


def audit_split(split_dir: Path, split_name: str, ball_threshold: float):
    images_dir = split_dir / "images" / split_name
    labels_dir = split_dir / "labels" / split_name

    if not images_dir.exists():
        print(f"[!] Split '{split_name}' images folder not found at: {images_dir}")
        return None

    image_files = list(images_dir.glob("*.*"))
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    class_counts = defaultdict(int)
    corrupted_boxes = 0
    images_with_balls = 0
    total_boxes = 0

    for lbl_path in label_files:
        has_ball = False
        with open(lbl_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    corrupted_boxes += 1
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:])

                    # Validate bounds
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        corrupted_boxes += 1
                        continue

                    class_counts[cls_id] += 1
                    total_boxes += 1

                    if cls_id == 3:
                        has_ball = True
                except ValueError:
                    corrupted_boxes += 1

        if has_ball:
            images_with_balls += 1

    print(f"\n================ Split: {split_name.upper()} ================")
    print(f" Images: {len(image_files)} | Labels: {len(label_files)} | Total Bounding Boxes: {total_boxes}")

    if corrupted_boxes > 0:
        print(f" [!] WARNING: Found {corrupted_boxes} malformed/out-of-bounds bounding boxes in {split_name}!")

    print("\n Per-Class Instance Distribution:")
    for cls_id in range(4):
        name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        count = class_counts[cls_id]
        pct = (count / total_boxes * 100.0) if total_boxes > 0 else 0.0
        print(f"   [{cls_id}] {name:<12}: {count:>6} instances ({pct:>5.1f}%)")

    # Ball to player check
    player_count = class_counts[0]
    ball_count = class_counts[3]
    ratio = (ball_count / player_count) if player_count > 0 else 0.0

    print(f"\n Ball-to-Player Ratio: {ratio:.3f} ({ball_count} balls vs {player_count} players)")
    print(f" Images containing at least one ball: {images_with_balls} / {len(image_files)}")

    if ratio < ball_threshold:
        print(f"\n [!] CRITICAL WARNING: Ball instances are below {ball_threshold * 100:.0f}% of player instances!")
        print(f"     Recommended Mitigation for Phase 1 Training:")
        print(f"     1. Enable weighted oversampling of images with balls (ratio factor ~{max(2, int(round(1.0 / max(0.01, ratio))))}x).")
        print(f"     2. Use image resolution imgsz=960 or imgsz=1280 to preserve small ball pixel gradients.")
        print(f"     3. Explicitly evaluate class-3 AP (Ball AP50) in addition to aggregate mAP.")
    else:
        print(f" [PASS] Ball representation meets minimum threshold ({ball_threshold * 100:.0f}%).")

    return {
        "images": len(image_files),
        "boxes": total_boxes,
        "classes": dict(class_counts),
        "ball_ratio": ratio
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("==================================================")
    print(f" AUDITING DATASET: {data_dir.resolve()}")
    print("==================================================")

    splits = ["train", "val", "test"]
    audits = {}
    for split in splits:
        audits[split] = audit_split(data_dir, split, args.ball_threshold)

    print("\n==================================================")
    print(" Dataset Audit Summary Finished.")
    print("==================================================")


if __name__ == "__main__":
    main()
