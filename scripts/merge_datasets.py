#!/usr/bin/env python3
"""
merge_datasets.py

Consolidates raw datasets (SoccerNet-V3, Roboflow exports, NPL annotated footage)
into a unified staging directory, remapping all bounding boxes to the strict 4-class schema:
  0: player
  1: goalkeeper
  2: referee
  3: ball

Generates and maintains `data/manifest.csv` tracking provenance:
  image_path, source, match_id, venue, license, split
"""

import os
import sys
import re
import csv
import shutil
import argparse
import yaml
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


TARGET_CLASSES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball"
}

# Standard name-to-target-ID dictionary
CLASS_SYNONYMS = {
    "player": 0,
    "players": 0,
    "outfield": 0,
    "goalkeeper": 1,
    "gk": 1,
    "goalie": 1,
    "referee": 2,
    "ref": 2,
    "referees": 2,
    "ball": 3,
    "soccer ball": 3,
    "football": 3
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge datasets and remap classes to 4-class schema.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Root folder of raw datasets")
    parser.add_argument("--staging-dir", type=str, default="data/staging", help="Staging folder before split")
    parser.add_argument("--manifest-path", type=str, default="data/manifest.csv", help="Path to master manifest CSV")
    return parser.parse_args()


def load_yaml_names(yaml_path: Path):
    """Loads class names from a dataset.yaml or data.yaml file if present."""
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        names = d.get("names", {})
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    except Exception as e:
        print(f"[!] Warning: could not parse {yaml_path}: {e}")
    return None


def extract_match_id(img_path: Path, source_name: str) -> str:
    """
    Deterministically extracts or synthesizes a match_id to guarantee match-level isolation.
    """
    stem = img_path.stem

    # Pattern: {venue}_{cam}_{match_id}_f000123
    match_tag = re.search(r"(match[_-]?[a-zA-Z0-9]+|game[_-]?[a-zA-Z0-9]+|sn[_-]?[0-9]+)", stem, re.IGNORECASE)
    if match_tag:
        return match_tag.group(1).lower()

    # If from a specific subfolder (e.g. raw/roboflow/project_v1/...)
    parent_folder = img_path.parent.name
    grandparent = img_path.parent.parent.name
    if parent_folder not in ["images", "train", "val", "test"]:
        return f"{source_name}_{parent_folder}"
    elif grandparent not in ["raw", "data", "images"]:
        return f"{source_name}_{grandparent}"

    # Fallback to source prefix + sequence hash
    return f"{source_name}_seq_{abs(hash(stem[:8])) % 1000:03d}"


def remap_and_copy_labels(src_label_path: Path, dst_label_path: Path, source_class_map: dict):
    """
    Reads source YOLO label file, translates class IDs to the 4-class schema,
    and writes out the clean label file. Drops any unknown classes.
    """
    if not src_label_path.exists():
        # Negative sample (background frame with no boxes)
        dst_label_path.touch()
        return 0

    valid_lines = []
    box_count = 0
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_class_id = int(float(parts[0]))
            coords = parts[1:5]

            target_class_id = None
            if source_class_map:
                source_name = source_class_map.get(orig_class_id, "").lower()
                target_class_id = CLASS_SYNONYMS.get(source_name)
            else:
                # Direct assumption if 4 classes already
                if orig_class_id in [0, 1, 2, 3]:
                    target_class_id = orig_class_id

            if target_class_id is not None:
                valid_lines.append(f"{target_class_id} {' '.join(coords)}\n")
                box_count += 1

    with open(dst_label_path, "w") as f:
        f.writelines(valid_lines)

    return box_count


def process_source(source_dir: Path, source_name: str, default_license: str,
                   staging_images: Path, staging_labels: Path) -> list:
    records = []
    if not source_dir.exists():
        print(f"[*] Skipping {source_name}: folder does not exist ({source_dir})")
        return records

    # Look for dataset config
    data_yaml = None
    for yml in source_dir.glob("**/*.yaml"):
        if yml.name in ["data.yaml", "dataset.yaml"]:
            data_yaml = yml
            break

    source_names_map = load_yaml_names(data_yaml) if data_yaml else None
    print(f"[*] Processing source '{source_name}' (Class mapping from config: {source_names_map is not None})")

    # Locate image files
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = [p for p in source_dir.glob("**/*") if p.suffix.lower() in image_exts]

    for img_path in tqdm(image_files, desc=f"Merging {source_name}"):
        match_id = extract_match_id(img_path, source_name)
        unique_name = f"{source_name}_{img_path.name}"

        dst_img = staging_images / unique_name
        dst_lbl = staging_labels / (Path(unique_name).stem + ".txt")

        # Copy image
        shutil.copy2(img_path, dst_img)

        # Find corresponding label file
        # Try same folder with .txt or parallel 'labels' folder
        # Universal YOLO label path resolution
        str_path = str(img_path)
        lbl_candidates = [
            img_path.with_suffix(".txt"),
            Path(str_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt") if f"{os.sep}images{os.sep}" in str_path else None,
            img_path.parents[1] / "labels" / (img_path.stem + ".txt") if len(img_path.parents) > 1 else None,
            img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt") if len(img_path.parents) > 2 else None,
        ]

        actual_lbl = None
        for cand in lbl_candidates:
            if cand and cand.exists():
                actual_lbl = cand
                break

        box_count = 0
        if actual_lbl:
            box_count = remap_and_copy_labels(actual_lbl, dst_lbl, source_names_map)
        else:
            dst_lbl.touch()  # empty label file

        records.append({
            "image_path": str(dst_img),
            "source": source_name,
            "match_id": match_id,
            "venue": "unspecified",
            "license": default_license,
            "split": "unassigned",
            "box_count": box_count
        })

    return records


def main():
    args = parse_args()
    raw_root = Path(args.raw_dir)
    staging_dir = Path(args.staging_dir)
    staging_images = staging_dir / "images"
    staging_labels = staging_dir / "labels"

    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    sources = [
        ("open_soccer", raw_root / "open_soccer", "CC-BY-4.0"),
        ("soccernet", raw_root / "soccernet", "MIT / GPL-3.0"),
        ("roboflow", raw_root / "roboflow", "CC-BY-4.0"),
        ("npfl_footage", raw_root / "npfl_footage", "Proprietary/Internal"),
        ("npl_footage", raw_root / "npl_footage", "Proprietary/Internal"),
        ("sample_soccer", raw_root / "sample_soccer", "Open-Access")
    ]

    all_records = []
    for s_name, s_dir, s_lic in sources:
        recs = process_source(s_dir, s_name, s_lic, staging_images, staging_labels)
        all_records.extend(recs)

    # Write manifest CSV
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "source", "match_id", "venue", "license", "split", "box_count"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("==================================================")
    print(f"[+] Staging and remapping complete!")
    print(f" Total images processed : {len(all_records)}")
    print(f" Staged images          : {staging_images}")
    print(f" Staged labels          : {staging_labels}")
    print(f" Master manifest        : {manifest_path}")
    print(" Next step: Run split_by_match.py to partition into train/val/test splits.")
    print("==================================================")


if __name__ == "__main__":
    main()
