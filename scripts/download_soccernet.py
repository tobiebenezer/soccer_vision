#!/usr/bin/env python3
"""
download_soccernet.py

Downloads and converts SoccerNet-GSR (Game State Reconstruction / Detection) dataset
from Hugging Face Hub (SoccerNet/SN-GSR-2024 or SoccerNet/SN-GSR-2025).

The dataset is stored on HF as zip archives (valid.zip, train.zip).
This script:
1. Downloads the specified zip archive (defaults to 'valid.zip' which is fast and compact).
2. Selectively extracts only up to --max-matches sequences (saving massive disk space on Kaggle).
3. Converts SoccerNet GameState annotations to standard 4-class YOLO format:
     0: player, 1: goalkeeper, 2: referee, 3: ball
"""

import os
import sys
import json
import zipfile
import shutil
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="Download & convert SoccerNet GameState detection data from HuggingFace.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="SoccerNet/SN-GSR-2024",
        help="HuggingFace dataset repository ID (default: SoccerNet/SN-GSR-2024)"
    )
    parser.add_argument(
        "--archive",
        type=str,
        default="valid.zip",
        choices=["valid.zip", "train.zip", "test.zip"],
        help="Which split zip to download from HF (default: valid.zip ~3GB; train.zip is ~25GB)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/soccernet",
        help="Target output directory for converted images and labels"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=5,
        help="Maximum number of match sequences to extract and convert (default: 5; use 0 or -1 for all)"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip archive after extraction (default: deletes zip to conserve disk space)"
    )
    return parser.parse_args()


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """
    Converts [x_min, y_min, width, height] to YOLO normalized format:
    x_center, y_center, width, height (0.0 to 1.0)
    """
    x_min, y_min, w, h = bbox
    xc = (x_min + w / 2.0) / img_w
    yc = (y_min + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h

    # Clip to valid [0.0, 1.0]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))

    return xc, yc, wn, hn


def resolve_class_id(ann, categories_map):
    """
    Resolves category from SoccerNet GameState annotations to 4 classes:
      0: player
      1: goalkeeper
      2: referee
      3: ball
    """
    # 1. Check role in attributes
    attributes = ann.get("attributes", {})
    if isinstance(attributes, dict):
        role = str(attributes.get("role", "")).lower()
        if "goalkeeper" in role or role == "gk":
            return 1
        elif "referee" in role or role == "ref":
            return 2
        elif "player" in role:
            return 0

    # 2. Check category name from categories mapping
    cat_id = ann.get("category_id")
    cat_name = str(categories_map.get(cat_id, "")).lower()
    if "goalkeeper" in cat_name or cat_name == "gk":
        return 1
    elif "referee" in cat_name or cat_name == "ref":
        return 2
    elif "ball" in cat_name:
        return 3
    elif "player" in cat_name or "person" in cat_name:
        return 0

    return None


def extract_and_convert_soccernet(zip_path: Path, output_dir: Path, max_matches: int = 5):
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Inspecting zip archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()

        # Find all JSON label files
        label_files = [f for f in all_files if f.endswith("Labels-gamestate.json") or f.endswith("labels.json")]
        if not label_files:
            # Check for any json file
            label_files = [f for f in all_files if f.endswith(".json") and not f.startswith("__MACOSX")]

        print(f"[*] Found {len(label_files)} match sequences inside archive.")

        # Limit matches if requested
        if max_matches and max_matches > 0 and len(label_files) > max_matches:
            label_files = label_files[:max_matches]
            print(f"[*] Extracting subset: first {len(label_files)} match sequences...")

        total_converted = 0
        match_count = 0

        for lbl_file_in_zip in label_files:
            match_folder = str(Path(lbl_file_in_zip).parent)
            match_id = f"soccernet_{match_folder.replace('/', '_')}"
            match_count += 1

            # Read JSON directly from zip
            try:
                with z.open(lbl_file_in_zip) as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"[!] Could not read {lbl_file_in_zip}: {e}")
                continue

            images_info = {img["image_id"]: img for img in data.get("images", [])}
            annotations = data.get("annotations", [])
            categories_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

            # Group annotations by image
            img_annos = {}
            for ann in annotations:
                img_id = ann.get("image_id")
                if img_id not in img_annos:
                    img_annos[img_id] = []
                img_annos[img_id].append(ann)

            print(f"[{match_count}/{len(label_files)}] Processing match: {match_id} ({len(images_info)} frames)...")

            for img_id, img_info in images_info.items():
                file_name = img_info.get("file_name", "")
                img_w = img_info.get("width", 1920)
                img_h = img_info.get("height", 1080)

                # Candidate paths inside zip
                possible_paths = [
                    f"{match_folder}/{file_name}",
                    f"{match_folder}/img1/{file_name}",
                    f"{match_folder}/{Path(file_name).name}"
                ]
                actual_zip_path = None
                for p in possible_paths:
                    if p in all_files:
                        actual_zip_path = p
                        break

                if not actual_zip_path:
                    continue

                clean_img_name = f"{match_id}_f{img_id:06d}.jpg"
                dst_img = images_out / clean_img_name
                dst_lbl = labels_out / (Path(clean_img_name).stem + ".txt")

                # Extract image to destination
                with z.open(actual_zip_path) as src_f, open(dst_img, "wb") as out_f:
                    shutil.copyfileobj(src_f, out_f)

                # Build YOLO label lines
                yolo_lines = []
                for ann in img_annos.get(img_id, []):
                    cls_id = resolve_class_id(ann, categories_map)
                    if cls_id is None:
                        continue

                    bbox = ann.get("bbox")  # [x, y, w, h]
                    if bbox and len(bbox) == 4:
                        xc, yc, wn, hn = convert_bbox_to_yolo(bbox, img_w, img_h)
                        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

                with open(dst_lbl, "w") as lf:
                    lf.writelines(yolo_lines)

                total_converted += 1

    print("==================================================")
    print(f"[+] SoccerNet Ingestion Complete!")
    print(f" Matches converted : {match_count}")
    print(f" Total frames      : {total_converted}")
    print(f" Output images     : {images_out}")
    print(f" Output labels     : {labels_out}")
    print("==================================================")


def main():
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] download_soccernet.py requires huggingface_hub.")
        print("        Install with: pip install huggingface_hub")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(f" Repository : {args.repo_id}")
    print(f" Archive    : {args.archive}")
    print(f" Max Matches: {args.max_matches}")
    print(f" Target Dir : {out_dir}")
    print("==================================================")

    print(f"[*] Downloading {args.archive} from Hugging Face Hub ({args.repo_id})...")
    try:
        downloaded_zip = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.archive,
            repo_type="dataset",
            local_dir=cache_dir,
            resume_download=True
        )
        print(f"[+] Download complete: {downloaded_zip}")
        zip_path = Path(downloaded_zip)

        extract_and_convert_soccernet(zip_path, out_dir, args.max_matches)

        if not args.keep_zip:
            print(f"[*] Removing downloaded zip to conserve disk space ({zip_path.stat().st_size / 1e9:.2f} GB)...")
            zip_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"\n[ERROR] Failed during SoccerNet download/extraction: {e}")
        print("\nPossible solutions:")
        print("1. Ensure internet access is enabled (in Kaggle: Settings -> Internet -> ON).")
        print("2. Run with --archive valid.zip for the fastest and most reliable download.")
        print("3. Alternatively, run: python scripts/download_open_dataset.py for a direct 1-click dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
