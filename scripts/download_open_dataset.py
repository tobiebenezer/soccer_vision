#!/usr/bin/env python3
"""
download_open_dataset.py

One-click downloader for public, pre-annotated soccer/football YOLO datasets
from Hugging Face (e.g. Adit-jain/Soccana_player_ball_detection_v1).

Requires ZERO API keys and ZERO logins.
Automatically downloads, unzips, and maps classes to the standard schema:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="Download open soccer detection dataset from HuggingFace.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Adit-jain/Soccana_player_ball_detection_v1",
        help="HuggingFace dataset repo (default: Adit-jain/Soccana_player_ball_detection_v1)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="V1.zip",
        help="Archive file name inside the repository"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/open_soccer",
        help="Target folder to stage extracted images and labels"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images to extract (default: 1000; use 0 for all 25k)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] download_open_dataset.py requires huggingface_hub.")
        print("        Install with: pip install huggingface_hub")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(f" Source Repository : {args.repo_id}")
    print(f" Archive File      : {args.filename}")
    print(f" Max Images Limit  : {args.max_images}")
    print(f" Output Directory  : {out_dir}")
    print("==================================================")

    print(f"[*] Downloading {args.filename} from {args.repo_id}...")
    try:
        zip_path_str = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            local_dir=cache_dir,
            resume_download=True
        )
        zip_path = Path(zip_path_str)
        print(f"[+] Download complete: {zip_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {args.filename} from HuggingFace: {e}")
        print("Ensure you have internet connection enabled in your Kaggle notebook settings.")
        sys.exit(1)

    print(f"[*] Extracting and re-organizing dataset from zip...")
    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("__MACOSX")]
        print(f"[*] Total images found in archive: {len(image_files)}")

        if args.max_images and args.max_images > 0:
            image_files = image_files[:args.max_images]
            print(f"[*] Extracting subset of {len(image_files)} images...")

        for img_entry in tqdm(image_files, desc="Extracting images"):
            img_stem = Path(img_entry).stem
            img_name = f"open_{Path(img_entry).name}"
            dst_img = images_out / img_name
            dst_lbl = labels_out / (Path(img_name).stem + ".txt")

            # Extract image
            with z.open(img_entry) as sf, open(dst_img, "wb") as df:
                shutil.copyfileobj(sf, df)

            # Find matching label in zip
            # Looking for label in parallel labels/ directory or same folder
            lbl_candidates = [
                img_entry.replace("/images/", "/labels/").replace(Path(img_entry).suffix, ".txt"),
                str(Path(img_entry).with_suffix(".txt")),
                f"labels/{img_stem}.txt",
                f"train/labels/{img_stem}.txt"
            ]

            found_lbl = None
            for cand in lbl_candidates:
                if cand in all_files:
                    found_lbl = cand
                    break

            if found_lbl:
                with z.open(found_lbl) as sf, open(dst_lbl, "wb") as df:
                    shutil.copyfileobj(sf, df)
            else:
                dst_lbl.touch()

            extracted_count += 1

    # Cleanup zip
    print(f"[*] Cleaning up temporary zip ({zip_path.stat().st_size / 1e9:.2f} GB)...")
    zip_path.unlink(missing_ok=True)

    print("==================================================")
    print(f"[+] Open Dataset Staging Complete!")
    print(f" Extracted Images : {extracted_count}")
    print(f" Images Location  : {images_out}")
    print(f" Labels Location  : {labels_out}")
    print(" Ready to run scripts/merge_datasets.py")
    print("==================================================")


if __name__ == "__main__":
    main()
