#!/usr/bin/env python3
"""
download_soccernet.py

Downloads detection assets from the public SoccerNet-V3 / SoccerNet GameState dataset
hosted on Hugging Face Hub (MIT license, no NDA required).

Converts annotations into standard YOLO format:
  class_id x_center y_center width height (normalized 0.0 - 1.0)
Mapping:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable



CLASS_MAPPING = {
    "player": 0,
    "players": 0,
    "goalkeeper": 1,
    "goalkeepers": 1,
    "gk": 1,
    "referee": 2,
    "referees": 2,
    "ref": 2,
    "ball": 3,
    "soccer ball": 3
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download SoccerNet detection data from HuggingFace.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="SoccerNet/sn-gamestate",
        help="HuggingFace dataset repository ID (default: SoccerNet/sn-gamestate)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/soccernet",
        help="Target output directory for raw soccernet data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test", "all"],
        help="Split to download (default: train)"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Optional maximum number of matches/sequences to process"
    )
    return parser.parse_args()


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """
    Converts [x, y, w, h] or [x1, y1, x2, y2] to YOLO normalized format:
    x_center, y_center, width, height (0.0 to 1.0)
    """
    # Assuming input is [x_min, y_min, width, height]
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Clip to valid [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm


def process_gamestate_annotations(dataset_path: Path, output_dir: Path, max_matches: int = None):
    """
    Processes SoccerNet-GameState sequences and converts annotations to YOLO format.
    """
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Scanning for sequences in: {dataset_path}")
    # Search for Gamestate annotation JSON files or Mot-style labels
    annotation_files = list(dataset_path.glob("**/*Labels-gamestate.json")) + \
                       list(dataset_path.glob("**/Labels-GameState.json")) + \
                       list(dataset_path.glob("**/labels.json"))

    if not annotation_files:
        print(f"[!] No standard JSON labels found directly. Checking for image folders...")
        image_files = list(dataset_path.glob("**/*.jpg")) + list(dataset_path.glob("**/*.png"))
        print(f"[*] Found {len(image_files)} raw images. Copying directly for staging...")
        for img in tqdm(image_files[: (max_matches * 100 if max_matches else None)], desc="Copying images"):
            dest = images_out / f"soccernet_{img.name}"
            shutil.copy2(img, dest)
        print(f"[+] Direct images staged into {images_out}")
        return

    processed_count = 0
    total_images_converted = 0

    for anno_file in annotation_files:
        if max_matches and processed_count >= max_matches:
            break

        match_id = f"soccernet_{anno_file.parent.name}"
        try:
            with open(anno_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to parse {anno_file}: {e}")
            continue

        images_info = {img["image_id"]: img for img in data.get("images", [])}
        annotations = data.get("annotations", [])
        categories = {cat["id"]: cat["name"].lower() for cat in data.get("categories", [])}

        # Group annotations by image
        img_annos = {}
        for ann in annotations:
            img_id = ann.get("image_id")
            if img_id not in img_annos:
                img_annos[img_id] = []
            img_annos[img_id].append(ann)

        for img_id, anns in img_annos.items():
            img_info = images_info.get(img_id)
            if not img_info:
                continue

            img_w = img_info.get("width", 1920)
            img_h = img_info.get("height", 1080)
            file_name = img_info.get("file_name")

            src_img_path = anno_file.parent / file_name
            if not src_img_path.exists():
                src_img_path = anno_file.parent / "img1" / file_name

            if not src_img_path.exists():
                continue

            target_img_name = f"{match_id}_{file_name.replace('/', '_')}"
            target_lbl_name = Path(target_img_name).stem + ".txt"

            shutil.copy2(src_img_path, images_out / target_img_name)

            yolo_lines = []
            for ann in anns:
                cat_id = ann.get("category_id")
                cat_name = categories.get(cat_id, "player")
                class_id = CLASS_MAPPING.get(cat_name, 0)

                bbox = ann.get("bbox")  # [x, y, w, h]
                if bbox and len(bbox) == 4:
                    xc, yc, w, h = convert_bbox_to_yolo(bbox, img_w, img_h)
                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            with open(labels_out / target_lbl_name, "w") as f:
                f.writelines(yolo_lines)

            total_images_converted += 1

        processed_count += 1

    print(f"[+] Finished! Converted {total_images_converted} images across {processed_count} matches.")
    print(f"[+] Output directory: {output_dir}")


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] download_soccernet.py requires huggingface_hub.")
        print("        Install with: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading dataset {args.repo_id} from Hugging Face Hub (Split: {args.split})...")
    allow_patterns = ["*.json", "*.jpg", "*.png", "*.txt"]
    if args.split != "all":
        allow_patterns.append(f"*{args.split}*")

    try:
        download_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=output_dir / "hf_cache",
            resume_download=True
        )
        print(f"[+] Download complete: {download_path}")
        process_gamestate_annotations(Path(download_path), output_dir, args.max_matches)
    except Exception as e:
        print(f"[!] Error downloading from Hugging Face Hub: {e}")
        print(f"[*] Note: If running without internet or credentials, you can stage raw files directly in {output_dir}")


if __name__ == "__main__":
    main()
