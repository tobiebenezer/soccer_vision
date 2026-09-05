#!/usr/bin/env python3
"""
create_sample_data.py

Generates realistic sample soccer match frames and YOLO annotations locally.
Useful for smoke-testing the pipeline (merge -> split -> validate -> train)
without waiting for heavy multi-gigabyte downloads.
"""

import os
import sys
import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic soccer match dataset for instant pipeline testing.")
    parser.add_argument("--output-dir", type=str, default="data/raw/sample_soccer", help="Target output folder")
    parser.add_argument("--num-matches", type=int, default=3, help="Number of distinct matches (default: 3)")
    parser.add_argument("--frames-per-match", type=int, default=20, help="Frames per match (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Optional PIL import for creating actual image files
    try:
        from PIL import Image, ImageDraw
        has_pil = True
    except ImportError:
        has_pil = False

    print(f"[*] Generating {args.num_matches * args.frames_per_match} sample frames across {args.num_matches} matches...")

    total_boxes = 0
    total_balls = 0

    for m in range(1, args.num_matches + 1):
        match_id = f"sample_match_{m:02d}"
        for f in range(1, args.frames_per_match + 1):
            frame_name = f"{match_id}_f{f:06d}"
            img_path = images_dir / f"{frame_name}.jpg"
            lbl_path = labels_dir / f"{frame_name}.txt"

            # Create an image (either green soccer pitch image or binary dummy)
            if has_pil:
                img = Image.new("RGB", (1280, 720), color=(34, 139, 34)) # Forest green pitch
                draw = ImageDraw.Draw(img)
                # Draw white pitch lines
                draw.rectangle([50, 50, 1230, 670], outline=(255, 255, 255), width=3)
                draw.line([(640, 50), (640, 670)], fill=(255, 255, 255), width=3)
                draw.ellipse([540, 310, 740, 410], outline=(255, 255, 255), width=3)
                img.save(img_path, quality=85)
            else:
                # Create a blank minimal JPEG
                with open(img_path, "wb") as f_out:
                    f_out.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9')

            # Generate YOLO annotations
            # Realistic soccer setup: ~18-22 players (0), 2 goalkeepers (1), 1-3 referees (2), 1 ball (3)
            lines = []

            # 1. Players (15 - 20)
            num_players = random.randint(15, 20)
            for _ in range(num_players):
                xc = random.uniform(0.1, 0.9)
                yc = random.uniform(0.15, 0.85)
                w = random.uniform(0.02, 0.04)
                h = random.uniform(0.06, 0.10)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                total_boxes += 1

            # 2. Goalkeepers (2)
            lines.append(f"1 0.080000 0.500000 0.035000 0.090000\n")
            lines.append(f"1 0.920000 0.500000 0.035000 0.090000\n")
            total_boxes += 2

            # 3. Referee (1)
            lines.append(f"2 {random.uniform(0.3, 0.7):.6f} {random.uniform(0.3, 0.7):.6f} 0.030000 0.080000\n")
            total_boxes += 1

            # 4. Ball (1)
            b_xc = random.uniform(0.2, 0.8)
            b_yc = random.uniform(0.2, 0.8)
            lines.append(f"3 {b_xc:.6f} {b_yc:.6f} 0.015000 0.020000\n")
            total_boxes += 1
            total_balls += 1

            with open(lbl_path, "w") as lf:
                lf.writelines(lines)

    print("==================================================")
    print(f"[+] Sample Soccer Dataset Created Successfully!")
    print(f" Matches Generated: {args.num_matches}")
    print(f" Total Frames     : {args.num_matches * args.frames_per_match}")
    print(f" Bounding Boxes   : {total_boxes} (including {total_balls} balls)")
    print(f" Destination      : {out_dir}")
    print(" You can now run scripts/merge_datasets.py to stage and split.")
    print("==================================================")


if __name__ == "__main__":
    main()
