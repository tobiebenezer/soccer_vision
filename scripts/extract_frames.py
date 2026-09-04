#!/usr/bin/env python3
"""
extract_frames.py

Extracts representative video frames from raw NPL match footage (.mp4, .mov, .mkv, .avi)
for downstream calibration, auto-labeling, and CVAT / AnyLabeling annotation.

Frames are named systematically:
  {venue}_{cam}_{match_id}_f{frame_idx:06d}.jpg
"""

import os
import sys
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from soccer match footage.")
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to source video file (.mp4, .mov, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted_frames",
        help="Directory where extracted frames will be saved"
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="mobolaji_johnson_arena",
        help="Name/identifier of the venue (e.g., mobolaji_johnson_arena, remo_stars_stadium, godswill_akpabio_stadium)"
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="cam_main",
        help="Camera identifier (e.g., cam_main, cam_tactical, cam_gantry)"
    )
    parser.add_argument(
        "--match-id",
        type=str,
        default="npfl_match_01",
        help="Unique match identifier (e.g. sporting_lagos_vs_enyimba_w01)"
    )
    parser.add_argument(
        "--sample-rate-fps",
        type=float,
        default=1.0,
        help="How many frames to extract per second of video (default: 1.0)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Extract every Nth frame directly (overrides --sample-rate-fps if set)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop extracting after N frames"
    )
    return parser.parse_args()


def extract_frames(video_path: Path, output_dir: Path, venue: str, cam: str, match_id: str,
                   sample_rate_fps: float = 1.0, stride: int = None, max_frames: int = None):
    try:
        import cv2
    except ImportError:
        print("[ERROR] extract_frames requires opencv-python.")
        print("        Install with: pip install opencv-python-headless")
        sys.exit(1)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file with OpenCV: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / video_fps

    # Target folder: data/extracted_frames/{venue}_{cam}
    target_dir = output_dir / f"{venue}_{cam}"
    target_dir.mkdir(parents=True, exist_ok=True)

    if stride is None:
        # Compute stride from target FPS
        stride = max(1, int(round(video_fps / sample_rate_fps)))

    print("==================================================")
    print(f" Source Video : {video_path.name}")
    print(f" Total Frames : {total_frames} ({duration_sec:.1f}s at {video_fps:.2f} fps)")
    print(f" Stride       : Every {stride} frames (~{video_fps / stride:.2f} FPS)")
    print(f" Output Dir   : {target_dir}")
    print("==================================================")

    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_filename = f"{venue}_{cam}_{match_id}_f{frame_idx:06d}.jpg"
            out_file = target_dir / frame_filename
            # Save frame in high quality JPEG
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    print(f"\n[+] Extraction Complete! Saved {saved_count} frames to {target_dir}")


def main():
    args = parse_args()
    extract_frames(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        venue=args.venue,
        cam=args.cam,
        match_id=args.match_id,
        sample_rate_fps=args.sample_rate_fps,
        stride=args.stride,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
