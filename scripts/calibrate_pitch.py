#!/usr/bin/env python3
"""
calibrate_pitch.py

Computes 3x3 Homography Matrix mapping 2D camera image coordinates (u, v in pixels)
to real-world 2D pitch coordinates (X, Y in meters, standard pitch = 105m x 68m).

Supports:
1. Standard FIFA pitch landmark dictionary:
   - top_left_corner: (0.0, 0.0)
   - bottom_left_corner: (0.0, 68.0)
   - top_right_corner: (105.0, 0.0)
   - bottom_right_corner: (105.0, 68.0)
   - center_spot: (52.5, 34.0)
   - left_penalty_spot: (11.0, 34.0)
   - right_penalty_spot: (94.0, 34.0)
   - left_box_top: (16.5, 13.84)
   - left_box_bottom: (16.5, 54.16)
   - right_box_top: (88.5, 13.84)
   - right_box_bottom: (88.5, 54.16)
   - halfway_line_top: (52.5, 0.0)
   - halfway_line_bottom: (52.5, 68.0)
2. Solving via cv2.findHomography with cv2.RANSAC for N >= 4 points.
3. Saves calibration to data/calibration/{venue}_{cam}_homography.json with reprojection metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# FIFA standard pitch dimensions in meters (origin at top-left corner)
STANDARD_LANDMARKS = {
    "top_left_corner": [0.0, 0.0],
    "bottom_left_corner": [0.0, 68.0],
    "top_right_corner": [105.0, 0.0],
    "bottom_right_corner": [105.0, 68.0],
    "center_spot": [52.5, 34.0],
    "halfway_line_top": [52.5, 0.0],
    "halfway_line_bottom": [52.5, 68.0],
    "left_penalty_spot": [11.0, 34.0],
    "right_penalty_spot": [94.0, 34.0],
    "left_box_top": [16.5, 13.84],
    "left_box_bottom": [16.5, 54.16],
    "right_box_top": [88.5, 13.84],
    "right_box_bottom": [88.5, 54.16],
    "left_6yd_top": [5.5, 24.84],
    "left_6yd_bottom": [5.5, 43.16],
    "right_6yd_top": [99.5, 24.84],
    "right_6yd_bottom": [99.5, 43.16]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute pitch homography calibration.")
    parser.add_argument("--image", type=str, help="Representative frame from the camera")
    parser.add_argument("--venue", type=str, default="venue01", help="Venue identifier")
    parser.add_argument("--cam", type=str, default="cam_main", help="Camera identifier")
    parser.add_argument("--output-dir", type=str, default="data/calibration", help="Calibration output folder")
    parser.add_argument(
        "--points-file",
        type=str,
        default=None,
        help="JSON file containing landmark mappings: {'landmark_name': [pixel_x, pixel_y]}"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive OpenCV click-to-calibrate GUI (requires DISPLAY environment)"
    )
    return parser.parse_args()


def check_dependencies():
    try:
        import numpy as np
        import cv2
        return np, cv2
    except ImportError:
        print("[ERROR] calibrate_pitch.py requires numpy and opencv-python.")
        print("        Install with: pip install numpy opencv-python-headless")
        sys.exit(1)


def compute_homography(pixel_points, pitch_points):
    np, cv2 = check_dependencies()
    """
    Computes 3x3 homography matrix from pixel coordinates to pitch meters.
    pixel_points: Nx2 numpy array
    pitch_points: Nx2 numpy array
    """
    if len(pixel_points) < 4:
        raise ValueError(f"At least 4 correspondence points required, got {len(pixel_points)}")

    H, mask = cv2.findHomography(pixel_points, pitch_points, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None:
        raise RuntimeError("Failed to compute homography matrix.")

    # Calculate reprojection error in meters
    src_homo = np.hstack([pixel_points, np.ones((len(pixel_points), 1))])
    projected = (H @ src_homo.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - pitch_points, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    return H, mean_error, max_error, errors.tolist()


def run_interactive(image_path: Path):
    np, cv2 = check_dependencies()
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    clone = img.copy()

    landmarks = list(STANDARD_LANDMARKS.keys())
    current_idx = 0
    collected_points = {}

    def click_event(event, x, y, flags, param):
        nonlocal current_idx
        if event == cv2.EVENT_LBUTTONDOWN and current_idx < len(landmarks):
            landmark = landmarks[current_idx]
            collected_points[landmark] = [x, y]
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{current_idx + 1}: {landmark}", (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Pitch Calibration", img)
            print(f"[+] Selected {landmark} at ({x}, {y})")
            current_idx += 1

    cv2.namedWindow("Pitch Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pitch Calibration", click_event)

    print("\n--- Interactive Calibration Instructions ---")
    print("Click on pitch landmarks as prompted.")
    print("Press 's' to skip current landmark.")
    print("Press 'q' when done (minimum 4 points required).\n")

    while True:
        cv2.imshow("Pitch Calibration", img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            current_idx += 1
            if current_idx >= len(landmarks):
                break

    cv2.destroyAllWindows()
    return collected_points


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.venue}_{args.cam}_homography.json"

    point_map = {}

    if args.points_file:
        with open(args.points_file, "r") as f:
            point_map = json.load(f)
    elif args.interactive:
        if not args.image:
            print("[ERROR] --image is required for interactive mode")
            sys.exit(1)
        point_map = run_interactive(Path(args.image))
    else:
        # Default mock / demo landmarks for reference template
        print("[*] No points specified. Writing a template points file...")
        template_file = out_dir / "template_landmarks.json"
        example_data = {
            "top_left_corner": [120, 85],
            "bottom_left_corner": [45, 950],
            "halfway_line_top": [960, 95],
            "halfway_line_bottom": [960, 960],
            "top_right_corner": [1800, 90],
            "bottom_right_corner": [1880, 955],
            "center_spot": [960, 520]
        }
        with open(template_file, "w") as f:
            json.dump(example_data, f, indent=2)
        print(f"[+] Example landmarks file created at: {template_file}")
        print(f"[*] Use --points-file {template_file} to calibrate.")
        return

    # Filter landmarks present in standard dictionary
    pixel_pts = []
    pitch_pts = []
    valid_landmarks = {}

    for name, pixel_coord in point_map.items():
        if name in STANDARD_LANDMARKS:
            pixel_pts.append(pixel_coord)
            pitch_pts.append(STANDARD_LANDMARKS[name])
            valid_landmarks[name] = {
                "pixel": pixel_coord,
                "pitch_meters": STANDARD_LANDMARKS[name]
            }

    if len(pixel_pts) < 4:
        print(f"[ERROR] Need at least 4 valid landmarks from standard dictionary. Found: {len(pixel_pts)}")
        print(f"Available landmarks: {list(STANDARD_LANDMARKS.keys())}")
        sys.exit(1)

    np, _ = check_dependencies()
    pixel_pts = np.array(pixel_pts, dtype=np.float32)
    pitch_pts = np.array(pitch_pts, dtype=np.float32)

    H, mean_err, max_err, errors = compute_homography(pixel_pts, pitch_pts)

    result = {
        "venue": args.venue,
        "cam": args.cam,
        "pitch_dimensions": {"length_m": 105.0, "width_m": 68.0},
        "num_correspondences": len(pixel_pts),
        "mean_reprojection_error_m": round(mean_err, 3),
        "max_reprojection_error_m": round(max_err, 3),
        "homography_matrix": H.tolist(),
        "inverse_homography_matrix": np.linalg.inv(H).tolist(),
        "landmarks": valid_landmarks
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print("==================================================")
    print(f"[+] Calibration Successfully Computed!")
    print(f" Output File    : {out_file}")
    print(f" Correspondences: {len(pixel_pts)} points")
    print(f" Mean Error     : {mean_err:.3f} meters")
    print(f" Max Error      : {max_err:.3f} meters")
    if mean_err > 1.0:
        print("[WARNING] Mean reprojection error is > 1.0m. Review landmark clicking accuracy!")
    else:
        print("[PASS] Calibration error is within tactical analytics tolerance (< 1.0m).")
    print("==================================================")


if __name__ == "__main__":
    main()
