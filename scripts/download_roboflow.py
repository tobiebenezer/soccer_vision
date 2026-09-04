#!/usr/bin/env python3
"""
download_roboflow.py

Downloads soccer/football datasets from Roboflow Universe while strictly enforcing
permissive open-source licenses (CC-BY 4.0, MIT, Apache 2.0, Public Domain).

STRICT POLICY: Automatically rejects any dataset marked CC-BY-NC (Non-Commercial)
or proprietary to protect production and commercial downstream compliance.
"""

import os
import sys
import argparse
import requests
from pathlib import Path



PERMISSIBLE_LICENSES = [
    "mit",
    "cc by 4.0",
    "cc-by-4.0",
    "cc-by",
    "apache 2.0",
    "apache-2.0",
    "public domain",
    "cc0",
    "cc0 1.0"
]

FORBIDDEN_KEYWORDS = [
    "-nc",
    " nc",
    "non-commercial",
    "noncommercial",
    "no commercial",
    "proprietary"
]


def check_license_compliance(license_str: str) -> tuple[bool, str]:
    """
    Returns (is_compliant, reason)
    """
    if not license_str:
        return False, "No license specified (unlicensed/proprietary default)."

    lic_lower = license_str.strip().lower()

    # Check forbidden keywords first
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lic_lower:
            return False, f"Non-commercial restriction detected: '{license_str}'"

    # Check permissible licenses
    for perm in PERMISSIBLE_LICENSES:
        if perm in lic_lower:
            return True, f"Permissible open license verified: '{license_str}'"

    return False, f"License '{license_str}' is not in permissible open list (MIT, CC-BY, Apache, CC0)."


def verify_roboflow_project_license(api_key: str, workspace: str, project: str) -> tuple[bool, str]:
    """
    Queries Roboflow REST API for project metadata to verify license before download.
    """
    url = f"https://api.roboflow.com/{workspace}/{project}?api_key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return False, f"Roboflow API returned status {resp.status_code}: {resp.text}"
        data = resp.json()
        project_info = data.get("project", {})
        license_str = project_info.get("license", "")
        return check_license_compliance(license_str)
    except Exception as e:
        return False, f"Failed to contact Roboflow API to check license: {e}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download license-checked soccer datasets from Roboflow.")
    parser.add_argument(
        "--workspace",
        type=str,
        default="roboflow-jvuqo",
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="football-players-detection-3zvbc",
        help="Roboflow project name"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Roboflow dataset version number"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/roboflow",
        help="Target folder to stage downloaded dataset"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Bypass license rejection (NOT recommended for commercial projects)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        print("[!] ROBOFLOW_API_KEY not found in environment or arguments.")
        print("[*] To obtain an API key, sign up for free at https://roboflow.com")
        print("[*] Example: python scripts/download_roboflow.py --api-key YOUR_KEY --workspace roboflow-jvuqo --project football-players-detection-3zvbc --version 1")
        sys.exit(1)

    print(f"[*] Checking license metadata for {args.workspace}/{args.project} (v{args.version})...")
    is_compliant, reason = verify_roboflow_project_license(args.api_key, args.workspace, args.project)

    if not is_compliant:
        print(f"[REJECTED] License Compliance Check Failed!")
        print(f"           Reason: {reason}")
        if not args.force_download:
            print("[ABORTING] To prevent license contamination, this dataset will not be downloaded.")
            print("           Provide an MIT or CC-BY licensed dataset instead.")
            sys.exit(2)
        else:
            print("[WARNING] --force-download was set! Proceeding despite non-compliant license...")
    else:
        print(f"[APPROVED] {reason}")

    out_path = Path(args.output_dir) / f"{args.project}_v{args.version}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing Roboflow client and downloading YOLOv8 export...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] download_roboflow.py requires roboflow.")
        print("        Install with: pip install roboflow")
        sys.exit(1)

    try:
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download(
            model_format="yolov8",
            location=str(out_path)
        )
        print(f"[+] Successfully downloaded dataset to: {dataset.location}")
    except Exception as e:
        print(f"[ERROR] Failed to download from Roboflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
