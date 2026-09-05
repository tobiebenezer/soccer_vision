# Changelog

All notable changes to the **Football-CV: NPFL Soccer Video Analytics Pipeline** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] - 2026-09-05

### Added
- **Self-Contained Kaggle Environment Initialization**:
  - Notebook Cell 1 now automatically detects if running inside a Kaggle session (`/kaggle/working`), clones `soccer_vision` from GitHub, navigates into the repo directory, and checks for GPU acceleration.
  - Guarantees zero `FileNotFoundError` or script path errors when importing the notebook into Kaggle.
- **Root Runner Notebook** (`kaggle_run.ipynb`):
  - Created a mirror of the notebook at the repository root for convenient drag-and-drop upload into Kaggle.
- **1-Click Model Weight Export for Kaggle UI**:
  - Added an automated export cell copying trained `best.pt` and `best.onnx` into `/kaggle/working/exported_models/`, making model weights immediately downloadable from Kaggle's "Output" tab without navigating subfolders.

---

## [0.2.0] - 2026-09-05

### Added
- **1-Click Open Soccer Dataset Downloader** (`scripts/download_open_dataset.py`):
  - Directly downloads pre-annotated soccer YOLO detection data (`Adit-jain/Soccana_player_ball_detection_v1`) from Hugging Face.
  - Zero API keys and zero authentication required.
  - Automatically extracts and stages up to 1,000 images with players, referees, and balls.
- **Offline Synthetic Sample Data Generator** (`scripts/create_sample_data.py`):
  - Generates realistic soccer pitch images and valid YOLO bounding boxes for instant offline smoke-testing in 5 seconds.
- **`open_soccer` and `sample_soccer` Ingestion** (`scripts/merge_datasets.py`):
  - Ingests and remaps new open dataset sources into unified staging and `manifest.csv`.

### Fixed & Enhanced
- **SoccerNet-GSR Zip Extraction Overhaul** (`scripts/download_soccernet.py`):
  - Fixed repository endpoint to official `SoccerNet/SN-GSR-2024`.
  - Switched default download from 25 GB training set to compact `valid.zip` (~3 GB).
  - Implemented selective Zip stream extraction: extracts only the first $N$ match sequences requested (`--max-matches`), preventing Kaggle disk space exhaustion.
  - Added automatic `.zip` cleanup after extraction.
- **Kaggle Notebook Polish** (`notebooks/kaggle_phase0_phase1.ipynb`):
  - Rebuilt all 21 notebook cells with strict cell-type validation (`markdown` vs `code`).
  - Added explicit choices for Option A (1-click open dataset), Option B (SoccerNet GSR), Option C (Offline sample test), and Option D (NPFL YouTube video extraction via `yt-dlp`).
  - Added ONNX export and per-class AP50 validation display.

---

## [0.1.0] - 2026-09-04

### Added

#### Core Architecture & Pipeline
- **4-Class Team-Agnostic Schema** (`configs/dataset.yaml`):
  - `0: player`
  - `1: goalkeeper`
  - `2: referee`
  - `3: ball`
  - Decoupled team identity from detection to ensure models generalize across arbitrary kit colors.
- **Kaggle High-Resolution Training Config** (`configs/train_kaggle.yaml`):
  - Configured `imgsz: 960` with AdamW optimizer and soccer-specific augmentations (mosaic, horizontal flip) to preserve small-object ball features.
- **Relational Database Schema** (`schema/schema.sql`):
  - SQLite/PostgreSQL schema mapping anonymous CV tracks to exact player identities:
    - `teams`: Team registry with primary/secondary kit HEX codes for K-Means color clustering.
    - `players`: Master roster with full names, known names, jersey numbers, and positions.
    - `matches` & `match_lineups`: Matchday fixtures and Starting XI / substitute sheets.
    - `track_identities`: Binds `(match_id, track_id)` to `player_id` via Kickoff Formation Matching, Jersey Number OCR, or UI Tagging.
    - `player_tracking_frames` & `ball_tracking_frames`: Pitch coordinates in meters with speed and acceleration.
    - `match_events`: Derived spatial-temporal events (touches, passes, shots, turnovers).

#### Phase 0 & Phase 1 Tooling (`scripts/`)
- **`download_soccernet.py`**:
  - Fetches SoccerNet-V3 / GameState detection assets from Hugging Face Hub (MIT license, no NDA required).
  - Automatically converts annotations to normalized YOLO format `(class_id, xc, yc, w, h)`.
- **`download_roboflow.py`**:
  - Queries Roboflow project metadata API and strictly enforces permissive open-source licenses (`CC-BY`, `MIT`, `Apache 2.0`, `CC0`).
  - Automatically rejects datasets containing `NC` (Non-Commercial) or proprietary licenses to guarantee commercial compliance.
- **`extract_frames.py`**:
  - Extracts frames from venue match videos with customizable sampling rates (FPS or frame stride).
  - Enforces systematic naming: `{venue}_{cam}_{match_id}_f{frame_idx:06d}.jpg`.
- **`calibrate_pitch.py`**:
  - Solves $3 \times 3$ projective homography matrix mapping image pixels $(u, v)$ to real-world pitch coordinates $(X, Y)$ on a standard $105 \times 68\text{m}$ pitch.
  - Employs RANSAC on $N \ge 4$ correspondence points and validates mean reprojection error in meters.
  - Includes interactive GUI mode and template coordinate JSON generator.
- **`merge_datasets.py`**:
  - Merges diverse raw sources (`soccernet`, `roboflow`, `npfl_footage`).
  - Remaps diverse source class naming to the unified 4-class schema.
  - Maintains `data/manifest.csv` tracking image provenance, source licenses, and match IDs.
- **`split_by_match.py`**:
  - Partitions dataset into train, validation, and test splits strictly at the **match level** (zero frame leakage across splits).
- **`validate_dataset.py`**:
  - Scans labeled datasets and computes per-class instance distributions.
  - Performs ball-to-player ratio validation and triggers a critical warning if ball instances $< 10\%$ of player instances.

#### Remote Execution Scaffold
- **`notebooks/kaggle_phase0_phase1.ipynb`**:
  - End-to-end Jupyter notebook structured for execution on Kaggle GPU (P100 / T4 x2).
  - Covers setup, downloading, remapping, match splitting, dataset auditing, YOLOv8s fine-tuning, per-class AP inspection, and ONNX export.
- **`requirements.txt`**:
  - Pinned core dependencies (`ultralytics`, `opencv-python-headless`, `roboflow`, `huggingface_hub`, `pandas`, `pyyaml`, `yt-dlp`).
- **`.gitignore`**:
  - Protects repository from committing raw videos, image caches, and weights while tracking the complete directory skeleton via `.gitkeep`.

---

### Changed

- **Domain Realignment to Nigeria Premier Football League (NPFL)**:
  - Updated footage staging directory to `data/raw/npfl_footage/`.
  - Added documentation and download workflows for official NPFL YouTube broadcasts, the NPFL Live App, and club media channels (Sporting Lagos FC, Remo Stars SC, Enyimba FC, Kano Pillars FC, Rangers International FC).
  - Configured default venue benchmarks in extraction tools for Nigerian stadia (*Mobolaji Johnson Arena*, *Remo Stars Stadium*, *Godswill Akpabio Stadium*, *Enyimba International Stadium*).
- **Deferred Library Imports**:
  - Refactored heavy dependencies (`cv2`, `numpy`, `huggingface_hub`, `roboflow`) across all scripts to be imported on demand inside runtime functions.
  - Enables all scripts to execute `--help` cleanly on local, lightweight machines without installing GPU dependencies.
- **Universal YOLO Label Path Resolution**:
  - Upgraded label lookup in `merge_datasets.py` to use path replacement (`/images/` $\to$ `/labels/`) alongside dynamic parent checks to handle any standard YOLO dataset folder hierarchy.
- **Match-Split Boundary Guards**:
  - Hardened `split_by_match.py` calculation of split boundaries to prevent partition overflow on datasets with few matches ($N \le 3$).
