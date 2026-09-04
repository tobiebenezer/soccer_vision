# Football-CV: Soccer Video Analytics Pipeline

A modular, production-ready computer vision and spatial-temporal analytics pipeline for soccer footage (targeting NPL, grassroots, and broadcast video).

Designed to run lightweight locally, push to Git, and execute heavy data acquisition and training on **Kaggle GPU**.

---

## Repository Structure

```text
soccer_monitor/
├── .gitignore                      # Excludes large datasets and model weights
├── README.md                       # Comprehensive guide and documentation
├── requirements.txt                # Pip dependencies
├── configs/
│   ├── dataset.yaml                # 4-class YOLO configuration
│   └── train_kaggle.yaml           # High-resolution (960px) training hyperparameters
├── schema/
│   └── schema.sql                  # Relational DB schema connecting CV tracks to exact player names
├── data/
│   ├── raw/                        # Raw source datasets (SoccerNet, Roboflow, NPL)
│   ├── extracted_frames/           # Frames extracted from venue footage
│   ├── staging/                    # Consolidated staging before split
│   ├── labeled/                    # Final YOLO dataset partitioned by match
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── calibration/                # Homography matrices per venue/camera
│   └── manifest.csv                # Master image tracking & provenance
├── scripts/
│   ├── download_soccernet.py       # Public SoccerNet-V3 / GameState downloader (MIT)
│   ├── download_roboflow.py        # Roboflow downloader with strict CC-BY/MIT license check
│   ├── extract_frames.py           # NPL video frame extraction with systematic naming
│   ├── calibrate_pitch.py          # N >= 4 points RANSAC pitch homography calculator
│   ├── merge_datasets.py           # 4-class remapper & manifest generator
│   ├── split_by_match.py           # Strict match-level partition (zero frame leakage)
│   └── validate_dataset.py         # Class balance check & ball ratio audit (<10% warning)
└── notebooks/
    └── kaggle_phase0_phase1.ipynb  # End-to-end Kaggle GPU execution notebook
```

---

## Labeling Schema (4 Classes)

Detection classes are strictly **team-agnostic**:
```yaml
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
```

> [!NOTE]
> **Why team identity is not in the detector**: Training YOLO on `player_team_a` / `player_team_b` fails when evaluating matches where teams wear different kit colors (e.g. green vs yellow). Team assignment is performed downstream via kit-color clustering on bounding box crops, and mapped to exact player rosters.

---

## Database Architecture: Mapping CV Tracks to Exact Player Names

The computer vision models produce anonymous detections: `(frame_idx, track_id=4, pitch_x=45.2m, pitch_y=28.1m)`.

The database schema in [`schema/schema.sql`](file:///home/sifu/Documents/soccer_monitor/schema/schema.sql) links these tracks to real players:
1. **`players`**: Registry of exact player full names, known names, jersey numbers, and primary positions.
2. **`match_lineups`**: Starting XI and substitutes for each match.
3. **`track_identities`**: Binds `(match_id, track_id)` $\to$ `player_id`.
   - **Kickoff Formation Matching**: Automated matching of the 11 tracks to starting lineup coordinates at 00:00 via Hungarian algorithm.
   - **Jersey Number OCR**: Reads jersey numbers when players face away from the camera.
   - **1-Click Analyst Tagging**: Analyst clicks on track #4 in UI and selects the player name from the roster.
4. **`player_tracking_frames`**: Calibrated pitch coordinates $(x, y)$ in meters with calculated speed and acceleration.
5. **`match_events`**: Derived touches, passes (with sender and receiver player IDs), shots, and transitions.

To initialize the SQLite database:
```bash
sqlite3 soccer_analytics.db < schema/schema.sql
```

---

## Sourcing NPL Footage Data

1. **State Federation YouTube Channels (1080p Public Broadcasts)**:
   - [Football NSW YouTube](https://www.youtube.com/@footballnsw) (Men's & Women's full matches)
   - [Football Victoria YouTube](https://www.youtube.com/@footballfedvic)
   - [BarTV Sports](https://www.youtube.com)
   - Download sample match clips using `yt-dlp`:
     ```bash
     yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" \
       --download-sections "*00:10:00-00:20:00" \
       "<YOUTUBE_URL>" -o "npl_clip.mp4"
     ```
2. **NPL.tv**:
   - Official OTT platform ([npl.tv](https://www.npl.tv)) with match replays across multiple state federations.
3. **Club Sideline Cameras (Veo / Hudl)**:
   - High-tripod wide-angle tactical footage obtained from club coaches/analysts in exchange for match reports.

---

## Recommended Bounding Box Labeling Workflow

For labeling your own NPL footage:
1. Extract frames at 1 frame every 2 seconds:
   ```bash
   python scripts/extract_frames.py --video-path match.mp4 --venue lambert_park --cam cam_main --match-id match01 --sample-rate-fps 0.5
   ```
2. Upload video or extracted frames to **[CVAT (cvat.ai)](https://cvat.ai)** or use **[AnyLabeling](https://github.com/vietanhdev/anylabeling)** desktop app.
3. Use CVAT's **linear interpolation** across frames (draw a box on frame 0 and frame 40, CVAT interpolates frames 1–39 automatically).
4. Export annotations in **YOLO 1.1** format and place into `data/raw/npl_footage/`.

---

## Running on Kaggle GPU

Since local disk and memory are constrained, run the heavy pipeline on Kaggle:

### 1. Push Repository to GitHub
```bash
git add .
git commit -m "Initialize Football-CV Phase 0 & Phase 1 pipeline"
git push origin main
```

### 2. In Kaggle Notebook
1. Create a new notebook with GPU accelerator enabled (T4 or P100).
2. Upload or clone the repository:
   ```python
   !git clone https://github.com/<YOUR_USERNAME>/soccer_monitor.git
   %cd soccer_monitor
   ```
3. Run `notebooks/kaggle_phase0_phase1.ipynb` cell-by-cell:
   - Downloads public datasets.
   - Remaps to 4 classes and enforces match-level train/val/test split.
   - Audits ball-to-player ratio (<10% warning).
   - Trains YOLOv8s at `imgsz=960` with ball-specific AP metrics.
   - Exports the fine-tuned model to ONNX.
