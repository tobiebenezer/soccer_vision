-- =======================================================================
-- Soccer Analytics Pipeline Database Schema
-- Decouples Computer Vision tracks from Exact Player Identity & Rosters
-- =======================================================================

PRAGMA foreign_keys = ON;

-- 1. Teams Table
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT,
    primary_color TEXT,      -- HEX code, e.g. '#FFFFFF' (used for kit clustering)
    secondary_color TEXT,    -- HEX code, e.g. '#0000FF'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Players Table (Master Registry of Exact Player Names)
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    team_id TEXT REFERENCES teams(team_id) ON DELETE SET NULL,
    full_name TEXT NOT NULL,
    known_name TEXT,         -- e.g. 'Son', 'Neymar Jr'
    jersey_number INTEGER NOT NULL,
    primary_position TEXT,    -- 'GK', 'CB', 'LB', 'RB', 'CM', 'LW', 'RW', 'ST'
    height_cm REAL,
    dominant_foot TEXT,      -- 'left', 'right', 'both'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Matches Table
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    home_team_id TEXT REFERENCES teams(team_id),
    away_team_id TEXT REFERENCES teams(team_id),
    venue TEXT NOT NULL,
    match_date TIMESTAMP NOT NULL,
    video_fps REAL DEFAULT 25.0,
    pitch_length_m REAL DEFAULT 105.0,
    pitch_width_m REAL DEFAULT 68.0,
    homography_calib_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Match Lineups (Roster per Match: Starters & Subs)
CREATE TABLE IF NOT EXISTS match_lineups (
    match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
    player_id TEXT REFERENCES players(player_id) ON DELETE CASCADE,
    team_id TEXT REFERENCES teams(team_id),
    jersey_number INTEGER NOT NULL,
    starter_position TEXT,   -- 'GK', 'LB', 'CB', 'RB', 'CM', 'CF', 'SUB'
    is_starter BOOLEAN NOT NULL DEFAULT 1,
    subbed_in_minute INTEGER,
    subbed_out_minute INTEGER,
    PRIMARY KEY (match_id, player_id)
);

-- 5. Track Identity Mapping
-- Maps anonymous CV track_ids to exact players via OCR, Formation Matching, or UI Tagging
CREATE TABLE IF NOT EXISTS track_identities (
    match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    player_id TEXT REFERENCES players(player_id) ON DELETE SET NULL,
    team_id TEXT REFERENCES teams(team_id) ON DELETE SET NULL,
    assigned_via TEXT NOT NULL, -- 'jersey_ocr', 'kickoff_formation', 'manual_tag', 'reid'
    confidence REAL DEFAULT 1.0,
    assigned_at_frame INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (match_id, track_id)
);

-- 6. Player Tracking Frame Data (Time-series coordinates on calibrated pitch)
CREATE TABLE IF NOT EXISTS player_tracking_frames (
    match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    track_id INTEGER NOT NULL,
    player_id TEXT REFERENCES players(player_id) ON DELETE SET NULL,
    team_id TEXT REFERENCES teams(team_id) ON DELETE SET NULL,
    pitch_x REAL NOT NULL,     -- 0.0 to 105.0 meters (standard pitch)
    pitch_y REAL NOT NULL,     -- 0.0 to 68.0 meters
    speed_m_s REAL DEFAULT 0.0,
    accel_m_s2 REAL DEFAULT 0.0,
    raw_bbox_xywh TEXT,        -- "[x, y, w, h]" in pixel coordinates for debug
    PRIMARY KEY (match_id, frame_idx, track_id)
);

-- 7. Ball Tracking Frame Data (Specialized Kalman filtered coordinates)
CREATE TABLE IF NOT EXISTS ball_tracking_frames (
    match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    pitch_x REAL NOT NULL,
    pitch_y REAL NOT NULL,
    speed_m_s REAL DEFAULT 0.0,
    is_interpolated BOOLEAN DEFAULT 0, -- 1 if imputed via Kalman during occlusion
    PRIMARY KEY (match_id, frame_idx)
);

-- 8. Derived Match Events (Touches, Passes, Transitions, Shots)
CREATE TABLE IF NOT EXISTS match_events (
    event_id TEXT PRIMARY KEY,
    match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    event_type TEXT NOT NULL,  -- 'touch', 'pass', 'transition', 'shot', 'turnover'
    player_id TEXT REFERENCES players(player_id) ON DELETE SET NULL,
    team_id TEXT REFERENCES teams(team_id) ON DELETE SET NULL,
    start_pitch_x REAL NOT NULL,
    start_pitch_y REAL NOT NULL,
    end_pitch_x REAL,
    end_pitch_y REAL,
    receiver_player_id TEXT REFERENCES players(player_id) ON DELETE SET NULL,
    success BOOLEAN DEFAULT 1,
    details_json TEXT
);

-- Indexes for lightning fast spatial-temporal analytics queries
CREATE INDEX IF NOT EXISTS idx_tracking_match_player ON player_tracking_frames(match_id, player_id);
CREATE INDEX IF NOT EXISTS idx_tracking_frame ON player_tracking_frames(match_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_events_match_player ON match_events(match_id, player_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON match_events(match_id, event_type);
