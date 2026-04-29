# utils.py
# Helper & redundant-utility functions for EMG Facial Muscle Detection System

from __future__ import annotations

import csv
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from config import CLASSES, CLASS_COLORS_BGR, DB_CONFIG, PREPROCESS_CONFIG, THRESHOLDS
from logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIGNAL PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply bandpass filter (20–450 Hz) to remove noise outside EMG band."""
    b, a = butter_bandpass(
        PREPROCESS_CONFIG["bandpass_low"],
        PREPROCESS_CONFIG["bandpass_high"],
        fs,
    )
    return filtfilt(b, a, signal, axis=0)


def apply_notch(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply notch filter to remove powerline interference (50 Hz)."""
    freq = PREPROCESS_CONFIG["notch_freq_hz"]
    b, a = iirnotch(freq, Q=30, fs=fs)
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize EMG signal per channel."""
    if method == "zscore":
        mean = signal.mean(axis=0)
        std = signal.std(axis=0) + 1e-8
        return (signal - mean) / std
    elif method == "minmax":
        mn = signal.min(axis=0)
        mx = signal.max(axis=0)
        return (signal - mn) / (mx - mn + 1e-8)
    return signal


# ══════════════════════════════════════════════════════════════════════════════
# 2. HAND-CRAFTED EMG FEATURES (per window, per channel)
# ══════════════════════════════════════════════════════════════════════════════

def feat_rms(window: np.ndarray) -> np.ndarray:
    """Root Mean Square — proxy for muscle contraction force."""
    return np.sqrt(np.mean(window ** 2, axis=0))


def feat_mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value — average rectified EMG amplitude."""
    return np.mean(np.abs(window), axis=0)


def feat_wl(window: np.ndarray) -> np.ndarray:
    """Waveform Length — cumulative length of the signal, measures complexity."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def feat_zc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Zero Crossings — counts sign changes (frequency content indicator)."""
    sign_changes = np.diff(np.sign(window), axis=0)
    return np.sum(np.abs(sign_changes) > threshold, axis=0).astype(float)


def feat_ssc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Slope Sign Changes — firing rate proxy."""
    diff1 = np.diff(window, axis=0)
    ssc = ((diff1[:-1] * diff1[1:]) < -threshold)
    return ssc.sum(axis=0).astype(float)


def feat_var(window: np.ndarray) -> np.ndarray:
    """Variance of the EMG signal."""
    return np.var(window, axis=0)


FEATURE_FUNCS = {
    "rms": feat_rms,
    "mav": feat_mav,
    "wl":  feat_wl,
    "zc":  feat_zc,
    "ssc": feat_ssc,
    "var": feat_var,
}


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract the configured feature set from a single EMG window.

    Args:
        window: shape (window_size, n_channels)

    Returns:
        feature_vector: shape (n_features * n_channels,)
    """
    feats = []
    for feat_name in PREPROCESS_CONFIG["feature_set"]:
        fn = FEATURE_FUNCS.get(feat_name)
        if fn:
            feats.append(fn(window))
    return np.concatenate(feats)  # flat vector


def is_active_signal(window: np.ndarray) -> bool:
    """Return True if RMS exceeds noise floor (muscle is active)."""
    rms = feat_rms(window)
    return bool(np.any(rms > THRESHOLDS["rms_noise_floor"]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROLLING STRESS SCORE
# ══════════════════════════════════════════════════════════════════════════════

class StressScoreTracker:
    """
    Maintains a rolling window of emotion predictions and computes
    a cumulative stress score (fraction of stressed/fatigued frames).
    """

    def __init__(self, window_sec: int, sample_rate_hz: int):
        capacity = window_sec * sample_rate_hz
        self._buf: deque = deque(maxlen=capacity)

    def update(self, label: str) -> float:
        """Add new label, return current stress score (0.0 – 1.0)."""
        self._buf.append(label)
        if not self._buf:
            return 0.0
        stress_labels = {"Stressed", "Fatigued", "Concentrated"}
        score = sum(1 for l in self._buf if l in stress_labels) / len(self._buf)
        return round(score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """Create SQLite tables if they don't exist."""
    db_path = Path(DB_CONFIG["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_CONFIG['table_events']} (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            timestamp    TEXT    NOT NULL,
            emotion      TEXT    NOT NULL,
            confidence   REAL,
            stress_score REAL,
            active_channels INTEGER
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_CONFIG['table_sessions']} (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at   TEXT,
            model_used TEXT,
            notes      TEXT
        )
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", db_path)


def save_event(session_id: str, emotion: str, confidence: float,
               stress_score: float, active_channels: int) -> None:
    """Persist an emotion detection event to SQLite."""
    try:
        conn = sqlite3.connect(DB_CONFIG["path"])
        conn.execute(
            f"INSERT INTO {DB_CONFIG['table_events']} "
            "(session_id, timestamp, emotion, confidence, stress_score, active_channels) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, datetime.utcnow().isoformat(),
             emotion, round(confidence, 4), round(stress_score, 4), active_channels),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log.error("DB write failed: %s", e)


def fetch_events(limit: int = 200) -> List[Dict]:
    """Retrieve recent emotion events for the dashboard."""
    try:
        conn = sqlite3.connect(DB_CONFIG["path"])
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {DB_CONFIG['table_events']} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        log.error("DB read failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_emg_dashboard(frame: np.ndarray,
                          label: str,
                          confidence: float,
                          stress_score: float,
                          channel_rms: np.ndarray,
                          fps: float,
                          inference_ms: float) -> np.ndarray:
    """
    Overlay EMG analysis results onto a display frame.
    Frame can be a blank canvas (np.zeros) or a camera image.

    Overlays:
      - Emotion label + confidence bar
      - Stress score gauge
      - Per-channel RMS bar chart
      - FPS + inference time (submission requirement)
    """
    import cv2

    h, w = frame.shape[:2]
    color = CLASS_COLORS_BGR.get(label, (200, 200, 200))

    # ── Top banner ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 30), -1)
    cv2.putText(frame, f"EMG Emotion: {label}",
                (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Confidence bar
    bar_w = int((w - 350) * confidence)
    cv2.rectangle(frame, (340, 18), (340 + bar_w, 44), color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (340 + bar_w + 8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # ── FPS + Inference time (required by submission spec) ────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Inference: {inference_ms:.1f}ms",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ── Stress score gauge ────────────────────────────────────────────────
    gauge_x, gauge_y = 10, 80
    cv2.putText(frame, f"Stress Score: {stress_score:.2f}",
                (gauge_x, gauge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    gauge_len = 200
    filled = int(gauge_len * stress_score)
    g_color = (0, 0, 220) if stress_score > THRESHOLDS["alert_stress_score"] else (0, 180, 80)
    cv2.rectangle(frame, (gauge_x, gauge_y + 8), (gauge_x + gauge_len, gauge_y + 22), (60, 60, 60), -1)
    cv2.rectangle(frame, (gauge_x, gauge_y + 8), (gauge_x + filled, gauge_y + 22), g_color, -1)

    # ── Per-channel RMS bars ───────────────────────────────────────────────
    bar_start_y = 130
    bar_h = 16
    bar_gap = 4
    max_rms = max(channel_rms.max(), 0.05)
    for i, rms_val in enumerate(channel_rms):
        by = bar_start_y + i * (bar_h + bar_gap)
        bw = int(180 * rms_val / max_rms)
        act = rms_val > THRESHOLDS["rms_noise_floor"]
        bar_color = (0, 200, 120) if act else (60, 60, 60)
        cv2.putText(frame, f"CH{i+1}", (10, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.rectangle(frame, (44, by), (44 + bw, by + bar_h), bar_color, -1)
        cv2.putText(frame, f"{rms_val:.4f}", (44 + bw + 6, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# 6. FPS COUNTER
# ══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling FPS estimator over a sliding window."""
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0] + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MISC
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def export_signal_csv(path: str, row: List) -> None:
    """Append one row of signal data to a CSV log file."""
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            header = ["timestamp", "emotion", "confidence", "stress_score"] + \
                     [f"ch{i+1}_rms" for i in range(8)]
            w.writerow(header)
        w.writerow(row)
