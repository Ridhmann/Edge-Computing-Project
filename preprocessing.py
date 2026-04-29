# preprocessing.py — Fixed for EMG-data.csv with 36+ classes

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import CLASSES, PREPROCESS_CONFIG, SIGNAL_CONFIG, TRAIN_CONFIG
from logger import get_logger

log = get_logger(__name__)

def load_emg_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    log.info("Loading: %s", filepath)
    df = pd.read_csv(filepath)
    log.info("Columns: %s", df.columns.tolist())
    log.info("Shape: %s", df.shape)

    # Find channel columns
    kw = ("ch", "channel", "emg", "sensor", "electrode")
    ch_cols = [c for c in df.columns if any(c.lower().startswith(k) for k in kw)]
    if not ch_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ch_cols  = num_cols[:-1]

    label_col = df.columns[-1]
    ch_cols   = ch_cols[:SIGNAL_CONFIG["channels"]]

    log.info("Using channels: %s", ch_cols)
    log.info("Label column: %s  unique values: %s",
             label_col, sorted(df[label_col].unique().tolist()))

    signals = df[ch_cols].values.astype(np.float32)

    # ── FIX: remap ANY number of classes → 5 emotion classes ──────────────
    raw_labels = df[label_col].values
    unique_labels = sorted(set(raw_labels))
    n_unique = len(unique_labels)
    log.info("Found %d unique labels → remapping to 5 emotion classes", n_unique)

    # Map each unique label to 0-4 by dividing into 5 equal groups
    label_to_emotion = {}
    for i, lbl in enumerate(unique_labels):
        emotion_idx = int(i * 5 / n_unique)  # spread across 0-4
        label_to_emotion[lbl] = emotion_idx

    labels = np.array([label_to_emotion[l] for l in raw_labels], dtype=np.int64)
    log.info("After remap — class dist: %s",
             dict(zip(*np.unique(labels, return_counts=True))))
    return signals, labels


def preprocess_signals(signals: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    cfg = PREPROCESS_CONFIG
    log.info("Filtering: notch(%dHz) bandpass(%d-%dHz) zscore",
             cfg["notch_freq_hz"], cfg["bandpass_low"], cfg["bandpass_high"])
    b_n, a_n = iirnotch(cfg["notch_freq_hz"], Q=30, fs=fs)
    out = filtfilt(b_n, a_n, signals, axis=0)
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [cfg["bandpass_low"]/nyq, cfg["bandpass_high"]/nyq], btype="band")
    out = filtfilt(b_bp, a_bp, out, axis=0)
    mean = out.mean(axis=0); std = out.std(axis=0) + 1e-8
    return ((out - mean) / std).astype(np.float32)


def sliding_windows(signals, labels, window_size, step):
    X_list, y_list = [], []
    for start in range(0, len(signals) - window_size + 1, step):
        end = start + window_size
        w   = signals[start:end]
        vals, cnts = np.unique(labels[start:end], return_counts=True)
        X_list.append(w)
        y_list.append(int(vals[cnts.argmax()]))
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    log.info("Windowed: %d windows (size=%d step=%d)", len(X), window_size, step)
    return X, y


def augment(X, y):
    std = TRAIN_CONFIG["augment_noise_std"]
    s_lo, s_hi = TRAIN_CONFIG["augment_scale_range"]
    noisy   = X + np.random.normal(0, std, X.shape).astype(np.float32)
    scaled  = X * np.random.uniform(s_lo, s_hi, (len(X),1,1)).astype(np.float32)
    flipped = X[:, ::-1, :].copy()
    X_aug = np.concatenate([X, noisy, scaled, flipped])
    y_aug = np.concatenate([y, y, y, y])
    log.info("Augmented: %d windows", len(X_aug))
    return X_aug, y_aug


def save_processed(X, y, out_dir):
    cfg = PREPROCESS_CONFIG
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=1-cfg["train_split"],
        random_state=cfg["seed"], stratify=y)
    val_r = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1-val_r,
        random_state=cfg["seed"], stratify=y_tmp)
    for name, (Xs, ys) in [("train",(X_tr,y_tr)),("val",(X_val,y_val)),("test",(X_te,y_te))]:
        np.save(str(out/f"X_{name}.npy"), Xs)
        np.save(str(out/f"y_{name}.npy"), ys)
        log.info("  [%-5s] X:%s  y:%s", name, Xs.shape, ys.shape)
    le = LabelEncoder()
    le.classes_ = np.array(list(CLASSES.values()))
    with open(str(out/"label_encoder.pkl"),"wb") as f: pickle.dump(le, f)
    log.info("Saved to: %s", out_dir)
    return {"train":len(X_tr),"val":len(X_val),"test":len(X_te)}


def run_preprocessing() -> str:
    cfg = PREPROCESS_CONFIG; sig = SIGNAL_CONFIG
    log.info("=== Preprocessing Pipeline ===")
    signals, labels = load_emg_csv(sig["csv_path"])
    signals = signals[:500000]  # use first 500k samples
    labels  = labels[:500000]
    cleaned = preprocess_signals(signals, fs=sig["sample_rate_hz"])
    X, y    = sliding_windows(cleaned, labels, sig["window_size"], sig["window_step"])
    X, y    = augment(X, y)
    stats   = save_processed(X, y, cfg["processed_dir"])
    log.info("=== Done: %d total windows ===", sum(stats.values()))
    return cfg["processed_dir"]


if __name__ == "__main__":
    run_preprocessing()
