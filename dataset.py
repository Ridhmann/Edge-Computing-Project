# dataset.py
# Kaggle Dataset Download + Loading + Preparation
# EMG-Based Facial Muscle Signal Detection — Emotion & Stress Recognition
#
# 3 Kaggle Datasets:
#  1. kyr0gane/emg-data-for-gestures        (8-channel, 36 subjects)
#  2. caesarlupum/emgdatasetactionrecognition (8-channel forearm EMG)
#  3. meowmeowmeowmeow/ninapro-db5           (benchmark, use first 8ch)
#
# Setup (one-time):
#   pip install kaggle
#   mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# Download:
#   python dataset.py --download-all
#   python dataset.py --build

from __future__ import annotations
import json, os, pickle, subprocess, zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import INPUT_CONFIG, PREPROCESS_CONFIG, CLASSES, TRAIN_CONFIG
from logger import get_logger

log = get_logger(__name__)

# ─── Dataset Registry ─────────────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "emg_gestures": {
        "slug":        "kyr0gane/emg-data-for-gestures",
        "description": "8-channel sEMG gesture dataset — 36 subjects",
        "channels":    8,
        "label_col":   "label",
    },
    "emg_actions": {
        "slug":        "caesarlupum/emgdatasetactionrecognition",
        "description": "8-channel forearm EMG — aggressive vs normal actions",
        "channels":    8,
        "label_col":   "label",
    },
    "ninapro_db5": {
        "slug":        "meowmeowmeowmeow/ninapro-db5",
        "description": "Ninapro DB5 benchmark — use first 8 of 16 channels",
        "channels":    8,
        "label_col":   "restimulus",
    },
}

# Raw label → 5 emotion classes (remap mod-5 for any dataset)
LABEL_REMAP = {0:"Neutral",1:"Happy",2:"Stressed",3:"Concentrated",4:"Fatigued"}
CLASS_TO_INT = {v:k for k,v in CLASSES.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_dataset(key: str, dest: str = "data/raw") -> str:
    ds  = KAGGLE_DATASETS[key]
    out = Path(dest) / key
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.csv")):
        log.info("'%s' already downloaded (%s)", key, out); return str(out)
    log.info("Downloading: %s → %s", ds["slug"], out)
    try:
        subprocess.run(
            ["kaggle","datasets","download","-d",ds["slug"],"-p",str(out),"--unzip"],
            check=True, capture_output=True, text=True)
        log.info("Download complete: %s", out)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error("Download failed: %s\nInstall kaggle CLI + set ~/.kaggle/kaggle.json", e)
        raise
    return str(out)


def download_all_datasets(dest: str = "data/raw") -> Dict[str, str]:
    paths = {}
    for key in KAGGLE_DATASETS:
        try:
            paths[key] = download_dataset(key, dest)
        except Exception as e:
            log.warning("Skipping %s: %s", key, e)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def _find_ch_cols(df: pd.DataFrame, label_col: str) -> List[str]:
    kw = ("ch","channel","emg","sensor","electrode")
    cols = [c for c in df.columns if any(c.lower().startswith(k) for k in kw) and c != label_col]
    if not cols:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in num if c != label_col][1:]  # drop time col
    return cols


def load_csv(filepath: str, n_channels: int, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    if label_col not in df.columns:
        raise ValueError(f"Label col '{label_col}' not found in {filepath}")
    ch_cols = _find_ch_cols(df, label_col)[:n_channels]
    signals = df[ch_cols].values.astype(np.float32)
    labels  = np.array([int(l) % 5 for l in df[label_col].values], dtype=np.int64)
    return signals, labels


def load_dataset(key: str, data_dir: str = "data/raw") -> Tuple[np.ndarray, np.ndarray]:
    ds = KAGGLE_DATASETS[key]
    folder = Path(data_dir) / key
    files  = sorted(folder.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}. Run: python dataset.py --download {key}")
    all_s, all_l = [], []
    for f in files:
        try:
            s, l = load_csv(str(f), ds["channels"], ds["label_col"])
            all_s.append(s); all_l.append(l)
        except Exception as e:
            log.warning("  Skip %s: %s", f.name, e)
    signals = np.concatenate(all_s); labels = np.concatenate(all_l)
    log.info("'%s': %d samples | dist=%s", key, len(signals),
             dict(zip(*np.unique(labels, return_counts=True))))
    return signals, labels


def load_combined(data_dir: str = "data/raw") -> Tuple[np.ndarray, np.ndarray]:
    n_ch = INPUT_CONFIG["channels"]
    all_s, all_l = [], []
    for key in KAGGLE_DATASETS:
        folder = Path(data_dir) / key
        if not folder.exists() or not list(folder.glob("*.csv")):
            log.warning("'%s' not found — skip. Run: python dataset.py --download %s", key, key)
            continue
        try:
            s, l = load_dataset(key, data_dir)
            if s.shape[1] < n_ch:
                s = np.concatenate([s, np.zeros((len(s), n_ch-s.shape[1]), dtype=np.float32)], 1)
            elif s.shape[1] > n_ch:
                s = s[:, :n_ch]
            all_s.append(s); all_l.append(l)
        except Exception as e:
            log.warning("Failed '%s': %s", key, e)
    if not all_s:
        raise RuntimeError("No datasets found. Run: python dataset.py --download-all")
    signals = np.concatenate(all_s); labels = np.concatenate(all_l)
    log.info("Combined: %d samples total", len(signals))
    return signals, labels


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_emg(signals: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    cfg = PREPROCESS_CONFIG
    log.info("Preprocessing: notch(%dHz) → bandpass(%d–%dHz) → z-score",
             cfg["notch_freq_hz"], cfg["bandpass_low"], cfg["bandpass_high"])
    b_n, a_n = iirnotch(cfg["notch_freq_hz"], Q=30, fs=fs)
    out = filtfilt(b_n, a_n, signals, axis=0)
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [cfg["bandpass_low"]/nyq, cfg["bandpass_high"]/nyq], btype="band")
    out = filtfilt(b_bp, a_bp, out, axis=0)
    mean = out.mean(0); std = out.std(0) + 1e-8
    return ((out - mean) / std).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. WINDOWING
# ══════════════════════════════════════════════════════════════════════════════

def sliding_windows(signals: np.ndarray, labels: np.ndarray,
                    window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for start in range(0, len(signals) - window_size + 1, step):
        end = start + window_size
        w   = signals[start:end]
        vals, cnts = np.unique(labels[start:end], return_counts=True)
        X_list.append(w); y_list.append(int(vals[cnts.argmax()]))
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    log.info("Windowed → %d windows (size=%d step=%d)", len(X), window_size, step)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    std  = TRAIN_CONFIG["augment_noise_std"]
    s_lo, s_hi = TRAIN_CONFIG["augment_scale_range"]
    noisy   = X + np.random.normal(0, std, X.shape).astype(np.float32)
    scaled  = X * np.random.uniform(s_lo, s_hi, (len(X),1,1)).astype(np.float32)
    flipped = X[:, ::-1, :].copy()
    X_aug = np.concatenate([X, noisy, scaled, flipped])
    y_aug = np.concatenate([y, y, y, y])
    log.info("Augmented: %d windows (4×)", len(X_aug))
    return X_aug, y_aug


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def save_processed(X: np.ndarray, y: np.ndarray, out_dir: str) -> Dict:
    cfg = PREPROCESS_CONFIG; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=1-cfg["train_split"],
                                                  random_state=cfg["seed"], stratify=y)
    val_r = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=1-val_r,
                                                  random_state=cfg["seed"], stratify=y_tmp)
    for name, (Xs, ys) in [("train",(X_tr,y_tr)),("val",(X_val,y_val)),("test",(X_te,y_te))]:
        np.save(str(out/f"X_{name}.npy"), Xs)
        np.save(str(out/f"y_{name}.npy"), ys)
        log.info("  [%-5s]  X:%s  y:%s", name, Xs.shape, ys.shape)
    le = LabelEncoder(); le.classes_ = np.array(list(CLASSES.values()))
    with open(str(out/"label_encoder.pkl"),"wb") as f: pickle.dump(le, f)
    stats = {"train":len(X_tr),"val":len(X_val),"test":len(X_te),"total":len(X),
             "n_channels":X.shape[2],"window_size":X.shape[1],
             "class_dist":{CLASSES[i]:int(np.sum(y==i)) for i in CLASSES}}
    with open(str(out/"dataset_stats.json"),"w") as f: json.dump(stats, f, indent=2)
    log.info("Class distribution: %s", stats["class_dist"])
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISE
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_dist(y: np.ndarray, save_path: str) -> None:
    names  = [CLASSES[i] for i in sorted(CLASSES)]
    counts = [int(np.sum(y==i)) for i in sorted(CLASSES)]
    colors = ["#8b949e","#3fb950","#f85149","#e3b341","#a371f7"]
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
    bars = ax.bar(names, counts, color=colors, edgecolor="#30363d")
    ax.set_title("EMG Dataset — Class Distribution", color="white", fontweight="bold")
    ax.set_ylabel("Sample Count", color="white"); ax.tick_params(colors="white")
    for b,c in zip(bars,counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+50, str(c),
                ha="center", fontsize=9, color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    log.info("Class dist plot: %s", save_path)


def plot_emg_sample(window: np.ndarray, label: int, save_path: str) -> None:
    n_ch = window.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, n_ch*1.2), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(f"EMG Window Sample — {CLASSES[label]}", color="white",
                 fontsize=12, fontweight="bold")
    clrs = ["#58a6ff","#3fb950","#f85149","#e3b341","#a371f7","#79c0ff","#ffa657","#ff7b72"]
    for i, ax in enumerate(axes):
        ax.plot(window[:,i], color=clrs[i%len(clrs)], linewidth=0.8)
        ax.set_ylabel(f"CH{i+1}", fontsize=8, color="white")
        ax.set_facecolor("#0d1117"); ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    axes[-1].set_xlabel("Sample Index", color="white")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    log.info("EMG sample plot: %s", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(raw_dir="data/raw", processed_dir="data/processed", runs_dir="runs") -> str:
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    log.info("=== Dataset Build Pipeline ===")
    signals, labels = load_combined(raw_dir)
    cleaned = preprocess_emg(signals, fs=INPUT_CONFIG["sample_rate_hz"])
    X, y    = sliding_windows(cleaned, labels, INPUT_CONFIG["window_size"], INPUT_CONFIG["window_step"])
    plot_class_dist(y, str(Path(runs_dir)/"class_distribution.png"))
    idx = int(np.random.randint(len(X)))
    plot_emg_sample(X[idx], int(y[idx]), str(Path(runs_dir)/"emg_sample.png"))
    X, y = augment(X, y)
    stats = save_processed(X, y, processed_dir)
    log.info("=== Dataset ready — %d total windows ===", stats["total"])
    return processed_dir


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EMG Kaggle Dataset Manager")
    parser.add_argument("--download",     metavar="KEY", help="Download one dataset")
    parser.add_argument("--download-all", action="store_true", help="Download all 3 datasets")
    parser.add_argument("--build",        action="store_true", help="Preprocess + window + split")
    parser.add_argument("--info",         action="store_true", help="Show dataset registry")
    parser.add_argument("--raw-dir",      default="data/raw")
    parser.add_argument("--processed-dir",default="data/processed")
    args = parser.parse_args()

    if args.info:
        print("\n📊 EMG Kaggle Dataset Registry\n" + "─"*55)
        for k, ds in KAGGLE_DATASETS.items():
            print(f"\n  Key  : {k}\n  Slug : {ds['slug']}\n  Info : {ds['description']}")
            print(f"  CMD  : kaggle datasets download -d {ds['slug']} -p data/raw/{k}/")
        print()
    if args.download:     download_dataset(args.download, args.raw_dir)
    if args.download_all: download_all_datasets(args.raw_dir)
    if args.build:        build_dataset(args.raw_dir, args.processed_dir)
    if not any([args.download, args.download_all, args.build, args.info]):
        parser.print_help()
