# config.py
# EMG-Based Facial Muscle Signal Detection
# Pipeline: YOLOv8 Face Detection → ROI Crop → 3 Model Emotion Classification
# Models: CNN-1D | Bi-LSTM | Transformer

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ─── Input Source ──────────────────────────────────────────────────────────────
INPUT_CONFIG = {
    "source":        0,            # 0 = webcam | "video.mp4" | "rtsp://..."
    "session_id":    "SESSION_01",
    "fps_target":    15,
    "frame_width":   1280,
    "frame_height":  720,
}

# ─── YOLOv8 Face Detection ────────────────────────────────────────────────────
YOLO_CONFIG = {
    "model_path":     str(BASE_DIR / "models" / "yolov8n.pt"),  # auto-downloads
    "conf_threshold": 0.50,
    "iou_threshold":  0.45,
    "imgsz":          640,
    "device":         "cpu",       # "cpu" | "cuda:0"
    "classes":        [0],         # class 0 = person (COCO)
    "roi_padding":    20,          # pixels padding around detected face bbox
}

# ─── EMG Signal Config (from CSV dataset) ─────────────────────────────────────
SIGNAL_CONFIG = {
    "csv_path":       str(BASE_DIR / "data" / "raw" / "emg_gestures" / "EMG-data.csv"),
    "channels":       8,
    "sample_rate_hz": 1000,
    "window_size":    200,
    "window_step":    50,
}

# ─── Model Paths ───────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "cnn1d":       str(BASE_DIR / "models" / "cnn1d_best.pt"),
    "lstm":        str(BASE_DIR / "models" / "lstm_best.pt"),
    "transformer": str(BASE_DIR / "models" / "transformer_best.pt"),
}

# Active model for inference
ACTIVE_MODEL = "cnn1d"   # "cnn1d" | "lstm" | "transformer"

# ─── Model Architectures ───────────────────────────────────────────────────────
MODEL_CONFIG = {
    "cnn1d": {
        "in_channels":  8,
        "num_classes":  5,
        "kernel_sizes": [3, 5, 7],
        "filters":      [64, 128, 256],
        "dropout":      0.4,
    },
    "lstm": {
        "input_size":    8,
        "hidden_size":   128,
        "num_layers":    2,
        "num_classes":   5,
        "dropout":       0.3,
        "bidirectional": True,
    },
    "transformer": {
        "input_size":      8,
        "d_model":         64,
        "nhead":           4,
        "num_layers":      3,
        "num_classes":     5,
        "dropout":         0.3,
        "dim_feedforward": 256,
        "max_seq_len":     200,
    },
}

# ─── Emotion Classes ───────────────────────────────────────────────────────────
CLASSES = {
    0: "Neutral",
    1: "Happy",
    2: "Stressed",
    3: "Concentrated",
    4: "Fatigued",
}

CLASS_COLORS_BGR = {
    "Neutral":       (160, 160, 160),
    "Happy":         (0,   200,  80),
    "Stressed":      (0,     0, 220),
    "Concentrated":  (220, 140,   0),
    "Fatigued":      (100, 100, 200),
}

# ─── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "confidence_min":    0.55,
    "stress_window_sec": 5,
    "alert_stress_score":0.70,
}

# ─── Preprocessing ────────────────────────────────────────────────────────────
PREPROCESS_CONFIG = {
    "raw_data_dir":  str(BASE_DIR / "data" / "raw"),
    "processed_dir": str(BASE_DIR / "data" / "processed"),
    "train_split":   0.70,
    "val_split":     0.15,
    "test_split":    0.15,
    "seed":          42,
    "notch_freq_hz": 50,
    "bandpass_low":  20,
    "bandpass_high": 450,
    "normalize":     "zscore",
    "feature_set":   ["rms", "mav", "wl", "zc", "ssc", "var"],
}

# ─── Training ─────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs":              20,
    "batch_size":          32,
    "lr":                  1e-3,
    "lr_scheduler":        "cosine",
    "weight_decay":        1e-4,
    "patience":            3,
    "device":              "cpu",
    "save_dir":            str(BASE_DIR / "models"),
    "results_dir":         str(BASE_DIR / "runs"),
    "num_workers":         0,       # 0 = safer on Windows
    "augment":             True,
    "augment_noise_std":   0.005,
    "augment_scale_range": (0.9, 1.1),
}

# ─── Database ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "path":           str(BASE_DIR / "data" / "emg_results.db"),
    "table_events":   "emotion_events",
    "table_sessions": "sessions",
}

# ─── Flask Dashboard ──────────────────────────────────────────────────────────
FLASK_CONFIG = {
    "host":       "0.0.0.0",
    "port":       5000,
    "debug":      False,
    "secret_key": os.environ.get("FLASK_SECRET", "emg-yolo-secret"),
}

# ─── Log File Path ────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "log_dir":      str(BASE_DIR / "logs"),
    "log_file":     "emg_system.log",
    "level":        "INFO",
    "max_bytes":    5 * 1024 * 1024,
    "backup_count": 3,
}
SIGNAL_CONFIG = {

    "csv_path": r"C:\Users\msi\Downloads\files (3)\data\raw\emg_gestures\EMG-data.csv",
    "channels": 8,
    "sample_rate_hz": 1000,
    "window_size": 200,
    "window_step": 50,
}

YOLO_CONFIG = {
    "model_path": "models/yolov8n.pt",
    "conf_threshold": 0.50,
    "iou_threshold": 0.45,
    "imgsz": 640,
    "device": "cpu",
    "classes": [0],
    "roi_padding": 20,
}
INPUT_CONFIG = {
    "source":       0,
    "session_id":   "SESSION_01",
    "fps_target":   15,
    "frame_width":  640,
    "frame_height": 360,
    "channels":     8,
    "sample_rate_hz": 1000,
    "window_size":  200,
    "window_step":  50,
}