# training.py
# 3 Model Architectures for EMG Emotion Recognition
# MODEL 1: CNN-1D  | MODEL 2: Bi-LSTM  | MODEL 3: Transformer
#
# python training.py --model cnn1d
# python training.py --model lstm
# python training.py --model transformer
# python training.py --all

from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from config import CLASSES, MODEL_CONFIG, TRAIN_CONFIG, PREPROCESS_CONFIG
from logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — CNN-1D
# ══════════════════════════════════════════════════════════════════════════════
class EMG_CNN1D(nn.Module):
    """
    Multi-Scale 1D CNN — 3 parallel branches with kernel sizes 3, 5, 7.
    Input:  (B, 200, 8)
    Output: (B, num_classes)
    """
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg["in_channels"]
        nc    = cfg["num_classes"]
        ks    = cfg["kernel_sizes"]
        filt  = cfg["filters"]
        drop  = cfg["dropout"]

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, filt[0], k, padding=k//2),
                nn.BatchNorm1d(filt[0]), nn.ReLU(),
                nn.Conv1d(filt[0], filt[1], k, padding=k//2),
                nn.BatchNorm1d(filt[1]), nn.ReLU(),
                nn.AdaptiveAvgPool1d(32),
            ) for k in ks
        ])
        merged = filt[1] * len(ks)
        self.head = nn.Sequential(
            nn.Conv1d(merged, filt[2], 3, padding=1),
            nn.BatchNorm1d(filt[2]), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(filt[2], 128), nn.ReLU(),
            nn.Dropout(drop/2), nn.Linear(128, nc),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Bi-LSTM
# ══════════════════════════════════════════════════════════════════════════════
class EMG_BiLSTM(nn.Module):
    """
    Bidirectional LSTM — captures temporal EMG patterns.
    Input:  (B, 200, 8)
    Output: (B, num_classes)
    """
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0,
            bidirectional=cfg["bidirectional"],
        )
        fc_in = cfg["hidden_size"] * (2 if cfg["bidirectional"] else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(fc_in, 128), nn.ReLU(),
            nn.Dropout(cfg["dropout"]/2),
            nn.Linear(128, cfg["num_classes"]),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Transformer
# ══════════════════════════════════════════════════════════════════════════════
class EMG_Transformer(nn.Module):
    """
    Transformer Encoder — self-attention on EMG time steps.
    Input:  (B, 200, 8)
    Output: (B, num_classes)
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.proj      = nn.Linear(cfg["input_size"], d)
        self.pos_embed = nn.Embedding(cfg["max_seq_len"], d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg["num_layers"])
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(cfg["dropout"]),
            nn.Linear(d, cfg["num_classes"]))

    def forward(self, x):
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x   = self.proj(x) + self.pos_embed(pos)
        x   = self.encoder(x).mean(dim=1)
        return self.head(x)


# ─── Registry ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "cnn1d":       EMG_CNN1D,
    "lstm":        EMG_BiLSTM,
    "transformer": EMG_Transformer,
}

def build_model(name, device):
    m = MODEL_REGISTRY[name](MODEL_CONFIG[name]).to(device)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log.info("Built %-12s | Params: %s", name.upper(), f"{n:,}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def get_loaders(processed_dir, batch_size):
    def ld(split):
        d = Path(processed_dir)
        X = np.load(str(d / f"X_{split}.npy"))
        y = np.load(str(d / f"y_{split}.npy"))
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))
    nw = TRAIN_CONFIG["num_workers"]
    tr = DataLoader(ld("train"), batch_size, shuffle=True,  num_workers=nw)
    va = DataLoader(ld("val"),   batch_size, shuffle=False, num_workers=nw)
    te = DataLoader(ld("test"),  batch_size, shuffle=False, num_workers=nw)
    log.info("DataLoaders | train:%d val:%d test:%d",
             len(tr.dataset), len(va.dataset), len(te.dataset))
    return tr, va, te


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_one_model(model_name, processed_dir, device):
    cfg = TRAIN_CONFIG
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, device)
    tr_ld, va_ld, te_ld = get_loaders(processed_dir, cfg["batch_size"])

    # ── Check actual number of classes in data ────────────────────────────
    all_labels = []
    for _, yb in tr_ld:
        all_labels.extend(yb.numpy().tolist())
    actual_classes = sorted(set(all_labels))
    n_actual = len(actual_classes)
    log.info("Actual classes in data: %s (n=%d)", actual_classes, n_actual)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_va   = 0.0
    patience  = 0
    best_path = str(Path(cfg["save_dir"]) / f"{model_name}_best.pt")
    t0 = time.time()

    log.info("="*55)
    log.info("Training %-12s | epochs=%d | device=%s",
             model_name.upper(), cfg["epochs"], device)
    log.info("="*55)

    for ep in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        tl = tc = tt = 0
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            tt += len(yb)

        # Validate
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for Xb, yb in va_ld:
                Xb, yb = Xb.to(device), yb.to(device)
                out  = model(Xb)
                loss = criterion(out, yb)
                vl += loss.item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(yb)

        ta  = tc/tt;  va  = vc/vt
        tl_ = tl/tt;  vl_ = vl/vt

        hist["train_loss"].append(tl_)
        hist["val_loss"].append(vl_)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        scheduler.step()

        log.info("Ep%3d/%d | tr_loss=%.4f tr_acc=%.3f | vl_loss=%.4f vl_acc=%.3f",
                 ep, cfg["epochs"], tl_, ta, vl_, va)

        if va > best_va:
            best_va   = va
            patience  = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                log.info("Early stopping at epoch %d", ep)
                break

    elapsed = time.time() - t0
    log.info("Training time: %.1fs", elapsed)

    # ── Test evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, preds, trues = _eval_full(model, te_ld, device)
    avg_ms = _benchmark(model, device)

    log.info("%-12s | BestVal=%.4f | Test=%.4f | Inf=%.2fms",
             model_name.upper(), best_va, test_acc, avg_ms)

    # ── Safe classification report (handles missing classes) ──────────────
    unique_in_test = sorted(set(trues))
    safe_names     = [CLASSES.get(i, str(i)) for i in unique_in_test]
    try:
        report_str = classification_report(
            trues, preds,
            labels=unique_in_test,
            target_names=safe_names,
            zero_division=0)
        log.info("\n%s", report_str)
        report_dict = classification_report(
            trues, preds,
            labels=unique_in_test,
            target_names=safe_names,
            zero_division=0,
            output_dict=True)
    except Exception as e:
        log.warning("Classification report skipped: %s", e)
        report_dict = {}

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_training(hist, model_name, cfg["results_dir"])
    _plot_confusion(trues, preds, model_name, cfg["results_dir"], unique_in_test, safe_names)

    results = {
        "model":        model_name,
        "best_val_acc": round(best_va, 4),
        "test_acc":     round(test_acc, 4),
        "avg_inf_ms":   round(avg_ms, 2),
        "train_time_s": round(elapsed, 1),
        "epochs_run":   len(hist["train_loss"]),
        "classification_report": report_dict,
    }
    with open(str(Path(cfg["results_dir"]) / f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved to runs/%s_results.json", model_name)
    return results


def _eval_full(model, loader, device):
    model.eval(); preds = []; trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            p = model(Xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(yb.numpy().tolist())
    acc = sum(p == t for p, t in zip(preds, trues)) / max(len(trues), 1)
    return acc, preds, trues


def _benchmark(model, device, n=50):
    model.eval()
    dummy = torch.randn(1, 200, 8).to(device)
    times = []
    with torch.no_grad():
        for _ in range(n):
            t = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t) * 1000)
    avg = float(np.mean(times[5:]))
    log.info("Avg inference: %.2fms | FPS equiv: %.1f", avg, 1000/avg)
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def _dark_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")


def _plot_training(hist, name, results_dir):
    eps = range(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(f"Training Curves — {name.upper()}",
                 fontsize=14, fontweight="bold", color="white")
    for ax in (ax1, ax2):
        _dark_ax(ax)

    ax1.plot(eps, hist["train_loss"], color="#58a6ff", label="Train Loss", linewidth=1.8)
    ax1.plot(eps, hist["val_loss"],   color="#f85149", label="Val Loss",
             linewidth=1.8, linestyle="--")
    ax1.set_title("Loss vs Epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(facecolor="#161b22", labelcolor="white")
    ax1.grid(alpha=0.15, color="#30363d")

    ax2.plot(eps, hist["train_acc"], color="#3fb950", label="Train Acc", linewidth=1.8)
    ax2.plot(eps, hist["val_acc"],   color="#e3b341", label="Val Acc",
             linewidth=1.8, linestyle="--")
    ax2.set_title("Accuracy vs Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend(facecolor="#161b22", labelcolor="white")
    ax2.grid(alpha=0.15, color="#30363d")

    plt.tight_layout()
    path = str(Path(results_dir) / f"{name}_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Training graph saved: %s", path)


def _plot_confusion(y_true, y_pred, name, results_dir, labels, label_names):
    cm   = confusion_matrix(y_true, y_pred, labels=labels)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    im = ax.imshow(cm_n, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax).ax.tick_params(colors="white")

    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right", color="white")
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.set_title(f"Confusion Matrix — {name.upper()}",
                 color="white", fontweight="bold")

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            v = cm_n[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v < 0.6 else "black")

    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    plt.tight_layout()
    path = str(Path(results_dir) / f"{name}_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved: %s", path)


def save_model_comparison(results: List[Dict], results_dir: str) -> None:
    names   = [r["model"].upper()  for r in results]
    val_acc = [r["best_val_acc"]   for r in results]
    tst_acc = [r["test_acc"]       for r in results]
    inf_ms  = [r["avg_inf_ms"]     for r in results]
    x = np.arange(len(names)); w = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Model Comparison — EMG Emotion Recognition",
                 fontsize=14, fontweight="bold", color="white")
    for ax in (ax1, ax2):
        _dark_ax(ax)
        ax.grid(alpha=0.15, color="#30363d", axis="y")

    b1 = ax1.bar(x - w/2, val_acc, w, label="Val Acc",  color="#58a6ff", alpha=0.9)
    b2 = ax1.bar(x + w/2, tst_acc, w, label="Test Acc", color="#3fb950", alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(names, color="white")
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison")
    ax1.legend(facecolor="#161b22", labelcolor="white")
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=8, color="white")

    b3 = ax2.bar(x, inf_ms, w * 1.8, color="#f85149", alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(names, color="white")
    ax2.set_ylabel("Avg Inference Time (ms)")
    ax2.set_title("Inference Speed (lower = better)")
    for b in b3:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                 f"{b.get_height():.1f}ms", ha="center", fontsize=9, color="white")

    plt.tight_layout()
    path = str(Path(results_dir) / "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Comparison chart saved: %s", path)

    log.info("\n┌─────────────────┬────────────┬───────────┬──────────────┐")
    log.info("│ Model           │ Val Acc    │ Test Acc  │ Inf Time ms  │")
    log.info("├─────────────────┼────────────┼───────────┼──────────────┤")
    for r in results:
        log.info("│ %-15s │ %.4f     │ %.4f    │ %-12.2f │",
                 r["model"].upper(), r["best_val_acc"], r["test_acc"], r["avg_inf_ms"])
    log.info("└─────────────────┴────────────┴───────────┴──────────────┘")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn1d", "lstm", "transformer"],
                        default="cnn1d")
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--data", default=PREPROCESS_CONFIG["processed_dir"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    to_train = ["cnn1d", "lstm", "transformer"] if args.all else [args.model]
    all_results = []
    for m in to_train:
        r = train_one_model(m, args.data, device)
        all_results.append(r)

    if len(all_results) > 1:
        save_model_comparison(all_results, TRAIN_CONFIG["results_dir"])

    log.info("Done! Check runs/ folder for graphs.")