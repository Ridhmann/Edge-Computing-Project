#!/usr/bin/env python3
# main.py
# EMG-Based Facial Muscle Signal Detection
# Pipeline: YOLOv8 Face Detection → EMG Signal → 3 Model Emotion Classification
#
# Commands:
#   py -3.11 main.py --model cnn1d
#   py -3.11 main.py --model lstm
#   py -3.11 main.py --model transformer
#   py -3.11 main.py --source video.mp4 --model cnn1d
#   py -3.11 main.py --preprocess
#   py -3.11 main.py --train --all
#   py -3.11 main.py --dashboard-only

from __future__ import annotations
import argparse, threading, time, webbrowser
from pathlib import Path

from config import ACTIVE_MODEL, FLASK_CONFIG, TRAIN_CONFIG
from logger import get_logger
from utils import ensure_dirs, init_db, fetch_events

log = get_logger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   EMG-Based Facial Muscle Signal Detection                   ║
║   YOLOv8 Face Detection + EMG Emotion Recognition            ║
║   Models: CNN-1D  |  Bi-LSTM  |  Transformer                 ║
╚══════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM INIT
# ══════════════════════════════════════════════════════════════════════════════

def initialise() -> str:
    ensure_dirs("models","data/raw","data/processed","logs","runs")
    init_db()
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            log.info("GPU: %s", torch.cuda.get_device_name())
        else:
            device = "cpu"
            log.info("Using CPU")
    except ImportError:
        device = "cpu"
    TRAIN_CONFIG["device"] = device
    return device


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLASK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    from flask import Flask, jsonify, render_template_string, request
    app = Flask(__name__)
    app.secret_key = FLASK_CONFIG["secret_key"]

    HTML = """
<!DOCTYPE html><html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EMG Facial Muscle Detection — Dashboard</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e0e0e0}
    header{background:#161b22;padding:18px 32px;border-bottom:2px solid #238636;
           display:flex;align-items:center;gap:16px}
    header h1{font-size:1.3rem;font-weight:700;color:#fff}
    .live{background:#238636;color:#fff;padding:4px 12px;border-radius:20px;
          font-size:.75rem;animation:pulse 2s infinite}
    .yolo{background:#1f6feb;color:#fff;padding:4px 12px;border-radius:20px;font-size:.75rem}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
    .container{max-width:1200px;margin:0 auto;padding:24px}
    .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px}
    .stat{background:#161b22;border-radius:10px;padding:18px;border-left:4px solid #238636;text-align:center}
    .stat .val{font-size:2rem;font-weight:700;color:#58a6ff}
    .stat .lbl{font-size:.78rem;color:#8b949e;margin-top:4px}
    table{width:100%;border-collapse:collapse}
    thead th{background:#161b22;padding:10px 14px;text-align:left;font-size:.78rem;
             text-transform:uppercase;color:#8b949e;border-bottom:1px solid #21262d}
    tbody tr{border-bottom:1px solid #161b22;transition:background .15s}
    tbody tr:hover{background:#161b22}
    tbody td{padding:10px 14px;font-size:.88rem}
    .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.73rem;font-weight:600}
    .Neutral{background:#21262d;color:#8b949e}
    .Happy{background:#0f2d1f;color:#3fb950}
    .Stressed{background:#2d0f0f;color:#f85149}
    .Concentrated{background:#2d1f0f;color:#e3b341}
    .Fatigued{background:#1a1a2d;color:#a371f7}
    .btn{background:#238636;color:#fff;border:none;padding:8px 18px;
         border-radius:6px;cursor:pointer;font-size:.83rem;margin-bottom:18px}
    .pipeline{background:#161b22;border-radius:10px;padding:16px;margin-bottom:24px;
              display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .pipe-step{background:#21262d;padding:8px 16px;border-radius:20px;font-size:.82rem}
    .pipe-arrow{color:#58a6ff;font-size:1.2rem}
  </style>
</head>
<body>
<header>
  <div>🧠</div>
  <h1>EMG Facial Muscle Detection Dashboard</h1>
  <span class="live">● LIVE</span>
  <span class="yolo">YOLOv8</span>
</header>
<div class="container">
  <div class="pipeline">
    <span class="pipe-step">📷 Webcam/Video</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🎯 YOLOv8 Detection</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">⚡ EMG Signal</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🧠 CNN-1D / BiLSTM / Transformer</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">😊 Emotion Output</span>
  </div>
  <div class="stats" id="stats"></div>
  <div style="margin-bottom:12px;font-size:1rem;font-weight:600;color:#c9d1d9">
    Recent Emotion Events
    <button class="btn" style="margin-left:12px" onclick="load()">↻ Refresh</button>
  </div>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr><th>#</th><th>Timestamp</th><th>Session</th>
            <th>Emotion</th><th>Confidence</th><th>Active CH</th></tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>
<script>
async function load(){
  const r=await fetch('/api/events');const d=await r.json();
  document.getElementById('tbody').innerHTML=d.events.map(e=>`
    <tr>
      <td>${e.id}</td><td>${e.timestamp}</td><td>${e.session_id}</td>
      <td><span class="badge ${e.emotion}">${e.emotion}</span></td>
      <td>${(e.confidence*100).toFixed(1)}%</td>
      <td>${e.active_channels}</td>
    </tr>`).join('');
  const s=d.stats;
  document.getElementById('stats').innerHTML=`
    <div class="stat"><div class="val">${s.total}</div><div class="lbl">Total Events</div></div>
    <div class="stat"><div class="val">${s.stressed}</div><div class="lbl">Stressed</div></div>
    <div class="stat"><div class="val">${s.happy}</div><div class="lbl">Happy</div></div>
    <div class="stat"><div class="val">${s.concentrated}</div><div class="lbl">Concentrated</div></div>
    <div class="stat"><div class="val">${s.fatigued}</div><div class="lbl">Fatigued</div></div>
    <div class="stat"><div class="val">${s.avg_conf}%</div><div class="lbl">Avg Confidence</div></div>`;
}
setInterval(load,4000);load();
</script>
</body></html>"""

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/api/events")
    def api_events():
        events = fetch_events(limit=200)
        confs  = [e["confidence"] for e in events if e.get("confidence")]
        stats  = {
            "total":        len(events),
            "stressed":     sum(1 for e in events if e["emotion"]=="Stressed"),
            "happy":        sum(1 for e in events if e["emotion"]=="Happy"),
            "concentrated": sum(1 for e in events if e["emotion"]=="Concentrated"),
            "fatigued":     sum(1 for e in events if e["emotion"]=="Fatigued"),
            "avg_conf":     round(sum(confs)/len(confs)*100,1) if confs else 0,
        }
        return jsonify({"events": events, "stats": stats})

    @app.route("/health")
    def health():
        return jsonify({"status":"running","time":time.time()})

    return app


# ══════════════════════════════════════════════════════════════════════════════
# 3. THREADS
# ══════════════════════════════════════════════════════════════════════════════

def start_dashboard(app):
    def _run():
        log.info("Dashboard → http://localhost:%d", FLASK_CONFIG["port"])
        app.run(host=FLASK_CONFIG["host"], port=FLASK_CONFIG["port"],
                debug=False, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True, name="flask")
    t.start()
    return t


def start_inference(source, model_name, show_window, save_output):
    from inference import run_inference
    def _run():
        try:
            run_inference(source=source, model_name=model_name,
                          show_window=show_window, save_output=save_output)
        except Exception as e:
            log.error("Inference error: %s", e, exc_info=True)
    t = threading.Thread(target=_run, daemon=True, name="inference")
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="EMG Facial Muscle Detection — YOLOv8 + 3 Models")
    parser.add_argument("--source",   default=None,
                        help="0=webcam, or path to video file")
    parser.add_argument("--model",    default=ACTIVE_MODEL,
                        choices=["cnn1d","lstm","transformer"])
    parser.add_argument("--no-window",      action="store_true")
    parser.add_argument("--save",           default=None,
                        help="Save output video to this path")
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--preprocess",     action="store_true")
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--all",            action="store_true",
                        help="Train all 3 models (use with --train)")
    args = parser.parse_args()

    # Preprocessing
    if args.preprocess:
        from preprocessing import run_preprocessing
        run_preprocessing(); return

    # Training
    if args.train:
        import torch
        from training import train_one_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from config import PREPROCESS_CONFIG
        models = ["cnn1d","lstm","transformer"] if args.all else [args.model]
        for m in models:
            train_one_model(m, PREPROCESS_CONFIG["processed_dir"], device)
        return

    # Normal run
    initialise()
    source = int(args.source) if args.source and args.source.isdigit() else args.source
    source = source if source is not None else 0
    show   = not args.no_window
    threads = []

    if not args.inference_only:
        app = build_app()
        threads.append(start_dashboard(app))
        if show:
            threading.Timer(2.0, lambda: webbrowser.open(
                f"http://localhost:{FLASK_CONFIG['port']}")).start()

    if not args.dashboard_only:
        threads.append(start_inference(source, args.model, show, args.save))

    log.info("Running. Press Ctrl+C to stop.")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutdown.")


if __name__ == "__main__":
    main()