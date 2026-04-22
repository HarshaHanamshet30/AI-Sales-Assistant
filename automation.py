"""
automation.py - Automated pipeline scheduler
Runs lead scoring every X minutes without manual trigger
"""

import os
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from model import (
    generate_sample_data, train_model, score_leads,
    save_model, load_model, model_exists
)
from utils import (
    generate_bulk_insights, save_results, compute_kpis,
    get_file_hash, timestamp_now, send_email_notification
)

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pipeline state (JSON log)
# ─────────────────────────────────────────────
STATE_FILE = "logs/pipeline_state.json"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "runs": 0, "last_hash": ""}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    input_path: str = None,
    output_path: str = "output/scored_leads.csv",
    retrain: bool = False,
    send_email: bool = False,
    use_mock_ai: bool = True,
):
    """
    Full end-to-end pipeline:
    1. Load / generate data
    2. Train or load model
    3. Score leads
    4. Generate AI recommendations
    5. Save results
    6. (Optional) send email
    """
    state = load_state()
    log.info("=" * 55)
    log.info(f"🚀 Pipeline run #{state['runs'] + 1} started at {timestamp_now()}")

    # ── 1. Load data ──────────────────────────────────────
    if input_path and os.path.exists(input_path):
        current_hash = get_file_hash(input_path)
        if current_hash == state.get("last_hash") and not retrain:
            log.info("⏭  Input file unchanged — skipping run.")
            return None
        df = pd.read_csv(input_path)
        log.info(f"📂 Loaded {len(df)} leads from {input_path}")
        state["last_hash"] = current_hash
    else:
        log.info("📊 No input file — generating synthetic demo data (200 leads)")
        df = generate_sample_data(n=200)

    # ── 2. Train or load model ────────────────────────────
    if retrain or not model_exists():
        log.info("🧠 Training RandomForest model …")
        model, scaler, metrics, feature_names, encoders = train_model(df)
        save_model(model, scaler, feature_names, encoders)
        log.info(
            f"✅ Model trained | Accuracy: {metrics['accuracy']} | AUC: {metrics['roc_auc']}"
        )
        # Save metrics
        os.makedirs("output", exist_ok=True)
        with open("output/model_metrics.json", "w") as f:
            # confusion matrix is not JSON-serialisable as np array → already list
            json.dump(
                {k: v for k, v in metrics.items() if k != "classification_report"},
                f, indent=2,
            )
    else:
        log.info("📦 Loading existing model from artifacts/")
        model, scaler, feature_names, encoders = load_model()

    # ── 3. Score leads ────────────────────────────────────
    log.info("📈 Scoring leads …")
    scored_df = score_leads(df, model, scaler, feature_names, encoders)

    # ── 4. AI recommendations ─────────────────────────────
    log.info("🤖 Generating AI recommendations …")
    scored_df = generate_bulk_insights(scored_df, use_mock=use_mock_ai)

    # ── 5. Save results ───────────────────────────────────
    path = save_results(scored_df, output_path)
    kpis = compute_kpis(scored_df)

    log.info(
        f"💾 Results saved to {path} | "
        f"High: {kpis['high_priority']} | "
        f"Medium: {kpis['medium_priority']} | "
        f"Low: {kpis['low_priority']}"
    )

    # ── 6. Email notification ─────────────────────────────
    if send_email:
        ok, msg = send_email_notification(scored_df)
        log.info(f"📧 Email: {msg}")

    # ── Update state ──────────────────────────────────────
    state["last_run"] = timestamp_now()
    state["runs"] += 1
    save_state(state)

    log.info(f"✅ Pipeline run #{state['runs']} complete.")
    return scored_df


# ─────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────

def start_scheduler(
    interval_minutes: int = 5,
    input_path: str = None,
    send_email: bool = False,
):
    """
    Runs the pipeline every `interval_minutes` minutes indefinitely.
    Runs once immediately, then on schedule.
    """
    log.info(f"⏰ Scheduler started — pipeline runs every {interval_minutes} minute(s)")

    def _loop():
        while True:
            try:
                run_pipeline(
                    input_path=input_path,
                    retrain=False,
                    send_email=send_email,
                )
            except Exception as exc:
                log.error(f"❌ Pipeline error: {exc}", exc_info=True)
            log.info(f"💤 Sleeping {interval_minutes} min …")
            time.sleep(interval_minutes * 60)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Sales Assistant — Automation Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to leads CSV")
    parser.add_argument("--output", type=str, default="output/scored_leads.csv")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between runs")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--email", action="store_true", help="Send email after run")
    args = parser.parse_args()

    if args.once:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            retrain=args.retrain,
            send_email=args.email,
        )
    else:
        thread = start_scheduler(
            interval_minutes=args.interval,
            input_path=args.input,
            send_email=args.email,
        )
        # Keep main thread alive
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("🛑 Scheduler stopped by user.")
