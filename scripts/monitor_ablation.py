#!/usr/bin/env python3
"""Monitor PRX-TG ablation study running on HuggingFace Jobs.

Usage:
    python scripts/monitor_ablation.py              # One-shot status check
    python scripts/monitor_ablation.py --watch      # Poll every 5 minutes
    python scripts/monitor_ablation.py --logs A     # Show last 30 log lines for arm A
    python scripts/monitor_ablation.py --logs all   # Show last 10 log lines per arm
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import UTC, datetime

JOBS = {
    "A-baseline":    "69effb98d70108f37ace0b88",
    "B-tread-adamw": "69efeedbd2c8bd8662bd1537",
    "C-tread-muon":  "69f16215d2c8bd8662bd2846",
    "D-full-stack":  "69eeaf63d2c8bd8662bd0575",
}

TOTAL_STEPS = 5000


def get_api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set. Source .env first.", file=sys.stderr)
        sys.exit(1)
    return HfApi(token=token)


def parse_progress(logs: list[str]) -> dict:
    """Extract latest training progress from log lines."""
    info = {"step": 0, "loss": "?", "grad": "?", "lr": "?", "s_per_it": "?"}
    for line in reversed(logs):
        if "Training:" in line and "/5000" in line:
            m = re.search(r"(\d+)/5000", line)
            if m:
                info["step"] = int(m.group(1))
            for key, pat in [
                ("loss", r"loss=([\d.]+)"),
                ("grad", r"grad=([\d.]+)"),
                ("lr", r"lr=([\d.e+-]+)"),
                ("s_per_it", r"([\d.]+)s/it"),
            ]:
                m2 = re.search(pat, line)
                if m2:
                    info[key] = m2.group(1)
            break
    return info


def check_status(api) -> list[dict]:
    results = []
    for name, jid in JOBS.items():
        try:
            job = api.inspect_job(job_id=jid)
            logs = list(api.fetch_job_logs(job_id=jid))
            progress = parse_progress(logs)

            step = progress["step"]
            pct = step / TOTAL_STEPS * 100
            s_per_it = progress["s_per_it"]

            eta_str = "?"
            if s_per_it != "?" and step < TOTAL_STEPS:
                remaining_s = (TOTAL_STEPS - step) * float(s_per_it)
                remaining_h = remaining_s / 3600
                eta_str = f"{remaining_h:.1f}h"
            elif step >= TOTAL_STEPS:
                eta_str = "done"

            results.append({
                "name": name,
                "stage": job.status.stage,
                "step": step,
                "pct": pct,
                "loss": progress["loss"],
                "grad": progress["grad"],
                "lr": progress["lr"],
                "s_per_it": s_per_it,
                "eta": eta_str,
            })
        except Exception as e:
            results.append({
                "name": name,
                "stage": f"ERROR: {str(e)[:60]}",
                "step": 0, "pct": 0, "loss": "?", "grad": "?",
                "lr": "?", "s_per_it": "?", "eta": "?",
            })
        time.sleep(3)
    return results


def print_status(results: list[dict]):
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*80}")
    print(f"PRX-TG Ablation Study — {now}")
    print(f"{'='*80}")
    print(f"{'Arm':<18} {'Status':<10} {'Step':>6} {'%':>6} {'Loss':>10} "
          f"{'Grad':>8} {'s/it':>7} {'ETA':>8}")
    print(f"{'-'*18} {'-'*10} {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    for r in results:
        bar = f"{r['pct']:5.1f}%"
        print(f"{r['name']:<18} {r['stage']:<10} {r['step']:>6} {bar:>6} "
              f"{r['loss']:>10} {r['grad']:>8} {r['s_per_it']:>7} {r['eta']:>8}")
    print()


def show_logs(api, arm_name: str, num_lines: int = 30):
    if arm_name.lower() == "all":
        for name, jid in JOBS.items():
            logs = list(api.fetch_job_logs(job_id=jid))
            print(f"\n=== {name} (last 10 lines) ===")
            for line in logs[-10:]:
                print(f"  {line.rstrip()[:150]}")
            time.sleep(3)
    else:
        key = None
        for k in JOBS:
            if arm_name.upper() in k.upper() or k.startswith(arm_name):
                key = k
                break
        if not key:
            print(f"Unknown arm: {arm_name}. Use A, B, C, D, or 'all'.")
            sys.exit(1)
        logs = list(api.fetch_job_logs(job_id=JOBS[key]))
        print(f"\n=== {key} (last {num_lines} lines) ===")
        for line in logs[-num_lines:]:
            print(f"  {line.rstrip()[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Monitor PRX-TG ablation study")
    parser.add_argument("--watch", action="store_true", help="Poll every 5 minutes")
    parser.add_argument("--logs", type=str, help="Show logs for arm (A/B/C/D/all)")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds")
    args = parser.parse_args()

    api = get_api()

    if args.logs:
        show_logs(api, args.logs)
        return

    if args.watch:
        try:
            while True:
                results = check_status(api)
                print_status(results)
                all_done = all(r["stage"] in ("COMPLETED", "ERROR") for r in results)
                if all_done:
                    print("All jobs finished!")
                    break
                print(f"Next check in {args.interval}s... (Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        results = check_status(api)
        print_status(results)


if __name__ == "__main__":
    main()
