#!/usr/bin/env python3
"""Monitor PRX-TG ablation study running on HuggingFace Jobs.

Resolves the currently active job for each arm using HF Job Labels
(``experiment`` + ``arm``), so monitors survive job restarts without
needing to update hardcoded IDs.

Usage:
    python scripts/monitor_ablation.py              # One-shot status check
    python scripts/monitor_ablation.py --watch      # Poll every 5 minutes
    python scripts/monitor_ablation.py --logs A     # Show last 30 log lines for arm A
    python scripts/monitor_ablation.py --logs all   # Show last 10 log lines per arm
    python scripts/monitor_ablation.py --experiment my-experiment  # Custom experiment name
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import UTC, datetime

# Arm names that this monitor tracks.
ARM_NAMES = [
    "A-baseline",
    "B-tread-adamw",
    "C-tread-muon",
    "D-full-stack",
]

# Default experiment label used when submitting jobs via ratiocinator fleet.
DEFAULT_EXPERIMENT = "prx-tg-ablation"

TOTAL_STEPS = 5000


def get_api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set. Source .env first.", file=sys.stderr)
        sys.exit(1)
    return HfApi(token=token)


def resolve_jobs(api, experiment: str, namespace: str | None = None) -> dict[str, str | None]:
    """Resolve arm names to job IDs using HF Job Labels.

    For each arm, finds the most recent job with matching labels
    ``experiment=<experiment>`` and ``arm=<arm_name>``.
    Prefers non-terminal (active) jobs over completed/failed ones.

    Returns a dict mapping arm name to job ID (or None if not found).
    """
    jobs = list(api.list_jobs(namespace=namespace))

    resolved: dict[str, str | None] = {}
    for arm_name in ARM_NAMES:
        matches = [
            j for j in jobs
            if hasattr(j, "labels") and j.labels
            and j.labels.get("experiment") == experiment
            and j.labels.get("arm") == arm_name
        ]
        if not matches:
            resolved[arm_name] = None
            continue

        # Separate active vs terminal
        active = [
            j for j in matches
            if hasattr(j, "status") and j.status
            and getattr(j.status, "stage", None) not in (
                "COMPLETED", "ERROR", "FAILED", "CANCELLED", "DELETED",
            )
        ]
        if active:
            active.sort(
                key=lambda j: getattr(j, "created_at", None) or datetime.min,
                reverse=True,
            )
            resolved[arm_name] = active[0].id
        else:
            matches.sort(
                key=lambda j: getattr(j, "created_at", None) or datetime.min,
                reverse=True,
            )
            resolved[arm_name] = matches[0].id

    return resolved


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


def check_status(api, jobs: dict[str, str | None]) -> list[dict]:
    results = []
    for name, jid in jobs.items():
        if jid is None:
            results.append({
                "name": name,
                "stage": "NOT FOUND",
                "step": 0, "pct": 0, "loss": "?", "grad": "?",
                "lr": "?", "s_per_it": "?", "eta": "?",
            })
            continue
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


def show_logs(api, jobs: dict[str, str | None], arm_name: str, num_lines: int = 30):
    if arm_name.lower() == "all":
        for name, jid in jobs.items():
            if jid is None:
                print(f"\n=== {name}: no job found ===")
                continue
            logs = list(api.fetch_job_logs(job_id=jid))
            print(f"\n=== {name} (last 10 lines) ===")
            for line in logs[-10:]:
                print(f"  {line.rstrip()[:150]}")
            time.sleep(3)
    else:
        key = None
        for k in jobs:
            if arm_name.upper() in k.upper() or k.startswith(arm_name):
                key = k
                break
        if not key:
            print(f"Unknown arm: {arm_name}. Use A, B, C, D, or 'all'.")
            sys.exit(1)
        jid = jobs[key]
        if jid is None:
            print(f"No active job found for arm {key}.")
            sys.exit(1)
        logs = list(api.fetch_job_logs(job_id=jid))
        print(f"\n=== {key} (last {num_lines} lines) ===")
        for line in logs[-num_lines:]:
            print(f"  {line.rstrip()[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Monitor PRX-TG ablation study")
    parser.add_argument("--watch", action="store_true", help="Poll every 5 minutes")
    parser.add_argument("--logs", type=str, help="Show logs for arm (A/B/C/D/all)")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds")
    parser.add_argument(
        "--experiment", type=str, default=DEFAULT_EXPERIMENT,
        help="Experiment label to filter jobs by",
    )
    parser.add_argument("--namespace", type=str, default=None, help="HF namespace to filter jobs")
    args = parser.parse_args()

    api = get_api()

    print("Resolving jobs by labels...")
    jobs = resolve_jobs(api, args.experiment, namespace=args.namespace)
    found = sum(1 for v in jobs.values() if v is not None)
    print(f"Resolved {found}/{len(ARM_NAMES)} arms via labels (experiment={args.experiment})")

    if args.logs:
        show_logs(api, jobs, args.logs)
        return

    if args.watch:
        try:
            while True:
                results = check_status(api, jobs)
                print_status(results)
                all_done = all(
                    r["stage"] in ("COMPLETED", "ERROR", "NOT FOUND")
                    for r in results
                )
                if all_done:
                    print("All jobs finished!")
                    break
                print(f"Next check in {args.interval}s... (Ctrl+C to stop)")
                time.sleep(args.interval)
                # Re-resolve in case jobs were recreated
                jobs = resolve_jobs(api, args.experiment, namespace=args.namespace)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        results = check_status(api, jobs)
        print_status(results)


if __name__ == "__main__":
    main()
