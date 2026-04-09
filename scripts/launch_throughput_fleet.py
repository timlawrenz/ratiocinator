#!/usr/bin/env python3
"""Fleet orchestrator: provisions 7 Vast.ai 4090 instances in parallel,
syncs data, dispatches one throughput experiment arm per instance, and
collects results.

Usage (presigned S3 URLs — recommended):
    python scripts/launch_throughput_fleet.py \
        --data-urls urls.txt \
        [--api-key <vast_api_key>]
        [--max-dph 0.50]
        [--dry-run]

    Where urls.txt has one presigned URL per line for each .tar shard.

Usage (rsync from remote host):
    python scripts/launch_throughput_fleet.py \
        --data-server root@50.173.30.254:/workspace/prx-tg/data/shards/faces7k \
        --data-ssh-port 25706 \
        [--max-shards 5]

Environment:
    VAST_API_KEY    Vast.ai API key (or use --api-key)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ratiocinator.infra.vast_client import InstanceStatus, VastClient
from ratiocinator.observability import init_sentry

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ARMS = [
    ("config_0_baseline", "Baseline"),
    ("config_1_runtime_flags", "+RuntimeFlags (tf32/cudnn/non_blocking/set_to_none)"),
    ("config_2_fused_optimizer", "+FusedOptimizer"),
    ("config_3_no_grad_ckpt", "+NoGradCheckpoint"),
    ("config_4_compile", "+torch.compile(max-autotune)"),
    ("config_5_vram_dataset", "+VRAMDataset"),
    ("config_6_cached_embeds", "+CachedEmbeddings"),
]

IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]

BOOT_TIMEOUT_S = 600
BOOT_POLL_S = 10
TRAIN_TIMEOUT_S = 7200


def _parse_remote_traceback(stderr_text: str) -> tuple[list[dict], str, str]:
    """Parse a Python traceback from remote stderr into Sentry-compatible frames.

    Returns (frames, exception_type, exception_value).
    """
    import re

    frames = []
    exc_type = "RemoteTrainingError"
    exc_value = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else "Unknown error"

    # Match: File "path", line N, in func
    frame_re = re.compile(
        r'^\s*File "([^"]+)", line (\d+), in (.+)$'
    )
    for line in stderr_text.splitlines():
        m = frame_re.match(line)
        if m:
            frames.append({
                "filename": m.group(1),
                "lineno": int(m.group(2)),
                "function": m.group(3),
            })

    # Extract actual exception type + message from last line
    # e.g. "AttributeError: module 'torch.optim' has no attribute 'Muon'"
    last_line = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else ""
    if ": " in last_line and not last_line.startswith(" "):
        exc_type, _, exc_value = last_line.partition(": ")

    return frames, exc_type, exc_value


def _report_remote_crash(
    *,
    arm_name: str,
    arm_idx: int,
    exit_code: int,
    stderr_text: str,
    stdout_tail: str,
    gpu_info: str,
    instance_id: int | None,
) -> None:
    """Send a remote training crash to Sentry as a structured exception event."""
    if not sentry_sdk:
        return

    frames, exc_type, exc_value = _parse_remote_traceback(stderr_text)

    event: dict = {
        "level": "error",
        "transaction": f"fleet/arm/{arm_name}",
        "tags": {
            "arm": arm_name,
            "arm_idx": str(arm_idx),
            "exit_code": str(exit_code),
        },
        "contexts": {
            "fleet": {
                "arm_name": arm_name,
                "arm_idx": arm_idx,
                "exit_code": exit_code,
                "instance_id": instance_id,
                "gpu_info": gpu_info[:200] if gpu_info else "",
            },
        },
        "extra": {
            "stderr": stderr_text[-3000:],
            "stdout_tail": stdout_tail,
        },
    }

    if frames:
        event["exception"] = {
            "values": [{
                "type": exc_type,
                "value": exc_value,
                "stacktrace": {"frames": frames},
                "mechanism": {
                    "type": "remote_ssh",
                    "handled": True,
                    "description": f"Remote training crash on Vast.ai instance {instance_id}",
                },
            }],
        }
    else:
        event["exception"] = {
            "values": [{
                "type": "RemoteTrainingError",
                "value": f"Arm {arm_name} failed (exit {exit_code}): {stderr_text[-500:]}",
                "mechanism": {"type": "remote_ssh", "handled": True},
            }],
        }

    sentry_sdk.capture_event(event)


@dataclass
class ArmResult:
    arm_name: str
    description: str
    instance_id: int | None = None
    gpu_info: str = ""
    avg_iter_per_sec: float = 0.0
    peak_vram_gb: float = 0.0
    final_loss: float = float("inf")
    total_steps: int = 0
    total_time_s: float = 0.0
    exit_code: int = -1
    error: str = ""


async def find_matching_offers(
    client: VastClient, max_dph: float, num_needed: int = 7,
) -> list[dict]:
    """Find 4090 offers with comparable hardware (PCIe Gen4, >=64GB RAM)."""
    offers = await client.search_offers(
        gpu_name="RTX 4090", num_gpus=1, max_dph=max_dph, limit=50,
    )
    matched = []
    for o in offers:
        if o.get("pcie_bw", 0) < 20.0:
            continue
        if o.get("cpu_ram", 0) < 60000:
            continue
        matched.append(o)
        if len(matched) >= num_needed:
            break
    return matched


async def wait_for_boot(client: VastClient, instance_id: int) -> tuple[str, int]:
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            info = await client.get_instance(instance_id)
            if info.status == InstanceStatus.RUNNING and info.ssh_host and info.ssh_port:
                return info.ssh_host, info.ssh_port
            if info.status in (InstanceStatus.ERROR, InstanceStatus.EXITED):
                return "", 0
        except Exception:
            pass
        await asyncio.sleep(BOOT_POLL_S)
    return "", 0


async def wait_for_ssh(host: str, port: int, ssh_key: str, retries: int = 15) -> bool:
    for attempt in range(retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", *SSH_OPTIONS, "-i", ssh_key, "-p", str(port),
                f"root@{host}", "echo ok",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=20)
            if proc.returncode == 0:
                return True
        except Exception:
            pass
        await asyncio.sleep(10)
    return False


async def ssh_exec(
    host: str, port: int, ssh_key: str, command: str, timeout: int = TRAIN_TIMEOUT_S,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "ssh", *SSH_OPTIONS, "-i", ssh_key, "-p", str(port),
        f"root@{host}", command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except TimeoutError:
        proc.kill()
        return 124, "", f"Timed out after {timeout}s"


def parse_throughput_results(stdout: str) -> dict:
    results = {}
    in_block = False
    for line in stdout.splitlines():
        line = line.strip()
        if line == "--- THROUGHPUT RESULTS ---":
            in_block = True
            continue
        if line == "--- END THROUGHPUT RESULTS ---":
            break
        if in_block and ":" in line:
            key, val = line.split(":", 1)
            try:
                results[key.strip()] = float(val.strip())
            except ValueError:
                results[key.strip()] = val.strip()
    return results


async def run_arm(
    client: VastClient,
    offer: dict,
    arm_idx: int,
    arm_name: str,
    arm_desc: str,
    ssh_key: str,
    prx_repo: str,
    prx_branch: str,
    data_urls: list[str] | None = None,
    data_server: str = "",
    data_ssh_port: int = 22,
    max_shards: int = 5,
) -> ArmResult:
    """Run a single experiment arm on a Vast.ai instance."""
    result = ArmResult(arm_name=arm_name, description=arm_desc)
    instance_id = None

    # Helper: create a Sentry span if SDK available, otherwise a no-op context manager
    def _span(op: str, description: str):
        if sentry_sdk:
            return sentry_sdk.start_span(op=op, name=description)
        from contextlib import nullcontext
        return nullcontext()

    try:
        # Stagger instance creation to avoid Vast.ai 429 rate limits
        if arm_idx > 0:
            await asyncio.sleep(arm_idx * 5)

        with _span("vm.provision", f"provision {arm_name}") as span:
            onstart = "#!/bin/bash\necho 'ready' > /tmp/ready\n"
            instance_id = await client.create_instance(
                offer_id=offer["id"], image=IMAGE, onstart=onstart,
                label=f"throughput-{arm_name}", disk_gb=200.0,
            )
            result.instance_id = instance_id
            dph = offer.get("dph_total", 0)
            logger.info("[Arm %d] %s — instance %s @ $%.3f/hr",
                         arm_idx, arm_name, instance_id, dph)
            if span:
                span.set_data("instance_id", instance_id)
                span.set_data("offer_dph", dph)
                span.set_data("image", IMAGE)

        with _span("vm.boot", f"boot {arm_name}") as span:
            ssh_host, ssh_port = await wait_for_boot(client, instance_id)
            if not ssh_host:
                result.error = "Instance failed to boot"
                return result

            if not await wait_for_ssh(ssh_host, ssh_port, ssh_key):
                result.error = "SSH never became ready"
                return result

            logger.info("[Arm %d] SSH ready: %s:%d", arm_idx, ssh_host, ssh_port)
            if span:
                span.set_data("ssh_host", ssh_host)
                span.set_data("ssh_port", ssh_port)

        with _span("vm.hwinfo", f"hwinfo {arm_name}") as span:
            rc, gpu_info, _ = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                "nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,"
                "pcie.link.width.current --format=csv,noheader 2>/dev/null; "
                "echo '---'; free -h | head -2; lscpu | grep 'Model name'",
                timeout=30,
            )
            result.gpu_info = gpu_info.strip()
            if span:
                span.set_data("gpu_info", result.gpu_info)

        with _span("git.clone", f"clone {arm_name}") as span:
            logger.info("[Arm %d] Cloning repo...", arm_idx)
            rc, _, err = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                f"cd /workspace && git clone --branch {prx_branch} --depth 1 {prx_repo} prx-tg",
                timeout=120,
            )
            if rc != 0:
                result.error = f"Git clone failed: {err[:500]}"
                return result
            if span:
                span.set_data("branch", prx_branch)

        with _span("pip.install", f"deps {arm_name}") as span:
            # Install PyTorch with CUDA 13.0 wheels (provides torch.optim.Muon)
            logger.info("[Arm %d] Installing torch+torchvision (cu130)...", arm_idx)
            rc_torch, torch_out, torch_err = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                "pip install -q torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -5",
                timeout=600,
            )
            # Read installed torch version
            _, torch_ver, _ = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                "python -c \"import torch; print(torch.__version__)\"",
                timeout=30,
            )
            torch_ver = torch_ver.strip()
            logger.info("[Arm %d] torch version: %s (install rc=%d)",
                         arm_idx, torch_ver, rc_torch)

            # Install remaining deps
            logger.info("[Arm %d] Installing remaining deps...", arm_idx)
            await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                "cd /workspace/prx-tg && grep -v '^torch' production/requirements.txt "
                "| pip install -q -r /dev/stdin 2>&1 | tail -5",
                timeout=300,
            )

            if span:
                span.set_data("torch_version", torch_ver)
                span.set_data("torch_install_rc", rc_torch)
                span.set_data("cuda_index", "cu130")

        with _span("pip.verify", f"verify muon {arm_name}") as span:
            rc_muon, muon_out, muon_err = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                "python -c \"import torch; print('Muon' in dir(torch.optim)); "
                "print(torch.version.cuda); print(torch.__version__)\"",
                timeout=30,
            )
            muon_lines = muon_out.strip().splitlines()
            has_muon = muon_lines[0] == "True" if muon_lines else False
            cuda_ver = muon_lines[1] if len(muon_lines) > 1 else "unknown"
            if span:
                span.set_data("has_muon", has_muon)
                span.set_data("cuda_runtime", cuda_ver)
                span.set_data("torch_version", torch_ver)
            if not has_muon:
                result.error = (
                    f"torch.optim.Muon not available: torch={torch_ver} cuda={cuda_ver}"
                )
                logger.error("[Arm %d] %s", arm_idx, result.error)
                return result

        # --- Data provisioning ---
        data_dir = "/workspace/prx-tg/data/shards/faces7k"
        await ssh_exec(ssh_host, ssh_port, ssh_key, f"mkdir -p {data_dir}", timeout=15)

        with _span("data.provision", f"data {arm_name}") as span:
            if data_urls:
                num_shards = len(data_urls)
                logger.info("[Arm %d] Downloading %d shard(s) from presigned URLs...",
                             arm_idx, num_shards)
                script_lines = ["#!/bin/bash", "set -e"]
                seen_dirs = set()
                for url in data_urls:
                    url_path = url.split("?")[0]
                    parts = url_path.split("/")
                    bucket_idx = next(
                        (i for i, p in enumerate(parts) if p.startswith("bucket_")), None
                    )
                    if bucket_idx is not None:
                        rel_path = "/".join(parts[bucket_idx:])
                        bucket_dir = parts[bucket_idx]
                    else:
                        rel_path = parts[-1]
                        bucket_dir = None
                    if bucket_dir and bucket_dir not in seen_dirs:
                        script_lines.append(f"mkdir -p {data_dir}/{bucket_dir}")
                        seen_dirs.add(bucket_dir)
                    script_lines.append(f"wget -q -O '{data_dir}/{rel_path}' '{url}' &")
                script_lines.append("wait")
                script_content = "\n".join(script_lines) + "\n"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                    f.write(script_content)
                    local_script = f.name
                try:
                    scp_proc = await asyncio.create_subprocess_exec(
                        "scp", *SSH_OPTIONS, "-i", ssh_key, "-P", str(ssh_port),
                        local_script, f"root@{ssh_host}:/tmp/download_shards.sh",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(scp_proc.communicate(), timeout=30)
                finally:
                    os.unlink(local_script)

                rc, _, err = await ssh_exec(
                    ssh_host, ssh_port, ssh_key,
                    "chmod +x /tmp/download_shards.sh && /tmp/download_shards.sh",
                    timeout=3600,
                )
                if rc != 0:
                    result.error = f"Data download failed: {err[:500]}"
                    return result
                logger.info("[Arm %d] Downloaded %d shard(s)", arm_idx, num_shards)
                if span:
                    span.set_data("method", "presigned_urls")
                    span.set_data("num_shards", num_shards)

            elif data_server:
                stagger_delay = arm_idx * 15
                if stagger_delay > 0:
                    logger.info("[Arm %d] Waiting %ds before data sync (stagger)...",
                                 arm_idx, stagger_delay)
                    await asyncio.sleep(stagger_delay)

                logger.info("[Arm %d] Syncing data from %s (port %d, max %d shards)...",
                             arm_idx, data_server, data_ssh_port, max_shards)
                await ssh_exec(ssh_host, ssh_port, ssh_key, "mkdir -p /root/.ssh", timeout=15)
                scp_proc = await asyncio.create_subprocess_exec(
                    "scp", *SSH_OPTIONS, "-i", ssh_key, "-P", str(ssh_port),
                    ssh_key, f"root@{ssh_host}:/root/.ssh/data_key",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(scp_proc.communicate(), timeout=30)

                ssh_remote = (
                    f"ssh -p {data_ssh_port} -i /root/.ssh/data_key "
                    f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
                )
                ds_host, ds_path = data_server.split(":", 1)
                list_cmd = (
                    f"chmod 600 /root/.ssh/data_key && "
                    f'{ssh_remote} {ds_host} "ls {ds_path}/*.tar 2>/dev/null | head -{max_shards}"'
                )
                rc, shard_list, err = await ssh_exec(
                    ssh_host, ssh_port, ssh_key, list_cmd, timeout=60
                )
                if rc != 0 or not shard_list.strip():
                    result.error = f"Failed to list shards: {err[:300]}"
                    return result

                shard_files = [
                    os.path.basename(s.strip())
                    for s in shard_list.strip().splitlines() if s.strip()
                ]
                logger.info("[Arm %d] Syncing %d shard(s): %s", arm_idx, len(shard_files),
                             ", ".join(shard_files[:3]) + ("..." if len(shard_files) > 3 else ""))

                include_args = " ".join(f"--include='{f}'" for f in shard_files)
                rsync_cmd = (
                    f"rsync -xahP --inplace "
                    f'{include_args} --exclude="*" '
                    f'-e "{ssh_remote}" '
                    f"{data_server}/ {data_dir}/"
                )
                rc, _, err = await ssh_exec(
                    ssh_host, ssh_port, ssh_key, rsync_cmd, timeout=3600
                )
                if rc != 0:
                    result.error = f"Data sync failed: {err[:500]}"
                    return result
                logger.info("[Arm %d] Data synced (%d shards)", arm_idx, len(shard_files))
                if span:
                    span.set_data("method", "rsync")
                    span.set_data("num_shards", len(shard_files))
                    span.set_data("data_server", data_server)

            else:
                result.error = "No data source specified (need --data-urls or --data-server)"
                return result

        # Run experiment — capture stdout and stderr separately for better diagnostics
        config_path = f"experiments/throughput/{arm_name}.yaml"
        with _span("train.run", f"train {arm_name}") as span:
            logger.info("[Arm %d] Running: %s", arm_idx, config_path)
            if span:
                span.set_data("config_path", config_path)
                span.set_data("torch_version", torch_ver)
                span.set_data("cuda_runtime", cuda_ver)
                span.set_data("gpu_info", result.gpu_info)
                span.set_data("instance_id", instance_id)

            rc, stdout, stderr = await ssh_exec(
                ssh_host, ssh_port, ssh_key,
                f"cd /workspace/prx-tg && bash scripts/run_throughput_arm.sh {config_path}",
                timeout=TRAIN_TIMEOUT_S,
            )
            result.exit_code = rc

            parsed = parse_throughput_results(stdout)
            result.avg_iter_per_sec = parsed.get("avg_iter_per_sec", 0.0)
            result.peak_vram_gb = parsed.get("peak_vram_gb", 0.0)
            result.final_loss = parsed.get("final_loss", float("inf"))
            result.total_steps = int(parsed.get("total_steps", 0))
            result.total_time_s = parsed.get("total_training_time_s", 0.0)

            if span:
                span.set_data("exit_code", rc)
                span.set_data("avg_iter_per_sec", result.avg_iter_per_sec)
                span.set_data("peak_vram_gb", result.peak_vram_gb)
                span.set_data("final_loss", result.final_loss)
                span.set_data("total_steps", result.total_steps)
                span.set_data("total_time_s", result.total_time_s)

        if rc != 0:
            err_detail = stderr[-2000:] if stderr.strip() else ""
            out_detail = stdout[-1000:]
            combined = (
                f"STDERR: {err_detail}\nSTDOUT(tail): {out_detail}"
                if err_detail
                else out_detail
            )
            result.error = f"Training failed (exit {rc}): {combined}"
            logger.warning("[Arm %d] Failed (exit %d).\nSTDERR:\n%s\nSTDOUT(tail):\n%s",
                           arm_idx, rc, err_detail or "(empty)", out_detail[-500:])

            if sentry_sdk:
                _report_remote_crash(
                    arm_name=arm_name,
                    arm_idx=arm_idx,
                    exit_code=rc,
                    stderr_text=err_detail,
                    stdout_tail=out_detail[-500:],
                    gpu_info=result.gpu_info,
                    instance_id=instance_id,
                )
        else:
            logger.info("[Arm %d] %s — %.2f it/s, %.1f GB VRAM, loss=%.6f",
                         arm_idx, arm_name,
                         result.avg_iter_per_sec, result.peak_vram_gb, result.final_loss)

    except Exception as e:
        result.error = str(e)
        logger.exception("[Arm %d] Unexpected error", arm_idx)
        if sentry_sdk:
            sentry_sdk.capture_exception(e)

    finally:
        if instance_id is not None:
            try:
                await client.destroy_instance(instance_id)
                logger.info("[Arm %d] Destroyed instance %s", arm_idx, instance_id)
            except Exception:
                logger.exception("[Arm %d] Failed to destroy instance %s",
                                  arm_idx, instance_id)

    return result


def print_results_table(results: list[ArmResult]):
    print("\n" + "=" * 90)
    print("THROUGHPUT EXPERIMENT RESULTS")
    print("=" * 90)
    print(f"{'Arm':<30} {'it/s':>8} {'Speedup':>8} {'VRAM GB':>8} {'Loss':>12} {'Status':>8}")
    print("-" * 90)

    baseline_its = None
    for r in results:
        status = "OK" if r.exit_code == 0 else "ERROR"
        its_str = f"{r.avg_iter_per_sec:.3f}" if r.avg_iter_per_sec > 0 else "—"
        vram_str = f"{r.peak_vram_gb:.1f}" if r.peak_vram_gb > 0 else "—"
        loss_str = f"{r.final_loss:.6f}" if r.final_loss < float("inf") else "—"

        speedup = "1.00x"
        if baseline_its is None and r.avg_iter_per_sec > 0:
            baseline_its = r.avg_iter_per_sec
        elif baseline_its and r.avg_iter_per_sec > 0:
            speedup = f"{r.avg_iter_per_sec / baseline_its:.2f}x"

        print(f"{r.arm_name:<30} {its_str:>8} {speedup:>8} {vram_str:>8} {loss_str:>12} {status:>8}")

    print("=" * 90)

    print("\n--- JSON RESULTS ---")
    print(json.dumps([{
        "arm": r.arm_name, "description": r.description,
        "avg_iter_per_sec": r.avg_iter_per_sec, "peak_vram_gb": r.peak_vram_gb,
        "final_loss": r.final_loss, "total_steps": r.total_steps,
        "exit_code": r.exit_code, "gpu_info": r.gpu_info, "error": r.error,
    } for r in results], indent=2))
    print("--- END JSON RESULTS ---")


async def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    init_sentry(environment="fleet")

    api_key = args.api_key or os.environ.get("VAST_API_KEY", "")
    if not api_key:
        print("ERROR: Set VAST_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Load presigned URLs from file if provided
    data_urls = None
    if args.data_urls:
        url_path = Path(args.data_urls)
        if not url_path.exists():
            print(f"ERROR: URL file not found: {args.data_urls}", file=sys.stderr)
            sys.exit(1)
        data_urls = [line.strip() for line in url_path.read_text().splitlines() if line.strip()]
        logger.info("Loaded %d presigned URLs from %s", len(data_urls), args.data_urls)

    data_server = args.data_server or os.environ.get("DATA_SERVER", "")
    data_ssh_port = args.data_ssh_port or int(os.environ.get("DATA_SSH_PORT", "22"))

    if not data_urls and not data_server:
        print("ERROR: Provide --data-urls or --data-server", file=sys.stderr)
        sys.exit(1)

    txn = (
        sentry_sdk.start_transaction(op="fleet", name="throughput-fleet")
        if sentry_sdk
        else None
    )
    if txn:
        txn.__enter__()
        txn.set_data("fleet.num_arms", len(ARMS))
        txn.set_data("fleet.max_dph", args.max_dph)
        txn.set_data("fleet.prx_branch", args.prx_branch)

    try:
        async with VastClient(api_key) as client:
            num_arms = len(ARMS)
            logger.info("Searching for %d matching RTX 4090 offers (max $%.2f/hr)...",
                         num_arms, args.max_dph)
            offers = await find_matching_offers(client, args.max_dph, num_arms)

            if not offers:
                print("ERROR: No matching RTX 4090 offers found", file=sys.stderr)
                sys.exit(1)

            if len(offers) < num_arms:
                logger.warning("Found %d offers (need %d) — some arms will share hosts",
                               len(offers), num_arms)

            if args.dry_run:
                print(f"DRY RUN: Would launch {num_arms} arms on {len(offers)} instances")
                for i, (name, desc) in enumerate(ARMS):
                    o = offers[i % len(offers)]
                    print(f"  Arm {i}: {name} → offer {o['id']} "
                          f"(${o.get('dph_total', 0):.3f}/hr, "
                          f"PCIe {o.get('pcie_bw', 0):.0f} GB/s, "
                          f"RAM {o.get('cpu_ram', 0) / 1000:.0f} GB)")
                return

            logger.info("Launching %d experiment arms in parallel...", num_arms)
            tasks = [
                run_arm(
                    client=client, offer=offers[i % len(offers)],
                    arm_idx=i, arm_name=name, arm_desc=desc,
                    ssh_key=args.ssh_key, prx_repo=args.prx_repo,
                    prx_branch=args.prx_branch, data_urls=data_urls,
                    data_server=data_server, data_ssh_port=data_ssh_port,
                    max_shards=args.max_shards,
                )
                for i, (name, desc) in enumerate(ARMS)
            ]
            results = await asyncio.gather(*tasks)

            # Record per-arm results as Sentry data
            if txn:
                succeeded = sum(1 for r in results if r.exit_code == 0)
                txn.set_data("fleet.succeeded", succeeded)
                txn.set_data("fleet.failed", len(results) - succeeded)
                for r in results:
                    txn.set_data(f"arm.{r.arm_name}.exit_code", r.exit_code)
                    txn.set_data(f"arm.{r.arm_name}.it_per_sec", r.avg_iter_per_sec)
                    txn.set_data(f"arm.{r.arm_name}.peak_vram_gb", r.peak_vram_gb)

            print_results_table(results)
    except Exception:
        if txn:
            txn.set_status("internal_error")
        raise
    finally:
        if txn:
            txn.__exit__(None, None, None)


def cli():
    parser = argparse.ArgumentParser(
        description="Launch throughput experiment fleet on Vast.ai 4090 instances",
    )
    parser.add_argument("--data-urls",
                        help="File with one presigned S3 URL per line for each .tar shard (recommended)")
    parser.add_argument("--data-server",
                        help="rsync source (e.g. root@50.173.30.254:/workspace/prx-tg/data/shards/faces7k)")
    parser.add_argument("--data-ssh-port", type=int, default=22,
                        help="SSH port for the data server (default: 22)")
    parser.add_argument("--max-shards", type=int, default=5,
                        help="Max number of .tar shards to sync per instance (default: 5)")
    parser.add_argument("--api-key", help="Vast.ai API key (or VAST_API_KEY env)")
    parser.add_argument("--max-dph", type=float, default=0.50,
                        help="Max $/hr per instance (default: 0.50)")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_rsa"))
    parser.add_argument("--prx-repo",
                        default="https://github.com/timlawrenz/prx-tg.git")
    parser.add_argument("--prx-branch", default="experiment/throughput-ablation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
