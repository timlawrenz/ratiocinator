#!/usr/bin/env python3
"""Provision a persistent Vast.ai instance as LIDC-IDRI data server.

This creates a cheap GPU instance that:
1. Downloads LIDC-IDRI from TCIA (public, no API key needed)
2. Preprocesses DICOMs to 16-bit HU PNGs
3. Creates the index.csv + split manifest
4. Serves data via rsync/SSH for fleet experiment instances

Usage:
    # Provision and prep data (takes ~2-4 hours for full LIDC-IDRI)
    python scripts/dinox_data_server.py provision

    # Check status
    python scripts/dinox_data_server.py status

    # Get rsync command for fleet specs
    python scripts/dinox_data_server.py rsync-info

    # Tear down when done with experiments
    python scripts/dinox_data_server.py destroy

The instance ID is saved to .ratiocinator/dinox_data_server.json for
subsequent commands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = Path(".ratiocinator/dinox_data_server.json")


def _get_api_key() -> str:
    """Get Vast.ai API key from env var or ratiocinator config."""
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key
    try:
        from ratiocinator.config import load_config
        key = load_config().vast.api_key
        if key:
            return key
    except Exception:
        pass
    logger.error("VAST_API_KEY not set. Set env var or add to .ratiocinator/config.json")
    sys.exit(1)


# Cheap instance: small GPU is fine, we just need storage + bandwidth.
# 300GB disk for raw DICOMs (~124GB) + processed PNGs (~60GB) + headroom.
OFFER_QUERY = {
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "min_disk_gb": 300,
    "min_inet_down": 2000,
    "max_dph": 0.50,
}

DOCKER_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

# Shell script that runs on the data server after boot.
DATA_PREP_SCRIPT = r"""#!/bin/bash
set -euo pipefail

echo "=== DINO-X Data Server Setup ==="
cd /workspace

# Clone DINO-X repo for preprocessing scripts
if [ ! -d "DINO-X" ]; then
    git clone --depth 1 https://github.com/timlawrenz/DINO-X.git
fi
cd DINO-X

# Install deps
pip install --quiet numpy pillow pydicom

# Create data directories
mkdir -p data/raw data/processed/_index data/processed/_splits data/processed/_manifests

# Download LIDC-IDRI from TCIA (public collection, no API key)
echo "=== Downloading LIDC-IDRI from TCIA ==="
python scripts/phase2_tcia_download.py download-collection \
    --collection LIDC-IDRI \
    --modality CT \
    --out-root data/raw/lidc-idri

# Preprocess DICOMs to 16-bit HU PNGs
echo "=== Preprocessing DICOMs ==="
python scripts/phase2_preprocess_lidc_idri.py \
    --dicom-root data/raw/lidc-idri \
    --out-root data/processed/lidc-idri

# Create train/val split
echo "=== Creating split manifest ==="
python scripts/phase4_make_split_manifest.py \
    --index-csv data/processed/_index/index.csv \
    --seed 42

echo "=== Data prep complete ==="
echo "Processed slices: $(wc -l < data/processed/_index/index.csv)"
echo "Data size: $(du -sh data/processed/)"

# Create a marker file
echo "ready" > /workspace/DINO-X/data/.data_ready
echo "=== Data server ready for rsync ==="
"""


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


async def _provision() -> None:
    """Provision a Vast.ai instance and start data preparation."""
    from ratiocinator.infra.vast_client import VastClient

    api_key = _get_api_key()

    state = _load_state()
    if state.get("instance_id"):
        logger.warning(
            "Data server instance %s already exists. Use 'destroy' first or 'status' to check.",
            state["instance_id"],
        )
        sys.exit(1)

    async with VastClient(api_key=api_key) as client:
        logger.info("Searching for suitable offers...")
        offers = await client.search_offers(
            gpu_name=OFFER_QUERY["gpu_name"],
            num_gpus=OFFER_QUERY["num_gpus"],
            max_dph=OFFER_QUERY["max_dph"],
            limit=50,
        )

        # Filter for bandwidth + disk space
        viable = [
            o for o in offers
            if o.get("inet_down", 0) >= OFFER_QUERY["min_inet_down"]
            and o.get("disk_space", 0) >= OFFER_QUERY["min_disk_gb"]
        ]

        if not viable:
            logger.error("No viable offers found. Try relaxing constraints.")
            sys.exit(1)

        # Pick cheapest
        viable.sort(key=lambda o: o.get("dph_total", 999))
        offer = viable[0]
        logger.info(
            "Selected offer: id=%s, gpu=%s, dph=$%.3f, inet_down=%d Mbps, disk=%d GB",
            offer["id"],
            offer.get("gpu_name"),
            offer.get("dph_total", 0),
            offer.get("inet_down", 0),
            offer.get("disk_space", 0),
        )

        logger.info("Creating instance...")
        instance_id = await client.create_instance(
            offer_id=offer["id"],
            image=DOCKER_IMAGE,
            disk_gb=300,
            onstart=DATA_PREP_SCRIPT,
        )

        state = {
            "instance_id": instance_id,
            "offer_id": offer["id"],
            "dph": offer.get("dph_total", 0),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "provisioning",
        }
        _save_state(state)

        logger.info("Instance %s created. Data prep will start automatically on boot.", instance_id)
        logger.info("Run 'python scripts/dinox_data_server.py status' to check progress.")
        logger.info("Expected prep time: 2-4 hours (download ~1.5h, preprocess ~1h).")


async def _status() -> None:
    """Check data server status."""
    from ratiocinator.infra.vast_client import VastClient

    state = _load_state()
    if not state.get("instance_id"):
        logger.info("No data server provisioned. Run 'provision' first.")
        return

    api_key = _get_api_key()

    async with VastClient(api_key=api_key) as client:
        info = await client.get_instance(state["instance_id"])
        ssh_host = info.ssh_host
        ssh_port = info.ssh_port
        status = info.actual_status

        logger.info("Instance: %s", state["instance_id"])
        logger.info("Status: %s", status)
        logger.info("SSH: ssh -p %s root@%s", ssh_port, ssh_host)
        logger.info("Cost so far: ~$%.2f", state.get("dph", 0) * (time.time() - time.mktime(time.strptime(state["created_at"], "%Y-%m-%dT%H:%M:%SZ"))) / 3600)

        if ssh_host and ssh_port:
            state["ssh_host"] = ssh_host
            state["ssh_port"] = ssh_port
            _save_state(state)


async def _rsync_info() -> None:
    """Print rsync connection info for fleet specs."""
    state = _load_state()
    if not state.get("ssh_host"):
        logger.error("No SSH info available. Run 'status' first (instance must be booted).")
        sys.exit(1)

    host = state["ssh_host"]
    port = state["ssh_port"]
    rsync_server = f"root@{host}:/workspace/DINO-X/data"

    print(f"\n=== Data Server rsync Info ===")
    print(f"rsync server: {rsync_server}")
    print(f"SSH port: {port}")
    print()
    print("For fleet specs, use:")
    print(f'  data:')
    print(f'    source: rsync')
    print(f'    rsync_server: "{rsync_server}"')
    print(f'    rsync_port: {port}')
    print(f'    target: /workspace/experiment/data')
    print()
    print("Or pass via CLI:")
    print(f"  ratiocinator fleet run spec.yaml --data-server {rsync_server}")


async def _destroy() -> None:
    """Destroy the data server instance."""
    from ratiocinator.infra.vast_client import VastClient

    state = _load_state()
    if not state.get("instance_id"):
        logger.info("No data server to destroy.")
        return

    api_key = _get_api_key()

    instance_id = state["instance_id"]
    async with VastClient(api_key=api_key) as client:
        logger.info("Destroying instance %s...", instance_id)
        await client.destroy_instance(instance_id)

    STATE_FILE.unlink(missing_ok=True)
    logger.info("Instance %s destroyed. State file cleaned up.", instance_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="DINO-X LIDC-IDRI data server on Vast.ai")
    ap.add_argument("action", choices=["provision", "status", "rsync-info", "destroy"])
    args = ap.parse_args()

    actions = {
        "provision": _provision,
        "status": _status,
        "rsync-info": _rsync_info,
        "destroy": _destroy,
    }
    asyncio.run(actions[args.action]())


if __name__ == "__main__":
    main()
