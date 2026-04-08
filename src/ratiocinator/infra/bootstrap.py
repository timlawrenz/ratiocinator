"""Instance bootstrap: generates onstart scripts for Vast.ai instances."""

from __future__ import annotations

import textwrap


def generate_onstart(
    repo_url: str,
    branch: str,
    train_command: str = "python train.py",
    webhook_url: str | None = None,
    env: dict[str, str] | None = None,
    steps: int = 500,
) -> str:
    """Generate an onstart.sh script for a Vast.ai instance.

    The script clones the repo at a specific branch, installs deps,
    runs training, and reports results back via webhook.
    """
    env_exports = ""
    if env:
        env_exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())

    webhook_report = ""
    if webhook_url:
        webhook_report = textwrap.dedent(f"""\
            # Report results back via webhook
            METRICS_LINE=$(grep "^METRICS:" "$WORK_DIR/train.log" | tail -1)
            STDOUT_TAIL=$(tail -50 "$WORK_DIR/train.log" | base64 -w0)
            python3 -c "
            import json, sys
            d = dict(instance_id='$INSTANCE_ID',
                     exit_code=int('$TRAIN_EXIT_CODE'),
                     metrics='$METRICS_LINE',
                     stdout_tail='$STDOUT_TAIL')
            print(json.dumps(d))
            " > /tmp/payload.json
            curl -s -X POST "{webhook_url}" \\
                -H "Content-Type: application/json" \\
                -d @/tmp/payload.json
        """)

    return textwrap.dedent(f"""\
        #!/bin/bash
        set -e

        export INSTANCE_ID=$(hostname)
        export TRAIN_STEPS={steps}
        {env_exports}

        # Use /root as working directory (works on all Vast.ai images)
        WORK_DIR=${{WORKSPACE:-/root}}
        mkdir -p "$WORK_DIR"

        echo "=== Ratiocinator experiment bootstrap ==="
        echo "Instance: $INSTANCE_ID"
        echo "Branch: {branch}"
        echo "Steps: $TRAIN_STEPS"
        echo "Work dir: $WORK_DIR"

        # Clone repo at specific branch
        cd "$WORK_DIR"
        git clone --branch {branch} --depth 1 {repo_url} experiment
        cd experiment

        # Install dependencies
        if [ -f requirements.txt ]; then
            pip install -q -r requirements.txt
        elif [ -f pyproject.toml ]; then
            pip install -q -e .
        fi

        # Run training, capture output
        echo "=== Starting training ==="
        {train_command} 2>&1 | tee "$WORK_DIR/train.log"
        TRAIN_EXIT_CODE=${{PIPESTATUS[0]}}
        echo "=== Training finished with exit code $TRAIN_EXIT_CODE ==="

        {webhook_report}

        exit $TRAIN_EXIT_CODE
    """)
