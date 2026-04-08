"""Tests for instance bootstrap script generation."""

from __future__ import annotations

from ratiocinator.infra.bootstrap import generate_onstart


def test_basic_onstart():
    script = generate_onstart(
        repo_url="https://github.com/user/repo.git",
        branch="experiment-1",
    )
    assert "#!/bin/bash" in script
    assert "git clone --branch experiment-1" in script
    assert "https://github.com/user/repo.git" in script
    assert "TRAIN_STEPS=500" in script


def test_custom_command_and_steps():
    script = generate_onstart(
        repo_url="https://github.com/user/repo.git",
        branch="main",
        train_command="python main.py --fast",
        steps=1000,
    )
    assert "python main.py --fast" in script
    assert "TRAIN_STEPS=1000" in script


def test_env_vars():
    script = generate_onstart(
        repo_url="https://github.com/user/repo.git",
        branch="main",
        env={"LR": "0.001", "BATCH_SIZE": "32"},
    )
    assert 'export LR="0.001"' in script
    assert 'export BATCH_SIZE="32"' in script


def test_webhook_url():
    script = generate_onstart(
        repo_url="https://github.com/user/repo.git",
        branch="main",
        webhook_url="http://orchestrator:8765/",
    )
    assert "curl" in script
    assert "http://orchestrator:8765/" in script
