"""Tests for the config module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from ratiocinator.config import Config, load_config


def test_default_config():
    config = Config()
    assert config.llm.coding.model == "deepseek/deepseek-coder"
    assert config.llm.generalist.model == "ollama/llama3"
    assert config.sandbox.timeout_seconds == 600
    assert config.search.max_depth == 5
    assert config.safety.max_dollars_per_run == 10.0
    assert config.vast.api_key == ""
    assert config.publish.hf_token == ""


def test_load_config_defaults():
    config = load_config(None)
    assert isinstance(config, Config)


def test_load_config_from_file():
    data = {
        "llm": {"coding": {"model": "test-model", "temperature": 0.1}},
        "search": {"max_depth": 10},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        config = load_config(Path(f.name))

    assert config.llm.coding.model == "test-model"
    assert config.llm.coding.temperature == 0.1
    assert config.search.max_depth == 10
    # Defaults preserved for unset fields
    assert config.llm.generalist.model == "ollama/llama3"


def test_vast_config_from_file():
    data = {"vast": {"max_dph": 0.25, "disk_gb": 10.0}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        config = load_config(Path(f.name))
    assert config.vast.max_dph == 0.25
    assert config.vast.disk_gb == 10.0
    assert config.vast.default_image == "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"


def test_env_overrides_vast_api_key():
    env = {"VAST_API_KEY": "test-vast-key-123"}
    with patch.dict(os.environ, env, clear=False):
        config = load_config(None)
    assert config.vast.api_key == "test-vast-key-123"


def test_env_overrides_hf_token():
    env = {"HF_TOKEN": "hf_test_token_456"}
    with patch.dict(os.environ, env, clear=False):
        config = load_config(None)
    assert config.publish.hf_token == "hf_test_token_456"


def test_env_overrides_hf_repo_id():
    env = {"HF_REPO_ID": "user/my-dataset"}
    with patch.dict(os.environ, env, clear=False):
        config = load_config(None)
    assert config.publish.repo_id == "user/my-dataset"


def test_env_overrides_config_file():
    """Env vars take precedence over values in config file."""
    data = {"vast": {"api_key": "from-file"}}
    env = {"VAST_API_KEY": "from-env"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        with patch.dict(os.environ, env, clear=False):
            config = load_config(Path(f.name))
    assert config.vast.api_key == "from-env"
