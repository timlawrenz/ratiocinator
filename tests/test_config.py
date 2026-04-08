"""Tests for the config module."""

import json
import tempfile
from pathlib import Path

from ratiocinator.config import Config, load_config


def test_default_config():
    config = Config()
    assert config.llm.coding.model == "deepseek/deepseek-coder"
    assert config.llm.generalist.model == "ollama/llama3"
    assert config.sandbox.timeout_seconds == 600
    assert config.search.max_depth == 5
    assert config.safety.max_dollars_per_run == 10.0


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
