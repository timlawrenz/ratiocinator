"""Central configuration for ratiocinator."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class ModelRoute(BaseModel):
    """Maps a task type to a specific LLM model."""

    model: str
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    """LLM routing configuration."""

    coding: ModelRoute = ModelRoute(model="deepseek/deepseek-coder")
    generalist: ModelRoute = ModelRoute(model="ollama/llama3")


class SandboxConfig(BaseModel):
    """Docker sandbox settings."""

    timeout_seconds: int = 600
    memory_limit: str = "8g"
    gpu: bool = False


class SearchConfig(BaseModel):
    """Tree search budget controls."""

    max_depth: int = 5
    max_nodes: int = 50
    max_wall_clock_seconds: int = 3600
    branching_factor: int = 3


class SafetyConfig(BaseModel):
    """Hard-coded safety limits (not LLM-controllable)."""

    max_dollars_per_run: float = 10.0
    instance_ttl_seconds: int = 1800


class VastConfig(BaseModel):
    """Vast.ai remote execution settings."""

    api_key: str = ""
    default_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    max_dph: float = 0.15
    disk_gb: float = 5.0
    install_deps: bool = True


class PublishConfig(BaseModel):
    """HuggingFace Hub publishing settings."""

    hf_token: str = ""
    repo_id: str = ""


class HFConfig(BaseModel):
    """HuggingFace Jobs compute settings."""

    token: str = ""
    default_image: str = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
    default_flavor: str = "a100-large"
    namespace: str = ""
    bucket_prefix: str = ""
    max_timeout: str = "4h"


class Config(BaseModel):
    """Top-level ratiocinator configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    vast: VastConfig = Field(default_factory=VastConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)
    work_dir: Path = Path(".ratiocinator")


def _load_dotenv() -> None:
    """Load .env from cwd or parents if it exists."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def load_config(path: Path | None = None) -> Config:
    """Load config from a JSON file, or return defaults.

    Auto-discovers .ratiocinator/config.json in CWD if no path given.

    Environment variables override config file values:
        VAST_API_KEY  → config.vast.api_key
        HF_TOKEN      → config.publish.hf_token
    """
    _load_dotenv()

    if path is None:
        candidate = Path(".ratiocinator/config.json")
        if candidate.exists():
            path = candidate

    config = (
        Config.model_validate_json(path.read_text()) if path and path.exists() else Config()
    )

    # Env overrides (secrets should come from env, not config files)
    if api_key := os.environ.get("VAST_API_KEY"):
        config.vast.api_key = api_key
    if hf_token := os.environ.get("HF_TOKEN"):
        config.publish.hf_token = hf_token
        config.hf.token = hf_token
    if repo_id := os.environ.get("HF_REPO_ID"):
        config.publish.repo_id = repo_id
    if hf_namespace := os.environ.get("HF_NAMESPACE"):
        config.hf.namespace = hf_namespace

    return config
