"""Central configuration for ratiocinator."""

from __future__ import annotations

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


class Config(BaseModel):
    """Top-level ratiocinator configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    work_dir: Path = Path(".ratiocinator")


def load_config(path: Path | None = None) -> Config:
    """Load config from a JSON file, or return defaults."""
    if path and path.exists():
        return Config.model_validate_json(path.read_text())
    return Config()
