"""Research coordinator: autonomous ideation → fleet → analysis loop.

The ``ResearchCoordinator`` drives the autonomous research workflow:

1. Ask the LLM to propose diverse experiment configurations (arms).
2. Translate proposals into a ``fleet.ExperimentSpec`` — critically,
   each arm's config overrides are propagated as **environment variables**
   in ``ArmSpec.env`` so the ``FleetExecutor`` can forward them to
   remote training instances.
3. Execute the spec via the fleet framework.
4. Feed results back to the LLM for the next iteration.

The ``_translate_to_spec`` method is the linchpin: it converts
LLM-proposed ``config_overrides`` dicts into ``arm.env`` entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ratiocinator.config import Config
from ratiocinator.fleet.spec import (
    ArmSpec,
    BudgetSpec,
    DataSpec,
    DepsSpec,
    ExperimentSpec,
    HardwareSpec,
    MetricsSpec,
    RepoSpec,
)
from ratiocinator.llm.client import LLMClient

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Research spec — high-level YAML that drives the coordinator
# ---------------------------------------------------------------------------


class ResearchSpec(BaseModel):
    """High-level research specification loaded from YAML.

    Unlike an ``ExperimentSpec`` (which lists concrete arms), a
    ``ResearchSpec`` describes a research *question* and lets the LLM
    propose the concrete arms each iteration.
    """

    name: str
    description: str = ""
    repo: RepoSpec
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    deps: DepsSpec = Field(default_factory=DepsSpec)
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    # Research-specific fields
    base_command: str = "python train.py"
    base_config_path: str = ""
    num_arms: int = 6
    iterations: int = 3
    score_key: str = "loss"
    maximize: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResearchSpec:
        """Load a research spec from a YAML file."""
        import yaml

        text = Path(path).read_text()
        data = yaml.safe_load(text)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# LLM prompt template
# ---------------------------------------------------------------------------

IDEATION_SYSTEM_PROMPT = """\
You are an AI research assistant designing experiments.  Given a research
spec and optional prior results, propose {num_arms} diverse configurations.

Each arm must have:
- "name": short snake_case identifier
- "description": what this arm tests
- "config_overrides": dict of parameter overrides (these become environment
  variables on the remote instance, so values must be strings)

The config_overrides MUST vary across arms — each arm should test a
meaningfully different configuration.

Respond with valid JSON only:
{{
  "reasoning": "why these arms were chosen",
  "arms": [
    {{
      "name": "arm_name",
      "description": "what it tests",
      "config_overrides": {{"PARAM_NAME": "value", ...}}
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigDuplicateError(Exception):
    """Raised when all generated arms have identical config overrides."""


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class ResearchCoordinator:
    """Autonomous research loop: ideation → fleet → analysis.

    Usage::

        spec = ResearchSpec.from_yaml("research.yaml")
        coordinator = ResearchCoordinator(config, spec)
        results = await coordinator.run()
    """

    def __init__(
        self,
        config: Config,
        research_spec: ResearchSpec,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.research_spec = research_spec
        self.llm = llm or LLMClient(config.llm)
        self._base_config: dict[str, Any] = {}

        if research_spec.base_config_path:
            self._load_base_config(research_spec.base_config_path)

    # ------------------------------------------------------------------
    # Base config helpers
    # ------------------------------------------------------------------

    def _load_base_config(self, path: str) -> None:
        """Load the base YAML config for reference in LLM prompts."""
        import yaml

        config_path = Path(path)
        if config_path.exists():
            self._base_config = yaml.safe_load(config_path.read_text()) or {}
            logger.info("Loaded base config from %s", path)
        else:
            logger.warning("Base config %s not found", path)

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def propose_arms(
        self,
        *,
        prior_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the LLM to propose diverse experiment arms.

        Returns:
            List of arm dicts, each containing ``name``, ``description``,
            and ``config_overrides``.
        """
        prompt_parts = [
            f"## Research: {self.research_spec.name}",
            f"## Description: {self.research_spec.description}",
            f"## Base command: {self.research_spec.base_command}",
        ]

        if self._base_config:
            import yaml

            prompt_parts.append(
                f"## Base config:\n```yaml\n{yaml.dump(self._base_config)}\n```"
            )

        if prior_results:
            prompt_parts.append(
                f"## Prior results:\n```json\n"
                f"{json.dumps(prior_results, indent=2)}\n```\n"
                "Build on these results — explore promising directions "
                "and avoid repeating failures."
            )

        prompt = "\n\n".join(prompt_parts)
        system = IDEATION_SYSTEM_PROMPT.format(
            num_arms=self.research_spec.num_arms,
        )

        response = await self.llm.complete_json(
            prompt, task="generalist", system=system,
        )

        arms = response.get("arms", [])
        if not arms:
            raise ValueError("LLM returned no experiment arms")

        return arms

    # ------------------------------------------------------------------
    # Spec translation (the critical fix)
    # ------------------------------------------------------------------

    def _translate_to_spec(
        self,
        llm_arms: list[dict[str, Any]],
        iteration: int = 0,
    ) -> ExperimentSpec:
        """Translate LLM-proposed arms into a ``FleetExecutor``-compatible spec.

        Each arm's ``config_overrides`` are propagated as **environment
        variables** in ``ArmSpec.env`` so the ``FleetExecutor`` prepends
        them to the training command on the remote instance.

        Raises:
            ConfigDuplicateError: If all generated arms have identical
                env configurations (indicating the LLM failed to produce
                diverse proposals).
        """
        arm_specs: list[ArmSpec] = []

        for idx, arm_data in enumerate(llm_arms):
            name = arm_data.get("name", f"arm_{idx}")
            description = arm_data.get("description", "")
            config_overrides = arm_data.get("config_overrides", {})

            # Propagate every config override as an env var so the
            # FleetExecutor injects it into the remote command.
            env: dict[str, str] = {
                str(k): str(v) for k, v in config_overrides.items()
            }

            arm_specs.append(
                ArmSpec(
                    name=name,
                    description=description,
                    command=self.research_spec.base_command,
                    env=env,
                )
            )

        # Guard: all arms must have distinct configurations.
        self._assert_arms_differ(arm_specs)

        return ExperimentSpec(
            name=f"{self.research_spec.name}-iter{iteration}",
            description=self.research_spec.description,
            hardware=self.research_spec.hardware,
            data=self.research_spec.data,
            repo=self.research_spec.repo,
            deps=self.research_spec.deps,
            arms=arm_specs,
            metrics=self.research_spec.metrics,
            budget=self.research_spec.budget,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_arms_differ(arms: list[ArmSpec]) -> None:
        """Verify that generated arm configs are not all identical.

        Uses SHA-256 hashing of the sorted env dict to compare arms.

        Raises:
            ConfigDuplicateError: When every arm has the same env dict.
        """
        if len(arms) <= 1:
            return

        hashes: set[str] = set()
        for arm in arms:
            env_hash = hashlib.sha256(
                json.dumps(arm.env, sort_keys=True).encode()
            ).hexdigest()
            hashes.add(env_hash)

        if len(hashes) == 1:
            raise ConfigDuplicateError(
                f"All {len(arms)} arms have identical env configuration. "
                "The LLM failed to generate diverse configurations."
            )

    # ------------------------------------------------------------------
    # Top-level run loop
    # ------------------------------------------------------------------

    async def run(self) -> list[dict[str, Any]]:
        """Execute the full autonomous research loop.

        For each iteration:
        1. Propose arms via LLM (with prior results as context).
        2. Translate proposals into an ``ExperimentSpec``.
        3. Execute via ``FleetExecutor``.
        4. Collect and return cumulative results.
        """
        from ratiocinator.fleet.executor import FleetExecutor
        from ratiocinator.fleet.results import ResultStore

        store = ResultStore(
            self.config.work_dir / "results" / f"{self.research_spec.name}.json",
        )
        all_results: list[dict[str, Any]] = []

        for iteration in range(self.research_spec.iterations):
            logger.info(
                "=== Iteration %d/%d for %s ===",
                iteration + 1,
                self.research_spec.iterations,
                self.research_spec.name,
            )

            # 1. Propose arms
            prior = all_results if all_results else None
            llm_arms = await self.propose_arms(prior_results=prior)
            logger.info("LLM proposed %d arms", len(llm_arms))

            # 2. Translate to fleet spec
            spec = self._translate_to_spec(llm_arms, iteration=iteration)
            logger.info(
                "Fleet spec: %d arms, envs: %s",
                len(spec.arms),
                [list(a.env.keys()) for a in spec.arms],
            )

            # 3. Execute
            executor = FleetExecutor(
                spec=spec,
                config=self.config.vast,
                store=store,
            )
            arm_results = await executor.run()

            # 4. Collect results for next iteration
            for r in arm_results:
                all_results.append({
                    "iteration": iteration,
                    "arm": r.arm_name,
                    "success": r.success,
                    "metrics": r.metrics,
                    "error": r.error,
                })

        return all_results
