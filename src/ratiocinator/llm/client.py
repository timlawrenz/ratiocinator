"""LLM client with model routing via LiteLLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm

from ratiocinator.config import LLMConfig, ModelRoute

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    model: str
    usage: dict[str, int]


class LLMClient:
    """Unified LLM client that routes requests based on task type."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def _route(self, task: str) -> ModelRoute:
        if task == "coding":
            return self.config.coding
        return self.config.generalist

    async def complete(
        self,
        prompt: str,
        *,
        task: str = "generalist",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a completion request to the appropriate model."""
        route = self._route(task)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else route.temperature,
            "max_tokens": max_tokens or route.max_tokens,
        }
        if route.api_base:
            kwargs["api_base"] = route.api_base

        logger.debug("LLM request: model=%s task=%s", route.model, task)
        response = await litellm.acompletion(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )

    async def complete_json(
        self,
        prompt: str,
        *,
        task: str = "coding",
        system: str | None = None,
    ) -> dict[str, Any]:
        """Request a JSON response and parse it."""
        json_system = (system or "") + "\nRespond with valid JSON only. No markdown fences."
        resp = await self.complete(prompt, task=task, system=json_system.strip())
        return _extract_json(resp.content)


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from LLM output.

    Handles: thinking tags, markdown fences, trailing text, control
    characters, and common formatting issues from local models.
    """
    import re

    # Strip <think>...</think> blocks (qwen3 reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    # Try direct parse first (strict=False tolerates control chars in strings)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

    # Find matching closing brace
    depth = 0
    in_string = False
    escape_next = False
    for i in range(brace_start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : i + 1]
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    return json.loads(fixed, strict=False)

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")
