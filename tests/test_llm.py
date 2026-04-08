"""Tests for the LLM client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ratiocinator.config import LLMConfig
from ratiocinator.llm.client import LLMClient


@pytest.fixture
def client():
    return LLMClient(LLMConfig())


def _mock_response(content: str, model: str = "test-model"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


@pytest.mark.asyncio
async def test_complete_routes_coding_task(client):
    mock_resp = _mock_response("some code")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock:
        result = await client.complete("write code", task="coding")
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["model"] == "deepseek/deepseek-coder"
        assert result.content == "some code"


@pytest.mark.asyncio
async def test_complete_routes_generalist_task(client):
    mock_resp = _mock_response("general answer")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock:
        await client.complete("explain something", task="generalist")
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["model"] == "ollama/llama3"


@pytest.mark.asyncio
async def test_complete_json_parses_response(client):
    mock_resp = _mock_response('{"key": "value"}')
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.complete_json("give me json")
        assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_complete_json_strips_markdown_fences(client):
    mock_resp = _mock_response('```json\n{"key": "value"}\n```')
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.complete_json("give me json")
        assert result == {"key": "value"}


class TestExtractJson:
    """Test the robust JSON extraction from _extract_json."""

    def test_strips_think_tags(self):
        from ratiocinator.llm.client import _extract_json

        text = '<think>\nLet me reason about this...\n</think>\n{"key": "value"}'
        assert _extract_json(text) == {"key": "value"}

    def test_extracts_json_from_surrounding_text(self):
        from ratiocinator.llm.client import _extract_json

        text = 'Here is the result:\n{"key": "value"}\nHope that helps!'
        assert _extract_json(text) == {"key": "value"}

    def test_fixes_trailing_commas(self):
        from ratiocinator.llm.client import _extract_json

        text = '{"key": "value", "items": [1, 2,]}'
        assert _extract_json(text) == {"key": "value", "items": [1, 2]}

    def test_raises_on_no_json(self):
        from ratiocinator.llm.client import _extract_json

        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json("just plain text with no braces")
