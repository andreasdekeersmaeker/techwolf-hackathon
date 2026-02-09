"""Wrapper around the Anthropic SDK for structured LLM calls."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

log = logging.getLogger(__name__)


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or None)


def ask_claude(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> str:
    """Send a single-turn message and return the text response."""
    client = _get_client()
    resp = client.messages.create(
        model=model or CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


def ask_claude_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.1,
) -> Any:
    """Send a prompt expecting a JSON response. Extracts JSON from the reply."""
    raw = ask_claude(
        system_prompt,
        user_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _extract_json(raw)


def _extract_json(text: str) -> Any:
    """Extract JSON from a response that may include markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first [ or { and matching to end
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    log.warning("Failed to extract JSON from response: %s", text[:500])
    return {}
