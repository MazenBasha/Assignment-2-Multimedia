"""Thin abstraction over LLM-as-a-judge providers used to generate VQA
pairs. We keep this provider-agnostic so the experiment is reproducible
even when one provider goes down.

Set ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment. For a fully
local run, point `local_base_url` at an OpenAI-compatible server (vLLM,
TGI, llama.cpp).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol


class Judge(Protocol):
    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str: ...


@dataclass
class AnthropicJudge:
    model: str

    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        import anthropic  # imported lazily so a user without the SDK can still use OpenAI
        client = anthropic.Anthropic()
        # Simple retry on transient errors.
        for attempt in range(4):
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                # Concatenate any text blocks.
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            except Exception as e:  # network / 429 / 5xx
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


@dataclass
class OpenAIJudge:
    model: str
    base_url: str | None = None

    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url) if self.base_url else OpenAI()
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def make_judge(provider: str, model: str, *, local_base_url: str | None = None) -> Judge:
    provider = provider.lower()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        return AnthropicJudge(model=model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAIJudge(model=model)
    if provider == "local":
        return OpenAIJudge(model=model, base_url=local_base_url or "http://localhost:8000/v1")
    raise ValueError(f"Unknown provider: {provider}")


def parse_strict_json_array(text: str) -> list[dict]:
    """Extract the first JSON array from `text`. Tolerant to ```json fences."""
    if "```" in text:
        # take the largest fenced block
        parts = text.split("```")
        for i, part in enumerate(parts):
            if part.lstrip().startswith("json"):
                text = part.split("\n", 1)[1] if "\n" in part else ""
                break
            if part.lstrip().startswith("["):
                text = part
                break
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        out = json.loads(text[start:end + 1])
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        return []
