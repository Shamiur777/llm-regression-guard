"""LLM provider interface plus a deterministic mock and an optional Anthropic client."""
from __future__ import annotations

import os
import random
from typing import Callable, Protocol


class Provider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MockProvider:
    """Deterministic stand-in for a model. SIMULATED behaviour, not real model quality.

    `handler(system, user, rng)` decides the output. The rng is seeded from
    (seed, user text) so results do not depend on call order.
    """

    def __init__(self, handler: Callable[[str, str, random.Random], str], seed: int = 0):
        self.handler = handler
        self.seed = seed

    def complete(self, system: str, user: str) -> str:
        return self.handler(system, user, random.Random(f"{self.seed}:{user}"))


class ScriptedProvider:
    """Returns canned replies in order (or by a function); records the calls. For tests."""

    def __init__(self, replies):
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self.replies):
            return self.replies(system, user)
        return self.replies[len(self.calls) - 1]


class AnthropicProvider:
    """Real model via the `anthropic` package. Not exercised live in this repo's tests."""

    def __init__(self, model: str | None = None, max_tokens: int = 512,
                 temperature: float = 0.0, client=None):
        self.model = model or os.environ.get("REGRESSION_GUARD_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed in live mode

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str) -> str:
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def build_provider(cfg: dict, seed: int | None = None, noise: float | None = None) -> Provider:
    """Create a provider from the `provider` block of a feature config."""
    kind = cfg.get("type", "mock")
    if kind == "anthropic":
        return AnthropicProvider(model=cfg.get("model"), temperature=cfg.get("temperature", 0.0))
    if kind == "mock":
        from .examples.triage import make_simulator

        sim = cfg.get("simulator", "triage")
        if sim != "triage":
            raise ValueError(f"unknown simulator: {sim}")
        n = cfg.get("noise", 0.0) if noise is None else noise
        s = cfg.get("seed", 0) if seed is None else seed
        return MockProvider(make_simulator(n), seed=s)
    raise ValueError(f"unknown provider type: {kind}")
