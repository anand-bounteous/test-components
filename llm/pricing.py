"""Per-model LLM pricing — Phase 14.

Hardcoded table of USD per 1M tokens (input / output). Kept as a flat
constant so the project stays deterministic — callers update the table
manually when model pricing changes. Unknown models return zero cost
rather than raising, since cost is informational only.
"""
from __future__ import annotations


_PRICING_USD_PER_M_TOKENS: dict[str, dict[str, float]] = {
    # Anthropic Claude (4.x family).
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI.
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for the given (model, input_tokens, output_tokens).

    Returns 0.0 for unknown models — cost is informational only and the
    detector must not fail because of a missing pricing entry.
    """
    rates = _PRICING_USD_PER_M_TOKENS.get(model)
    if not rates:
        return 0.0
    return round(
        (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000,
        6,
    )


def known_models() -> tuple[str, ...]:
    return tuple(sorted(_PRICING_USD_PER_M_TOKENS))


__all__ = ["estimate_cost_usd", "known_models"]
