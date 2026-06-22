"""LLM client interface, mock implementation, and call record — Phase 14.

The detector's LLM-fallback path uses an abstract ``LLMClient`` so the
project stays SDK-agnostic. Real providers wrap their SDK in a thin
adapter (``anthropic_client.py``, ``openai_client.py``); tests inject a
``MockLLMClient`` that returns canned responses without touching the
network.

Every call returns a uniform ``LLMCallRecord`` capturing provider,
model, prompt, response, tokens, cost, latency, request id, and
timestamp. The detector logs and audits this record in full.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallRecord:
    """Structured record of a single LLM call.

    Fields are deliberately flat + JSON-serialisable so the audit trail
    can drop them straight into the response.
    """

    provider: str               # "anthropic" | "openai" | "mock"
    model: str
    prompt: dict                # the payload sent (already a dict)
    response_text: str          # raw text from the model
    parsed_response: dict       # JSON parsed; matches llm_output schema
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    request_id: str | None
    timestamp_utc: str          # ISO 8601
    error: str | None = None    # populated on failure


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM call surface."""

    def call(
        self,
        payload: dict,
        *,
        model: str,
        max_tokens: int = 4096,
    ) -> LLMCallRecord: ...


# ---------------------------------------------------------------------------
# Mock client — deterministic, no network. Tests use this exclusively.
# ---------------------------------------------------------------------------


@dataclass
class MockLLMClient:
    """Canned-response LLM client for tests.

    Callers either provide a list of pre-built parsed responses (one per
    expected call) or override ``default_response`` for a fixed reply.
    Each call records the prompt the detector sent so tests can assert
    the prompt shape.
    """

    responses: list[dict] = field(default_factory=list)
    default_response: dict | None = None
    provider: str = "mock"
    model: str = "mock-llm"
    fixed_input_tokens: int = 256
    fixed_output_tokens: int = 128
    fixed_latency_ms: float = 12.5
    invocations: list[LLMCallRecord] = field(default_factory=list)

    def call(
        self,
        payload: dict,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMCallRecord:
        if self.responses:
            parsed = self.responses.pop(0)
        elif self.default_response is not None:
            parsed = dict(self.default_response)
        else:
            raise RuntimeError(
                "MockLLMClient ran out of canned responses and no "
                "default_response was set."
            )
        used_model = model or self.model
        record = LLMCallRecord(
            provider=self.provider,
            model=used_model,
            prompt=payload,
            response_text=json.dumps(parsed),
            parsed_response=parsed,
            input_tokens=self.fixed_input_tokens,
            output_tokens=self.fixed_output_tokens,
            latency_ms=self.fixed_latency_ms,
            cost_usd=estimate_cost_usd(
                used_model, self.fixed_input_tokens, self.fixed_output_tokens
            ),
            request_id=f"mock-{len(self.invocations) + 1:04d}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.invocations.append(record)
        return record


# ---------------------------------------------------------------------------
# Common helpers used by real adapters
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def measure_latency_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def safe_parse_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM reply. The fallback prompt
    asks the model to emit JSON wrapped in ```json fences — strip them.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop fence + optional language tag.
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


__all__ = [
    "LLMCallRecord",
    "LLMClient",
    "MockLLMClient",
    "measure_latency_ms",
    "safe_parse_json",
    "utc_now_iso",
]
