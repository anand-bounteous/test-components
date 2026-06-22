"""Anthropic Claude adapter — Phase 14.

Gated import: the project does not require the ``anthropic`` SDK to be
installed. If a caller tries to instantiate ``AnthropicClient`` without
the SDK present, we raise a clear ``ImportError`` so the deterministic
path keeps working.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .client import (
    LLMCallRecord,
    measure_latency_ms,
    safe_parse_json,
    utc_now_iso,
)
from .pricing import estimate_cost_usd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_SYSTEM_INSTRUCTION = (
    "You review UK bank-statement salary detection candidates. "
    "Respond with a single JSON object matching the schema described "
    "in the user message. Do not include any prose outside the JSON. "
    "Wrap the JSON in a ```json``` fenced block."
)


@dataclass
class AnthropicClient:
    api_key: str
    default_model: str = _DEFAULT_MODEL
    provider: str = "anthropic"

    def __post_init__(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK not installed. Install with "
                "`pip install salary-extractor[llm]` or `pip install anthropic`."
            ) from exc

    def call(
        self,
        payload: dict,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMCallRecord:
        import anthropic

        used_model = model or self.default_model
        client = anthropic.Anthropic(api_key=self.api_key)
        user_message = json.dumps(payload, indent=2, default=str)

        start = time.perf_counter()
        try:
            message = client.messages.create(
                model=used_model,
                max_tokens=max_tokens,
                system=_SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("anthropic LLM call failed: %s", exc)
            return LLMCallRecord(
                provider=self.provider,
                model=used_model,
                prompt=payload,
                response_text="",
                parsed_response={},
                input_tokens=0,
                output_tokens=0,
                latency_ms=measure_latency_ms(start),
                cost_usd=0.0,
                request_id=None,
                timestamp_utc=utc_now_iso(),
                error=str(exc),
            )
        latency_ms = measure_latency_ms(start)

        # Anthropic returns content as a list of blocks; concatenate text.
        text = "".join(
            getattr(block, "text", "") for block in getattr(message, "content", [])
        )
        try:
            parsed = safe_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("anthropic LLM returned unparseable JSON: %s", exc)
            return LLMCallRecord(
                provider=self.provider,
                model=used_model,
                prompt=payload,
                response_text=text,
                parsed_response={},
                input_tokens=getattr(getattr(message, "usage", None), "input_tokens", 0) or 0,
                output_tokens=getattr(getattr(message, "usage", None), "output_tokens", 0) or 0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                request_id=getattr(message, "id", None),
                timestamp_utc=utc_now_iso(),
                error=f"unparseable JSON: {exc}",
            )

        usage = getattr(message, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return LLMCallRecord(
            provider=self.provider,
            model=used_model,
            prompt=payload,
            response_text=text,
            parsed_response=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost_usd(used_model, input_tokens, output_tokens),
            request_id=getattr(message, "id", None),
            timestamp_utc=utc_now_iso(),
        )


__all__ = ["AnthropicClient"]
