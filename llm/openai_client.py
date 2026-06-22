"""OpenAI adapter — Phase 14. Gated import; mirrors ``AnthropicClient``."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from .client import (
    LLMCallRecord,
    measure_latency_ms,
    safe_parse_json,
    utc_now_iso,
)
from .pricing import estimate_cost_usd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_SYSTEM_INSTRUCTION = (
    "You review UK bank-statement salary detection candidates. "
    "Respond with a single JSON object matching the schema described "
    "in the user message. Do not include any prose outside the JSON."
)


@dataclass
class OpenAIClient:
    api_key: str
    default_model: str = _DEFAULT_MODEL
    provider: str = "openai"

    def __post_init__(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai SDK not installed. Install with "
                "`pip install salary-extractor[llm]` or `pip install openai`."
            ) from exc

    def call(
        self,
        payload: dict,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMCallRecord:
        import openai

        used_model = model or self.default_model
        client = openai.OpenAI(api_key=self.api_key)
        user_message = json.dumps(payload, indent=2, default=str)

        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=used_model,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("openai LLM call failed: %s", exc)
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

        text = completion.choices[0].message.content or ""
        try:
            parsed = safe_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai LLM returned unparseable JSON: %s", exc)
            return LLMCallRecord(
                provider=self.provider,
                model=used_model,
                prompt=payload,
                response_text=text,
                parsed_response={},
                input_tokens=getattr(completion, "usage", None).prompt_tokens if getattr(completion, "usage", None) else 0,
                output_tokens=getattr(completion, "usage", None).completion_tokens if getattr(completion, "usage", None) else 0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                request_id=getattr(completion, "id", None),
                timestamp_utc=utc_now_iso(),
                error=f"unparseable JSON: {exc}",
            )

        usage = getattr(completion, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
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
            request_id=getattr(completion, "id", None),
            timestamp_utc=utc_now_iso(),
        )


__all__ = ["OpenAIClient"]
