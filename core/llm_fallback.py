"""LLM fallback orchestrator — Phase 14.

Decides whether the deterministic pipeline needs an LLM tie-break and,
when it does, builds the fallback payload, calls the injected
``LLMClient``, parses the response, and constructs an
``LLMReviewResult`` for the caller.

Trigger logic (only when ``enable_llm=True``):
    * empty candidate list + ``no_signal_detected`` warning
    * top candidate confidence < 0.55
    * ≥ 2 candidates within 0.05 confidence of each other
"""
from __future__ import annotations

import logging
from typing import Sequence

from ..llm.client import LLMClient, LLMCallRecord
from ..llm.fallback_prompt import build_fallback_payload
from ..models.audit import LLMReviewResult
from ..models.output import SalaryDetectionResult
from ..models.transaction import NormalisedTransaction

logger = logging.getLogger(__name__)

_AMBIGUITY_DELTA = 0.05
_LOW_CONFIDENCE_FLOOR = 0.55
_NO_SIGNAL_WARNING = "no_signal_detected"


def should_invoke_fallback(
    result: SalaryDetectionResult,
    *,
    enable_llm: bool,
) -> tuple[bool, str]:
    """Return ``(invoke, reason)``. ``reason`` is a short token."""
    if not enable_llm:
        return False, "disabled"
    if not result.candidate_sets:
        if any(w.code == _NO_SIGNAL_WARNING for w in result.warnings):
            return True, "no_signal_detected"
        return False, "no_candidates_no_warning"
    confidences = sorted(
        (float(c.confidence) for c in result.candidate_sets),
        reverse=True,
    )
    top = confidences[0]
    if top < _LOW_CONFIDENCE_FLOOR:
        return True, "low_top_confidence"
    if len(confidences) >= 2 and (confidences[0] - confidences[1]) <= _AMBIGUITY_DELTA:
        return True, "tied_candidates"
    return False, "clear_winner"


def parse_llm_review(
    call: LLMCallRecord,
) -> LLMReviewResult:
    """Map a raw ``LLMCallRecord`` into a structured ``LLMReviewResult``.

    Tolerates missing fields by falling back to safe defaults; the
    raw call is always attached as ``call_record`` so callers can
    inspect the unparsed reply.
    """
    parsed = call.parsed_response or {}
    selected = tuple(parsed.get("selected_candidate_set_ids") or ())
    rejected = tuple(parsed.get("rejected_candidate_set_ids") or ())
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    reasoning_raw = parsed.get("reasoning") or ()
    if isinstance(reasoning_raw, str):
        reasoning = (reasoning_raw,)
    else:
        reasoning = tuple(str(r) for r in reasoning_raw)
    additional_review = bool(parsed.get("additional_review_needed", False))
    return LLMReviewResult(
        selected_candidate_set_ids=selected,
        rejected_candidate_set_ids=rejected,
        confidence=confidence,
        reasoning=reasoning,
        additional_review_needed=additional_review,
        call_record=call,
    )


def invoke_llm_fallback(
    result: SalaryDetectionResult,
    transactions: Sequence[NormalisedTransaction],
    *,
    client: LLMClient,
    model: str | None = None,
    max_tokens: int = 4096,
) -> LLMReviewResult:
    """Build the prompt, call the LLM client, parse the reply."""
    payload = build_fallback_payload(result, transactions)
    logger.info(
        "llm_fallback.invoke",
        extra={
            "analysis_id": result.analysis_id,
            "candidate_count": len(result.candidate_sets),
            "transaction_count": sum(
                1 for t in transactions if t.direction == "credit"
            ),
            "model": model,
        },
    )
    call = client.call(
        payload,
        model=model or getattr(client, "default_model", "mock-model"),
        max_tokens=max_tokens,
    )
    if call.error:
        logger.warning(
            "llm_fallback.call_failed",
            extra={
                "analysis_id": result.analysis_id,
                "provider": call.provider,
                "model": call.model,
                "error": call.error,
            },
        )
    else:
        logger.info(
            "llm_fallback.call_complete",
            extra={
                "analysis_id": result.analysis_id,
                "provider": call.provider,
                "model": call.model,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "latency_ms": call.latency_ms,
                "cost_usd": call.cost_usd,
            },
        )
    return parse_llm_review(call)


__all__ = [
    "should_invoke_fallback",
    "parse_llm_review",
    "invoke_llm_fallback",
]
