"""LLM fallback prompt builder — Phase 14.

When the deterministic detector is ambiguous (no clear winner) the
detector pipeline calls back into an LLM with the *complete* transaction
list, together with a compact summary of the deterministic candidates,
and asks the LLM to pick the most plausible salary set.

The payload extends ``build_llm_payload`` (Phase 10) by appending a
``transactions`` block listing every credit transaction the detector
ingested — descriptions included so the LLM has full context — and
swapping the task description / instructions for the fallback role.
"""
from __future__ import annotations

from typing import Sequence

from ..models.candidate import SalaryCandidateSet
from ..models.output import SalaryDetectionResult
from ..models.transaction import NormalisedTransaction
from .payload_builder import build_llm_payload

_FALLBACK_INSTRUCTIONS: tuple[str, ...] = (
    "The deterministic detector could not pick a clear winner.",
    "Inspect the full credit transaction list provided below.",
    "Identify which transactions form the salary set, listing the "
    "transaction_id values you select.",
    "Score the most plausible interpretation in [0, 1] and provide "
    "concise reasoning lines.",
    "Set additional_review_needed=true when even with full context "
    "the salary set is not determinable — explain why.",
    "Treat induced description patterns and amount/date-only "
    "candidates as secondary; prefer explicit payroll signals when "
    "available.",
    "Do not invent transaction_ids — pick only from the provided list.",
)


def build_fallback_payload(
    result: SalaryDetectionResult,
    transactions: Sequence[NormalisedTransaction],
    *,
    include_descriptions: bool = True,
    extra_instructions: tuple[str, ...] | None = None,
) -> dict:
    """Build the prompt the LLM-fallback path sends.

    Descriptions are included by default here — the whole point of
    the fallback is that the LLM gets richer context than the
    candidate-only payload provides.
    """
    base = build_llm_payload(
        result,
        include_descriptions=include_descriptions,
        include_near_misses=len(result.rejected_or_near_miss_sets),
        extra_instructions=extra_instructions,
        validate=False,
    )
    base["task"] = (
        "The deterministic pipeline did not pick a clear salary "
        "candidate. Review the full credit transaction list and the "
        "ambiguous candidates, and return the salary set explicitly."
    )
    base["instructions"] = list(_FALLBACK_INSTRUCTIONS) + (
        list(extra_instructions) if extra_instructions else []
    )
    base["transactions"] = [
        {
            "transaction_id": t.id,
            "date": t.date.isoformat(),
            "description": t.raw_description if include_descriptions else "",
            "amount": float(t.amount),
            "direction": t.direction,
        }
        for t in transactions
        if t.direction == "credit"
    ]
    base["mode"] = "llm_fallback_full_transaction_review"
    return base


def summarise_candidates(candidates: Sequence[SalaryCandidateSet]) -> list[dict]:
    """Compact one-line-per-candidate summary used inside structured
    logs (so the prompt itself isn't echoed at INFO level)."""
    return [
        {
            "id": c.candidate_set_id,
            "type": c.candidate_type,
            "confidence": round(float(c.confidence), 4),
            "band": c.confidence_band,
            "n_transactions": len(c.transactions),
        }
        for c in candidates
    ]


__all__ = ["build_fallback_payload", "summarise_candidates"]
