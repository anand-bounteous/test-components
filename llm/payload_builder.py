"""LLM payload builder — Design §14.

Takes a ``SalaryDetectionResult`` and produces the compact JSON-serialisable
payload the downstream LLM reviewer expects. Key choices (per the Phase 5
notes and Requirement §8.5):

- **Send the minimum** — score breakdown + reasoning + risks, not the raw
  detection metadata. The audit trail stays in `.plan/reviews/phase9-outputs/`.
- **Redact descriptions by default** — the caller can opt in to including
  raw text via ``include_descriptions=True``; account numbers, sort codes
  and reference fields are never sent.
- **Cap near-misses** — only the top few near-misses go to the LLM; the
  rest stay in the detector output.

The builder always produces schema-valid output (asserted by
``validate_llm_payload`` from ``llm.schemas``).
"""
from __future__ import annotations

import json
from typing import Optional

from ..models.candidate import SalaryCandidateSet
from ..models.output import SalaryDetectionResult
from .schemas import validate_llm_payload

# Hard instructions handed to the LLM. Pinned in tests.
_DEFAULT_INSTRUCTIONS: tuple[str, ...] = (
    "Do not rely on amount alone.",
    "Prefer recurring salary-like credits over one-off high-value credits.",
    "Consider UK payroll patterns including Bacs, Faster Payments, "
    "working-day shifts, final pay and employer changes.",
    "Return selected candidate set IDs with relative confidence and reasoning.",
    "Inputs already filtered to credit transactions only — do not "
    "request debit transactions.",
    "Some candidates carry an `induced_pattern_signature` of the form "
    "`{prefix}*{suffix}` — this is a character-level pattern auto-"
    "discovered from the descriptions. Treat it as evidence of a stable "
    "employer / stream identity, but **secondary** to clear "
    "`possible_employer_tokens`.",
    "Candidates flagged with `amount_date_only: true` were detected "
    "purely from monthly cadence and amount band — the description gave "
    "no employer signal. Treat as `unknown_recurring_credit` unless "
    "context strongly suggests salary.",
    "Description-aligned candidates with explicit payroll hints "
    "(`PAYROLL`, `SALARY`, `WAGES`) and stable employer tokens outrank "
    "pattern-induced and amount/date-only candidates.",
)

# Default cap on near-misses included in the payload.
_DEFAULT_NEAR_MISS_LIMIT = 3


def build_llm_payload(
    result: SalaryDetectionResult,
    *,
    include_descriptions: bool = False,
    include_near_misses: int = _DEFAULT_NEAR_MISS_LIMIT,
    extra_instructions: Optional[tuple[str, ...]] = None,
    validate: bool = True,
) -> dict:
    """Build the LLM-review payload dict for ``result``.

    Args:
        result: Output of ``detect_salary_candidates``.
        include_descriptions: Default ``False`` — drop raw descriptions to
            avoid leaking employer / counterparty text. Set to ``True``
            when the caller has confirmed downstream consumers may see
            the description body.
        include_near_misses: Number of near-misses to include (capped at
            len(result.rejected_or_near_miss_sets)).
        extra_instructions: Extra instructions to append after the default
            block — useful when the caller needs to add a domain rule
            (e.g. "do not classify pensions as salary").
        validate: When ``True`` (default), the payload is validated
            against ``llm_payload.schema.json`` before being returned.
    """
    instructions = list(_DEFAULT_INSTRUCTIONS)
    if extra_instructions:
        instructions.extend(extra_instructions)

    payload: dict = {
        "task": (
            "Review candidate salary transaction sets and select the most "
            "likely salary set(s)."
        ),
        "jurisdiction": result.jurisdiction,
        "country_region": result.country_region,
        "analysis_id": result.analysis_id,
        "candidate_sets": [
            _candidate_dict(c, include_descriptions=include_descriptions)
            for c in result.candidate_sets
        ],
        "near_misses": [
            _candidate_dict(c, include_descriptions=include_descriptions)
            for c in result.rejected_or_near_miss_sets[: max(0, include_near_misses)]
        ],
        "applied_hints": _applied_hints_dict(result),
        "instructions": instructions,
        "metadata": {
            "config_versions": result.metadata.get("config_versions", {}),
            "input_summary": {
                "credit_count": result.input_summary.credit_count,
                "date_range_start": result.input_summary.date_range_start,
                "date_range_end": result.input_summary.date_range_end,
            },
        },
    }
    if validate:
        validate_llm_payload(payload)
    return payload


def serialise_payload(payload: dict) -> str:
    """Deterministic JSON string for byte-stable comparisons."""
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Per-candidate helpers
# ---------------------------------------------------------------------------


def _candidate_dict(candidate: SalaryCandidateSet, *, include_descriptions: bool) -> dict:
    out: dict = {
        "id": candidate.candidate_set_id,
        "candidate_type": candidate.candidate_type,
        "confidence": float(candidate.confidence),
        "confidence_band": candidate.confidence_band,
        "transactions": [
            _txn_compact(t, include_descriptions=include_descriptions)
            for t in candidate.transactions
        ],
        "score_breakdown": _score_breakdown_dict(candidate),
        "detected_pattern": {
            "frequency": candidate.detected_pattern.frequency,
            "payday_model": candidate.detected_pattern.payday_model,
            "amount_model": candidate.detected_pattern.amount_model,
            "payment_channel": candidate.detected_pattern.payment_channel,
            "possible_employer_tokens": list(candidate.detected_pattern.possible_employer_tokens),
            "coverage": candidate.detected_pattern.coverage,
        },
        "reasoning": list(candidate.reasoning),
        "risks": list(candidate.risks),
    }
    merged_sources = candidate.metadata.get("merged_sources")
    if merged_sources:
        out["merged_sources"] = list(merged_sources)
    if "anchor_in_cluster" in candidate.metadata:
        out["anchor_in_cluster"] = bool(candidate.metadata["anchor_in_cluster"])
    near_miss_reason = candidate.metadata.get("near_miss_reason")
    if near_miss_reason in {"cap_overflow", "below_threshold"}:
        out["near_miss_reason"] = near_miss_reason
    induced_signature = candidate.metadata.get("induced_pattern_signature")
    if induced_signature:
        out["induced_pattern_signature"] = induced_signature
    if candidate.metadata.get("amount_date_only"):
        out["amount_date_only"] = True
    transitions = candidate.metadata.get("possible_employment_transition")
    if transitions:
        out["possible_employment_transition"] = list(transitions)
    return out


def _txn_compact(t, *, include_descriptions: bool) -> dict:
    rec = {
        "transaction_id": t.transaction_id,
        "date": t.date,
        "amount": float(t.amount),
    }
    if include_descriptions:
        rec["description"] = t.description
    return rec


def _score_breakdown_dict(candidate: SalaryCandidateSet) -> dict:
    b = candidate.score_breakdown
    return {
        "date": float(b.date_pattern_score),
        "description": float(b.description_similarity_score),
        "amount": float(b.amount_consistency_score),
        "channel": float(b.payment_channel_score),
        "coverage": float(b.coverage_score),
        "context_bonus": float(b.context_bonus),
        "anchor_bonus": float(b.anchor_bonus),
        "salary_band_modifier": float(b.salary_band_modifier),
        "negative_penalty": float(b.negative_signal_penalty),
        "final": float(b.final_score),
    }


def _applied_hints_dict(result: SalaryDetectionResult) -> dict:
    applied = result.applied_hints
    return {
        "anchor_count": len(applied.anchors),
        "anchor_transaction_ids": [a.transaction_id for a in applied.anchors],
        "unmatched_hint_ids": list(applied.unmatched_hint_ids),
        "effective_monthly": (
            float(applied.effective_monthly)
            if applied.effective_monthly is not None
            else None
        ),
        "band_lower": (
            float(applied.band_lower) if applied.band_lower is not None else None
        ),
        "band_upper": (
            float(applied.band_upper) if applied.band_upper is not None else None
        ),
    }


__all__ = [
    "build_llm_payload",
    "serialise_payload",
]
