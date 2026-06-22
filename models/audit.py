"""Audit dataclasses — Phase 14.

Returned on ``SalaryDetectionResult.audit`` when the caller sets
``include_audit=True``. Captures every pipeline step, every per-
transaction feature, every per-candidate detail, every LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.client import LLMCallRecord
from .transaction import TransactionFeatures


@dataclass(frozen=True)
class PipelineStepRecord:
    """One pipeline step's inputs / outputs / duration."""

    name: str                         # e.g. "validate", "normalise"
    duration_ms: float
    input_summary: dict[str, Any]     # e.g. {"row_count": 100}
    output_summary: dict[str, Any]    # e.g. {"normalised_count": 99, "warnings": 1}
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAuditDetail:
    """Full per-candidate detail for the audit trail."""

    candidate_set_id: str
    candidate_type: str
    confidence: float
    confidence_band: str
    transaction_ids: tuple[str, ...]
    score_breakdown: dict[str, float]
    reasoning: tuple[str, ...]
    risks: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LLMReviewResult:
    """Parsed structured response from the LLM-fallback path."""

    selected_candidate_set_ids: tuple[str, ...]
    rejected_candidate_set_ids: tuple[str, ...]
    confidence: float
    reasoning: tuple[str, ...]
    additional_review_needed: bool
    call_record: LLMCallRecord


@dataclass(frozen=True)
class AuditRecord:
    """Full audit attached to ``SalaryDetectionResult.audit`` when
    ``include_audit=True``. Always returned as an immutable snapshot."""

    analysis_id: str
    timings_ms: dict[str, float]
    pipeline_steps: tuple[PipelineStepRecord, ...]
    per_transaction_features: tuple[TransactionFeatures, ...]
    candidate_details: tuple[CandidateAuditDetail, ...]
    llm_calls: tuple[LLMCallRecord, ...]
    config_versions: dict[str, str]
    effective_kwargs: dict[str, Any]


__all__ = [
    "AuditRecord",
    "CandidateAuditDetail",
    "LLMReviewResult",
    "PipelineStepRecord",
]
