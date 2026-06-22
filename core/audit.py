"""AuditBuilder — Phase 14.

Threaded through the detector pipeline. Accumulates pipeline-step
records, per-transaction features, per-candidate details, and LLM call
records. When the caller sets ``include_audit=True``, the builder
snapshots into an immutable ``AuditRecord``; otherwise it stays in
``noop`` mode and pays minimal allocation cost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.client import LLMCallRecord
from ..models.audit import (
    AuditRecord,
    CandidateAuditDetail,
    PipelineStepRecord,
)
from ..models.candidate import SalaryCandidateSet
from ..models.transaction import TransactionFeatures


@dataclass
class AuditBuilder:
    enabled: bool = False
    analysis_id: str = ""
    config_versions: dict[str, str] = field(default_factory=dict)
    effective_kwargs: dict[str, Any] = field(default_factory=dict)
    _steps: list[PipelineStepRecord] = field(default_factory=list)
    _features: tuple[TransactionFeatures, ...] = ()
    _candidate_details: list[CandidateAuditDetail] = field(default_factory=list)
    _llm_calls: list[LLMCallRecord] = field(default_factory=list)
    _timings: dict[str, float] = field(default_factory=dict)
    _step_starts: dict[str, float] = field(default_factory=dict)

    # --- Public API -----------------------------------------------------

    def start_step(self, name: str) -> None:
        if not self.enabled:
            return
        self._step_starts[name] = time.perf_counter()

    def record_step(
        self,
        name: str,
        *,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        start = self._step_starts.pop(name, time.perf_counter())
        duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
        self._timings[name] = duration_ms
        self._steps.append(
            PipelineStepRecord(
                name=name,
                duration_ms=duration_ms,
                input_summary=dict(input_summary or {}),
                output_summary=dict(output_summary or {}),
                extra=dict(extra or {}),
            )
        )

    def record_features(self, features: tuple[TransactionFeatures, ...]) -> None:
        if not self.enabled:
            return
        self._features = features

    def record_candidates(self, candidates: tuple[SalaryCandidateSet, ...]) -> None:
        if not self.enabled:
            return
        for c in candidates:
            self._candidate_details.append(_candidate_detail_from(c))

    def record_llm_call(self, call: LLMCallRecord) -> None:
        if not self.enabled:
            return
        self._llm_calls.append(call)

    def snapshot(self) -> AuditRecord | None:
        if not self.enabled:
            return None
        return AuditRecord(
            analysis_id=self.analysis_id,
            timings_ms=dict(self._timings),
            pipeline_steps=tuple(self._steps),
            per_transaction_features=self._features,
            candidate_details=tuple(self._candidate_details),
            llm_calls=tuple(self._llm_calls),
            config_versions=dict(self.config_versions),
            effective_kwargs=dict(self.effective_kwargs),
        )


def _candidate_detail_from(candidate: SalaryCandidateSet) -> CandidateAuditDetail:
    breakdown = candidate.score_breakdown
    return CandidateAuditDetail(
        candidate_set_id=candidate.candidate_set_id,
        candidate_type=candidate.candidate_type,
        confidence=float(candidate.confidence),
        confidence_band=candidate.confidence_band,
        transaction_ids=tuple(t.transaction_id for t in candidate.transactions),
        score_breakdown={
            "date_pattern_score": float(breakdown.date_pattern_score),
            "description_similarity_score": float(breakdown.description_similarity_score),
            "amount_consistency_score": float(breakdown.amount_consistency_score),
            "payment_channel_score": float(breakdown.payment_channel_score),
            "coverage_score": float(breakdown.coverage_score),
            "context_bonus": float(breakdown.context_bonus),
            "anchor_bonus": float(breakdown.anchor_bonus),
            "salary_band_modifier": float(breakdown.salary_band_modifier),
            "negative_signal_penalty": float(breakdown.negative_signal_penalty),
            "final_score": float(breakdown.final_score),
        },
        reasoning=tuple(candidate.reasoning),
        risks=tuple(candidate.risks),
        metadata=dict(candidate.metadata),
    )


__all__ = ["AuditBuilder"]
