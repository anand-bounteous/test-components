"""Amount + date-only fallback path — Phase 13.

The earlier candidate-generation paths (skeleton-grouping in Phase 5,
similarity-graph clustering in Phase 7) rely on shared tokens or shared
character patterns in the description. When a salary stream uses
descriptions that look like a random UUID per credit (no employer
name, no payroll hint, no shared substring), every prior path emits
nothing — even though the credits have a textbook monthly cadence on
matching amounts.

This module bridges the gap. It takes credits **not** in any earlier
candidate and clusters them by:

1. Amount band (median ±10 %).
2. Periodic cadence — the existing ``fit_best_recurrence_model`` from
   Phase 6.

Each surviving bucket emits a single low-confidence candidate with
``metadata.source = "amount_date_fallback"`` and a confidence cap of
0.55 (because the absence of description signal is fundamental
uncertainty). The LLM payload still surfaces these so the reviewer can
confirm or reject.

Triggered fixtures (Phase 13): ``realistic_10_uuid_clean_cadence``.
Not triggered (and correctly): ``realistic_11_uuid_no_signal`` — random
amounts + random dates fail the cadence check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

from ..models.candidate import (
    CandidateTransaction,
    DetectedPattern,
    SalaryCandidateSet,
    candidate_transaction_from,
)
from ..models.hints import AppliedHints
from ..models.score import ScoreBreakdown
from ..models.transaction import NormalisedTransaction
from .amount_features import cluster_stats
from .calendar_service import WorkingDayCalendar
from .config import AppConfig
from .recurrence_models import (
    PaydayModel,
    RecurrenceFit,
    fit_best_recurrence_model,
)

# Defaults (Phase 13 spec) — kept as constants so callers can swap them.
_BAND_TOLERANCE_PCT = 0.10        # amount-band half-width
_MIN_CADENCE_FIT = 0.60           # recurrence fit_score floor
_MIN_CREDITS_PER_BUCKET = 4       # too few = noise
_CONFIDENCE_CAP = 0.55            # max confidence allowed for this path


@dataclass
class AmountDateFallbackDetector:
    """Final pass clustering by amount band + monthly cadence."""

    config: AppConfig
    _band_tolerance: float = field(init=False, default=_BAND_TOLERANCE_PCT)
    _min_cadence_fit: float = field(init=False, default=_MIN_CADENCE_FIT)
    _min_credits: int = field(init=False, default=_MIN_CREDITS_PER_BUCKET)
    _confidence_cap: float = field(init=False, default=_CONFIDENCE_CAP)

    def __post_init__(self) -> None:
        amount_cfg = self.config.amount_config()
        # Reuse the moderate amount tolerance from the existing config.
        self._band_tolerance = float(
            amount_cfg.get("moderate_pct_tolerance", _BAND_TOLERANCE_PCT)
        )

    # --- Public API -----------------------------------------------------

    def emit_candidates(
        self,
        unaccounted: Sequence[NormalisedTransaction],
        calendar: WorkingDayCalendar,
        *,
        applied_hints: Optional[AppliedHints] = None,
        region: str = "EnglandAndWales",
    ) -> tuple[SalaryCandidateSet, ...]:
        if len(unaccounted) < self._min_credits:
            return ()

        buckets = self._bucket_by_amount(unaccounted)
        emitted: list[SalaryCandidateSet] = []
        anchor_ids: set[str] = set()
        if applied_hints is not None:
            anchor_ids = {a.transaction_id for a in applied_hints.anchors}

        for idx, members in enumerate(buckets):
            if len(members) < self._min_credits:
                continue
            fit = fit_best_recurrence_model(members, calendar=calendar, region=region)
            if fit.fit_score < self._min_cadence_fit:
                continue
            emitted.append(
                self._build_candidate(
                    idx,
                    members,
                    fit,
                    anchor_present=any(t.id in anchor_ids for t in members),
                )
            )
        return tuple(emitted)

    # --- Internal -------------------------------------------------------

    def _bucket_by_amount(
        self,
        credits: Sequence[NormalisedTransaction],
    ) -> list[list[NormalisedTransaction]]:
        """Greedy single-pass bucketing — credits land in the first
        existing bucket whose median is within ``±band_tolerance``;
        otherwise they start a new bucket."""
        buckets: list[list[NormalisedTransaction]] = []
        # Sort by amount so adjacent values land together deterministically.
        sorted_credits = sorted(credits, key=lambda t: t.amount)
        for txn in sorted_credits:
            placed = False
            for bucket in buckets:
                median_amount = sorted(b.amount for b in bucket)[len(bucket) // 2]
                if median_amount == 0:
                    continue
                relative = abs(txn.amount - median_amount) / median_amount
                if relative <= Decimal(str(self._band_tolerance)):
                    bucket.append(txn)
                    placed = True
                    break
            if not placed:
                buckets.append([txn])
        # Sort each bucket by date for downstream determinism.
        return [sorted(b, key=lambda t: t.date) for b in buckets]

    def _build_candidate(
        self,
        idx: int,
        members: Sequence[NormalisedTransaction],
        fit: RecurrenceFit,
        *,
        anchor_present: bool,
    ) -> SalaryCandidateSet:
        sorted_members = sorted(members, key=lambda t: t.date)
        txns: tuple[CandidateTransaction, ...] = tuple(
            candidate_transaction_from(t) for t in sorted_members
        )
        stats = cluster_stats([t.amount for t in sorted_members])
        confidence = round(
            min(
                self._confidence_cap,
                0.40 + 0.10 * fit.fit_score + (0.05 if anchor_present else 0.0),
            ),
            4,
        )
        pattern = DetectedPattern(
            frequency=_frequency_for(fit.model),
            payday_model=fit.model.value,
            payment_channel="unknown_credit",
            amount_model="stable" if stats.coefficient_of_variation <= 0.02 else "stable_with_minor_variation",
            possible_employer_tokens=(),
            coverage=f"{fit.matched_count}_of_{fit.expected_count}",
        )
        return SalaryCandidateSet(
            candidate_set_id=f"amount_date_{idx:03d}",
            candidate_type="unknown_recurring_credit",
            transactions=txns,
            score_breakdown=ScoreBreakdown(
                final_score=confidence,
                date_pattern_score=round(fit.fit_score, 4),
                amount_consistency_score=1.0
                if stats.coefficient_of_variation <= 0.02
                else round(max(0.0, 1.0 - 5 * stats.coefficient_of_variation), 4),
                coverage_score=round(fit.coverage, 4),
            ),
            detected_pattern=pattern,
            reasoning=(
                "Detected purely on amount + cadence — description provides "
                "no employer signal. The reviewer should confirm the "
                "context before treating this as salary.",
                f"Cadence: {fit.model.value} (fit {fit.fit_score:.2f}, "
                f"{fit.matched_count} of {fit.expected_count} expected periods).",
                f"Amounts cluster within ±{self._band_tolerance * 100:.0f}% "
                f"of median £{stats.median_amount}.",
            ),
            risks=(
                "No payroll keyword, employer token, or canonical channel "
                "marker was found in the descriptions.",
            ),
            confidence=confidence,
            confidence_band=_band_for_score(confidence),
            llm_review_recommendation="send_to_llm",
            metadata={
                "source": "amount_date_fallback",
                "amount_date_only": True,
                "anchor_in_cluster": anchor_present,
                "fallback_confidence_cap": self._confidence_cap,
            },
        )


# --- Helpers --------------------------------------------------------------


def _frequency_for(model: PaydayModel) -> str:
    if model == PaydayModel.WEEKLY_SAME_WEEKDAY:
        return "weekly"
    if model == PaydayModel.FORTNIGHTLY_SAME_WEEKDAY:
        return "fortnightly"
    if model == PaydayModel.FOUR_WEEKLY:
        return "four_weekly"
    if model == PaydayModel.IRREGULAR_SALARY_LIKE:
        return "irregular"
    return "monthly"


def _band_for_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium_high"
    if score >= 0.55:
        return "medium"
    if score >= 0.40:
        return "low"
    return "very_low"


__all__ = ["AmountDateFallbackDetector"]
