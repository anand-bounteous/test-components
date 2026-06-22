"""Normalised-description grouping path — FR-018, Design §10.5.

The simpler of the two candidate-generation paths. Runs **before** the
similarity-graph clustering of Phase 7. Steps:

1. Bucket normalised credits by their ``skeleton`` field (already computed
   in Phase 1 by ``normalisation.skeleton_of``).
2. Score each group's plausibility from four signals:
   - **date plausibility** — fraction of credits aligned on a monthly
     cadence (one credit per month, gaps in the 26–32 day window);
   - **amount plausibility** — derived from coefficient of variation,
     then multiplied by the FR-017 salary-band modifier when a band
     is supplied;
   - **anchor presence** — ``1.0`` when any FR-016 anchor lands inside
     the group, else ``0.0``;
   - **1 − negative-signal share** — penalises groups dominated by
     pension / DWP / own-transfer / refund / etc.
3. Emit groups whose plausibility ≥ ``threshold`` (default 0.45) as
   ``SalaryCandidateSet`` records carrying ``metadata.source =
   "skeleton_grouping"`` plus the skeleton key and a per-axis plausibility
   breakdown so downstream phases can audit the decision.

Output candidates carry ``candidate_type = "unknown_recurring_credit"``
intentionally — Phase 8 scoring is the canonical classifier. The
skeleton path's job is recall (find the plausible set) not classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from ..models.candidate import (
    CandidateTransaction,
    DetectedPattern,
    SalaryCandidateSet,
    candidate_transaction_from,
)
from ..models.hints import AppliedHints
from ..models.score import ScoreBreakdown
from ..models.transaction import NormalisedTransaction, TransactionFeatures
from .amount_features import cluster_stats
from .config import AppConfig
from .hints import SalaryBand, band_modifier, compute_salary_band

# Months are recognised by (year, month) tuples to handle Dec→Jan rollovers
# without arithmetic on raw day counts.

# Date plausibility tuning — wider than Phase 6 will be, because this is
# the recall-oriented first pass.
_GOOD_MONTHLY_GAP_DAYS = (26, 32)


@dataclass(frozen=True)
class PlausibilityScore:
    """Per-axis breakdown so reasoning can quote the evidence."""

    date_plausibility: float
    amount_plausibility: float
    anchor_presence: float
    negative_signal_share: float
    final: float


@dataclass
class SkeletonGroupingDetector:
    """Stateful detector — built once per detector run."""

    config: AppConfig
    _min_group_size: int = field(init=False, default=2)
    _threshold: float = field(init=False, default=0.45)

    def __post_init__(self) -> None:
        cg = self.config.candidate_generation()
        self._min_group_size = int(cg.get("min_skeleton_group_size", 2))
        self._threshold = float(cg.get("skeleton_plausibility_threshold", 0.45))

    # --- Public API ------------------------------------------------------

    def group_by_skeleton(
        self,
        credits: Iterable[NormalisedTransaction],
    ) -> dict[str, list[NormalisedTransaction]]:
        groups: dict[str, list[NormalisedTransaction]] = {}
        for c in credits:
            groups.setdefault(c.skeleton, []).append(c)
        return groups

    def score_group_plausibility(
        self,
        group: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
        *,
        band: Optional[SalaryBand] = None,
        applied_hints: Optional[AppliedHints] = None,
    ) -> PlausibilityScore:
        date_plaus = self._date_plausibility(group)
        amount_plaus = self._amount_plausibility(group, band)
        anchor_presence = self._anchor_presence(group, applied_hints)
        neg_share = self._negative_signal_share(group, features_by_id)

        base = (
            0.40 * date_plaus
            + 0.30 * amount_plaus
            + 0.15 * anchor_presence
            + 0.15 * (1.0 - neg_share)
        )

        # Anchor presence raises the floor — a confirmed payslip anchor
        # inside the group should never let the final dip below medium
        # (~0.55) just because the rest of the signal is noisy.
        final = base
        if anchor_presence >= 1.0:
            final = max(final, 0.55)

        return PlausibilityScore(
            date_plausibility=round(date_plaus, 4),
            amount_plausibility=round(amount_plaus, 4),
            anchor_presence=anchor_presence,
            negative_signal_share=round(neg_share, 4),
            final=round(final, 4),
        )

    def emit_plausible_sets(
        self,
        credits: Iterable[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
        *,
        applied_hints: Optional[AppliedHints] = None,
        threshold: Optional[float] = None,
    ) -> tuple[SalaryCandidateSet, ...]:
        """Run the full skeleton path end-to-end."""
        cutoff = self._threshold if threshold is None else float(threshold)
        band = self._band_from_applied_hints(applied_hints)

        groups = self.group_by_skeleton(credits)
        emitted: list[SalaryCandidateSet] = []
        for idx, (skeleton, members) in enumerate(_stable_sorted(groups)):
            if len(members) < self._min_group_size:
                continue
            score = self.score_group_plausibility(
                members,
                features_by_id,
                band=band,
                applied_hints=applied_hints,
            )
            if score.final < cutoff:
                continue
            emitted.append(self._build_candidate(idx, skeleton, members, score, features_by_id))
        return tuple(emitted)

    # --- Internal scoring helpers ---------------------------------------

    def _date_plausibility(self, group: Sequence[NormalisedTransaction]) -> float:
        if len(group) < 2:
            return 0.0
        sorted_group = sorted(group, key=lambda t: t.date)
        # Coverage: distinct months covered out of the calendar span the
        # group occupies.
        months = {(t.date.year, t.date.month) for t in sorted_group}
        first = sorted_group[0].date
        last = sorted_group[-1].date
        expected_months = max(
            1,
            (last.year - first.year) * 12 + (last.month - first.month) + 1,
        )
        coverage = min(1.0, len(months) / expected_months)

        # Gap regularity: count adjacent gaps in the 26–32 day window.
        good_gaps = 0
        total_gaps = 0
        for prev, nxt in zip(sorted_group, sorted_group[1:]):
            gap = (nxt.date - prev.date).days
            if gap == 0:
                # Two credits on the same day — neither helps nor hurts.
                continue
            total_gaps += 1
            if _GOOD_MONTHLY_GAP_DAYS[0] <= gap <= _GOOD_MONTHLY_GAP_DAYS[1]:
                good_gaps += 1
        gap_regularity = (good_gaps / total_gaps) if total_gaps else 0.0

        return round(0.5 * coverage + 0.5 * gap_regularity, 6)

    def _amount_plausibility(
        self,
        group: Sequence[NormalisedTransaction],
        band: Optional[SalaryBand],
    ) -> float:
        amounts = [t.amount for t in group]
        if not amounts:
            return 0.0
        stats = cluster_stats(amounts)
        cv = stats.coefficient_of_variation
        # Decay CV-based score linearly: cv=0 → 1.0; cv=0.20 → 0.0; cap
        # at 0.0 for very-noisy series. Salary bonuses and commission
        # series will land in the 0.4–0.7 range.
        if cv <= 0.02:
            cv_score = 1.0
        elif cv >= 0.20:
            cv_score = 0.0
        else:
            cv_score = 1.0 - (cv - 0.02) / 0.18

        if band is not None:
            cv_score *= band_modifier(stats.median_amount, band)

        return round(cv_score, 6)

    @staticmethod
    def _anchor_presence(
        group: Sequence[NormalisedTransaction],
        applied_hints: Optional[AppliedHints],
    ) -> float:
        if not applied_hints or not applied_hints.anchors:
            return 0.0
        anchor_ids = {a.transaction_id for a in applied_hints.anchors}
        for t in group:
            if t.id in anchor_ids:
                return 1.0
        return 0.0

    @staticmethod
    def _negative_signal_share(
        group: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
    ) -> float:
        if not group:
            return 0.0
        flagged = 0
        for t in group:
            f = features_by_id.get(t.id)
            if f is not None and f.negative_tokens:
                flagged += 1
        return flagged / len(group)

    @staticmethod
    def _band_from_applied_hints(applied: Optional[AppliedHints]) -> Optional[SalaryBand]:
        if applied is None or applied.effective_monthly is None:
            return None
        # ``apply_hints`` already computed the band; reconstruct the
        # SalaryBand dataclass here so we can call ``contains``.
        return SalaryBand(
            effective_monthly=applied.effective_monthly,
            lower=applied.band_lower or Decimal("0"),
            upper=applied.band_upper or Decimal("0"),
            tolerance_pct=0.0,  # used only inside compute_salary_band; not needed downstream
        )

    @staticmethod
    def _build_candidate(
        idx: int,
        skeleton: str,
        members: Sequence[NormalisedTransaction],
        score: PlausibilityScore,
        features_by_id: dict[str, TransactionFeatures],
    ) -> SalaryCandidateSet:
        sorted_members = sorted(members, key=lambda t: t.date)
        txns: tuple[CandidateTransaction, ...] = tuple(
            candidate_transaction_from(t) for t in sorted_members
        )
        # Employer tokens are the intersection of per-record employer
        # tokens — Phase 3 already extracted them.
        token_sets = [
            set(features_by_id[t.id].possible_employer_tokens)
            for t in sorted_members
            if t.id in features_by_id
        ]
        if token_sets:
            common = sorted(set.intersection(*token_sets)) if all(token_sets) else []
        else:
            common = []

        # Channel: pick the dominant channel across the group.
        channels: dict[str, int] = {}
        for t in sorted_members:
            f = features_by_id.get(t.id)
            if f is not None:
                channels[f.payment_channel] = channels.get(f.payment_channel, 0) + 1
        dominant_channel = max(channels.items(), key=lambda kv: kv[1])[0] if channels else "unknown_credit"

        pattern = DetectedPattern(
            frequency="monthly_candidate",
            payday_model="unknown_pending_phase6",
            payment_channel=dominant_channel,
            amount_model="unknown_pending_phase6",
            possible_employer_tokens=tuple(common),
            coverage=f"{len(sorted_members)}_credits_across_{_distinct_months(sorted_members)}_months",
        )

        return SalaryCandidateSet(
            candidate_set_id=f"skeleton_{idx:03d}",
            candidate_type="unknown_recurring_credit",
            transactions=txns,
            score_breakdown=ScoreBreakdown(),
            detected_pattern=pattern,
            reasoning=(
                f"Skeleton-grouping plausibility {score.final:.2f}: "
                f"date={score.date_plausibility:.2f}, amount={score.amount_plausibility:.2f}, "
                f"anchor={score.anchor_presence:.2f}, negative_share={score.negative_signal_share:.2f}.",
            ),
            risks=(),
            confidence=score.final,
            confidence_band=_band_for_score(score.final),
            llm_review_recommendation="send_to_llm" if score.final >= 0.55 else "skip",
            metadata={
                "source": "skeleton_grouping",
                "skeleton_key": skeleton,
                "anchor_in_cluster": score.anchor_presence >= 1.0,
                "plausibility": {
                    "date": score.date_plausibility,
                    "amount": score.amount_plausibility,
                    "anchor": score.anchor_presence,
                    "negative_share": score.negative_signal_share,
                    "final": score.final,
                },
            },
        )


# --- Module-level helpers ------------------------------------------------


def _stable_sorted(groups: dict[str, list[NormalisedTransaction]]):
    """Deterministic group order — sorted by skeleton string."""
    return sorted(groups.items(), key=lambda kv: kv[0])


def _distinct_months(members: Sequence[NormalisedTransaction]) -> int:
    return len({(t.date.year, t.date.month) for t in members})


def _band_for_score(final: float) -> str:
    # Phase 5 confidence bands intentionally borrow the Phase 8 cuts so the
    # detector output is uniform once Phase 9 wires everything together.
    if final >= 0.85:
        return "high"
    if final >= 0.70:
        return "medium_high"
    if final >= 0.55:
        return "medium"
    if final >= 0.40:
        return "low"
    return "very_low"
