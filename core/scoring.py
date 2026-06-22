"""Phase 8 — final scoring (Design §11).

Builds ``ScoreBreakdown`` from the per-cluster signals produced by
Phases 3, 4, 5, 6 and 7:

```
raw_score =
    W_date    * date_pattern_score          # Phase 6 RecurrenceFit
  + W_desc    * description_similarity_score # average pairwise Jaccard
  + W_amount  * amount_consistency_score     # Phase 6 AmountFit × FR-017 band modifier
  + W_channel * payment_channel_score        # Phase 3 channel × payroll-hint blend
  + W_coverage * coverage_score              # RecurrenceFit.coverage
  + W_context * context_bonus                # small bump when employer + hint align

final_score = clamp(raw_score + anchor_bonus + negative_signal_penalty, 0, 1)
```

Penalty is **clamped** at ``max_negative_signal_penalty`` (default -0.20),
per FR-013. Anchor bonus is **clamped** at ``max_anchor_bonus`` (default
+0.15) per Design §11.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from ..models.candidate import SalaryCandidateSet
from ..models.hints import AppliedHints, SalaryHints
from ..models.score import ScoreBreakdown
from ..models.transaction import NormalisedTransaction, TransactionFeatures
from .amount_models import AmountFit, classify_amount_pattern
from .clustering import description_similarity, token_sequence_match_score
from .config import AppConfig
from .hints import band_modifier, compute_salary_band
from .recurrence_models import RecurrenceFit, fit_best_recurrence_model
from .calendar_service import WorkingDayCalendar


# Confidence bands — Design §6.4 / Requirement §6.4.
_BAND_THRESHOLDS = (
    ("high", 0.85),
    ("medium_high", 0.70),
    ("medium", 0.55),
    ("low", 0.40),
    ("very_low", 0.0),
)


def confidence_band(score: float) -> str:
    for band, cutoff in _BAND_THRESHOLDS:
        if score >= cutoff:
            return band
    return "very_low"


# --- Sub-scores ----------------------------------------------------------


def description_score(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
) -> float:
    """Average pairwise description similarity within the cluster.

    Adds a small bonus when the same employer-token survives across every
    record (the strongest description signal we have).
    """
    if len(cluster) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            total += description_similarity(
                cluster[i].description_tokens, cluster[j].description_tokens
            )
            pairs += 1
    jaccard = total / pairs if pairs else 0.0

    # Employer-token alignment bonus.
    token_sets = [
        set(features_by_id[t.id].possible_employer_tokens)
        for t in cluster
        if t.id in features_by_id and features_by_id[t.id].possible_employer_tokens
    ]
    if len(token_sets) >= 2:
        common = set.intersection(*token_sets)
        if common:
            jaccard = min(1.0, jaccard + 0.05)
    return round(jaccard, 6)


def sequence_consistency_score(cluster: Sequence[NormalisedTransaction]) -> float:
    """Average pairwise LCS-ratio across all pairs in the cluster.

    Returns 1.0 when all tokens appear in the same order; lower values
    indicate that token order varies across transactions.
    """
    if len(cluster) < 2:
        return 1.0
    pairs = 0
    total = 0.0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            total += token_sequence_match_score(
                cluster[i].description_tokens, cluster[j].description_tokens
            )
            pairs += 1
    return round(total / pairs if pairs else 1.0, 6)


def _build_gap_summary(recurrence, frequency: str, transactions_range: Optional[dict]) -> dict:
    """Build a structured gap summary for candidate metadata (Enhancement B1).

    Identifies which calendar months were expected but had no matching credit,
    and incorporates an optional caller-supplied ``transactions_range`` to
    override the denominator.
    """
    from .recurrence_models import PaydayModel as _PM

    missing_months = [
        f"{d.year}-{d.month:02d}"
        for d in recurrence.monthly_details
        if d.actual is None and d.predicted is not None
    ]

    # Date range from the monthly_details span.
    if recurrence.monthly_details:
        first_detail = recurrence.monthly_details[0]
        last_detail = recurrence.monthly_details[-1]
        range_start = f"{first_detail.year}-{first_detail.month:02d}"
        range_end = f"{last_detail.year}-{last_detail.month:02d}"
    else:
        range_start = ""
        range_end = ""

    # Caller-supplied range overrides the denominator reasoning label.
    supplied_start = None
    supplied_end = None
    if transactions_range:
        supplied_start = transactions_range.get("start") or transactions_range.get("from")
        supplied_end = transactions_range.get("end") or transactions_range.get("to")

    expected = recurrence.expected_count
    matched = recurrence.matched_count
    missing_count = expected - matched

    parts: list[str] = [
        f"{matched} of {expected} expected periods matched."
    ]
    if missing_months:
        month_labels = ", ".join(missing_months[:5])
        if len(missing_months) > 5:
            month_labels += f" (and {len(missing_months) - 5} more)"
        parts.append(f"Missing: {month_labels}.")
    if supplied_start or supplied_end:
        parts.append(
            f"Payload transactions_range: {supplied_start or '?'} → {supplied_end or '?'}."
        )

    return {
        "expected_periods": expected,
        "matched_periods": matched,
        "missing_months": missing_months,
        "missing_count": missing_count,
        "salary_frequency": frequency,
        "transactions_range_start": range_start,
        "transactions_range_end": range_end,
        "reasoning": " ".join(parts),
    }


def channel_score(
    features: Sequence[TransactionFeatures],
) -> float:
    """Payment-channel score per Design §11.2."""
    if not features:
        return 0.30
    channels = [f.payment_channel for f in features]
    has_payroll_hint = any(
        any(tok in {"PAYROLL", "SALARY", "WAGES", "WAGE", "STAFF PAY", "MONTHLY PAY"} for tok in f.salary_hint_tokens)
        for f in features
    )

    # Use the dominant channel for the band.
    from collections import Counter

    dominant, _ = Counter(channels).most_common(1)[0]
    if dominant == "bacs_or_direct_credit":
        return 0.95 if has_payroll_hint else 0.65
    if dominant == "faster_payment":
        return 0.80 if has_payroll_hint else 0.55
    if dominant == "chaps":
        return 0.55 if has_payroll_hint else 0.40
    if dominant in {"cheque", "cash_or_counter_credit"}:
        return 0.45 if has_payroll_hint else 0.35
    if dominant in {"internal_transfer", "standing_order_credit"}:
        return 0.20
    if dominant == "international_credit":
        return 0.60 if has_payroll_hint else 0.40
    # unknown_credit
    return 0.40 if has_payroll_hint else 0.30


def context_bonus(
    features: Sequence[TransactionFeatures],
) -> float:
    """Small bonus when the cluster has aligned positive signals."""
    if not features:
        return 0.0
    has_employer = sum(1 for f in features if f.possible_employer_tokens) / len(features)
    has_strong_hint = sum(
        1
        for f in features
        if any(t in {"PAYROLL", "SALARY", "WAGES", "WAGE"} for t in f.salary_hint_tokens)
    ) / len(features)
    bonus = 0.0
    if has_employer >= 0.8:
        bonus += 0.02
    if has_strong_hint >= 0.5:
        bonus += 0.03
    return round(min(0.05, bonus), 4)


def negative_signal_penalty(
    features: Sequence[TransactionFeatures],
    *,
    max_penalty: float,
    category_weights: Optional[dict[str, float]] = None,
) -> float:
    """Aggregate negative-signal penalty, clamped at ``max_penalty`` (negative)."""
    if not features:
        return 0.0
    # Penalty per transaction = severity proportional to fraction of credits
    # carrying a negative signal.
    flagged = sum(1 for f in features if f.negative_tokens)
    if flagged == 0:
        return 0.0
    share = flagged / len(features)
    # Linear from 0 (no signal) to max_penalty (all flagged).
    penalty = max_penalty * share
    return round(max(max_penalty, min(0.0, penalty)), 6)


def anchor_bonus(
    applied_hints: Optional[AppliedHints],
    cluster_txn_ids: set[str],
    *,
    per_anchor: float,
    max_bonus: float,
) -> float:
    if not applied_hints or not applied_hints.anchors:
        return 0.0
    matched = sum(1 for a in applied_hints.anchors if a.transaction_id in cluster_txn_ids)
    if matched == 0:
        return 0.0
    return round(min(max_bonus, per_anchor * matched), 6)


# --- Composite -----------------------------------------------------------


@dataclass(frozen=True)
class ScoringInputs:
    recurrence: RecurrenceFit
    amount: AmountFit
    band_modifier_value: float = 1.0
    channel: float = 0.0
    description: float = 0.0
    context: float = 0.0
    coverage: float = 0.0
    negative_penalty: float = 0.0
    anchor: float = 0.0


def compute_score(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
    *,
    calendar: WorkingDayCalendar,
    config: AppConfig,
    applied_hints: Optional[AppliedHints] = None,
    salary_hints: Optional[SalaryHints] = None,
) -> tuple[ScoreBreakdown, ScoringInputs]:
    """Compute a full ScoreBreakdown for a candidate cluster.

    Returns the breakdown plus the intermediate ``ScoringInputs`` so the
    Phase 8 candidate-type classifier and reasoning builder can quote the
    same evidence.
    """
    weights = {
        k: float(v) for k, v in config.data.get("weights", {}).items()
    }
    penalties = config.penalties()
    bonuses = config.bonuses()

    recurrence = fit_best_recurrence_model(cluster, calendar=calendar)
    amount_fit = classify_amount_pattern(cluster, config=config)
    cluster_features = [features_by_id[t.id] for t in cluster if t.id in features_by_id]

    band = compute_salary_band(salary_hints) if salary_hints else None
    if band is None and applied_hints and applied_hints.effective_monthly is not None:
        # Reconstruct band from applied hints record (the band edges are
        # already there from `apply_hints`).
        from .hints import SalaryBand
        band = SalaryBand(
            effective_monthly=applied_hints.effective_monthly,
            lower=applied_hints.band_lower or applied_hints.effective_monthly,
            upper=applied_hints.band_upper or applied_hints.effective_monthly,
            tolerance_pct=0.0,
        )
    mod = band_modifier(amount_fit.median_amount, band) if band else 1.0

    date_pattern_score = recurrence.fit_score
    description_score_value = description_score(cluster, features_by_id)
    channel_score_value = channel_score(cluster_features)
    coverage_score_value = recurrence.coverage
    context_bonus_value = context_bonus(cluster_features)
    amount_score_adjusted = amount_fit.score * mod

    negative_pen = negative_signal_penalty(
        cluster_features,
        max_penalty=float(penalties.get("max_negative_signal_penalty", -0.20)),
    )
    cluster_ids = {t.id for t in cluster}
    anchor_b = anchor_bonus(
        applied_hints,
        cluster_ids,
        per_anchor=float(bonuses.get("per_anchor", 0.05)),
        max_bonus=float(bonuses.get("max_anchor_bonus", 0.15)),
    )

    raw_score = (
        weights.get("date_pattern_score", 0.30) * date_pattern_score
        + weights.get("description_similarity_score", 0.25) * description_score_value
        + weights.get("amount_consistency_score", 0.20) * amount_score_adjusted
        + weights.get("payment_channel_score", 0.10) * channel_score_value
        + weights.get("coverage_score", 0.10) * coverage_score_value
        + weights.get("context_bonus", 0.05) * context_bonus_value
    )
    final = max(0.0, min(1.0, raw_score + anchor_b + negative_pen))

    breakdown = ScoreBreakdown(
        date_pattern_score=round(date_pattern_score, 4),
        description_similarity_score=round(description_score_value, 4),
        amount_consistency_score=round(amount_score_adjusted, 4),
        payment_channel_score=round(channel_score_value, 4),
        coverage_score=round(coverage_score_value, 4),
        context_bonus=round(context_bonus_value, 4),
        negative_signal_penalty=round(negative_pen, 4),
        anchor_bonus=round(anchor_b, 4),
        salary_band_modifier=round(mod, 4),
        final_score=round(final, 4),
    )
    inputs = ScoringInputs(
        recurrence=recurrence,
        amount=amount_fit,
        band_modifier_value=mod,
        channel=channel_score_value,
        description=description_score_value,
        context=context_bonus_value,
        coverage=coverage_score_value,
        negative_penalty=negative_pen,
        anchor=anchor_b,
    )
    return breakdown, inputs


def score_candidate(
    candidate: SalaryCandidateSet,
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
    *,
    calendar: WorkingDayCalendar,
    config: AppConfig,
    applied_hints: Optional[AppliedHints] = None,
    salary_hints: Optional[SalaryHints] = None,
    transactions_range: Optional[dict] = None,
) -> SalaryCandidateSet:
    """Compose scoring + classification + reasoning into a final candidate.

    Takes a placeholder candidate from Phase 5/7 (which carries
    ``candidate_type = "unknown_recurring_credit"`` and an empty
    breakdown) and returns a fully scored ``SalaryCandidateSet`` with:

    - filled ``ScoreBreakdown``,
    - reclassified ``candidate_type``,
    - rendered ``reasoning`` and ``risks``,
    - confidence + band derived from ``final_score``.
    """
    from .candidate_type import classify_candidate_type
    from .reasoning import build_reasoning, build_risks

    breakdown, inputs = compute_score(
        cluster,
        features_by_id,
        calendar=calendar,
        config=config,
        applied_hints=applied_hints,
        salary_hints=salary_hints,
    )
    cluster_features = [features_by_id[t.id] for t in cluster if t.id in features_by_id]
    classification = classify_candidate_type(breakdown, inputs, cluster_features)
    positive = build_reasoning(cluster, features_by_id, breakdown, inputs)
    risks = build_risks(cluster, features_by_id, breakdown, inputs)

    frequency = _frequency_for(inputs.recurrence)
    pattern = replace(
        candidate.detected_pattern,
        frequency=frequency,
        payday_model=inputs.recurrence.model.value,
        amount_model=inputs.amount.model.value,
    )

    # Enhancement C: detect sequence ordering mismatch (same tokens, different order).
    seq_score = sequence_consistency_score(cluster)
    jaccard = breakdown.description_similarity_score
    sequence_extra_risks: list[str] = []
    if jaccard >= 0.6 and seq_score < 0.7 and len(cluster) >= 3:
        sequence_extra_risks.append(
            "Description token order varies across transactions; "
            "slight confidence reduction applied (sequence consistency "
            f"{seq_score:.0%})."
        )

    merged_reasoning = tuple(list(candidate.reasoning) + list(positive))
    merged_risks = tuple(list(risks) + sequence_extra_risks)

    # Enhancement B1: attach structured gap summary to metadata.
    gap_summary = _build_gap_summary(inputs.recurrence, frequency, transactions_range)

    # Phase 13: candidates produced by the amount + date-only fallback
    # never claim probable_salary. Clamp the final score at the fallback
    # cap and keep the unknown_recurring_credit type — Phase 8's
    # classification cascade would otherwise re-promote them into
    # probable_salary on a clean fit.
    final_score = breakdown.final_score
    final_type = classification.candidate_type
    if candidate.metadata.get("source") == "amount_date_fallback":
        cap = float(candidate.metadata.get("fallback_confidence_cap", 0.55))
        final_score = min(final_score, cap)
        final_type = "unknown_recurring_credit"
        breakdown = replace(breakdown, final_score=final_score)

    return SalaryCandidateSet(
        candidate_set_id=candidate.candidate_set_id,
        candidate_type=final_type,
        transactions=candidate.transactions,
        score_breakdown=breakdown,
        detected_pattern=pattern,
        reasoning=merged_reasoning,
        risks=merged_risks,
        confidence=final_score,
        confidence_band=confidence_band(final_score),
        llm_review_recommendation=(
            "send_to_llm" if final_score >= 0.55 else "skip"
        ),
        metadata={
            **candidate.metadata,
            "classification_reason_key": classification.reason_key,
            "gap_summary": gap_summary,
            "sequence_consistency": round(seq_score, 4),
        },
    )


def _frequency_for(recurrence) -> str:
    from .recurrence_models import PaydayModel as _PM

    if recurrence.model == _PM.WEEKLY_SAME_WEEKDAY:
        return "weekly"
    if recurrence.model == _PM.FORTNIGHTLY_SAME_WEEKDAY:
        return "fortnightly"
    if recurrence.model == _PM.FOUR_WEEKLY:
        return "four_weekly"
    if recurrence.model == _PM.IRREGULAR_SALARY_LIKE:
        return "irregular"
    return "monthly"


__all__ = [
    "ScoringInputs",
    "anchor_bonus",
    "channel_score",
    "compute_score",
    "confidence_band",
    "context_bonus",
    "description_score",
    "negative_signal_penalty",
    "score_candidate",
    "sequence_consistency_score",
]
