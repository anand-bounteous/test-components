"""Reasoning builder — Design §13.

Renders the positive-reason and risk templates against the actual
evidence in a cluster. Output strings are short, audit-quotable
sentences that mention the concrete facts (employer tokens, missing
months, payday model name, etc.) rather than narrating internal state.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..models.score import ScoreBreakdown
from ..models.transaction import NormalisedTransaction, TransactionFeatures
from .amount_models import AmountFit, AmountModel
from .recurrence_models import PaydayModel, RecurrenceFit
from .scoring import ScoringInputs


_PAYDAY_MODEL_LABELS = {
    PaydayModel.FIXED_DAY_N: "fixed day-of-month",
    PaydayModel.PREVIOUS_WORKING_DAY_OF_N: "previous working day of a fixed date",
    PaydayModel.NEXT_WORKING_DAY_OF_N: "next working day of a fixed date",
    PaydayModel.NEAREST_WORKING_DAY_OF_N: "nearest working day of a fixed date",
    PaydayModel.LAST_CALENDAR_DAY: "last calendar day of month",
    PaydayModel.LAST_WORKING_DAY: "last working day of month",
    PaydayModel.FIRST_WORKING_DAY: "first working day of month",
    PaydayModel.WEEKLY_SAME_WEEKDAY: "weekly on the same weekday",
    PaydayModel.FORTNIGHTLY_SAME_WEEKDAY: "fortnightly on the same weekday",
    PaydayModel.FOUR_WEEKLY: "four-weekly",
    PaydayModel.IRREGULAR_SALARY_LIKE: "irregular but recurring",
}

_AMOUNT_MODEL_LABELS = {
    AmountModel.STABLE: "stable across the cluster",
    AmountModel.STABLE_WITH_MINOR_VARIATION: "stable with minor month-on-month variation",
    AmountModel.STABLE_WITH_STEP_CHANGE: "stable with one step change",
    AmountModel.VARIABLE_BUT_RECURRING: "variable but recurring (commission-like)",
    AmountModel.ONE_OFF_FINAL_PAY_OUTLIER: "stable with a single tail outlier (possible final pay)",
    AmountModel.BONUS_OR_EXTRA_PAY_OUTLIER: "stable with one inside-cluster bonus-like outlier",
    AmountModel.TOO_VARIABLE: "highly variable",
}


def build_reasoning(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
    score: ScoreBreakdown,
    inputs: ScoringInputs,
) -> tuple[str, ...]:
    """Positive-reason list — what makes this set look salary-like."""
    out: list[str] = []
    recurrence: RecurrenceFit = inputs.recurrence
    amount: AmountFit = inputs.amount

    # Date / recurrence reasoning.
    if recurrence.model != PaydayModel.IRREGULAR_SALARY_LIKE:
        label = _PAYDAY_MODEL_LABELS[recurrence.model]
        out.append(
            f"Credits fit the {label} model "
            f"({recurrence.matched_count} of {recurrence.expected_count} expected periods, "
            f"average deviation {recurrence.mean_deviation_days:.1f} days)."
        )
    elif recurrence.matched_count >= 3:
        out.append(
            f"{recurrence.matched_count} recurring credits — no clean payday model fits "
            f"but the cadence is salary-like."
        )

    # Description / employer reasoning.
    employer_tokens = _common_employer_tokens(cluster, features_by_id)
    if employer_tokens:
        out.append(
            "Descriptions share stable employer-like tokens: "
            + ", ".join(employer_tokens)
            + "."
        )
    hint_tokens = _dominant_salary_hints(cluster, features_by_id)
    if hint_tokens:
        out.append(
            "Descriptions contain payroll hints: " + ", ".join(hint_tokens) + "."
        )

    # Amount reasoning.
    out.append(
        f"Amounts are {_AMOUNT_MODEL_LABELS[amount.model]} "
        f"(median £{amount.median_amount}, CV {amount.coefficient_of_variation:.2f})."
    )
    if amount.model == AmountModel.STABLE_WITH_STEP_CHANGE and amount.segments:
        seg_text = " → ".join(f"£{seg.median}" for seg in amount.segments)
        out.append(f"Amount step change observed: {seg_text}.")

    # Channel reasoning.
    channel_counts = Counter(
        features_by_id[t.id].payment_channel
        for t in cluster
        if t.id in features_by_id
    )
    if channel_counts:
        dominant_channel, _ = channel_counts.most_common(1)[0]
        if dominant_channel == "bacs_or_direct_credit":
            out.append("Payment channel is Bacs/Direct Credit, commonly used for UK payroll.")
        elif dominant_channel == "faster_payment":
            out.append("Payment channel is Faster Payments.")
        elif dominant_channel == "chaps":
            out.append("Payment channel is CHAPS.")

    # Anchor evidence.
    if score.anchor_bonus > 0:
        out.append(
            f"Payslip anchor confirmed for {round(score.anchor_bonus / 0.05)} month(s) "
            "— amount and integer-part match exactly."
        )

    return tuple(out)


def build_risks(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
    score: ScoreBreakdown,
    inputs: ScoringInputs,
) -> tuple[str, ...]:
    """Risk list — what could make this set NOT salary."""
    out: list[str] = []
    recurrence: RecurrenceFit = inputs.recurrence

    # Missing months.
    missing = recurrence.expected_count - recurrence.matched_count
    if missing > 0:
        out.append(
            f"{missing} expected payday period(s) were missing or unmatched."
        )

    # Negative signals — cite the categories.
    neg_seen = set()
    for f in (features_by_id[t.id] for t in cluster if t.id in features_by_id):
        for tok in f.negative_tokens:
            neg_seen.add(tok)
    if neg_seen:
        out.append(
            "Negative-signal tokens detected in the cluster: "
            + ", ".join(sorted(neg_seen))
            + "."
        )

    # Bacs caveat — only when channel is BACS and we labelled the candidate
    # as salary-ish.
    channel_counts = Counter(
        features_by_id[t.id].payment_channel
        for t in cluster
        if t.id in features_by_id
    )
    dominant_channel = channel_counts.most_common(1)[0][0] if channel_counts else "unknown_credit"
    if dominant_channel == "bacs_or_direct_credit" and score.final_score >= 0.55:
        out.append(
            "Bacs/Direct Credit is also used for pensions, expenses, refunds and "
            "dividends, so channel alone is not conclusive."
        )

    # Amount variability risk.
    if inputs.amount.model == AmountModel.TOO_VARIABLE:
        out.append(
            "Amount varies significantly, which may indicate commission, overtime, "
            "unpaid leave or non-salary income."
        )

    # Salary-band modifier visibility.
    if score.salary_band_modifier < 1.0:
        out.append(
            f"Cluster median falls outside the supplied approx-salary band "
            f"(band-modifier {score.salary_band_modifier:.2f})."
        )

    return tuple(out)


# --- Helpers --------------------------------------------------------------


def _common_employer_tokens(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
) -> tuple[str, ...]:
    token_sets = [
        set(features_by_id[t.id].possible_employer_tokens)
        for t in cluster
        if t.id in features_by_id and features_by_id[t.id].possible_employer_tokens
    ]
    if not token_sets:
        return ()
    common = set.intersection(*token_sets)
    return tuple(sorted(common))


def _dominant_salary_hints(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for t in cluster:
        f = features_by_id.get(t.id)
        if not f:
            continue
        for hint in f.salary_hint_tokens:
            if hint in {"PAYROLL", "SALARY", "WAGES", "WAGE", "STAFF PAY", "MONTHLY PAY", "STAFF"}:
                counts[hint] += 1
    if not counts:
        return ()
    threshold = len(cluster) * 0.5
    return tuple(sorted(t for t, c in counts.items() if c >= threshold))


__all__ = [
    "build_reasoning",
    "build_risks",
]
