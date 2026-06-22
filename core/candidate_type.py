"""Candidate-type classifier — Design §12.

```text
probable_salary
possible_salary
recurring_income_not_salary_candidate
bonus_or_extra_pay_candidate
expense_reimbursement_candidate
final_pay_candidate
unknown_recurring_credit
```

The classifier sees the full scoring evidence so its decisions are
auditable in the reasoning step that follows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models.score import ScoreBreakdown
from ..models.transaction import TransactionFeatures
from .amount_models import AmountFit, AmountModel
from .recurrence_models import PaydayModel, RecurrenceFit
from .scoring import ScoringInputs


# Negative-signal categories that should pull us toward
# recurring_income_not_salary_candidate.
_NON_SALARY_RECURRING_CATEGORIES = frozenset(
    {
        "government_benefits",
        "tax_refund",
        "pension",
        "investment_income",
        "own_transfers",
    }
)


# Strong negative-token names (what extract emits as the matched token text).
_NON_SALARY_RECURRING_TOKENS = frozenset(
    {
        "DWP",
        "UNIVERSAL CREDIT",
        "CHILD BENEFIT",
        "PIP",
        "HMRC",
        "TAX REFUND",
        "PENSION",
        "RETIREMENT",
        "DIVIDEND",
        "INTEREST",
        "OWN ACCOUNT",
        "SAVINGS",
        "TRANSFER FROM SAVINGS",
        "TRANSFER FROM OWN",
        "RENT",
        "INSURANCE",
        "MARKETPLACE",
    }
)


@dataclass(frozen=True)
class CandidateClassification:
    candidate_type: str
    reason_key: str  # short code used by reasoning builder


def classify_candidate_type(
    score: ScoreBreakdown,
    inputs: ScoringInputs,
    features: Sequence[TransactionFeatures],
) -> CandidateClassification:
    final = score.final_score
    amount: AmountFit = inputs.amount
    recurrence: RecurrenceFit = inputs.recurrence

    # 1. Strong non-salary recurring marker — wins over everything else,
    # regardless of the final score, because we never want to claim a
    # pension or DWP credit is salary.
    if _has_non_salary_recurring_marker(features):
        return CandidateClassification(
            candidate_type="recurring_income_not_salary_candidate",
            reason_key="non_salary_recurring_marker",
        )

    # 2. Single-credit outlier branches.
    if amount.model == AmountModel.ONE_OFF_FINAL_PAY_OUTLIER:
        return CandidateClassification(
            candidate_type="final_pay_candidate",
            reason_key="final_pay_outlier",
        )
    if amount.model == AmountModel.BONUS_OR_EXTRA_PAY_OUTLIER:
        return CandidateClassification(
            candidate_type="bonus_or_extra_pay_candidate",
            reason_key="bonus_outlier",
        )

    # 3. Expense-reimbursement pattern — same-employer cluster but TOO_VARIABLE
    # amounts with smaller median than what we'd expect for salary. We use a
    # loose heuristic: TOO_VARIABLE + low channel score + employer tokens
    # present. Phase 11 will calibrate further.
    if (
        amount.model == AmountModel.TOO_VARIABLE
        and inputs.channel <= 0.50
        and _has_employer_tokens(features)
    ):
        return CandidateClassification(
            candidate_type="expense_reimbursement_candidate",
            reason_key="expense_reimbursement_pattern",
        )

    # 4. Probable / possible salary by score band.
    if final >= 0.70:
        return CandidateClassification(
            candidate_type="probable_salary",
            reason_key="high_confidence_salary",
        )
    if final >= 0.55:
        return CandidateClassification(
            candidate_type="possible_salary",
            reason_key="medium_confidence_salary",
        )

    # 5. Default — recurring credit we can't classify confidently.
    return CandidateClassification(
        candidate_type="unknown_recurring_credit",
        reason_key="below_salary_threshold",
    )


# --- Helpers --------------------------------------------------------------


def _has_non_salary_recurring_marker(features: Sequence[TransactionFeatures]) -> bool:
    if not features:
        return False
    flagged = 0
    for f in features:
        if any(tok in _NON_SALARY_RECURRING_TOKENS for tok in f.negative_tokens):
            flagged += 1
    # Treat as non-salary-recurring when the majority of cluster credits
    # carry the marker — a one-off pension lookup in a stable salary
    # series doesn't taint the whole cluster.
    return flagged / len(features) >= 0.5


def _has_employer_tokens(features: Sequence[TransactionFeatures]) -> bool:
    return any(bool(f.possible_employer_tokens) for f in features)


__all__ = [
    "CandidateClassification",
    "classify_candidate_type",
]
