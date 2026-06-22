"""Candidate set output models — Design §4.4, §6.2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .score import ScoreBreakdown
from .transaction import NormalisedTransaction


# Allowed candidate types — Requirement §6.3
CANDIDATE_TYPES = frozenset(
    {
        "probable_salary",
        "possible_salary",
        "recurring_income_not_salary_candidate",
        "bonus_or_extra_pay_candidate",
        "expense_reimbursement_candidate",
        "final_pay_candidate",
        "unknown_recurring_credit",
    }
)

CONFIDENCE_BANDS = ("high", "medium_high", "medium", "low", "very_low")


@dataclass(frozen=True)
class DetectedPattern:
    frequency: str = "unknown"
    payday_model: str = "unknown"
    payment_channel: str = "unknown_credit"
    amount_model: str = "unknown"
    possible_employer_tokens: tuple[str, ...] = ()
    coverage: str = ""


@dataclass(frozen=True)
class CandidateTransaction:
    transaction_id: str
    date: str
    description: str
    amount: float
    direction: str
    role_in_set: str = "main_salary"


@dataclass(frozen=True)
class SalaryCandidateSet:
    candidate_set_id: str
    candidate_type: str
    transactions: tuple[CandidateTransaction, ...]
    score_breakdown: ScoreBreakdown
    detected_pattern: DetectedPattern
    reasoning: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    confidence: float = 0.0
    confidence_band: str = "very_low"
    llm_review_recommendation: str = "skip"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.candidate_type not in CANDIDATE_TYPES:
            raise ValueError(
                f"candidate_type {self.candidate_type!r} not in {sorted(CANDIDATE_TYPES)}"
            )
        if self.confidence_band not in CONFIDENCE_BANDS:
            raise ValueError(
                f"confidence_band {self.confidence_band!r} not in {CONFIDENCE_BANDS}"
            )


def candidate_transaction_from(nt: NormalisedTransaction, role: str = "main_salary") -> CandidateTransaction:
    return CandidateTransaction(
        transaction_id=nt.id,
        date=nt.date.isoformat(),
        description=nt.raw_description,
        amount=float(nt.amount),
        direction=nt.direction,
        role_in_set=role,
    )
