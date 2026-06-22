"""Transaction domain models — Design §4.1, §4.2, §4.3."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Optional

Direction = Literal["credit", "debit"]


@dataclass(frozen=True)
class RawTransaction:
    """Caller-supplied transaction record. Design §4.1."""

    id: str
    date: str
    description: str
    amount: Decimal
    direction: Optional[Direction] = None
    balance: Optional[Decimal] = None
    reference: Optional[str] = None
    counterparty_name: Optional[str] = None
    transaction_code: Optional[str] = None
    source_bank: Optional[str] = None
    currency: Optional[str] = None
    account_id: Optional[str] = None
    raw_category: Optional[str] = None
    booking_date: Optional[str] = None
    value_date: Optional[str] = None


@dataclass(frozen=True)
class NormalisedTransaction:
    """Canonical credit transaction used by every downstream phase. Design §4.2."""

    id: str
    date: date
    amount: Decimal
    direction: Direction
    raw_description: str
    normalised_description: str
    description_tokens: tuple[str, ...]
    skeleton: str
    source_bank: Optional[str] = None
    transaction_code: Optional[str] = None
    counterparty_name: Optional[str] = None
    currency: str = "GBP"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionFeatures:
    """Per-transaction feature record. Design §4.3.

    Populated in Phase 3; the model is declared in Phase 1 so cross-phase
    imports work without circular references.
    """

    transaction_id: str
    payment_channel: str = "unknown_credit"
    payment_channel_confidence: float = 0.0
    payment_channel_matched_tokens: tuple[str, ...] = ()
    salary_hint_tokens: tuple[str, ...] = ()
    negative_tokens: tuple[str, ...] = ()
    possible_employer_tokens: tuple[str, ...] = ()
    day_of_month: int = 0
    weekday: int = 0
    is_month_end: bool = False
    is_last_working_day: bool = False
    is_first_working_day: bool = False
    is_bank_holiday_adjacent: bool = False
    amount_bucket: str = ""
    is_anchor: bool = False
    anchor_hint_id: Optional[str] = None
    ambiguous_anchor: bool = False
