"""Hint inputs — FR-016, FR-017, Design §4.6.

Three small immutable types:

- ``PayslipHint`` — one (month, exact_amount) anchor.
- ``SalaryHints`` — optional approx-monthly / approx-yearly band.
- ``HintInputs`` — wrapper passed alongside the transaction list.

An ``AppliedHints`` record describes what actually matched at runtime; it is
returned by the hint application step in Phase 4 and embedded in the detector
output metadata for audit (Design §6.4).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _coerce_decimal(value, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except Exception as exc:
            raise ValueError(f"{field_name} is not a valid decimal: {value!r}") from exc
    raise TypeError(f"{field_name} must be Decimal/int/float/str, got {type(value).__name__}")


@dataclass(frozen=True)
class PayslipHint:
    """A single payslip anchor.

    Both checks at match time are independent (FR-016): ``abs(txn.amount) ==
    hint.amount`` AND ``int(txn.amount) == int(hint.amount)``. Either failing
    means no anchor.
    """

    month: str
    amount: Decimal
    hint_id: str = ""

    def __post_init__(self):
        if not isinstance(self.month, str) or not _MONTH_RE.match(self.month):
            raise ValueError(f"month must be YYYY-MM, got {self.month!r}")
        object.__setattr__(self, "amount", _coerce_decimal(self.amount, "amount"))
        if not self.hint_id:
            object.__setattr__(self, "hint_id", f"hint_{self.month}_{self.amount}")


@dataclass(frozen=True)
class SalaryHints:
    """Approximate monthly/yearly salary band — FR-017.

    Soft modifier only. ``tolerance_pct`` defaults to 0.20 (±20 %).
    """

    approx_monthly: Optional[Decimal] = None
    approx_yearly: Optional[Decimal] = None
    tolerance_pct: float = 0.20

    def __post_init__(self):
        if self.approx_monthly is not None:
            object.__setattr__(self, "approx_monthly", _coerce_decimal(self.approx_monthly, "approx_monthly"))
        if self.approx_yearly is not None:
            object.__setattr__(self, "approx_yearly", _coerce_decimal(self.approx_yearly, "approx_yearly"))
        if not 0.0 <= self.tolerance_pct <= 1.0:
            raise ValueError(f"tolerance_pct must be in [0, 1], got {self.tolerance_pct}")

    @property
    def effective_monthly(self) -> Optional[Decimal]:
        """Effective monthly value, derived from yearly when monthly is absent."""
        if self.approx_monthly is not None:
            return self.approx_monthly
        if self.approx_yearly is not None:
            return self.approx_yearly / Decimal(12)
        return None

    def is_empty(self) -> bool:
        return self.approx_monthly is None and self.approx_yearly is None


@dataclass(frozen=True)
class HintInputs:
    """Caller-supplied optional hints — Design §4.6."""

    payslip_hints: tuple[PayslipHint, ...] = ()
    salary_hints: Optional[SalaryHints] = None

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "HintInputs":
        if not payload:
            return cls()
        ps = tuple(
            PayslipHint(month=p["month"], amount=p["amount"], hint_id=p.get("hint_id", ""))
            for p in payload.get("payslip_hints", [])
        )
        sh_dict = payload.get("salary_hints")
        sh = (
            SalaryHints(
                approx_monthly=sh_dict.get("approx_monthly"),
                approx_yearly=sh_dict.get("approx_yearly"),
                tolerance_pct=sh_dict.get("tolerance_pct", 0.20),
            )
            if sh_dict is not None
            else None
        )
        return cls(payslip_hints=ps, salary_hints=sh)


@dataclass(frozen=True)
class AnchorMatch:
    """One anchor match emitted by the hint application step."""

    hint_id: str
    transaction_id: str
    month: str
    amount: Decimal
    ambiguous: bool = False


@dataclass(frozen=True)
class AppliedHints:
    """Audit record returned by hint application — Design §6.4."""

    anchors: tuple[AnchorMatch, ...] = ()
    unmatched_hint_ids: tuple[str, ...] = ()
    effective_monthly: Optional[Decimal] = None
    band_lower: Optional[Decimal] = None
    band_upper: Optional[Decimal] = None
