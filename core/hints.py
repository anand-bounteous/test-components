"""Optional-hint application — FR-016, FR-017, Design §6.4 + §9.5 + §11.5.

Two responsibilities:

1. **Payslip anchor matching** — for each ``PayslipHint`` find credits in
   the hint's month where **both**

   - ``abs(txn.amount) == hint.amount`` (Decimal exact equality), AND
   - ``int(txn.amount) == int(hint.amount)``

   hold. Either failing means no match. The two checks deliberately ride
   the same vehicle so callers get a contract that catches both whole-pound
   typos and decimal-position typos (Design §6.4).

   When two or more credits match the same hint we don't pick one — both
   become anchors tagged ``ambiguous=True`` and disambiguation falls to
   the LLM-review stage.

2. **Approx-salary band modifier** — given ``SalaryHints`` and a cluster's
   median amount, return a factor in ``[0.5, 1.0]`` that the Phase 8
   scoring step multiplies into ``amount_consistency_score``. Inside the
   band the modifier is 1.0; outside, it decays linearly to a floor of
   0.5 (never a hard exclusion, per FR-017 acceptance criterion #3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from ..models.hints import AnchorMatch, AppliedHints, PayslipHint, SalaryHints
from ..models.transaction import NormalisedTransaction


# --- Payslip anchor matching --------------------------------------------


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def apply_payslip_hints(
    credits: Iterable[NormalisedTransaction],
    hints: Iterable[PayslipHint],
) -> AppliedHints:
    """Match each payslip hint against the credit list.

    Returns an ``AppliedHints`` whose ``anchors`` lists every confirmed
    match (with the ``ambiguous`` flag set when more than one credit
    qualifies in a single hint month), and whose ``unmatched_hint_ids``
    records hints that found nothing.
    """
    credit_list = list(credits)
    by_month: dict[str, list[NormalisedTransaction]] = {}
    for c in credit_list:
        by_month.setdefault(_month_key(c.date), []).append(c)

    anchors: list[AnchorMatch] = []
    unmatched: list[str] = []

    for hint in hints:
        candidates = by_month.get(hint.month, [])
        matches = [c for c in candidates if _anchor_match(c, hint)]
        if not matches:
            unmatched.append(hint.hint_id)
            continue
        ambiguous = len(matches) > 1
        for c in matches:
            anchors.append(
                AnchorMatch(
                    hint_id=hint.hint_id,
                    transaction_id=c.id,
                    month=hint.month,
                    amount=hint.amount,
                    ambiguous=ambiguous,
                )
            )

    band_lower, band_upper, effective_monthly = None, None, None
    return AppliedHints(
        anchors=tuple(anchors),
        unmatched_hint_ids=tuple(unmatched),
        effective_monthly=effective_monthly,
        band_lower=band_lower,
        band_upper=band_upper,
    )


def _anchor_match(txn: NormalisedTransaction, hint: PayslipHint) -> bool:
    """Both conditions must hold — never just one. See FR-016."""
    abs_match = _decimal_equal(abs(txn.amount), hint.amount)
    int_match = int(abs(txn.amount)) == int(hint.amount)
    return abs_match and int_match


def _decimal_equal(a: Decimal, b: Decimal) -> bool:
    """Numeric equality that tolerates equivalent representations such as
    ``2450.32`` vs ``2450.320`` (both equal). Uses ``Decimal.compare`` so
    we don't get bitten by trailing-zero scale differences.
    """
    return a == b


# --- Approx-salary band -------------------------------------------------


@dataclass(frozen=True)
class SalaryBand:
    effective_monthly: Decimal
    lower: Decimal
    upper: Decimal
    tolerance_pct: float

    def contains(self, amount: Decimal) -> bool:
        return self.lower <= amount <= self.upper


def compute_salary_band(hints: Optional[SalaryHints]) -> Optional[SalaryBand]:
    """Resolve the effective monthly value and the ±tolerance band.

    Returns ``None`` when no hint is provided (or both fields are empty).
    """
    if hints is None or hints.is_empty():
        return None
    effective = hints.effective_monthly
    assert effective is not None
    tol = Decimal(str(hints.tolerance_pct))
    lower = effective * (Decimal("1") - tol)
    upper = effective * (Decimal("1") + tol)
    return SalaryBand(
        effective_monthly=effective,
        lower=lower,
        upper=upper,
        tolerance_pct=hints.tolerance_pct,
    )


def band_modifier(cluster_median: Decimal, band: Optional[SalaryBand]) -> float:
    """Return the amount-score modifier for FR-017.

    Rules:

    - ``None`` band (no hint) → ``1.0`` no-op.
    - Median inside the band → ``1.0``.
    - Median outside the band → linear decay from 1.0 (at the band edge)
      down to 0.5 (at one full ``effective_monthly`` away). Never < 0.5.
    """
    if band is None:
        return 1.0
    if band.contains(cluster_median):
        return 1.0
    # Distance OUTSIDE the band (positive number).
    if cluster_median < band.lower:
        distance = band.lower - cluster_median
    else:
        distance = cluster_median - band.upper
    relative = float(distance) / float(band.effective_monthly)
    # Decay 1.0 → 0.5 across one full effective_monthly outside the band.
    modifier = 1.0 - 0.5 * relative
    return max(0.5, min(1.0, modifier))


# --- Top-level orchestration -------------------------------------------


def apply_hints(
    credits: Iterable[NormalisedTransaction],
    *,
    payslip_hints: Iterable[PayslipHint] = (),
    salary_hints: Optional[SalaryHints] = None,
) -> AppliedHints:
    """Single entry point Phase 9's detector will call.

    Currently returns only anchor information plus the resolved band
    boundaries; the band is consumed later (Phase 8) when amount scoring
    happens, but the band edges are recorded here for audit.
    """
    base = apply_payslip_hints(credits, payslip_hints)
    band = compute_salary_band(salary_hints)
    if band is None:
        return base
    return AppliedHints(
        anchors=base.anchors,
        unmatched_hint_ids=base.unmatched_hint_ids,
        effective_monthly=band.effective_monthly,
        band_lower=band.lower,
        band_upper=band.upper,
    )


def anchor_txn_ids(applied: AppliedHints) -> dict[str, str]:
    """Build the ``{transaction_id: hint_id}`` map ``attach_anchor_flags``
    expects (from ``feature_extraction.attach_anchor_flags``)."""
    out: dict[str, str] = {}
    for anchor in applied.anchors:
        out[anchor.transaction_id] = anchor.hint_id
    return out


def ambiguous_anchor_ids(applied: AppliedHints) -> set[str]:
    return {a.transaction_id for a in applied.anchors if a.ambiguous}
