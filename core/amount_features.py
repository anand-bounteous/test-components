"""Amount feature extraction — Design §6.3, §9.

This module produces two distinct things:

1. **Per-transaction features**: a discrete amount bucket and an integer
   part. The integer part is what FR-016 needs in Phase 4 to match a
   payslip hint deterministically.
2. **Cluster statistics**: median / mean / MAD / coefficient of variation
   — these are used in Phase 6 to classify amount-regime models. Kept
   here so Phase 3 can already wire the data shape into
   ``TransactionFeatures``.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean, median
from typing import Iterable, Sequence


# Bucket thresholds in £; chosen to spread typical UK net-pay amounts plus
# small reimbursements. The boundaries are inclusive on the lower side.
_BUCKET_BOUNDARIES_GBP = (
    (Decimal("0"),      "under_100"),
    (Decimal("100"),    "100_to_500"),
    (Decimal("500"),    "500_to_1500"),
    (Decimal("1500"),   "1500_to_3000"),
    (Decimal("3000"),   "3000_to_6000"),
    (Decimal("6000"),   "over_6000"),
)


def bucket_for(amount: Decimal) -> str:
    """Return a coarse amount bucket suitable for pre-grouping."""
    if amount is None:
        return "unknown"
    a = abs(amount)
    label = "under_100"
    for boundary, bucket_label in _BUCKET_BOUNDARIES_GBP:
        if a >= boundary:
            label = bucket_label
    return label


def integer_part(amount: Decimal) -> int:
    """Integer (pre-decimal) part of an amount as a plain int.

    Used by FR-016 — both ``abs(amount) == hint.amount`` AND
    ``int(amount) == int(hint.amount)`` must hold for an anchor to attach.
    """
    if amount is None:
        return 0
    return int(abs(amount))


def round_two_dp(amount: Decimal) -> Decimal:
    """Round to 2 dp using banker-safe half-up rounding."""
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class ClusterAmountStats:
    """Coarse statistics that Phase 6 will refine into amount-regime models."""

    count: int
    mean_amount: Decimal
    median_amount: Decimal
    min_amount: Decimal
    max_amount: Decimal
    coefficient_of_variation: float
    median_absolute_deviation: Decimal


def cluster_stats(amounts: Sequence[Decimal]) -> ClusterAmountStats:
    if not amounts:
        raise ValueError("cluster_stats requires at least one amount")
    abs_amounts = [abs(a) for a in amounts]
    floats = [float(a) for a in abs_amounts]
    m = Decimal(str(mean(floats)))
    med = Decimal(str(median(floats)))
    mn = min(abs_amounts)
    mx = max(abs_amounts)
    if len(amounts) >= 2 and float(m) > 0:
        # Population CV; small samples in our domain rarely justify sample
        # variance correction and our scoring is monotonic in CV anyway.
        squared = [(float(a) - float(m)) ** 2 for a in floats]
        variance = sum(squared) / len(squared)
        cv = (variance ** 0.5) / float(m)
    else:
        cv = 0.0
    deviations = sorted(abs(float(a) - float(med)) for a in floats)
    mad = Decimal(str(median(deviations) if deviations else 0.0))
    return ClusterAmountStats(
        count=len(amounts),
        mean_amount=round_two_dp(m),
        median_amount=round_two_dp(med),
        min_amount=round_two_dp(mn),
        max_amount=round_two_dp(mx),
        coefficient_of_variation=round(cv, 6),
        median_absolute_deviation=round_two_dp(mad),
    )


__all__ = [
    "ClusterAmountStats",
    "bucket_for",
    "cluster_stats",
    "integer_part",
    "round_two_dp",
]
