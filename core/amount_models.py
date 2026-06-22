"""Amount-regime classification — Design §9.

Seven classes:

- ``stable``                       — CV ≤ 2 % around the median.
- ``stable_with_minor_variation``  — CV ≤ 10 %.
- ``stable_with_step_change``      — two stable segments separated by a ≥ 5 % step.
- ``variable_but_recurring``       — CV in the moderate band (≤ 20 %) without
  a clean step.
- ``one_off_final_pay_outlier``    — one anomaly at the tail of an otherwise
  stable series.
- ``bonus_or_extra_pay_outlier``   — one anomaly **inside** the series.
- ``too_variable``                 — anything else.

The classifier returns ``AmountFit`` carrying segment medians,
identified outliers, the chosen model, and a 0..1 amount-consistency
score Phase 8 will multiply into the scoring blend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from statistics import mean, median
from typing import Optional, Sequence

from ..models.transaction import NormalisedTransaction
from .amount_features import cluster_stats, round_two_dp
from .config import AppConfig


class AmountModel(str, Enum):
    STABLE = "stable"
    STABLE_WITH_MINOR_VARIATION = "stable_with_minor_variation"
    STABLE_WITH_STEP_CHANGE = "stable_with_step_change"
    VARIABLE_BUT_RECURRING = "variable_but_recurring"
    ONE_OFF_FINAL_PAY_OUTLIER = "one_off_final_pay_outlier"
    BONUS_OR_EXTRA_PAY_OUTLIER = "bonus_or_extra_pay_outlier"
    TOO_VARIABLE = "too_variable"


@dataclass(frozen=True)
class AmountSegment:
    """A stable amount-segment within a cluster (used by step-change)."""

    start_index: int
    end_index: int  # inclusive
    median: Decimal


@dataclass(frozen=True)
class AmountFit:
    model: AmountModel
    score: float
    median_amount: Decimal
    coefficient_of_variation: float
    outlier_ids: tuple[str, ...] = ()
    segments: tuple[AmountSegment, ...] = ()


# --- Defaults (overridable by config) ------------------------------------


_DEFAULTS = {
    "stable_pct_tolerance": 0.02,
    "moderate_pct_tolerance": 0.10,
    "step_change_min_pct": 0.05,
    "variable_max_cv": 0.30,
    "outlier_mad_ratio": 3.0,
}


def _settings(config: Optional[AppConfig]) -> dict:
    if config is None:
        return dict(_DEFAULTS)
    out = dict(_DEFAULTS)
    for key, default in _DEFAULTS.items():
        value = config.get("amount", key, default=default)
        out[key] = float(value)
    return out


# --- Top-level classifier ------------------------------------------------


def classify_amount_pattern(
    cluster: Sequence[NormalisedTransaction],
    *,
    config: Optional[AppConfig] = None,
) -> AmountFit:
    """Pick the amount regime that best describes the cluster."""
    if not cluster:
        return AmountFit(
            model=AmountModel.TOO_VARIABLE,
            score=0.0,
            median_amount=Decimal("0"),
            coefficient_of_variation=0.0,
        )

    settings = _settings(config)
    sorted_cluster = sorted(cluster, key=lambda t: t.date)
    amounts = [t.amount for t in sorted_cluster]
    stats = cluster_stats(amounts)
    cv = stats.coefficient_of_variation
    med = stats.median_amount

    # 1) Stable / minor variation paths.
    if cv <= settings["stable_pct_tolerance"]:
        return AmountFit(
            model=AmountModel.STABLE,
            score=1.0,
            median_amount=med,
            coefficient_of_variation=cv,
        )

    # 2) Single-outlier paths — try this BEFORE relaxing to minor variation
    # so a stable salary with one final-pay anomaly isn't lost in CV.
    outliers = _detect_outliers(sorted_cluster, settings)
    if len(outliers) == 1:
        non_outlier_amounts = [
            t.amount for t in sorted_cluster if t.id not in outliers
        ]
        if len(non_outlier_amounts) >= 2:
            non_outlier_cv = cluster_stats(non_outlier_amounts).coefficient_of_variation
            if non_outlier_cv <= settings["moderate_pct_tolerance"]:
                # Decide whether the outlier sits at the end (final-pay) or
                # earlier inside the cluster (bonus / extra pay).
                outlier_index = next(
                    i for i, t in enumerate(sorted_cluster) if t.id in outliers
                )
                if outlier_index in (0, len(sorted_cluster) - 1):
                    return AmountFit(
                        model=AmountModel.ONE_OFF_FINAL_PAY_OUTLIER,
                        score=0.80,
                        median_amount=med,
                        coefficient_of_variation=cv,
                        outlier_ids=tuple(outliers),
                    )
                return AmountFit(
                    model=AmountModel.BONUS_OR_EXTRA_PAY_OUTLIER,
                    score=0.75,
                    median_amount=med,
                    coefficient_of_variation=cv,
                    outlier_ids=tuple(outliers),
                )

    # 3) Step-change.
    segment_split = _detect_step_change(sorted_cluster, settings)
    if segment_split is not None:
        return AmountFit(
            model=AmountModel.STABLE_WITH_STEP_CHANGE,
            score=0.85,
            median_amount=med,
            coefficient_of_variation=cv,
            segments=segment_split,
        )

    # 4) Minor variation.
    if cv <= settings["moderate_pct_tolerance"]:
        return AmountFit(
            model=AmountModel.STABLE_WITH_MINOR_VARIATION,
            score=0.80,
            median_amount=med,
            coefficient_of_variation=cv,
        )

    # 5) Variable but recurring.
    if cv <= settings["variable_max_cv"]:
        return AmountFit(
            model=AmountModel.VARIABLE_BUT_RECURRING,
            score=0.55,
            median_amount=med,
            coefficient_of_variation=cv,
        )

    # 6) Anything else.
    return AmountFit(
        model=AmountModel.TOO_VARIABLE,
        score=0.20,
        median_amount=med,
        coefficient_of_variation=cv,
    )


# --- Helpers --------------------------------------------------------------


def _detect_outliers(
    sorted_cluster: Sequence[NormalisedTransaction],
    settings: dict,
) -> list[str]:
    """MAD-based outlier detection. Returns transaction ids."""
    if len(sorted_cluster) < 4:
        return []
    floats = [float(t.amount) for t in sorted_cluster]
    med = median(floats)
    deviations = [abs(x - med) for x in floats]
    mad = median(deviations)
    if mad == 0:
        # All-equal cluster: a single value off the median is the outlier.
        return [t.id for t in sorted_cluster if float(t.amount) != med]
    ratio_cutoff = settings["outlier_mad_ratio"]
    return [
        t.id
        for t, dev in zip(sorted_cluster, deviations)
        if (dev / mad) >= ratio_cutoff
    ]


def _detect_step_change(
    sorted_cluster: Sequence[NormalisedTransaction],
    settings: dict,
) -> Optional[tuple[AmountSegment, AmountSegment]]:
    """Try each candidate split. Return two segments if both halves are
    stable and their medians differ by ≥ step_change_min_pct.
    """
    n = len(sorted_cluster)
    if n < 4:
        return None
    floats = [float(t.amount) for t in sorted_cluster]
    stable_tol = settings["stable_pct_tolerance"]
    moderate_tol = settings["moderate_pct_tolerance"]
    min_step = settings["step_change_min_pct"]

    best_split: Optional[int] = None
    best_step_delta = 0.0
    for split in range(2, n - 1):  # at least 2 elements each side
        left = floats[:split]
        right = floats[split:]
        left_med = median(left)
        right_med = median(right)
        left_cv = _cv(left)
        right_cv = _cv(right)
        if left_cv > moderate_tol or right_cv > moderate_tol:
            continue
        if left_med == 0:
            continue
        delta = abs(right_med - left_med) / left_med
        if delta < min_step:
            continue
        # Prefer the split with the cleanest segments and the biggest delta.
        if delta > best_step_delta:
            best_step_delta = delta
            best_split = split

    if best_split is None:
        return None

    left = floats[:best_split]
    right = floats[best_split:]
    seg_left = AmountSegment(
        start_index=0,
        end_index=best_split - 1,
        median=round_two_dp(Decimal(str(median(left)))),
    )
    seg_right = AmountSegment(
        start_index=best_split,
        end_index=n - 1,
        median=round_two_dp(Decimal(str(median(right)))),
    )
    return (seg_left, seg_right)


def _cv(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    if m == 0:
        return 0.0
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / abs(m)


__all__ = [
    "AmountFit",
    "AmountModel",
    "AmountSegment",
    "classify_amount_pattern",
]
