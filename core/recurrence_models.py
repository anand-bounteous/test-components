"""Recurrence-model fitting — Design §7.2, §6.4 (calendar) consumer.

Eleven payday models from Design §7.2. The detector tries each and returns
the best-fitting one for a cluster of credits.

Model categories:

- **Monthly** — produce one predicted payday per month in the cluster's
  date range. Match each actual credit to the predicted date for its
  month, score by date deviation + coverage. Weekend / bank-holiday
  shifts are recognised so the deviation can be "explained" rather than
  penalised.
- **Periodic** — predict an interval (7 / 14 / 28 days) and score the
  group by interval regularity + weekday consistency.
- **Irregular** — fallback that still counts a series as salary-like if
  it has enough recurrence, even when no clean model fits.

Returns a ``RecurrenceFit`` so reasoning and scoring can inspect
predicted dates, deviations, and which deviations were explained by UK
working-day rules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from statistics import median
from typing import Callable, Optional, Sequence

from ..models.transaction import NormalisedTransaction
from .calendar_service import WorkingDayCalendar


class PaydayModel(str, Enum):
    FIXED_DAY_N = "fixed_day_n"
    PREVIOUS_WORKING_DAY_OF_N = "previous_working_day_of_n"
    NEXT_WORKING_DAY_OF_N = "next_working_day_of_n"
    NEAREST_WORKING_DAY_OF_N = "nearest_working_day_of_n"
    LAST_CALENDAR_DAY = "last_calendar_day"
    LAST_WORKING_DAY = "last_working_day"
    FIRST_WORKING_DAY = "first_working_day"
    WEEKLY_SAME_WEEKDAY = "weekly_same_weekday"
    FORTNIGHTLY_SAME_WEEKDAY = "fortnightly_same_weekday"
    FOUR_WEEKLY = "four_weekly"
    IRREGULAR_SALARY_LIKE = "irregular_salary_like"


# Maximum date deviation (days) the fit score still grants any credit.
_MAX_DEVIATION_DAYS = 7

# Default UK region used when the caller doesn't specify one.
_DEFAULT_REGION = "EnglandAndWales"


@dataclass(frozen=True)
class MonthFitDetail:
    """Per-month detail of a monthly-recurrence fit."""

    year: int
    month: int
    predicted: Optional[date]
    actual: Optional[date]
    deviation_days: Optional[int]
    explained_by_calendar: bool = False


@dataclass(frozen=True)
class RecurrenceFit:
    model: PaydayModel
    fit_score: float
    coverage: float
    mean_deviation_days: float
    matched_count: int
    expected_count: int
    monthly_details: tuple[MonthFitDetail, ...] = ()
    period_days: Optional[int] = None
    weekday: Optional[int] = None
    parameter_n: Optional[int] = None  # day-of-month for fixed-day models

    @property
    def coverage_str(self) -> str:
        return f"{self.matched_count}_of_{self.expected_count}"


# --- Top-level selector --------------------------------------------------


def fit_best_recurrence_model(
    cluster: Sequence[NormalisedTransaction],
    *,
    calendar: WorkingDayCalendar,
    region: str = _DEFAULT_REGION,
) -> RecurrenceFit:
    """Try every model; return the one with the highest fit score.

    Ties broken by preferring the more specific model — e.g. last_working_day
    beats last_calendar_day when both fit equally well.
    """
    if len(cluster) < 2:
        return _empty_fit(PaydayModel.IRREGULAR_SALARY_LIKE)

    candidates: list[RecurrenceFit] = []

    # Monthly models — work out the "N" (day-of-month) from the cluster
    # itself for fixed-day variants.
    n = _infer_fixed_day(cluster)
    candidates.append(_fit_last_calendar_day(cluster, calendar, region))
    candidates.append(_fit_last_working_day(cluster, calendar, region))
    candidates.append(_fit_first_working_day(cluster, calendar, region))
    if n is not None:
        candidates.append(_fit_fixed_day(cluster, n))
        candidates.append(_fit_prev_working_day_of_n(cluster, calendar, region, n))
        candidates.append(_fit_next_working_day_of_n(cluster, calendar, region, n))
        candidates.append(_fit_nearest_working_day_of_n(cluster, calendar, region, n))

    # Periodic models.
    candidates.append(_fit_periodic(cluster, period_days=7, model=PaydayModel.WEEKLY_SAME_WEEKDAY))
    candidates.append(_fit_periodic(cluster, period_days=14, model=PaydayModel.FORTNIGHTLY_SAME_WEEKDAY))
    candidates.append(_fit_periodic(cluster, period_days=28, model=PaydayModel.FOUR_WEEKLY))

    # Irregular fallback — counts the credits, gives a small baseline so
    # we never return a totally empty fit when there is data.
    candidates.append(_fit_irregular(cluster))

    candidates.sort(
        key=lambda f: (f.fit_score, _specificity(f.model), f.matched_count),
        reverse=True,
    )
    return candidates[0]


# --- Monthly fits --------------------------------------------------------


def _months_between(first: date, last: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        months.append((y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return months


def _fit_monthly_model(
    cluster: Sequence[NormalisedTransaction],
    model: PaydayModel,
    predict: Callable[[int, int], Optional[date]],
    *,
    explain_calendar_shift: Callable[[date, date], bool] = lambda predicted, actual: False,
    parameter_n: Optional[int] = None,
) -> RecurrenceFit:
    sorted_cluster = sorted(cluster, key=lambda t: t.date)
    first, last = sorted_cluster[0].date, sorted_cluster[-1].date
    months = _months_between(first, last)
    if not months:
        return _empty_fit(model)

    txns_by_month: dict[tuple[int, int], list[NormalisedTransaction]] = {}
    for t in sorted_cluster:
        txns_by_month.setdefault((t.date.year, t.date.month), []).append(t)

    details: list[MonthFitDetail] = []
    matched_count = 0
    total_deviation_days = 0
    matched_deviations: list[int] = []
    for ym in months:
        predicted = predict(ym[0], ym[1])
        candidates = txns_by_month.get(ym, [])
        if not candidates or predicted is None:
            details.append(
                MonthFitDetail(
                    year=ym[0],
                    month=ym[1],
                    predicted=predicted,
                    actual=None,
                    deviation_days=None,
                )
            )
            continue
        # Pick the credit closest to the predicted date.
        best = min(candidates, key=lambda t: abs((t.date - predicted).days))
        deviation = abs((best.date - predicted).days)
        explained = explain_calendar_shift(predicted, best.date) if deviation > 0 else False
        if explained:
            effective_dev = 0
        else:
            effective_dev = deviation
        matched_deviations.append(effective_dev)
        total_deviation_days += effective_dev
        matched_count += 1
        details.append(
            MonthFitDetail(
                year=ym[0],
                month=ym[1],
                predicted=predicted,
                actual=best.date,
                deviation_days=deviation,
                explained_by_calendar=explained,
            )
        )

    expected_count = len(months)
    coverage = matched_count / expected_count if expected_count else 0.0
    mean_deviation = (
        sum(matched_deviations) / len(matched_deviations) if matched_deviations else float("inf")
    )

    # Date-fit component: deviation 0 → 1.0; deviation _MAX → 0.0.
    if mean_deviation == float("inf"):
        date_fit = 0.0
    else:
        date_fit = max(0.0, 1.0 - mean_deviation / _MAX_DEVIATION_DAYS)
    score = 0.60 * date_fit + 0.40 * coverage

    return RecurrenceFit(
        model=model,
        fit_score=round(score, 6),
        coverage=round(coverage, 6),
        mean_deviation_days=round(mean_deviation, 4) if mean_deviation != float("inf") else 999.0,
        matched_count=matched_count,
        expected_count=expected_count,
        monthly_details=tuple(details),
        parameter_n=parameter_n,
    )


def _fit_last_calendar_day(cluster, calendar, region) -> RecurrenceFit:
    from calendar import monthrange

    def predict(y, m):
        return date(y, m, monthrange(y, m)[1])

    return _fit_monthly_model(cluster, PaydayModel.LAST_CALENDAR_DAY, predict)


def _fit_last_working_day(cluster, calendar, region) -> RecurrenceFit:
    def predict(y, m):
        return calendar.last_working_day_of_month(y, m, region)

    def explain(predicted: date, actual: date) -> bool:
        # last_working_day collapses the holiday/weekend cases inside the
        # predictor, so any drift here is "unexplained".
        return False

    return _fit_monthly_model(
        cluster, PaydayModel.LAST_WORKING_DAY, predict, explain_calendar_shift=explain
    )


def _fit_first_working_day(cluster, calendar, region) -> RecurrenceFit:
    def predict(y, m):
        return calendar.first_working_day_of_month(y, m, region)

    return _fit_monthly_model(cluster, PaydayModel.FIRST_WORKING_DAY, predict)


def _fit_fixed_day(cluster, n: int) -> RecurrenceFit:
    from calendar import monthrange

    def predict(y, m):
        last = monthrange(y, m)[1]
        return date(y, m, min(n, last))

    return _fit_monthly_model(cluster, PaydayModel.FIXED_DAY_N, predict, parameter_n=n)


def _fit_prev_working_day_of_n(cluster, calendar, region, n) -> RecurrenceFit:
    from calendar import monthrange

    def predict(y, m):
        last = monthrange(y, m)[1]
        target = date(y, m, min(n, last))
        return calendar.previous_working_day(target, region)

    return _fit_monthly_model(
        cluster, PaydayModel.PREVIOUS_WORKING_DAY_OF_N, predict, parameter_n=n
    )


def _fit_next_working_day_of_n(cluster, calendar, region, n) -> RecurrenceFit:
    from calendar import monthrange

    def predict(y, m):
        last = monthrange(y, m)[1]
        target = date(y, m, min(n, last))
        return calendar.next_working_day(target, region)

    return _fit_monthly_model(
        cluster, PaydayModel.NEXT_WORKING_DAY_OF_N, predict, parameter_n=n
    )


def _fit_nearest_working_day_of_n(cluster, calendar, region, n) -> RecurrenceFit:
    from calendar import monthrange

    def predict(y, m):
        last = monthrange(y, m)[1]
        target = date(y, m, min(n, last))
        return calendar.nearest_working_day(target, region)

    return _fit_monthly_model(
        cluster, PaydayModel.NEAREST_WORKING_DAY_OF_N, predict, parameter_n=n
    )


# --- Periodic fits -------------------------------------------------------


def _fit_periodic(
    cluster: Sequence[NormalisedTransaction],
    *,
    period_days: int,
    model: PaydayModel,
) -> RecurrenceFit:
    sorted_cluster = sorted(cluster, key=lambda t: t.date)
    if len(sorted_cluster) < 2:
        return _empty_fit(model)

    gaps = [
        (sorted_cluster[i + 1].date - sorted_cluster[i].date).days
        for i in range(len(sorted_cluster) - 1)
    ]
    deviations = [abs(g - period_days) for g in gaps]
    mean_gap_dev = sum(deviations) / len(deviations) if deviations else float("inf")

    weekdays = [t.date.weekday() for t in sorted_cluster]
    dominant_weekday, dominant_count = Counter(weekdays).most_common(1)[0]
    weekday_consistency = dominant_count / len(weekdays)

    # Expected count = span / period + 1
    span_days = (sorted_cluster[-1].date - sorted_cluster[0].date).days
    expected_count = max(1, span_days // period_days + 1)
    coverage = min(1.0, len(sorted_cluster) / expected_count)

    # Periodic fit allows a 2-day slack before deviation hurts.
    slack = 2
    if mean_gap_dev <= slack:
        gap_fit = 1.0
    else:
        gap_fit = max(0.0, 1.0 - (mean_gap_dev - slack) / period_days)
    score = 0.55 * gap_fit + 0.25 * weekday_consistency + 0.20 * coverage

    return RecurrenceFit(
        model=model,
        fit_score=round(score, 6),
        coverage=round(coverage, 6),
        mean_deviation_days=round(mean_gap_dev, 4),
        matched_count=len(sorted_cluster),
        expected_count=expected_count,
        period_days=period_days,
        weekday=dominant_weekday,
    )


# --- Irregular fallback --------------------------------------------------


def _fit_irregular(cluster: Sequence[NormalisedTransaction]) -> RecurrenceFit:
    """Baseline so a credit list always has *some* fit returned."""
    if len(cluster) < 3:
        return _empty_fit(PaydayModel.IRREGULAR_SALARY_LIKE)
    # Minimal score so it's a fallback the others can outrank.
    return RecurrenceFit(
        model=PaydayModel.IRREGULAR_SALARY_LIKE,
        fit_score=0.30,
        coverage=1.0,
        mean_deviation_days=0.0,
        matched_count=len(cluster),
        expected_count=len(cluster),
    )


def _empty_fit(model: PaydayModel) -> RecurrenceFit:
    return RecurrenceFit(
        model=model,
        fit_score=0.0,
        coverage=0.0,
        mean_deviation_days=999.0,
        matched_count=0,
        expected_count=0,
    )


# --- Helpers --------------------------------------------------------------


def _infer_fixed_day(cluster: Sequence[NormalisedTransaction]) -> Optional[int]:
    """Pick the modal day-of-month from the cluster as the candidate N."""
    days = [t.date.day for t in cluster]
    if not days:
        return None
    n, _ = Counter(days).most_common(1)[0]
    return n


# Specificity ranking for tie-breaking — more specific models win.
_SPECIFICITY: dict[PaydayModel, int] = {
    PaydayModel.LAST_WORKING_DAY: 10,
    PaydayModel.FIRST_WORKING_DAY: 10,
    PaydayModel.PREVIOUS_WORKING_DAY_OF_N: 9,
    PaydayModel.NEXT_WORKING_DAY_OF_N: 9,
    PaydayModel.NEAREST_WORKING_DAY_OF_N: 9,
    PaydayModel.LAST_CALENDAR_DAY: 6,
    PaydayModel.FIXED_DAY_N: 5,
    PaydayModel.WEEKLY_SAME_WEEKDAY: 8,
    PaydayModel.FORTNIGHTLY_SAME_WEEKDAY: 8,
    PaydayModel.FOUR_WEEKLY: 8,
    PaydayModel.IRREGULAR_SALARY_LIKE: 0,
}


def _specificity(model: PaydayModel) -> int:
    return _SPECIFICITY.get(model, 0)


__all__ = [
    "MonthFitDetail",
    "PaydayModel",
    "RecurrenceFit",
    "fit_best_recurrence_model",
]
