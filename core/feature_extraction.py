"""Per-transaction feature extraction orchestrator — Design §6.3.

Composes:

- payment-channel detection (``payment_channel.py``)
- description features (``description_features.py``)
- amount features (``amount_features.py``)
- calendar features (``calendar_service.py``)

Produces an immutable ``TransactionFeatures`` per credit. The hint
application step (Phase 4) will replace the ``is_anchor`` /
``anchor_hint_id`` defaults later in the pipeline.
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

from ..models.transaction import NormalisedTransaction, TransactionFeatures
from .amount_features import bucket_for
from .calendar_service import WorkingDayCalendar
from .config import AppConfig
from .description_features import DescriptionFeatureExtractor
from .payment_channel import PaymentChannelDetector


@dataclass
class FeatureExtractor:
    config: AppConfig
    calendar: WorkingDayCalendar
    region: str = "EnglandAndWales"
    channel_detector: PaymentChannelDetector | None = None
    description_extractor: DescriptionFeatureExtractor | None = None

    def __post_init__(self) -> None:
        if self.channel_detector is None:
            self.channel_detector = PaymentChannelDetector(config=self.config)
        if self.description_extractor is None:
            self.description_extractor = DescriptionFeatureExtractor(config=self.config)

    def extract(self, txn: NormalisedTransaction) -> TransactionFeatures:
        channel = self.channel_detector.detect(txn.description_tokens)
        desc = self.description_extractor.extract(txn.description_tokens)
        d = txn.date

        # Calendar features.
        is_lwd = d == self.calendar.last_working_day_of_month(d.year, d.month, self.region)
        is_fwd = d == self.calendar.first_working_day_of_month(d.year, d.month, self.region)
        last_day = monthrange(d.year, d.month)[1]
        is_month_end = d.day == last_day

        is_bh_adjacent = self._is_bank_holiday_adjacent(d)

        return TransactionFeatures(
            transaction_id=txn.id,
            payment_channel=channel.channel,
            payment_channel_confidence=channel.confidence,
            payment_channel_matched_tokens=channel.matched_tokens,
            salary_hint_tokens=desc.salary_hint_tokens,
            negative_tokens=desc.negative_tokens,
            possible_employer_tokens=desc.possible_employer_tokens,
            day_of_month=d.day,
            weekday=d.weekday(),
            is_month_end=is_month_end,
            is_last_working_day=is_lwd,
            is_first_working_day=is_fwd,
            is_bank_holiday_adjacent=is_bh_adjacent,
            amount_bucket=bucket_for(txn.amount),
            is_anchor=False,
            anchor_hint_id=None,
            ambiguous_anchor=False,
        )

    def extract_many(self, txns: Iterable[NormalisedTransaction]) -> tuple[TransactionFeatures, ...]:
        return tuple(self.extract(t) for t in txns)

    # --- Internal --------------------------------------------------------

    def _is_bank_holiday_adjacent(self, d: date) -> bool:
        """True if the day before or after this date is a UK bank holiday."""
        for delta in (-1, 1):
            neighbour = d + timedelta(days=delta)
            if not self.calendar.is_working_day(neighbour, self.region) and neighbour.weekday() < 5:
                return True
        return False


def attach_anchor_flags(
    features: tuple[TransactionFeatures, ...],
    anchor_txn_ids: dict[str, str],
    *,
    ambiguous_ids: set[str] | None = None,
) -> tuple[TransactionFeatures, ...]:
    """Phase-4 helper that returns a new tuple with anchor flags set.

    ``anchor_txn_ids`` maps ``transaction_id`` → ``hint_id``.
    Ambiguous matches are flagged separately so the LLM payload can show
    the conflict.
    """
    ambiguous_ids = ambiguous_ids or set()
    updated: list[TransactionFeatures] = []
    for f in features:
        if f.transaction_id in anchor_txn_ids:
            updated.append(
                replace(
                    f,
                    is_anchor=True,
                    anchor_hint_id=anchor_txn_ids[f.transaction_id],
                    ambiguous_anchor=f.transaction_id in ambiguous_ids,
                )
            )
        else:
            updated.append(f)
    return tuple(updated)
