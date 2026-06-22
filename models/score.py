"""Score breakdown — Design §4.5, §11."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreBreakdown:
    date_pattern_score: float = 0.0
    description_similarity_score: float = 0.0
    amount_consistency_score: float = 0.0
    payment_channel_score: float = 0.0
    coverage_score: float = 0.0
    context_bonus: float = 0.0
    negative_signal_penalty: float = 0.0
    anchor_bonus: float = 0.0
    salary_band_modifier: float = 1.0
    final_score: float = 0.0
