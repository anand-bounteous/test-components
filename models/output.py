"""Top-level detector output — Requirement §6.1.

Phase 14 extends with two optional fields: ``audit`` (populated when
the caller passes ``include_audit=True``) and ``llm_review`` (populated
when ``enable_llm=True`` AND the deterministic pipeline triggers the
LLM-fallback path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .candidate import SalaryCandidateSet
from .hints import AppliedHints

if TYPE_CHECKING:
    from .audit import AuditRecord, LLMReviewResult


@dataclass(frozen=True)
class InputSummary:
    transaction_count: int
    credit_count: int
    date_range_start: str = ""
    date_range_end: str = ""


@dataclass(frozen=True)
class ValidationWarning:
    code: str
    transaction_id: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SalaryDetectionResult:
    analysis_id: str
    jurisdiction: str
    country_region: str
    input_summary: InputSummary
    candidate_sets: tuple[SalaryCandidateSet, ...] = ()
    rejected_or_near_miss_sets: tuple[SalaryCandidateSet, ...] = ()
    warnings: tuple[ValidationWarning, ...] = ()
    applied_hints: AppliedHints = field(default_factory=AppliedHints)
    # Enhancement A: populated when country_region was NOT explicitly provided;
    # lists the UK calendar region(s) whose holiday rules best explain the
    # observed payday dates (union across all candidate sets).
    inferred_country_regions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    audit: Optional["AuditRecord"] = None
    llm_review: Optional["LLMReviewResult"] = None
