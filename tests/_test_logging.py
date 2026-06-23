"""Shared logging helpers for feature tests.

INFO level  — one-line summary per test case: scenario id, candidates found,
              transaction counts, warnings.
DEBUG level — full detail: input transactions, expected block, per-candidate
              breakdown (type, confidence, reasoning, risks, transaction ids),
              and audit step durations / LLM call records when present.

Usage in a test::

    from tests.feature._test_logging import get_feature_logger, log_test_summary, log_test_detail

    logger = get_feature_logger(__name__)

    def test_something(path):
        payload = yaml.safe_load(path.read_text())
        result = detect_salary_candidates(payload, include_audit=True)
        # ... assertions ...
        log_test_summary(logger, payload.get("analysis_id", path.stem), result)
        log_test_detail(logger, payload, payload.get("expected", {}), result)
"""
from __future__ import annotations

import json
import yaml
import logging
from typing import Any


def get_feature_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_test_summary(logger: logging.Logger, scenario_id: str, result) -> None:
    """Log a structured INFO summary for the test case including the full result shape."""
    candidates = result.candidate_sets
    near_misses = result.rejected_or_near_miss_sets
    warning_codes = [w.code for w in result.warnings]

    # Top candidate shorthand (None if no candidates)
    top = candidates[0] if candidates else None
    top_summary = (
        f"{top.candidate_type}  conf={top.confidence:.3f}  band={top.confidence_band}"
        f"  txns={len(top.transactions)}"
        if top else "none"
    )

    # All candidates one-liner
    all_summary = " | ".join(
        f"{c.candidate_type}(conf={c.confidence:.3f} band={c.confidence_band} txns={len(c.transactions)})"
        for c in candidates
    ) or "none"

    logger.info(
        "\n"
        "  ┌─ SalaryDetectionResult ─────────────────────────────────────\n"
        "  │  analysis_id   : %s\n"
        "  │  jurisdiction  : %s / %s\n"
        "  │  input         : %d rows  →  %d credits  [%s … %s]\n"
        "  │  candidates    : %d  [%s]\n"
        "  │  top candidate : %s\n"
        "  │  near_misses   : %d\n"
        "  │  warnings      : %s\n"
        "  └─────────────────────────────────────────────────────────────",
        result.analysis_id,
        getattr(result, "jurisdiction", "GB"),
        getattr(result, "country_region", "EnglandAndWales"),
        result.input_summary.transaction_count,
        result.input_summary.credit_count,
        result.input_summary.date_range_start or "—",
        result.input_summary.date_range_end or "—",
        len(candidates),
        all_summary,
        top_summary,
        len(near_misses),
        warning_codes or [],
    )


def log_test_detail(
    logger: logging.Logger,
    payload: dict,
    expected: dict,
    result,
) -> None:
    """Log full DEBUG detail: input, expected block, result with audit/reasoning."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    # Input
    transactions = payload.get("transactions") or []
    logger.debug(
        "--- INPUT  scenario=%s  transactions=%d ---\n%s",
        payload.get("analysis_id", "<unknown>"),
        len(transactions),
        json.dumps(transactions, indent=2, default=str),
    )

    # Expected block
    if expected:
        logger.debug(
            "--- EXPECTED ---\n%s",
            json.dumps(expected, indent=2, default=str),
        )

    # Candidates
    for i, c in enumerate(result.candidate_sets):
        logger.debug(
            "--- CANDIDATE %d  id=%s  type=%s  conf=%.3f  band=%s ---\n"
            "  transaction_ids: %s\n"
            "  reasoning:       %s\n"
            "  risks:           %s\n"
            "  score_breakdown: %s\n"
            "  metadata:        %s",
            i,
            c.candidate_set_id,
            c.candidate_type,
            c.confidence,
            c.confidence_band,
            [t.transaction_id for t in c.transactions],
            list(c.reasoning),
            list(c.risks),
            _score_summary(c.score_breakdown),
            c.metadata,
        )

    # Near misses
    for c in result.rejected_or_near_miss_sets:
        logger.debug(
            "--- NEAR MISS  id=%s  type=%s  conf=%.3f  reason=%s ---",
            c.candidate_set_id,
            c.candidate_type,
            c.confidence,
            c.metadata.get("near_miss_reason"),
        )

    # Audit
    if result.audit is not None:
        audit = result.audit
        timings = {k: f"{v:.1f}ms" for k, v in audit.timings_ms.items()}
        logger.debug("--- AUDIT  steps=%d  timings=%s ---", len(audit.pipeline_steps), timings)
        for step in audit.pipeline_steps:
            logger.debug(
                "  step=%s  duration=%.1fms  in=%s  out=%s",
                step.name,
                step.duration_ms,
                step.input_summary,
                step.output_summary,
            )
        if audit.llm_calls:
            for call in audit.llm_calls:
                logger.debug(
                    "  llm_call  provider=%s  model=%s  tokens_in=%d  tokens_out=%d"
                    "  cost_usd=%.6f  latency_ms=%.0f  error=%s",
                    call.provider,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cost_usd,
                    call.latency_ms,
                    call.error,
                )

    # LLM review
    if result.llm_review is not None:
        r = result.llm_review
        logger.debug(
            "--- LLM REVIEW  selected=%s  conf=%.3f  additional_review_needed=%s ---\n"
            "  reasoning: %s",
            list(r.selected_candidate_set_ids),
            r.confidence,
            r.additional_review_needed,
            list(r.reasoning),
        )


def _score_summary(sb) -> dict[str, Any]:
    """Return a compact dict of the score breakdown fields."""
    try:
        from dataclasses import asdict
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(sb).items()}
    except Exception:
        return {}
