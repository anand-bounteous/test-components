"""Real-world scenario fixture runner.

Each JSON file in tests/realworld/scenarios/ is an independent test case.
The fixture schema is identical to the existing feature-test fixtures::

    {
      "analysis_id": "...",
      "scenario_description": "...",
      "scenario_tags": ["tag1", "tag2"],
      "validation_mode": "lenient",
      "transactions": [...],
      "expected": {                 # optional
        "candidate_count_min": 1,
        "top_candidate_type": "probable_salary",
        "top_confidence_band": "high",
        "must_contain_transaction_ids": ["t01", "t02"],
        "must_not_classify_as_probable_salary": false,
        "expected_warnings": []
      }
    }

Run all::

    pytest tests/realworld -m realworld -v

Run a single fixture::

    pytest tests/realworld -k acme_jan_2025 -v

Run by tag::

    pytest tests/realworld --rw-tag=monthly --rw-tag=bacs -v

List fixtures::

    pytest tests/realworld --rw-list

Debug a single fixture (full audit trace)::

    pytest tests/realworld -k acme_jan_2025 -s --log-cli-level=DEBUG
    # or via the standalone runner:
    python -m tests.realworld.runner --debug acme_jan_2025
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from salary_extractor.core.detector import detect_salary_candidates
from tests.feature._test_logging import get_feature_logger, log_test_detail, log_test_summary

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / ".plan" / "reviews" / "realworld-outputs"

logger = get_feature_logger(__name__)


def _validate_expected_block(path_stem: str, expected: dict, result) -> None:
    """Assert the labelled expectations carried in the fixture's ``expected`` block.

    Supported keys
    --------------
    Candidate assertions:
      candidate_count_min            int    at least this many candidates
      candidate_count_max            int    at most this many candidates
      candidate_count_exact          int    exactly this many candidates
      top_candidate_type             str    type of highest-confidence candidate
      top_confidence_band            str    band of highest-confidence candidate
      must_not_classify_as_probable_salary  bool  no probable_salary anywhere
      must_contain_transaction_ids   [str]  these ids must appear in some candidate

    Warning / negative-case assertions:
      expected_warnings              [str]  these warning codes must be present
                                           (alias: must_have_warning_codes)
      must_have_no_candidates        bool   assert zero candidates returned
                                           (use for negative / error-path fixtures)

    Pipeline phase assertions:
      expected_pipeline_phases       [str]  audit must record exactly these step
                                           names (order-insensitive); requires
                                           include_audit=True
    """
    if not expected:
        return

    candidates = list(result.candidate_sets)
    near_misses = list(result.rejected_or_near_miss_sets)
    every = candidates + near_misses

    # ── Candidate count ──────────────────────────────────────────────────────
    if "candidate_count_min" in expected:
        assert len(candidates) >= expected["candidate_count_min"], (
            f"{path_stem}: expected ≥ {expected['candidate_count_min']} candidates, "
            f"got {len(candidates)}"
        )
    if "candidate_count_max" in expected:
        assert len(candidates) <= expected["candidate_count_max"], (
            f"{path_stem}: expected ≤ {expected['candidate_count_max']} candidates, "
            f"got {len(candidates)}"
        )
    if "candidate_count_exact" in expected:
        assert len(candidates) == expected["candidate_count_exact"], (
            f"{path_stem}: expected exactly {expected['candidate_count_exact']} candidates, "
            f"got {len(candidates)}"
        )
    if expected.get("must_have_no_candidates") is True:
        assert len(candidates) == 0, (
            f"{path_stem}: expected zero candidates (negative case), "
            f"got {len(candidates)}: {[c.candidate_type for c in candidates]}"
        )

    # ── Top candidate shape ───────────────────────────────────────────────────
    if "top_candidate_type" in expected:
        assert candidates, f"{path_stem}: expected a candidate but none emitted"
        assert candidates[0].candidate_type == expected["top_candidate_type"], (
            f"{path_stem}: top type {candidates[0].candidate_type!r} ≠ "
            f"expected {expected['top_candidate_type']!r}"
        )
    if "top_confidence_band" in expected:
        assert candidates, f"{path_stem}: expected a candidate but none emitted"
        assert candidates[0].confidence_band == expected["top_confidence_band"], (
            f"{path_stem}: top band {candidates[0].confidence_band!r} ≠ "
            f"expected {expected['top_confidence_band']!r}"
        )
    if expected.get("must_not_classify_as_probable_salary") is True:
        assert all(c.candidate_type != "probable_salary" for c in every), (
            f"{path_stem}: at least one candidate is probable_salary but fixture forbids it"
        )

    # ── Transaction membership ────────────────────────────────────────────────
    must_contain = expected.get("must_contain_transaction_ids", [])
    if must_contain:
        if expected.get("must_contain_transaction_ids_ordered", False):
            # Ordered check: IDs must appear in the given sequence within the
            # top candidate's transaction list (subsequence, not exact match).
            assert candidates, f"{path_stem}: must_contain_transaction_ids_ordered requires at least one candidate"
            top_ids = [t.transaction_id for t in candidates[0].transactions]
            it = iter(top_ids)
            for tid in must_contain:
                assert any(seen_id == tid for seen_id in it), (
                    f"{path_stem}: transaction id {tid!r} not found in order "
                    f"within top candidate (top ids: {top_ids})"
                )
        else:
            seen = {t.transaction_id for c in every for t in c.transactions}
            for tid in must_contain:
                assert tid in seen, f"{path_stem}: expected txn id {tid!r} in some candidate"

    # ── Warning codes (both key names accepted) ───────────────────────────────
    required_warnings = expected.get("expected_warnings") or expected.get("must_have_warning_codes", [])
    if required_warnings:
        warning_codes = {w.code for w in result.warnings}
        for code in required_warnings:
            assert code in warning_codes, (
                f"{path_stem}: expected warning code {code!r}, "
                f"got {sorted(warning_codes)}"
            )

    # ── Pipeline phase assertions ─────────────────────────────────────────────
    expected_phases = expected.get("expected_pipeline_phases", [])
    if expected_phases:
        assert result.audit is not None, (
            f"{path_stem}: expected_pipeline_phases requires include_audit=True"
        )
        actual_phases = {s.name for s in result.audit.pipeline_steps}
        for phase in expected_phases:
            assert phase in actual_phases, (
                f"{path_stem}: expected pipeline phase {phase!r}, "
                f"got {sorted(actual_phases)}"
            )

    # ── Region inference assertions (Enhancement A) ───────────────────────────
    expected_inferred = expected.get("expected_inferred_regions", [])
    if expected_inferred:
        actual_inferred = set(result.inferred_country_regions)
        for region in expected_inferred:
            assert region in actual_inferred, (
                f"{path_stem}: expected inferred region {region!r} in "
                f"result.inferred_country_regions, got {sorted(actual_inferred)}"
            )

    # ── Gap summary assertions (Enhancement B1) ───────────────────────────────
    expected_missing = expected.get("expected_missing_months", [])
    if expected_missing:
        all_gap_summaries = [
            c.metadata.get("gap_summary", {})
            for c in list(result.candidate_sets) + list(result.rejected_or_near_miss_sets)
        ]
        all_missing = {m for gs in all_gap_summaries for m in gs.get("missing_months", [])}
        for month in expected_missing:
            assert month in all_missing, (
                f"{path_stem}: expected missing month {month!r} in gap_summary, "
                f"got {sorted(all_missing)}"
            )


@pytest.mark.realworld
def test_realworld_fixture(rw_scenario_path: Path) -> None:
    """Run a single real-world fixture end-to-end.

    The fixture file is discovered by conftest.pytest_generate_tests.
    Filtering via --rw-fixture, --rw-tag, or -k is supported.
    """
    payload = yaml.safe_load(rw_scenario_path.read_text()) or {}
    result = detect_salary_candidates(payload, include_audit=True)

    # Save evidence
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {
        "analysis_id": result.analysis_id,
        "scenario": rw_scenario_path.stem,
        "input_summary": {
            "transaction_count": result.input_summary.transaction_count,
            "credit_count": result.input_summary.credit_count,
            "date_range_start": result.input_summary.date_range_start,
            "date_range_end": result.input_summary.date_range_end,
        },
        "candidates": [
            {
                "candidate_set_id": c.candidate_set_id,
                "candidate_type": c.candidate_type,
                "confidence": c.confidence,
                "confidence_band": c.confidence_band,
                "transaction_ids": [t.transaction_id for t in c.transactions],
                "reasoning": list(c.reasoning),
                "risks": list(c.risks),
            }
            for c in result.candidate_sets
        ],
        "near_misses": [
            {
                "candidate_set_id": c.candidate_set_id,
                "candidate_type": c.candidate_type,
                "confidence": c.confidence,
                "near_miss_reason": c.metadata.get("near_miss_reason"),
            }
            for c in result.rejected_or_near_miss_sets
        ],
        "warnings": [{"code": w.code, "detail": w.detail} for w in result.warnings],
        "inferred_country_regions": list(result.inferred_country_regions),
    }
    (EVIDENCE_DIR / f"{rw_scenario_path.stem}.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n"
    )

    # Validate expected block
    expected = payload.get("expected", {})
    _validate_expected_block(rw_scenario_path.stem, expected, result)

    # Logging
    scenario_id = payload.get("analysis_id", rw_scenario_path.stem)
    log_test_summary(logger, scenario_id, result)
    log_test_detail(logger, payload, expected, result)

    # Log the final matched transactions (top candidate) so they appear inline
    # with --log-cli-level=INFO/DEBUG (no -s needed).
    if result.candidate_sets:
        top = result.candidate_sets[0]
        lines = [
            f"--- FINAL MATCHED TRANSACTIONS  {rw_scenario_path.stem}  "
            f"{top.candidate_type}  conf={top.confidence:.3f} ({top.confidence_band})  "
            f"{len(top.transactions)} transactions ---"
        ]
        lines += [
            f"  {t.date}  {t.description:<28} {t.amount:>10.2f}  [{t.transaction_id}]"
            for t in top.transactions
        ]
        logger.info("\n".join(lines))
    else:
        logger.info("--- FINAL MATCHED TRANSACTIONS  %s  (no candidate found) ---", rw_scenario_path.stem)
