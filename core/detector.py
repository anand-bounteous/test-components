"""End-to-end detector orchestrator — Phase 9.

Composes every earlier phase into a single ``detect_salary_candidates``
entry point that takes an input payload (matching ``input.schema.json``)
and returns a ``SalaryDetectionResult``.

Pipeline:

1. Schema + row-level validation (Phase 1, FR-001 / FR-002).
2. Normalisation (Phase 1).
3. Per-transaction feature extraction (Phase 3).
4. Hint application — payslip anchor + approx-salary band (Phase 4,
   FR-016 / FR-017).
5. Skeleton-grouping candidate generation (Phase 5, FR-018).
6. Similarity-graph clustering candidate generation (Phase 7).
7. Pool dedup by transaction-id overlap (Phase 7, Design §10.5).
8. Per-candidate scoring + candidate-type classification + reasoning
   (Phase 8, Design §11 / §12 / §13).
9. **Edge-case linking** (this phase, Design §15) — employment-switch
   linkage, final-pay attachment, bonus / expense separation.
10. Ranking + return — top N candidates and near-misses, plus the
    applied-hints audit record and the configuration versions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date as _date
from typing import Optional

from .._logging import setup_trace
from ..llm.client import LLMClient
from ..models.candidate import SalaryCandidateSet
from ..models.hints import HintInputs
from ..models.output import (
    InputSummary,
    SalaryDetectionResult,
    ValidationWarning,
)
from .amount_date_fallback import AmountDateFallbackDetector
from .audit import AuditBuilder
from .calendar_service import WorkingDayCalendar
from .clustering import GraphClusteringDetector, dedup_candidate_pool
from .config import AppConfig, load_config
from .feature_extraction import FeatureExtractor, attach_anchor_flags
from .hints import (
    ambiguous_anchor_ids,
    anchor_txn_ids,
    apply_hints,
)
from .llm_fallback import invoke_llm_fallback, should_invoke_fallback
from .normalisation import normalise_many
from .scoring import score_candidate
from .skeleton_grouping import SkeletonGroupingDetector
from .validation import validate_and_normalise_rows, validate_schema

setup_trace()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def detect_salary_candidates(
    payload: dict,
    *,
    config: Optional[AppConfig] = None,
    calendar: Optional[WorkingDayCalendar] = None,
    mode: Optional[str] = None,
    max_candidates: Optional[int] = None,
    max_near_misses: Optional[int] = None,
    return_threshold: Optional[float] = None,
    enable_llm: Optional[bool] = None,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = None,
    include_audit: Optional[bool] = None,
) -> SalaryDetectionResult:
    """End-to-end detector. Single call returns the final result.

    ``payload`` is the input JSON described by ``input.schema.json``;
    ``mode`` overrides the payload's ``validation_mode`` if supplied.

    Phase 14 kwargs (each falls back to the payload's ``options`` block
    when left as ``None``; defaults are conservative):

    - ``enable_llm`` — when ``True`` AND the deterministic result is
      ambiguous (top conf < 0.55, two candidates within 0.05, or no
      candidates with ``no_signal_detected``), the detector calls
      ``llm_client.call(...)`` with the full credit list. Default
      ``False``. Strict no-op when ``False``.
    - ``llm_client`` — injected ``LLMClient`` implementation. Required
      when ``enable_llm=True``.
    - ``llm_model`` — override the client's ``default_model``.
    - ``include_audit`` — when ``True`` the result carries a full
      ``AuditRecord`` with per-step durations, every feature record,
      every per-candidate detail, every LLM call. Default ``False``.
    """
    cfg = config or load_config()
    cal = calendar or WorkingDayCalendar()
    validation_mode = mode or payload.get("validation_mode", "lenient")

    options = payload.get("options") or {}
    if enable_llm is None:
        enable_llm = bool(options.get("enable_llm", False))
    if include_audit is None:
        include_audit = bool(options.get("include_audit", False))
    if llm_model is None:
        llm_model = options.get("llm_model")

    versions = cfg.versions()
    analysis_id = payload.get("analysis_id", "salary_analysis")
    logger.info(
        "salary_detector.start analysis_id=%s rows=%d mode=%s versions=%s "
        "enable_llm=%s include_audit=%s",
        analysis_id,
        len(payload.get("transactions") or []),
        validation_mode,
        versions,
        enable_llm,
        include_audit,
    )

    audit = AuditBuilder(
        enabled=bool(include_audit),
        analysis_id=analysis_id,
        config_versions=dict(versions),
        effective_kwargs={
            "validation_mode": validation_mode,
            "max_candidates": max_candidates,
            "max_near_misses": max_near_misses,
            "return_threshold": return_threshold,
            "enable_llm": enable_llm,
            "include_audit": include_audit,
            "llm_model": llm_model,
        },
    )

    warnings: list[ValidationWarning] = []

    # 1. Schema validation (lenient — surface every issue as a warning so
    #    a single bad row doesn't kill the whole run).
    audit.start_step("validate")
    schema_warnings = validate_schema(payload, strict=False)
    warnings.extend(schema_warnings)

    # 2. Row-level + credit-only.
    raw_transactions = payload.get("transactions")
    if raw_transactions is None:
        detail = (
            "'transactions' key is missing from the payload"
            if "transactions" not in payload
            else "'transactions' is null — expected a list of transaction objects"
        )
        warnings.append(ValidationWarning(
            code="no_transactions",
            transaction_id="",
            detail=detail,
        ))
        audit.record_step(
            "validate",
            input_summary={"row_count": 0},
            output_summary={"credit_count": 0, "warnings": len(warnings)},
        )
        return _early_exit(
            analysis_id, warnings, audit, include_audit,
            jurisdiction=payload.get("jurisdiction", "GB"),
            country_region=payload.get("country_region", cfg.default_region()),
        )
    validated = validate_and_normalise_rows(
        raw_transactions, mode=validation_mode
    )
    warnings.extend(validated.warnings)
    audit.record_step(
        "validate",
        input_summary={"row_count": len(payload.get("transactions", []))},
        output_summary={
            "credit_count": len(validated.transactions),
            "warnings": len(schema_warnings) + len(validated.warnings),
        },
    )

    # 3. Normalise.
    audit.start_step("normalise")
    normalised = normalise_many(validated.transactions)
    audit.record_step(
        "normalise",
        input_summary={"credit_count": len(validated.transactions)},
        output_summary={"normalised_count": len(normalised)},
    )

    # 4. Features.
    audit.start_step("feature_extraction")
    fe = FeatureExtractor(
        config=cfg,
        calendar=cal,
        region=payload.get("country_region", cfg.default_region()),
    )
    features = fe.extract_many(normalised)
    features_by_id = {f.transaction_id: f for f in features}
    audit.record_features(tuple(features))
    audit.record_step(
        "feature_extraction",
        input_summary={"transactions": len(normalised)},
        output_summary={"features": len(features)},
    )
    if logger.isEnabledFor(5):  # TRACE
        for f in features:
            logger.log(
                5,
                "feature_extraction.record id=%s channel=%s tokens=%s",
                f.transaction_id,
                f.payment_channel,
                f.salary_hint_tokens,
            )

    # 5. Hints.
    hint_inputs = HintInputs.from_dict(payload)
    applied_hints = apply_hints(
        normalised,
        payslip_hints=hint_inputs.payslip_hints,
        salary_hints=hint_inputs.salary_hints,
    )

    # 6. Attach anchor flags to features (idempotent helper from Phase 3).
    if applied_hints.anchors:
        flagged = attach_anchor_flags(
            features,
            anchor_txn_ids=anchor_txn_ids(applied_hints),
            ambiguous_ids=ambiguous_anchor_ids(applied_hints),
        )
        features_by_id = {f.transaction_id: f for f in flagged}
        if applied_hints.unmatched_hint_ids:
            for hint_id in applied_hints.unmatched_hint_ids:
                warnings.append(
                    ValidationWarning(
                        code="hint_unmatched",
                        transaction_id="",
                        detail=f"payslip hint {hint_id!r} did not match any credit",
                    )
                )

    # 7. Skeleton path.
    skeleton_candidates = SkeletonGroupingDetector(config=cfg).emit_plausible_sets(
        normalised, features_by_id, applied_hints=applied_hints
    )

    # 8. Graph path.
    graph_candidates = GraphClusteringDetector(config=cfg).emit_candidates(
        normalised, features_by_id, applied_hints=applied_hints
    )

    # 9. Dedup.
    cg = cfg.candidate_generation()
    overlap_threshold = float(cg.get("dedup_txn_id_overlap_pct", 0.90))
    pool = dedup_candidate_pool(
        skeleton_candidates,
        graph_candidates,
        overlap_threshold=overlap_threshold,
    )

    # 9b. Split off bonus / expense rows from any candidate that mixes
    # main-salary credits with bonus / expense markers (Design §15.5).
    pool = _split_bonus_and_expense_credits(pool, normalised)

    # 9c. Amount + date-only fallback (Phase 13). Catches UUID-only or
    # otherwise-opaque descriptions that the skeleton + graph paths
    # couldn't bucket. Surfaces as low-confidence candidates so the LLM
    # reviewer can confirm.
    accounted_ids = {t.transaction_id for c in pool for t in c.transactions}
    unaccounted = [t for t in normalised if t.id not in accounted_ids]
    if unaccounted:
        fallback_candidates = AmountDateFallbackDetector(config=cfg).emit_candidates(
            unaccounted,
            cal,
            applied_hints=applied_hints,
            region=payload.get("country_region", cfg.default_region()),
        )
        if fallback_candidates:
            pool = pool + fallback_candidates
            if not skeleton_candidates and not graph_candidates:
                # Surface a 'no_signal_detected' warning when the *only*
                # candidate comes from the fallback path — useful audit.
                warnings.append(
                    ValidationWarning(
                        code="no_signal_detected",
                        transaction_id="",
                        detail=(
                            "No description signal recovered; relying on "
                            "amount + cadence alone."
                        ),
                    )
                )
        else:
            # No fallback either AND no earlier candidates → genuinely
            # nothing to surface.
            if not skeleton_candidates and not graph_candidates and unaccounted:
                warnings.append(
                    ValidationWarning(
                        code="no_signal_detected",
                        transaction_id="",
                        detail=(
                            "No salary signal found in description, amount, "
                            "or cadence."
                        ),
                    )
                )

    # 10. Score + classify + reasoning per candidate.
    transactions_range = payload.get("transactions_range")  # optional caller-supplied range
    scored: list[SalaryCandidateSet] = []
    for cand in pool:
        cluster_ids = {t.transaction_id for t in cand.transactions}
        cluster = [t for t in normalised if t.id in cluster_ids]
        scored_candidate = score_candidate(
            cand,
            cluster,
            features_by_id,
            calendar=cal,
            config=cfg,
            applied_hints=applied_hints,
            salary_hints=hint_inputs.salary_hints,
            transactions_range=transactions_range,
        )
        scored.append(scored_candidate)

    # 10b. Region inference — when country_region was not explicitly provided,
    # compare how well the best payday model fits each UK region's calendar.
    # Winners are tagged in candidate metadata and collected at result level.
    explicit_region = "country_region" in payload
    inferred_country_regions: tuple[str, ...] = ()
    if not explicit_region:
        inferred_country_regions, scored, region_warnings = _infer_regions(
            scored, normalised, cal, warnings
        )
        warnings = region_warnings

    # 10c. Emit missing_salary_periods warnings for candidates with ≥ 2 missing months.
    for cand in scored:
        gs = cand.metadata.get("gap_summary", {})
        missing = gs.get("missing_months", [])
        if len(missing) >= 2:
            warnings.append(
                ValidationWarning(
                    code="missing_salary_periods",
                    transaction_id="",
                    detail=(
                        f"Candidate {cand.candidate_set_id!r}: "
                        f"{len(missing)} expected salary period(s) have no matching credit: "
                        + ", ".join(missing[:10])
                        + ("…" if len(missing) > 10 else "")
                        + "."
                    ),
                )
            )

    # 11. Edge-case linking (employment switch, final-pay attachment).
    scored, transition_warnings = _link_employment_transitions(scored, cfg)
    warnings = list(warnings) + transition_warnings
    scored = _override_split_subcandidate_types(scored)

    # 12. Rank and split into candidate set + near-miss set.
    scored.sort(key=lambda c: (-c.confidence, c.candidate_set_id))
    effective_return_threshold = (
        float(return_threshold)
        if return_threshold is not None
        else float(cfg.get("thresholds", "return_threshold", default=0.40))
    )
    effective_max_candidates = (
        int(max_candidates)
        if max_candidates is not None
        else int(cg.get("max_candidates_to_return", 10))
    )
    effective_max_near_misses = (
        int(max_near_misses)
        if max_near_misses is not None
        else int(cg.get("max_near_misses", 10))
    )

    above = [c for c in scored if c.confidence >= effective_return_threshold]
    below = [c for c in scored if c.confidence < effective_return_threshold]

    returned = above[:effective_max_candidates]
    cap_overflow = above[effective_max_candidates:]
    remaining_near_miss_slots = max(0, effective_max_near_misses - len(cap_overflow))
    below_kept = below[:remaining_near_miss_slots]

    near_misses_combined = [_tag_near_miss(c, "cap_overflow") for c in cap_overflow]
    near_misses_combined.extend(
        _tag_near_miss(c, "below_threshold") for c in below_kept
    )
    # Hard cap (defensive — cap_overflow already respects this through
    # remaining_near_miss_slots, but trim to be safe).
    near_misses_combined = near_misses_combined[:effective_max_near_misses]

    summary = _build_summary(payload, normalised)

    logger.info(
        "salary_detector.finish analysis_id=%s credits=%d candidates=%d "
        "near_misses=%d (cap_overflow=%d below_threshold=%d) warnings=%d",
        analysis_id,
        summary.credit_count,
        len(returned),
        len(near_misses_combined),
        sum(
            1
            for c in near_misses_combined
            if c.metadata.get("near_miss_reason") == "cap_overflow"
        ),
        sum(
            1
            for c in near_misses_combined
            if c.metadata.get("near_miss_reason") == "below_threshold"
        ),
        len(warnings),
    )

    audit.record_candidates(tuple(returned) + tuple(near_misses_combined))

    result = SalaryDetectionResult(
        analysis_id=analysis_id,
        jurisdiction=payload.get("jurisdiction", "GB"),
        country_region=payload.get("country_region", cfg.default_region()),
        input_summary=summary,
        candidate_sets=tuple(returned),
        rejected_or_near_miss_sets=tuple(near_misses_combined),
        warnings=tuple(warnings),
        applied_hints=applied_hints,
        inferred_country_regions=inferred_country_regions,
        metadata={
            "config_versions": versions,
            "return_threshold": effective_return_threshold,
            "max_candidates": effective_max_candidates,
            "max_near_misses": effective_max_near_misses,
            "validation_mode": validation_mode,
            "enable_llm": enable_llm,
            "include_audit": include_audit,
        },
    )

    # ---- Phase 14: LLM fallback ----
    llm_review = None
    if enable_llm:
        invoke, reason = should_invoke_fallback(result, enable_llm=True)
        logger.info(
            "salary_detector.llm_fallback_decision analysis_id=%s invoke=%s reason=%s",
            analysis_id,
            invoke,
            reason,
        )
        if invoke:
            if llm_client is None:
                logger.warning(
                    "salary_detector.llm_fallback_skipped analysis_id=%s "
                    "reason=missing_client",
                    analysis_id,
                )
            else:
                audit.start_step("llm_fallback")
                llm_review = invoke_llm_fallback(
                    result,
                    normalised,
                    client=llm_client,
                    model=llm_model,
                )
                audit.record_llm_call(llm_review.call_record)
                audit.record_step(
                    "llm_fallback",
                    input_summary={
                        "candidates": len(result.candidate_sets),
                        "transactions": len(normalised),
                        "reason": reason,
                    },
                    output_summary={
                        "selected": list(llm_review.selected_candidate_set_ids),
                        "confidence": llm_review.confidence,
                        "additional_review_needed": llm_review.additional_review_needed,
                        "cost_usd": llm_review.call_record.cost_usd,
                        "latency_ms": llm_review.call_record.latency_ms,
                    },
                )

    snapshot = audit.snapshot()
    if snapshot is not None or llm_review is not None:
        from dataclasses import replace as _replace
        result = _replace(result, audit=snapshot, llm_review=llm_review)
    return result


# ---------------------------------------------------------------------------
# Near-miss tagging (Phase 12)
# ---------------------------------------------------------------------------


def _tag_near_miss(candidate: SalaryCandidateSet, reason: str) -> SalaryCandidateSet:
    """Return a copy of ``candidate`` with ``metadata.near_miss_reason``
    set to ``reason`` (``"cap_overflow"`` or ``"below_threshold"``)."""
    new_meta = dict(candidate.metadata)
    new_meta["near_miss_reason"] = reason
    return replace(candidate, metadata=new_meta)


# ---------------------------------------------------------------------------
# Bonus / expense splitting (Design §15.5)
# ---------------------------------------------------------------------------


_BONUS_TOKENS = frozenset({"BONUS"})
_EXPENSE_TOKENS = frozenset({"EXPENSES", "EXPENSE", "REIMBURSEMENT"})


def _split_bonus_and_expense_credits(
    pool: tuple,
    normalised,
) -> tuple:
    """Split off bonus and expense rows from candidates that mix them with
    a main-salary stream.

    Triggers when:

    - the candidate contains both flagged rows (BONUS / EXPENSES) **and**
      non-flagged rows, and
    - the non-flagged subset has at least 2 transactions.

    The original candidate is replaced by:

    - a *main* sub-candidate carrying the non-flagged transactions
      (downstream scoring will classify the type — usually probable_salary);
    - a *bonus / expense* sub-candidate carrying the flagged transactions
      (downstream scoring will classify them as bonus_or_extra_pay or
      expense_reimbursement based on the amount model).
    """
    txn_by_id = {t.id: t for t in normalised}
    out: list = []
    next_split_idx = 0
    for cand in pool:
        flagged_ids: list[str] = []
        normal_ids: list[str] = []
        for ct in cand.transactions:
            txn = txn_by_id.get(ct.transaction_id)
            if txn is None:
                normal_ids.append(ct.transaction_id)
                continue
            tokens = set(txn.description_tokens)
            if tokens & _BONUS_TOKENS or tokens & _EXPENSE_TOKENS:
                flagged_ids.append(ct.transaction_id)
            else:
                normal_ids.append(ct.transaction_id)

        if not flagged_ids or len(normal_ids) < 2:
            out.append(cand)
            continue

        main_txns = tuple(t for t in cand.transactions if t.transaction_id in set(normal_ids))
        flagged_txns = tuple(t for t in cand.transactions if t.transaction_id in set(flagged_ids))

        # Decide flagged sub-candidate id / label.
        flagged_is_expense = any(
            set(txn_by_id[i].description_tokens) & _EXPENSE_TOKENS
            for i in flagged_ids
            if i in txn_by_id
        )
        sub_id_suffix = "expense" if flagged_is_expense else "bonus"

        main = replace(
            cand,
            candidate_set_id=f"{cand.candidate_set_id}_main",
            transactions=main_txns,
        )
        flagged = replace(
            cand,
            candidate_set_id=f"{cand.candidate_set_id}_{sub_id_suffix}_{next_split_idx:02d}",
            transactions=flagged_txns,
        )
        next_split_idx += 1
        out.append(main)
        out.append(flagged)
    return tuple(out)


# ---------------------------------------------------------------------------
# Edge-case linking (Design §15)
# ---------------------------------------------------------------------------


def _link_employment_transitions(
    candidates: list[SalaryCandidateSet],
    config: AppConfig,
) -> tuple[list[SalaryCandidateSet], list[ValidationWarning]]:
    """Tag candidate pairs that look like an employment transition
    (Design §15.1) and attach final-pay candidates to the salary
    candidate immediately preceding them in time (Design §15.4).

    Enhancement B2: also emits ValidationWarning records for transitions
    and detects same-month overlaps (mid-month job switches).

    Side-effect-free: returns (updated_candidates, new_warnings).
    """
    extra_warnings: list[ValidationWarning] = []
    if len(candidates) < 2:
        return list(candidates), extra_warnings

    # Bucket by (year, month) of last credit so we can detect adjacency.
    indexed: list[tuple[_date, _date, SalaryCandidateSet]] = []
    for cand in candidates:
        dates = sorted(_date.fromisoformat(t.date) for t in cand.transactions)
        if not dates:
            continue
        indexed.append((dates[0], dates[-1], cand))

    # Collect transition pairs to avoid duplicate warnings.
    emitted_transition_pairs: set[tuple[str, str]] = set()

    out: list[SalaryCandidateSet] = []
    for cand_idx, (first_a, last_a, cand_a) in enumerate(indexed):
        new_meta = dict(cand_a.metadata)
        # Look for candidates that *start* within 60 days after this one
        # *ends* with a different employer — flag a transition.
        for other_idx, (first_b, last_b, cand_b) in enumerate(indexed):
            if other_idx == cand_idx:
                continue
            if first_b < first_a:
                continue
            gap = (first_b - last_a).days
            if gap < 0 or gap > 62:
                continue
            if _same_employer(cand_a, cand_b):
                continue
            if cand_a.candidate_type == "probable_salary" and cand_b.candidate_type == "probable_salary":
                transition_info = {
                    "to_candidate_set_id": cand_b.candidate_set_id,
                    "transition_months": [
                        f"{last_a.year}-{last_a.month:02d}",
                        f"{first_b.year}-{first_b.month:02d}",
                    ],
                    "gap_days": gap,
                }
                new_meta.setdefault("possible_employment_transition", []).append(transition_info)

                # Emit a structured warning once per ordered pair.
                pair_key = (cand_a.candidate_set_id, cand_b.candidate_set_id)
                if pair_key not in emitted_transition_pairs:
                    emitted_transition_pairs.add(pair_key)
                    gap_desc = (
                        f"{gap} day gap"
                        if gap > 0
                        else "back-to-back (same month)"
                    )
                    extra_warnings.append(
                        ValidationWarning(
                            code="employment_transition_detected",
                            transaction_id="",
                            detail=(
                                f"Employment transition detected between "
                                f"{cand_a.candidate_set_id!r} "
                                f"(ends {last_a.year}-{last_a.month:02d}) and "
                                f"{cand_b.candidate_set_id!r} "
                                f"(starts {first_b.year}-{first_b.month:02d}); "
                                f"{gap_desc}."
                            ),
                        )
                    )

                # Detect same-month overlap (mid-month switch).
                if last_a.year == first_b.year and last_a.month == first_b.month:
                    new_meta["same_month_overlap"] = True

        # Attach final-pay candidates to the preceding salary cluster
        # when their date range overlaps the salary's last month.
        if cand_a.candidate_type == "final_pay_candidate":
            for (first_b, last_b, cand_b) in indexed:
                if cand_b is cand_a or cand_b.candidate_type != "probable_salary":
                    continue
                if _same_employer(cand_a, cand_b):
                    if first_a <= last_b + _DAYS(31):
                        new_meta.setdefault("attached_to_salary_candidate", cand_b.candidate_set_id)
                        break

        out.append(replace(cand_a, metadata=new_meta))
    return out, extra_warnings


def _override_split_subcandidate_types(
    candidates: list[SalaryCandidateSet],
) -> list[SalaryCandidateSet]:
    """The bonus / expense splitter creates sub-candidate ids with the
    suffixes ``_bonus_NN`` and ``_expense_NN``. After Phase 8 has finished
    its generic classification cascade, override those sub-candidates'
    types so they land in the canonical bucket from Requirement §6.3
    rather than ``unknown_recurring_credit``.
    """
    out: list[SalaryCandidateSet] = []
    for cand in candidates:
        suffix = cand.candidate_set_id.rsplit("_", 2)
        if len(suffix) >= 3 and suffix[-2] == "bonus":
            out.append(replace(cand, candidate_type="bonus_or_extra_pay_candidate"))
            continue
        if len(suffix) >= 3 and suffix[-2] == "expense":
            out.append(replace(cand, candidate_type="expense_reimbursement_candidate"))
            continue
        out.append(cand)
    return out


def _same_employer(a: SalaryCandidateSet, b: SalaryCandidateSet) -> bool:
    a_tokens = set(a.detected_pattern.possible_employer_tokens)
    b_tokens = set(b.detected_pattern.possible_employer_tokens)
    if not a_tokens or not b_tokens:
        # Treat unknown employer tokens as "could be the same" so we don't
        # over-flag transitions on noisy data.
        return False
    return bool(a_tokens & b_tokens)


# ---------------------------------------------------------------------------
# Input summary helper
# ---------------------------------------------------------------------------


def _build_summary(payload: dict, normalised) -> InputSummary:
    txn_count = len(payload.get("transactions", []))
    credit_count = len(normalised)
    if normalised:
        sorted_dates = sorted(n.date for n in normalised)
        start = sorted_dates[0].isoformat()
        end = sorted_dates[-1].isoformat()
    else:
        start = ""
        end = ""
    return InputSummary(
        transaction_count=txn_count,
        credit_count=credit_count,
        date_range_start=start,
        date_range_end=end,
    )


_ALL_REGIONS = ("EnglandAndWales", "Scotland", "NorthernIreland")
_REGION_INFERENCE_MIN_DELTA = 0.05  # score gap needed to prefer one region over another


def _infer_regions(
    scored: list[SalaryCandidateSet],
    normalised: list,
    cal: WorkingDayCalendar,
    warnings: list,
) -> tuple[tuple[str, ...], list[SalaryCandidateSet], list]:
    """Enhancement A — infer which UK calendar region(s) best explain the
    salary date pattern when the caller did not supply ``country_region``.

    For each monthly candidate, re-fits the best recurrence model under each
    of the three UK regions and compares scores.  If one region scores
    materially better (≥ REGION_INFERENCE_MIN_DELTA), it is tagged as the
    inferred region.  If multiple regions tie, all are listed.  If the
    regions conflict across candidates, an ``ambiguous_region`` warning is
    emitted.

    Returns (inferred_regions_union, updated_scored, updated_warnings).
    """
    from .recurrence_models import fit_best_recurrence_model, PaydayModel

    out_candidates: list[SalaryCandidateSet] = []
    per_candidate_regions: list[set[str]] = []

    for cand in scored:
        cluster_ids = {t.transaction_id for t in cand.transactions}
        cluster = [t for t in normalised if t.id in cluster_ids]
        if len(cluster) < 2:
            out_candidates.append(cand)
            per_candidate_regions.append(set())
            continue

        # Only run region inference for monthly-ish candidates.
        freq = cand.detected_pattern.frequency
        if freq not in ("monthly", "unknown"):
            out_candidates.append(cand)
            per_candidate_regions.append(set())
            continue

        region_scores: dict[str, float] = {}
        for region in _ALL_REGIONS:
            fit = fit_best_recurrence_model(cluster, calendar=cal, region=region)
            # Only compare monthly models — periodic models are region-agnostic.
            if fit.model not in (
                PaydayModel.WEEKLY_SAME_WEEKDAY,
                PaydayModel.FORTNIGHTLY_SAME_WEEKDAY,
                PaydayModel.FOUR_WEEKLY,
                PaydayModel.IRREGULAR_SALARY_LIKE,
            ):
                region_scores[region] = fit.fit_score
            else:
                region_scores[region] = 0.0

        if not region_scores or max(region_scores.values()) == 0.0:
            out_candidates.append(cand)
            per_candidate_regions.append(set())
            continue

        best_score = max(region_scores.values())
        # Regions within delta of the best are all equally valid.
        candidate_regions = {
            r for r, s in region_scores.items()
            if best_score - s <= _REGION_INFERENCE_MIN_DELTA
        }

        new_meta = dict(cand.metadata)
        new_meta["inferred_regions"] = sorted(candidate_regions)
        new_meta["region_fit_scores"] = {r: round(s, 4) for r, s in region_scores.items()}
        out_candidates.append(replace(cand, metadata=new_meta))
        per_candidate_regions.append(candidate_regions)

    # Union of inferred regions across all candidates.
    union_regions: set[str] = set()
    for r_set in per_candidate_regions:
        union_regions |= r_set

    # Detect conflict: if two candidates disagree and neither is a superset.
    non_empty = [r for r in per_candidate_regions if r]
    conflict = False
    if len(non_empty) >= 2:
        all_intersection = set.intersection(*non_empty)
        if not all_intersection:
            conflict = True

    updated_warnings = list(warnings)
    if conflict:
        updated_warnings.append(
            ValidationWarning(
                code="ambiguous_region",
                transaction_id="",
                detail=(
                    "Calendar region could not be determined uniquely — different "
                    "candidate sets best match different UK regions. "
                    f"Regions seen: {sorted(union_regions)}."
                ),
            )
        )

    return tuple(sorted(union_regions)), out_candidates, updated_warnings


def _early_exit(
    analysis_id: str,
    warnings: list,
    audit,
    include_audit,
    jurisdiction: str = "GB",
    country_region: str = "EnglandAndWales",
) -> SalaryDetectionResult:
    """Return an empty result when the pipeline cannot proceed (e.g. no transactions)."""
    return SalaryDetectionResult(
        analysis_id=analysis_id,
        jurisdiction=jurisdiction,
        country_region=country_region,
        input_summary=InputSummary(
            transaction_count=0,
            credit_count=0,
            date_range_start="",
            date_range_end="",
        ),
        candidate_sets=(),
        rejected_or_near_miss_sets=(),
        warnings=tuple(warnings),
        applied_hints=None,
        audit=audit.snapshot() if include_audit else None,
    )


# Tiny helper to keep the calendar dependency out of the linker.
def _DAYS(n: int):
    from datetime import timedelta

    return timedelta(days=n)


__all__ = [
    "detect_salary_candidates",
]
