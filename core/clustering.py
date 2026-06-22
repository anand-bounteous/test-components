"""Similarity-graph clustering — Design §10, the advanced candidate path.

Steps:

1. **Pre-grouping** — bucket credits by month / employer-token / amount band
   so the graph build does not explode to all-pairs comparisons.
2. **Edge build** — for every pair of credits inside the same pre-group
   compute the Design §10.2 edge score:

   ```
   edge_score = 0.35 * description_similarity
              + 0.25 * date_compatibility
              + 0.15 * amount_similarity
              + 0.15 * payment_channel_compatibility
              + 0.10 * counterparty_similarity
              - negative_conflict_penalty
   ```

3. **Cluster** — connected components over the threshold-pruned graph.
4. **Split / merge** — apply Design §10.4 heuristics: merge when employer
   tokens match across changed descriptions; split when two distinct
   employer-like token sets are present.
5. **Emit** — produce ``SalaryCandidateSet`` records with
   ``metadata.source = "graph_clustering"``.

A separate ``dedup_candidate_pool`` function merges skeleton-path and
graph-path candidates by transaction-id overlap (Design §10.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from ..models.candidate import (
    CandidateTransaction,
    DetectedPattern,
    SalaryCandidateSet,
    candidate_transaction_from,
)
from ..models.hints import AppliedHints
from ..models.score import ScoreBreakdown
from ..models.transaction import NormalisedTransaction, TransactionFeatures
from .config import AppConfig
from .pattern_induction import induce_pattern, matches_pattern

# ---------------------------------------------------------------------------
# Edge scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeBreakdown:
    description: float
    date: float
    amount: float
    channel: float
    counterparty: float
    negative_penalty: float
    final: float


def description_similarity(a_tokens: tuple[str, ...], b_tokens: tuple[str, ...]) -> float:
    """Symmetric Jaccard over description tokens — fast, dependency-free,
    and good enough for the Phase 7 scope. Stop-word filtering already
    happened during normalisation."""
    if not a_tokens and not b_tokens:
        return 0.0
    set_a, set_b = set(a_tokens), set(b_tokens)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def token_sequence_match_score(
    a_tokens: tuple[str, ...],
    b_tokens: tuple[str, ...],
) -> float:
    """LCS-ratio score for token ordering similarity (Enhancement C).

    Returns the length of the longest common subsequence divided by the
    length of the longer sequence.  A value of 1.0 means the tokens appear
    in exactly the same order; lower values indicate reordering.

    The score is used as a sub-component of description_similarity_with_sequence
    so that clusters where the same tokens appear in different orders are
    kept together (Jaccard stays high) but receive a small confidence penalty.
    """
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    m, n = len(a_tokens), len(b_tokens)
    # DP LCS — O(mn) but clusters are small (< 50 tokens typically).
    prev = [0] * (n + 1)
    for tok in a_tokens:
        curr = [0] * (n + 1)
        for j, btok in enumerate(b_tokens):
            if tok == btok:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    lcs_len = prev[n]
    return lcs_len / max(m, n)


def description_similarity_with_sequence(
    a_tokens: tuple[str, ...],
    b_tokens: tuple[str, ...],
) -> float:
    """Blended description score: 90% Jaccard set-similarity + 10% LCS ordering.

    Tokens in different orders receive a small penalty (≤ 10% of the
    description component) while remaining in the same cluster — per the
    design decision to keep the group unified rather than split on order alone.
    """
    jaccard = description_similarity(a_tokens, b_tokens)
    seq = token_sequence_match_score(a_tokens, b_tokens)
    return round(0.90 * jaccard + 0.10 * seq, 6)


def date_compatibility(a, b) -> float:
    """How "monthly-cadent" are these two dates relative to each other?

    - Same calendar month → low (likely same-month duplicates, not
      different paydays).
    - Gap close to 28–32 days → high (monthly).
    - Gap close to 7 / 14 / 28 days → high (periodic).
    - Anything else → low.
    """
    if a == b:
        return 0.0
    gap = abs((a - b).days)
    if 26 <= gap <= 32 or 12 <= gap <= 16 or 5 <= gap <= 9:
        return 1.0
    if 24 <= gap <= 34 or 10 <= gap <= 18 or 3 <= gap <= 11:
        return 0.6
    # Same-month pairs get the partial-mismatch score so the graph still
    # builds within-month edges for skeleton refinement.
    if a.year == b.year and a.month == b.month:
        return 0.3
    return 0.1


def amount_similarity(a: Decimal, b: Decimal, *, tolerance_pct: float = 0.10) -> float:
    """How similar are the two amounts?

    1.0 inside ±tolerance; linear decay outside; clamped at 0."""
    if a == 0 and b == 0:
        return 1.0
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    diff = abs(a - b)
    relative = float(diff) / float(base)
    if relative <= tolerance_pct:
        return 1.0
    # Decay 1.0 → 0.0 across one full magnitude.
    return max(0.0, 1.0 - (relative - tolerance_pct))


def payment_channel_compatibility(a: str, b: str) -> float:
    if a == b and a != "unknown_credit":
        return 1.0
    if "unknown_credit" in {a, b}:
        return 0.5
    return 0.2


def counterparty_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.4
    if a.strip().upper() == b.strip().upper():
        return 1.0
    return 0.2


def _negative_conflict_penalty(a_neg: tuple[str, ...], b_neg: tuple[str, ...]) -> float:
    """If exactly one side carries a negative signal, the pair is
    suspicious."""
    has_a, has_b = bool(a_neg), bool(b_neg)
    if has_a ^ has_b:
        return 0.20
    return 0.0


def compute_edge_score(
    txn_a: NormalisedTransaction,
    txn_b: NormalisedTransaction,
    feat_a: TransactionFeatures,
    feat_b: TransactionFeatures,
) -> EdgeBreakdown:
    desc = description_similarity_with_sequence(txn_a.description_tokens, txn_b.description_tokens)
    date_c = date_compatibility(txn_a.date, txn_b.date)
    amount = amount_similarity(txn_a.amount, txn_b.amount)
    channel = payment_channel_compatibility(feat_a.payment_channel, feat_b.payment_channel)
    cparty = counterparty_similarity(txn_a.counterparty_name, txn_b.counterparty_name)
    penalty = _negative_conflict_penalty(feat_a.negative_tokens, feat_b.negative_tokens)
    final = (
        0.35 * desc
        + 0.25 * date_c
        + 0.15 * amount
        + 0.15 * channel
        + 0.10 * cparty
        - penalty
    )
    return EdgeBreakdown(
        description=round(desc, 4),
        date=round(date_c, 4),
        amount=round(amount, 4),
        channel=round(channel, 4),
        counterparty=round(cparty, 4),
        negative_penalty=round(penalty, 4),
        final=round(final, 4),
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class GraphClusteringDetector:
    config: AppConfig
    _edge_threshold: float = field(init=False, default=0.55)

    def __post_init__(self) -> None:
        cg = self.config.candidate_generation()
        self._edge_threshold = float(cg.get("graph_edge_threshold", 0.55))

    # --- Public API ------------------------------------------------------

    def build_graph(
        self,
        credits: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
    ) -> dict[tuple[str, str], EdgeBreakdown]:
        """Return the edges (above threshold) as
        ``{(id_a, id_b): EdgeBreakdown}`` with ``id_a < id_b``.

        Pre-grouping (Design §10.2): credits are first bucketed by each
        of their ``possible_employer_tokens``; credits with no employer
        tokens fall back to a skeleton-keyed bucket. Two credits become
        an edge candidate when they share **at least one** token bucket;
        cross-bucket duplicates are de-duped so each pair is scored at
        most once.

        This keeps the payroll-provider-change scenario working — ADP-
        style and SAGE-style descriptions still share the ACME employer
        token and so still get compared — while cutting the comparison
        count on 5,000-credit inputs from ~12 M to a small multiple of
        the largest bucket.
        """
        edges: dict[tuple[str, str], EdgeBreakdown] = {}
        buckets: dict[str, list[NormalisedTransaction]] = {}
        for c in credits:
            f = features_by_id.get(c.id)
            tokens = list(f.possible_employer_tokens) if f else []
            if tokens:
                for tok in tokens:
                    buckets.setdefault(tok, []).append(c)
            else:
                buckets.setdefault(f"__skel__::{c.skeleton}", []).append(c)

        # Phase 13: pattern-induction pre-grouping. When credits share a
        # long character-level prefix / suffix despite no overlapping
        # tokens, group them so the all-pairs edge build still sees them.
        # Capped at 100 credits to keep runtime safe — the per-bucket
        # loop below is O(n²) within each bucket.
        if len(credits) <= 100:
            pattern = induce_pattern(
                [c.raw_description for c in credits],
                min_support=0.5,
                min_stable_chars=8,
            )
            if pattern is not None:
                bucket = [c for c in credits if matches_pattern(c.raw_description, pattern)]
                if len(bucket) >= 2:
                    buckets.setdefault(f"__pattern__::{pattern.signature}", []).extend(bucket)

        seen_pairs: set[tuple[str, str]] = set()
        for members in buckets.values():
            n = len(members)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = members[i], members[j]
                    pair = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    fa = features_by_id.get(a.id)
                    fb = features_by_id.get(b.id)
                    if fa is None or fb is None:
                        continue
                    edge = compute_edge_score(a, b, fa, fb)
                    if edge.final >= self._edge_threshold:
                        edges[pair] = edge
        return edges

    def cluster(
        self,
        credits: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
    ) -> list[list[NormalisedTransaction]]:
        edges = self.build_graph(credits, features_by_id)
        return self._connected_components(credits, edges)

    def emit_candidates(
        self,
        credits: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
        *,
        applied_hints: Optional[AppliedHints] = None,
    ) -> tuple[SalaryCandidateSet, ...]:
        clusters = self.cluster(credits, features_by_id)
        # Apply Design §10.4 split/merge using employer tokens.
        clusters = self._split_distinct_employers(clusters, features_by_id)
        candidates: list[SalaryCandidateSet] = []
        anchor_txn_ids = set()
        if applied_hints:
            anchor_txn_ids = {a.transaction_id for a in applied_hints.anchors}
        for idx, members in enumerate(clusters):
            if len(members) < 2:
                continue
            candidates.append(
                self._build_candidate(idx, members, features_by_id, anchor_txn_ids)
            )
        return tuple(candidates)

    # --- Internal --------------------------------------------------------

    @staticmethod
    def _connected_components(
        credits: Sequence[NormalisedTransaction],
        edges: dict[tuple[str, str], EdgeBreakdown],
    ) -> list[list[NormalisedTransaction]]:
        parent = {c.id: c.id for c in credits}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for (a, b) in edges.keys():
            union(a, b)

        groups: dict[str, list[NormalisedTransaction]] = {}
        for c in credits:
            root = find(c.id)
            groups.setdefault(root, []).append(c)

        # Sort clusters deterministically: by min transaction id within
        # the cluster, then by the credit count descending.
        ordered = sorted(groups.values(), key=lambda g: (min(t.id for t in g),))
        return [sorted(g, key=lambda t: t.date) for g in ordered]

    @staticmethod
    def _split_distinct_employers(
        clusters: list[list[NormalisedTransaction]],
        features_by_id: dict[str, TransactionFeatures],
    ) -> list[list[NormalisedTransaction]]:
        """Design §10.4: if a cluster contains two DISTINCT strong
        employer-token groups with no overlap, split them.
        """
        out: list[list[NormalisedTransaction]] = []
        for cluster in clusters:
            token_groups = _group_by_employer_tokens(cluster, features_by_id)
            if len(token_groups) <= 1:
                out.append(cluster)
                continue
            # Sort each subgroup deterministically before re-emitting.
            for members in token_groups:
                out.append(sorted(members, key=lambda t: t.date))
        return out

    def _build_candidate(
        self,
        idx: int,
        members: Sequence[NormalisedTransaction],
        features_by_id: dict[str, TransactionFeatures],
        anchor_txn_ids: set[str],
    ) -> SalaryCandidateSet:
        sorted_members = sorted(members, key=lambda t: t.date)
        txns: tuple[CandidateTransaction, ...] = tuple(
            candidate_transaction_from(t, role="main_salary") for t in sorted_members
        )

        token_sets = [
            set(features_by_id[t.id].possible_employer_tokens)
            for t in sorted_members
            if t.id in features_by_id and features_by_id[t.id].possible_employer_tokens
        ]
        common = sorted(set.intersection(*token_sets)) if token_sets else []

        channel_counts: dict[str, int] = {}
        for t in sorted_members:
            f = features_by_id.get(t.id)
            if f is not None:
                channel_counts[f.payment_channel] = channel_counts.get(f.payment_channel, 0) + 1
        dominant_channel = (
            max(channel_counts.items(), key=lambda kv: kv[1])[0]
            if channel_counts
            else "unknown_credit"
        )

        # Quick confidence — Phase 8 will recompute, but we want a
        # sensible interim value so this candidate sorts correctly when it
        # flows into the dedup pool.
        anchor_present = any(t.id in anchor_txn_ids for t in sorted_members)
        size_score = min(1.0, len(sorted_members) / 12.0)
        interim_confidence = 0.50 + 0.25 * size_score + (0.15 if anchor_present else 0.0)
        interim_confidence = round(min(1.0, interim_confidence), 4)

        pattern = DetectedPattern(
            frequency="monthly_candidate",
            payday_model="unknown_pending_phase8",
            payment_channel=dominant_channel,
            amount_model="unknown_pending_phase8",
            possible_employer_tokens=tuple(common),
            coverage=f"{len(sorted_members)}_credits",
        )

        # Phase 13: stamp the induced pattern signature (if any) on the
        # candidate so the LLM payload can surface it.
        induced = induce_pattern(
            [t.description for t in txns],
            min_support=0.6,
            min_stable_chars=8,
        )
        induced_signature = induced.signature if induced else None

        return SalaryCandidateSet(
            candidate_set_id=f"graph_{idx:03d}",
            candidate_type="unknown_recurring_credit",
            transactions=txns,
            score_breakdown=ScoreBreakdown(),
            detected_pattern=pattern,
            reasoning=(
                f"Graph clustering grouped {len(sorted_members)} credits with "
                f"common employer tokens {common!r} on the {dominant_channel} channel.",
            ),
            risks=(),
            confidence=interim_confidence,
            confidence_band=_band_for_score(interim_confidence),
            llm_review_recommendation="send_to_llm" if interim_confidence >= 0.55 else "skip",
            metadata={
                "source": "graph_clustering",
                "anchor_in_cluster": anchor_present,
                "edge_threshold": self._edge_threshold,
                **(
                    {"induced_pattern_signature": induced_signature}
                    if induced_signature
                    else {}
                ),
            },
        )


def _group_by_employer_tokens(
    cluster: Sequence[NormalisedTransaction],
    features_by_id: dict[str, TransactionFeatures],
) -> list[list[NormalisedTransaction]]:
    """Partition a cluster into sub-groups by employer-token overlap.

    Two transactions live in the same sub-group if their possible_employer_tokens
    sets intersect (or both are empty).
    """
    groups: list[tuple[set[str], list[NormalisedTransaction]]] = []
    for txn in cluster:
        feat = features_by_id.get(txn.id)
        tokens = set(feat.possible_employer_tokens) if feat else set()
        placed = False
        for tokens_set, members in groups:
            if (tokens and tokens_set and tokens & tokens_set) or (not tokens and not tokens_set):
                tokens_set |= tokens
                members.append(txn)
                placed = True
                break
        if not placed:
            groups.append((set(tokens), [txn]))
    return [members for _, members in groups]


# ---------------------------------------------------------------------------
# Pool de-duplication with skeleton candidates
# ---------------------------------------------------------------------------


def dedup_candidate_pool(
    skeleton_candidates: Sequence[SalaryCandidateSet],
    graph_candidates: Sequence[SalaryCandidateSet],
    *,
    overlap_threshold: float = 0.90,
) -> tuple[SalaryCandidateSet, ...]:
    """Merge skeleton and graph candidates by transaction-id overlap.

    Two candidates are considered the same income stream when ≥
    ``overlap_threshold`` of the smaller candidate's transaction-ids
    appear in the larger one. The merged candidate:

    - keeps the **union** of transactions (so a graph cluster that
      spans two skeleton groups absorbs both);
    - keeps the **larger** candidate's pattern + identity (more credits =
      more evidence);
    - records every merged source in ``metadata.merged_sources`` for the
      audit trail.

    The merge cascades: each incoming candidate is checked against every
    existing survivor; on any match all overlapping survivors collapse
    into the new candidate.
    """
    pool: list[SalaryCandidateSet] = list(skeleton_candidates) + list(graph_candidates)
    if not pool:
        return ()
    # Sort the input by descending transaction count so larger candidates
    # are the anchors that absorb smaller subsets.
    pool.sort(key=lambda c: (-len(c.transactions), -c.confidence, c.candidate_set_id))

    survivors: list[SalaryCandidateSet] = []
    for cand in pool:
        cand_ids = {t.transaction_id for t in cand.transactions}
        overlapping_indices: list[int] = []
        for i, survivor in enumerate(survivors):
            sid = {t.transaction_id for t in survivor.transactions}
            if _set_overlap(cand_ids, sid) >= overlap_threshold:
                overlapping_indices.append(i)

        if not overlapping_indices:
            survivors.append(cand)
            continue

        absorbed = [survivors[i] for i in overlapping_indices]
        merged = _absorb_all(cand, absorbed)
        for i in sorted(overlapping_indices, reverse=True):
            del survivors[i]
        survivors.append(merged)

    return tuple(sorted(survivors, key=lambda c: (-c.confidence, c.candidate_set_id)))


def _set_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return intersection / smaller


def _absorb_all(
    primary: SalaryCandidateSet,
    others: Sequence[SalaryCandidateSet],
) -> SalaryCandidateSet:
    """Combine ``primary`` and every entry of ``others`` into one record.

    The candidate with the most transactions wins identity (id, type,
    pattern). Reasoning lists are concatenated; risks are unioned;
    metadata ``merged_sources`` records every contributing source.
    Transactions are unioned (de-duped by ``transaction_id``); confidence
    is taken from the highest-confidence contributor.
    """
    everything: list[SalaryCandidateSet] = [primary, *others]
    # Identity: pick the largest by transaction count, then highest confidence.
    identity = max(
        everything,
        key=lambda c: (len(c.transactions), c.confidence, c.candidate_set_id),
    )
    # Union of transactions, preserving order from identity then appending
    # any not already present.
    seen: set[str] = set()
    merged_txns: list[CandidateTransaction] = []
    for source in (identity, *[c for c in everything if c is not identity]):
        for t in source.transactions:
            if t.transaction_id in seen:
                continue
            seen.add(t.transaction_id)
            merged_txns.append(t)
    # Reasoning: concatenate.
    merged_reasoning: list[str] = []
    for c in everything:
        for r in c.reasoning:
            if r not in merged_reasoning:
                merged_reasoning.append(r)
    merged_risks = tuple(sorted({risk for c in everything for risk in c.risks}))
    merged_meta = dict(identity.metadata)
    sources = [c.metadata.get("source", "unknown") for c in everything]
    merged_meta["merged_sources"] = sources
    confidence = max(c.confidence for c in everything)
    return SalaryCandidateSet(
        candidate_set_id=identity.candidate_set_id,
        candidate_type=identity.candidate_type,
        transactions=tuple(merged_txns),
        score_breakdown=identity.score_breakdown,
        detected_pattern=identity.detected_pattern,
        reasoning=tuple(merged_reasoning),
        risks=merged_risks,
        confidence=confidence,
        confidence_band=_band_for_score(confidence),
        llm_review_recommendation=identity.llm_review_recommendation,
        metadata=merged_meta,
    )


def _band_for_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium_high"
    if score >= 0.55:
        return "medium"
    if score >= 0.40:
        return "low"
    return "very_low"


__all__ = [
    "EdgeBreakdown",
    "GraphClusteringDetector",
    "amount_similarity",
    "compute_edge_score",
    "counterparty_similarity",
    "date_compatibility",
    "dedup_candidate_pool",
    "description_similarity",
    "payment_channel_compatibility",
]
