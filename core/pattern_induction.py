"""Pattern induction — Phase 13.

When a candidate cluster's descriptions share a long character-level
prefix or suffix but don't overlap at the token level, the token-based
similarity (Jaccard / employer-token bucketing) fails to recognise them
as the same income stream. This module exposes a pragmatic LCS-based
``induce_pattern`` that finds the longest common prefix + suffix across a
list of descriptions and returns a canonical ``{prefix}*{suffix}``
signature.

Example — six ACH-style salary credits whose tokens diverge but whose
character prefix is stable:

```
ACH C-SAL-ABCCorpINDL-ADISALJAN26-b4036e11cf45
ACH C-SAL-ABCCorpINDL-ADISALFEB26-d0cdd68ddafa
...
```

→ ``InducedPattern(prefix="ACH C-SAL-ABCCorpINDL-ADISAL", suffix="",
                     signature="ACH C-SAL-ABCCorpINDL-ADISAL*",
                     support_ratio=1.0)``

Phase 7 graph clustering uses the ``signature`` as an additional
pre-grouping key and as an extra ``pattern_similarity`` axis on the edge
score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class InducedPattern:
    """A `{prefix}*{suffix}` shape inferred from a set of descriptions."""

    prefix: str
    suffix: str
    variable_centres: tuple[str, ...]
    signature: str
    support_ratio: float


# Default thresholds — tuned for real UK bank descriptions where employer
# tokens typically sit in the first ~20 chars.
_DEFAULT_MIN_SUPPORT = 0.6
_DEFAULT_MIN_STABLE_CHARS = 6


# --- LCP / LCS helpers ---------------------------------------------------


def longest_common_prefix(strings: Sequence[str]) -> str:
    """Longest character prefix shared by every string in ``strings``.

    Returns ``""`` for an empty sequence or when the first characters
    don't match.
    """
    if not strings:
        return ""
    n = min(len(s) for s in strings)
    i = 0
    while i < n and all(s[i] == strings[0][i] for s in strings):
        i += 1
    return strings[0][:i]


def longest_common_suffix(strings: Sequence[str]) -> str:
    """Longest character suffix shared by every string."""
    if not strings:
        return ""
    n = min(len(s) for s in strings)
    i = 0
    while i < n and all(s[-1 - i] == strings[0][-1 - i] for s in strings):
        i += 1
    return strings[0][len(strings[0]) - i:] if i else ""


def longest_common_substring_pair(a: str, b: str) -> str:
    """Longest common substring between two strings (anywhere in the
    string). Standard O(|a|·|b|) dynamic programming — fine for cluster-
    sized inputs (we never call this on 5,000 strings)."""
    if not a or not b:
        return ""
    la, lb = len(a), len(b)
    # Rolling-row DP keeps memory at O(min(la, lb)).
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = [0] * (lb + 1)
    curr = [0] * (lb + 1)
    best_len = 0
    best_end = 0
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end = i
            else:
                curr[j] = 0
        prev, curr = curr, prev
        for j in range(lb + 1):
            curr[j] = 0
    return a[best_end - best_len:best_end]


# --- Induction -----------------------------------------------------------


def induce_pattern(
    descriptions: Sequence[str],
    *,
    min_support: float = _DEFAULT_MIN_SUPPORT,
    min_stable_chars: int = _DEFAULT_MIN_STABLE_CHARS,
) -> Optional[InducedPattern]:
    """Find the longest common prefix + suffix supported by at least
    ``min_support`` fraction of ``descriptions``.

    Returns ``None`` when the combined stable parts (prefix + suffix)
    are shorter than ``min_stable_chars`` — the pattern isn't strong
    enough to bucket on.
    """
    if not descriptions:
        return None
    cleaned = [s for s in descriptions if s]
    if not cleaned:
        return None

    # Start with the prefix shared by ALL strings. If too short, try a
    # support-relaxed version: pick the majority prefix.
    prefix = longest_common_prefix(cleaned)
    suffix = longest_common_suffix(cleaned)

    if len(prefix) + len(suffix) < min_stable_chars and min_support < 1.0:
        # Try to find a prefix that ≥ min_support of strings share.
        prefix = _majority_prefix(cleaned, min_support)
        suffix = _majority_suffix(cleaned, min_support)

    stable = len(prefix) + len(suffix)
    if stable < min_stable_chars:
        return None

    # Compute support — fraction of input strings that begin with `prefix`
    # AND end with `suffix`.
    support_count = sum(
        1 for s in cleaned if s.startswith(prefix) and s.endswith(suffix)
    )
    support_ratio = support_count / len(cleaned)
    if support_ratio < min_support:
        return None

    # Variable centres — what's left between prefix and suffix in each
    # matching description.
    centres: list[str] = []
    for s in cleaned:
        if s.startswith(prefix) and s.endswith(suffix):
            start = len(prefix)
            end = len(s) - len(suffix) if suffix else len(s)
            centres.append(s[start:end])

    signature = f"{prefix}*{suffix}" if suffix else f"{prefix}*"
    return InducedPattern(
        prefix=prefix,
        suffix=suffix,
        variable_centres=tuple(centres),
        signature=signature,
        support_ratio=round(support_ratio, 6),
    )


def matches_pattern(description: str, pattern: InducedPattern) -> bool:
    """Does ``description`` fit the induced `{prefix}*{suffix}` shape?"""
    if not description:
        return False
    if not description.startswith(pattern.prefix):
        return False
    if pattern.suffix and not description.endswith(pattern.suffix):
        return False
    return True


# --- Internal majority-prefix helpers -----------------------------------


def _majority_prefix(strings: Sequence[str], min_support: float) -> str:
    """Find a prefix shared by ≥ min_support of strings, taking the
    longest such prefix. Falls back to the empty string when no prefix
    meets the bar."""
    if not strings:
        return ""
    target = max(1, int(round(min_support * len(strings))))
    # Compute the longest prefix from each pair of consecutive sorted
    # strings — useful for adversarial sets. For typical inputs the
    # all-strings LCP is already a prefix of the majority.
    sorted_strings = sorted(strings)
    best = ""
    for i in range(len(sorted_strings)):
        for j in range(i + target - 1, len(sorted_strings)):
            candidate = longest_common_prefix([sorted_strings[i], sorted_strings[j]])
            if len(candidate) > len(best):
                # Verify support across the full list.
                count = sum(1 for s in sorted_strings if s.startswith(candidate))
                if count >= target:
                    best = candidate
    return best


def _majority_suffix(strings: Sequence[str], min_support: float) -> str:
    reversed_strings = [s[::-1] for s in strings]
    rev_prefix = _majority_prefix(reversed_strings, min_support)
    return rev_prefix[::-1]


__all__ = [
    "InducedPattern",
    "induce_pattern",
    "longest_common_prefix",
    "longest_common_substring_pair",
    "longest_common_suffix",
    "matches_pattern",
]
