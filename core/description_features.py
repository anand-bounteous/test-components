"""Description feature extraction — Design §6.3, §8, FR-009.

For one normalised transaction this module identifies:

- ``salary_hint_tokens``    — words that lift salary likelihood (PAYROLL, SALARY…)
- ``negative_tokens``       — words that lower salary likelihood (HMRC, DWP, …)
- ``possible_employer_tokens`` — words that look employer-like (proper nouns
  after the noise has been removed). These survive into clustering as the
  primary similarity signal.

The implementation reads from ``AppConfig`` so the lists stay configurable
per Requirement §8.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .config import AppConfig
from .normalisation import _GENERIC_STOP_TOKENS  # type: ignore[attr-defined]


# Channel-marker tokens we do NOT want bubbling up as "employer tokens".
# We seed with a baseline; the active config layers contribute more channel
# tokens at runtime via ``DescriptionFeatureExtractor``.
_CHANNEL_TOKEN_BASELINE = frozenset(
    {
        "BACS",
        "BAC",
        "BGC",
        "FPI",
        "FPS",
        "FP",
        "CHAPS",
        "CHP",
        "CHQ",
        "CASH",
        "CDM",
        "TLR",
        "STO",
        "ITL",
        "IBAN",
        "SWIFT",
        "DC",
        "TFR",
    }
)

# Action/role hint tokens; not employer signals.
_HINT_TOKEN_BASELINE = frozenset(
    {
        "PAYROLL",
        "SALARY",
        "WAGE",
        "WAGES",
        "PAY",
        "STAFF",
        "MONTHLY",
        "HR",
        "EMPLOYEE",
        "EMPLOYER",
        "CREDIT",
        "PAYMENT",
        "DIRECT",
        "AUTOMATED",
        "ONLINE",
    }
)

# Token "shapes" that are never employer signals — short generic verbs like
# `LTD` / `PLC` and pure-number tokens fall under this.
_LEGAL_SUFFIX_TOKENS = frozenset({"LTD", "PLC", "LLP", "LLC", "INC", "GROUP"})


@dataclass(frozen=True)
class DescriptionFeatureResult:
    salary_hint_tokens: tuple[str, ...] = ()
    salary_hint_strengths: dict[str, str] = field(default_factory=dict)
    negative_tokens: tuple[str, ...] = ()
    negative_categories: tuple[str, ...] = ()
    possible_employer_tokens: tuple[str, ...] = ()


@dataclass
class DescriptionFeatureExtractor:
    config: AppConfig
    # Single-token hint lookup (e.g. PAYROLL → strong).
    _hint_lookup: dict[str, str] = field(default_factory=dict, init=False)
    # Multi-token hint phrases as (phrase tuple, strength).
    _hint_phrases: list[tuple[tuple[str, ...], str]] = field(default_factory=list, init=False)
    # Flat list of (phrase tokens, category) tuples for negative signals.
    _negative_phrases: list[tuple[tuple[str, ...], str]] = field(default_factory=list, init=False)
    _channel_stop: frozenset[str] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        hints = self.config.salary_hints_keywords()
        for strength_key in ("strong", "medium", "weak"):
            for tok in hints.get(strength_key, []) or []:
                upper = tok.upper().strip()
                if not upper:
                    continue
                pieces = tuple(upper.split())
                if len(pieces) == 1:
                    self._hint_lookup[upper] = strength_key
                else:
                    self._hint_phrases.append((pieces, strength_key))
        # Longer phrases first so STAFF PAY beats PAY when both could match.
        self._hint_phrases.sort(key=lambda x: (-len(x[0]), x[1]))

        for category, tokens in self.config.negative_signals().items():
            for tok in tokens or []:
                self._negative_phrases.append((tuple(tok.upper().split()), category))
        # Longer phrases first so "OWN ACCOUNT" wins over "OWN".
        self._negative_phrases.sort(key=lambda x: (-len(x[0]), x[1]))

        # Channel stop list grows with whatever the active config (incl
        # bank overlay) calls out as channel tokens.
        extra_channel: set[str] = set(_CHANNEL_TOKEN_BASELINE)
        for channel_payload in self.config.channels().values():
            for tok in channel_payload.get("positive_tokens", []):
                for piece in tok.upper().split():
                    extra_channel.add(piece)
        object.__setattr__(self, "_channel_stop", frozenset(extra_channel))

    # --- Public ---------------------------------------------------------

    def extract(self, description_tokens: tuple[str, ...]) -> DescriptionFeatureResult:
        if not description_tokens:
            return DescriptionFeatureResult()

        hint_tokens: list[str] = []
        hint_strengths: dict[str, str] = {}
        token_index_for_hints = {t: i for i, t in enumerate(description_tokens)}
        # Match multi-word phrases first so longer wins.
        for phrase, strength in self._hint_phrases:
            if self._phrase_present(description_tokens, phrase, token_index_for_hints):
                joined = " ".join(phrase)
                if joined not in hint_strengths:
                    hint_tokens.append(joined)
                    hint_strengths[joined] = strength
        for tok in description_tokens:
            strength = self._hint_lookup.get(tok)
            if strength is not None and tok not in hint_strengths:
                hint_tokens.append(tok)
                hint_strengths[tok] = strength

        negative_tokens: list[str] = []
        negative_categories: list[str] = []
        token_index = {t: i for i, t in enumerate(description_tokens)}
        for phrase, category in self._negative_phrases:
            if self._phrase_present(description_tokens, phrase, token_index):
                joined = " ".join(phrase)
                if joined not in negative_tokens:
                    negative_tokens.append(joined)
                if category not in negative_categories:
                    negative_categories.append(category)

        employer_tokens = self._employer_tokens(description_tokens, hint_strengths)

        return DescriptionFeatureResult(
            salary_hint_tokens=tuple(hint_tokens),
            salary_hint_strengths=hint_strengths,
            negative_tokens=tuple(negative_tokens),
            negative_categories=tuple(negative_categories),
            possible_employer_tokens=employer_tokens,
        )

    # --- Internal -------------------------------------------------------

    def _employer_tokens(
        self,
        tokens: Iterable[str],
        hint_strengths: dict[str, str],
    ) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            if t in seen:
                continue
            if t in hint_strengths:
                continue
            if t in self._channel_stop:
                continue
            if t in _HINT_TOKEN_BASELINE:
                continue
            if t in _LEGAL_SUFFIX_TOKENS:
                continue
            if t in _GENERIC_STOP_TOKENS:
                continue
            if t.isdigit():
                continue
            if len(t) < 2:
                continue
            seen.add(t)
            out.append(t)
        return tuple(out)

    @staticmethod
    def _phrase_present(
        tokens: tuple[str, ...],
        phrase: tuple[str, ...],
        token_index: dict[str, int],
    ) -> bool:
        if len(phrase) == 1:
            return phrase[0] in token_index
        n, m = len(tokens), len(phrase)
        if m > n:
            return False
        for i in range(n - m + 1):
            if tokens[i : i + m] == phrase:
                return True
        return False
