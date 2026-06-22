"""Payment-channel detection — Design §5.1, §11.2.

Matches the longest configured token first. Token search uses **word
boundaries** on the normalised description, so a short alias like ``BAC`` in
``BACS`` does not falsely match — Phase 1's normaliser tokenises on
whitespace, so the match is implemented against the tuple of tokens, not
the raw string.

Confidence reflects two things:

1. **Specificity** — longer tokens (multi-word phrases) score higher than
   short abbreviations.
2. **Bank-specific match strength** — if the caller passes a ``source_bank``
   to ``load_config`` and the matched token came from the bank's overlay,
   confidence is bumped.

The detector always returns *one* canonical channel (the highest-scoring
match). Other matched channels go into ``alternative_channels`` so the
LLM payload can present ambiguity to the reviewer if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import AppConfig

CANONICAL_CHANNELS = (
    "bacs_or_direct_credit",
    "faster_payment",
    "chaps",
    "cheque",
    "cash_or_counter_credit",
    "internal_transfer",
    "international_credit",
    "standing_order_credit",
    "unknown_credit",
)


@dataclass(frozen=True)
class ChannelMatch:
    channel: str
    confidence: float
    matched_tokens: tuple[str, ...] = ()
    alternative_channels: tuple[str, ...] = ()


@dataclass
class PaymentChannelDetector:
    """Stateful detector — built once per config, used across transactions."""

    config: AppConfig
    # Mapping channel -> list of (token tuple, original token string).
    _channel_tokens: dict[str, list[tuple[tuple[str, ...], str]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for channel, payload in self.config.channels().items():
            if not isinstance(payload, dict):
                continue
            tokens = payload.get("positive_tokens", [])
            self._channel_tokens[channel] = []
            for tok in tokens:
                norm = tok.upper().strip()
                if not norm:
                    continue
                pieces = tuple(norm.split())
                self._channel_tokens[channel].append((pieces, norm))
            # Sort by length descending so longer matches are tried first.
            self._channel_tokens[channel].sort(key=lambda x: (-len(x[0]), x[1]))

    # --- Detection -------------------------------------------------------

    def detect(self, description_tokens: tuple[str, ...]) -> ChannelMatch:
        """Find the strongest channel signal in the normalised tokens."""
        if not description_tokens:
            return ChannelMatch(channel="unknown_credit", confidence=0.0)

        token_set_idx = {t: i for i, t in enumerate(description_tokens)}

        best_channel: Optional[str] = None
        best_specificity: int = 0
        best_token: str = ""
        all_matched: list[str] = []
        all_channels_with_match: set[str] = set()

        for channel, candidates in self._channel_tokens.items():
            for pieces, original in candidates:
                if self._matches(description_tokens, pieces, token_set_idx):
                    all_matched.append(original)
                    all_channels_with_match.add(channel)
                    if len(pieces) > best_specificity:
                        best_channel = channel
                        best_specificity = len(pieces)
                        best_token = original

        if best_channel is None:
            return ChannelMatch(channel="unknown_credit", confidence=0.30)

        # Confidence: 0.55 baseline + 0.15 per extra token in the matched
        # phrase, capped at 1.0.
        confidence = min(0.55 + 0.15 * (best_specificity - 1), 1.0)
        # If the bank-specific overlay was applied, give a small boost so
        # callers can see the bank match worked.
        if self.config.get("source_bank") is not None:
            confidence = min(confidence + 0.05, 1.0)

        alternatives = tuple(sorted(c for c in all_channels_with_match if c != best_channel))
        return ChannelMatch(
            channel=best_channel,
            confidence=confidence,
            matched_tokens=tuple(sorted(set(all_matched))),
            alternative_channels=alternatives,
        )

    # --- Internal --------------------------------------------------------

    @staticmethod
    def _matches(
        description_tokens: tuple[str, ...],
        phrase_tokens: tuple[str, ...],
        token_index: dict[str, int],
    ) -> bool:
        """Does ``phrase_tokens`` appear as a contiguous substring of
        ``description_tokens``?

        Word-boundary aware by construction — we already split on whitespace
        in the normaliser, so ``BAC`` will not match inside ``BACS``.
        """
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in token_index
        n, m = len(description_tokens), len(phrase_tokens)
        if m > n:
            return False
        for i in range(n - m + 1):
            if description_tokens[i : i + m] == phrase_tokens:
                return True
        return False
