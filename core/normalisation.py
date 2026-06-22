"""Transaction normalisation — Design §6.1, §8.1.

Two distinct text outputs per transaction:

- ``normalised_description``: human-readable, uppercased, punctuation→space,
  whitespace collapsed, long numeric references stripped. Used for similarity
  scoring in Phase 7.
- ``skeleton``: the more aggressive deterministic key from Design §10.5.
  Used by the skeleton-grouping path in Phase 5. Idempotent.

The skeleton normalisation deliberately keeps salary/channel hint tokens
(``BACS``, ``PAYROLL``, ``SALARY``, ``WAGES``…). Removing them would collapse
distinct income streams into a single bucket. Phase 3 separates them into
their own feature lists, but for grouping their presence is still useful.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Iterable

from ..models.transaction import NormalisedTransaction, RawTransaction

# Generic noise tokens dropped from both the normalised description and the
# skeleton. These are payment-route / banking glue words that carry no
# employer signal. The list mirrors Design §8.1 stop tokens.
_GENERIC_STOP_TOKENS = frozenset(
    {
        "REF",
        "REFERENCE",
        "PAYMENT",
        "CREDIT",
        "ONLINE",
        "TRANSFER",
        "TFR",
    }
)

# Legal-suffix collapse. After collapse, ``ACME LIMITED`` and ``ACME LTD``
# both produce the token ``LTD``, so grouping by skeleton merges them.
# Order matters: the longest phrase must be applied first so it isn't
# partially consumed by a shorter rule (PUBLIC LIMITED COMPANY → PLC must
# run before LIMITED → LTD).
_LEGAL_SUFFIX_MAP = (
    ("PUBLIC LIMITED COMPANY", "PLC"),
    ("LIMITED", "LTD"),
)

_PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")
_LONG_DIGIT_TOKEN_RE = re.compile(r"^\d{4,}$")
# Strip a trailing digit run only when letters precede it. This catches
# REF1234 → REF without consuming purely-numeric tokens like WAGE 12.
_TRAILING_DIGITS_RE = re.compile(r"(?<=[A-Z])\d{2,}$")


def normalise_text(raw: str) -> str:
    """Description normalisation — uppercase, punctuation→space, whitespace
    collapse. Preserves numeric references; the skeleton step removes them.
    """
    if not raw:
        return ""
    # NFKD strips diacritics consistently — banks occasionally feed accented
    # legal names through.
    decomposed = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    upper = ascii_only.upper()
    # Apply multi-word legal suffix collapse before stripping punctuation, so
    # we can match phrase boundaries.
    for long, short in _LEGAL_SUFFIX_MAP:
        upper = re.sub(rf"\b{long}\b", short, upper)
    cleaned = _PUNCT_RE.sub(" ", upper)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def tokenise(normalised: str) -> tuple[str, ...]:
    """Split the normalised description into tokens. Order preserved."""
    if not normalised:
        return ()
    return tuple(t for t in normalised.split(" ") if t)


def skeleton_of(raw: str) -> str:
    """Compute the deterministic, idempotent skeleton key — Design §10.5.

    Steps beyond ``normalise_text``:

    1. Drop tokens that are pure digits with ≥ 4 digits (long references).
    2. Drop generic banking glue words (``REF``, ``PAYMENT``…).
    3. Drop trailing ``#NNN`` or ``NNN`` patterns within tokens.
    """
    if raw is None:
        return "__EMPTY__"
    base = normalise_text(raw)
    if not base:
        return "__EMPTY__"
    tokens = []
    for tok in tokenise(base):
        if _LONG_DIGIT_TOKEN_RE.match(tok):
            continue
        if tok in _GENERIC_STOP_TOKENS:
            continue
        # Trim a trailing numeric suffix like #0042 or 1234 inside a token
        # such as REF1234.
        trimmed = _TRAILING_DIGITS_RE.sub("", tok)
        if not trimmed:
            continue
        # Re-check the stop list after digit stripping — REF1234 collapses
        # to REF, which is itself a generic glue word we should drop.
        if trimmed in _GENERIC_STOP_TOKENS:
            continue
        tokens.append(trimmed)
    if not tokens:
        return "__EMPTY__"
    return " ".join(tokens)


def normalise_transaction(raw: RawTransaction) -> NormalisedTransaction:
    """Project a ``RawTransaction`` into the canonical model used downstream.

    Caller has already validated the row — this function does not raise on
    malformed inputs.
    """
    parsed_date = date.fromisoformat(raw.date)
    normalised_desc = normalise_text(raw.description)
    tokens = tokenise(normalised_desc)
    skeleton = skeleton_of(raw.description)
    return NormalisedTransaction(
        id=raw.id,
        date=parsed_date,
        amount=raw.amount,
        direction=raw.direction or "credit",
        raw_description=raw.description,
        normalised_description=normalised_desc,
        description_tokens=tokens,
        skeleton=skeleton,
        source_bank=raw.source_bank,
        transaction_code=raw.transaction_code,
        counterparty_name=raw.counterparty_name,
        currency=raw.currency or "GBP",
    )


def normalise_many(rows: Iterable[RawTransaction]) -> tuple[NormalisedTransaction, ...]:
    return tuple(normalise_transaction(r) for r in rows)
