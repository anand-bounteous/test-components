"""Input validation — FR-001, FR-002.

Two responsibilities:

1. JSON-schema validation of the input payload against
   ``salary_extractor/schemas/input.schema.json``.
2. Row-level validation of ``RawTransaction`` records:
   - parseable ISO date,
   - numeric amount,
   - resolvable credit/debit direction,
   - non-empty description (replaced with a placeholder when missing),
   - unique transaction id,
   - credit-only contract (Requirement FR-002 updated).

Lenient mode (default) emits warnings and skips offending rows. Strict mode
raises ``ValidationError`` on the first failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

from jsonschema import Draft202012Validator

from ..models.output import ValidationWarning
from ..models.transaction import Direction, RawTransaction

EMPTY_DESCRIPTION_PLACEHOLDER = "__EMPTY__"


class ValidationError(ValueError):
    """Raised in strict mode when an input row fails validation."""


@dataclass(frozen=True)
class ValidationResult:
    transactions: tuple[RawTransaction, ...]
    warnings: tuple[ValidationWarning, ...]


def _load_input_schema() -> dict:
    schema_file = resources.files("salary_extractor.schemas") / "input.schema.json"
    return json.loads(schema_file.read_text())


_INPUT_SCHEMA: Optional[dict] = None
_INPUT_VALIDATOR: Optional[Draft202012Validator] = None


def input_schema() -> dict:
    global _INPUT_SCHEMA
    if _INPUT_SCHEMA is None:
        _INPUT_SCHEMA = _load_input_schema()
    return _INPUT_SCHEMA


def input_validator() -> Draft202012Validator:
    global _INPUT_VALIDATOR
    if _INPUT_VALIDATOR is None:
        _INPUT_VALIDATOR = Draft202012Validator(input_schema())
    return _INPUT_VALIDATOR


def validate_schema(payload: dict, *, strict: bool = True) -> tuple[ValidationWarning, ...]:
    """Validate the full input payload against the JSON schema.

    In strict mode the first error raises ``ValidationError``. In lenient mode
    each error is returned as a warning and the caller can continue.
    """
    errors = sorted(input_validator().iter_errors(payload), key=lambda e: list(e.path))
    if not errors:
        return ()
    if strict:
        first = errors[0]
        raise ValidationError(f"schema validation failed at {list(first.path)}: {first.message}")
    return tuple(
        ValidationWarning(
            code="schema_violation",
            transaction_id=_path_to_txn_id(payload, e),
            detail=f"{list(e.path)}: {e.message}",
        )
        for e in errors
    )


def _path_to_txn_id(payload: dict, error) -> str:
    path = list(error.path)
    if len(path) >= 2 and path[0] == "transactions":
        idx = path[1]
        try:
            return payload["transactions"][idx].get("id", "")
        except (IndexError, KeyError, TypeError):
            return ""
    return ""


def _parse_date(raw: Any) -> date:
    if not isinstance(raw, str):
        raise ValidationError(f"date must be a string, got {type(raw).__name__}")
    # Strictly enforce YYYY-MM-DD — looser parsers hide upstream bugs.
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"date {raw!r} is not ISO YYYY-MM-DD") from exc


def _resolve_direction(amount: Decimal, raw_direction: Optional[str]) -> Direction:
    if isinstance(raw_direction, str):
        d = raw_direction.lower()
        if d in {"credit", "cr", "+"}:
            return "credit"
        if d in {"debit", "dr", "-"}:
            return "debit"
        raise ValidationError(f"unrecognised direction {raw_direction!r}")
    # Fallback: signed amount.
    if amount > 0:
        return "credit"
    if amount < 0:
        return "debit"
    raise ValidationError("amount is zero and direction is missing — cannot resolve")


def _to_decimal(raw: Any) -> Decimal:
    if isinstance(raw, Decimal):
        return raw
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValidationError(f"amount {raw!r} is not numeric") from exc


def _normalise_description(raw: Any) -> str:
    if raw is None:
        return EMPTY_DESCRIPTION_PLACEHOLDER
    if not isinstance(raw, str):
        raw = str(raw)
    stripped = raw.strip()
    return stripped if stripped else EMPTY_DESCRIPTION_PLACEHOLDER


def validate_and_normalise_rows(
    rows: Iterable[dict],
    *,
    mode: str = "lenient",
) -> ValidationResult:
    """Apply row-level validation and credit-only enforcement.

    Returns successfully-validated ``RawTransaction`` objects plus warnings.
    Direction is normalised at this stage so downstream code can rely on the
    ``direction`` field being present and lower-case.
    """
    if mode not in {"lenient", "strict"}:
        raise ValueError(f"mode must be 'lenient' or 'strict', got {mode!r}")
    strict = mode == "strict"

    accepted: list[RawTransaction] = []
    warnings: list[ValidationWarning] = []
    seen_ids: set[str] = set()

    for raw in rows:
        tid = raw.get("id", "")
        try:
            txn = _validate_single(raw, tid)
        except ValidationError as exc:
            if strict:
                raise
            warnings.append(ValidationWarning(code="row_invalid", transaction_id=tid, detail=str(exc)))
            continue

        if txn.id in seen_ids:
            warning = ValidationWarning(
                code="duplicate_id",
                transaction_id=txn.id,
                detail=f"duplicate id {txn.id!r} — keeping the first occurrence",
            )
            if strict:
                raise ValidationError(warning.detail)
            warnings.append(warning)
            continue
        seen_ids.add(txn.id)

        if txn.direction != "credit":
            warning = ValidationWarning(
                code="non_credit_row_skipped",
                transaction_id=txn.id,
                detail=(
                    f"input contract is credit-only (FR-002); skipping {txn.direction} row {txn.id!r}"
                ),
            )
            if strict:
                raise ValidationError(warning.detail)
            warnings.append(warning)
            continue

        accepted.append(txn)

    return ValidationResult(transactions=tuple(accepted), warnings=tuple(warnings))


def _validate_single(raw: dict, tid: str) -> RawTransaction:
    if not tid or not isinstance(tid, str):
        raise ValidationError(f"transaction id is required and must be a non-empty string, got {tid!r}")
    amount = _to_decimal(raw.get("amount"))
    direction = _resolve_direction(amount, raw.get("direction"))
    parsed_date = _parse_date(raw.get("date"))
    description = _normalise_description(raw.get("description"))
    return RawTransaction(
        id=tid,
        date=parsed_date.isoformat(),
        description=description,
        amount=abs(amount),
        direction=direction,
        balance=_decimal_optional(raw.get("balance")),
        reference=raw.get("reference"),
        counterparty_name=raw.get("counterparty_name"),
        transaction_code=raw.get("transaction_code"),
        source_bank=raw.get("source_bank"),
        currency=raw.get("currency"),
        account_id=raw.get("account_id"),
        raw_category=raw.get("raw_category"),
        booking_date=raw.get("booking_date"),
        value_date=raw.get("value_date"),
    )


def _decimal_optional(raw: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    return _to_decimal(raw)


def load_payload(path: Path) -> dict:
    """Convenience loader used by feature tests and the CLI."""
    return json.loads(Path(path).read_text())
