"""Helpers around the LLM payload + output JSON Schemas."""
from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from jsonschema import Draft202012Validator


_PAYLOAD_VALIDATOR: Optional[Draft202012Validator] = None
_OUTPUT_VALIDATOR: Optional[Draft202012Validator] = None


def _load(name: str) -> dict:
    text = (resources.files("salary_extractor.schemas") / name).read_text()
    schema = json.loads(text)
    Draft202012Validator.check_schema(schema)
    return schema


def payload_schema() -> dict:
    return _load("llm_payload.schema.json")


def output_schema() -> dict:
    return _load("llm_output.schema.json")


def payload_validator() -> Draft202012Validator:
    global _PAYLOAD_VALIDATOR
    if _PAYLOAD_VALIDATOR is None:
        _PAYLOAD_VALIDATOR = Draft202012Validator(payload_schema())
    return _PAYLOAD_VALIDATOR


def output_validator() -> Draft202012Validator:
    global _OUTPUT_VALIDATOR
    if _OUTPUT_VALIDATOR is None:
        _OUTPUT_VALIDATOR = Draft202012Validator(output_schema())
    return _OUTPUT_VALIDATOR


class LLMOutputValidationError(ValueError):
    """Raised when an LLM reply doesn't match ``llm_output.schema.json``."""


def validate_llm_output(payload: dict) -> None:
    """Raise ``LLMOutputValidationError`` if ``payload`` is malformed."""
    errors = sorted(output_validator().iter_errors(payload), key=lambda e: list(e.path))
    if not errors:
        return
    first = errors[0]
    raise LLMOutputValidationError(
        f"LLM output validation failed at {list(first.path)}: {first.message}"
    )


def validate_llm_payload(payload: dict) -> None:
    """Raise ``LLMOutputValidationError`` if the *payload we send* is bad.

    Useful as a safety net before serialising for transmission.
    """
    errors = sorted(payload_validator().iter_errors(payload), key=lambda e: list(e.path))
    if not errors:
        return
    first = errors[0]
    raise LLMOutputValidationError(
        f"LLM payload validation failed at {list(first.path)}: {first.message}"
    )


__all__ = [
    "LLMOutputValidationError",
    "output_schema",
    "output_validator",
    "payload_schema",
    "payload_validator",
    "validate_llm_output",
    "validate_llm_payload",
]
