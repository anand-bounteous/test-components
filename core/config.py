"""Layered configuration loader — Design §5, plan.md Phase 2.

Three layers, applied in this order (later layers override earlier ones):

1. **Built-in defaults** — every YAML file under ``salary_extractor/config/``.
2. **File overrides** — caller passes a ``Path`` (or list of paths) to YAML
   files that are deep-merged on top of the defaults.
3. **Caller overrides** — a ``dict`` of overrides supplied at call time.

Bank-specific mappings are loaded if ``source_bank`` is provided. The
overlay merges into ``config['channels']`` and ``config['negative_signals']``
respectively, so callers can keep generic mappings as the baseline.

Deep merge rules:

- ``dict`` merges recursively;
- ``list`` appends without de-duplication (callers handle de-dup);
- everything else is a straight replacement.

The merged result is JSON-schema-validated against ``config.schema.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from jsonschema import Draft202012Validator

__all__ = [
    "AppConfig",
    "ConfigLoadError",
    "load_config",
    "deep_merge",
]


class ConfigLoadError(ValueError):
    """Raised when YAML loading or schema validation fails."""


@dataclass(frozen=True)
class AppConfig:
    """Wrapper exposing the merged config dict.

    Most callers should use the convenience accessors rather than poking the
    raw dict so that field renames are easy to track.
    """

    data: dict

    # --- Convenience accessors ------------------------------------------

    def weight(self, name: str) -> float:
        return float(self.data["weights"][name])

    def threshold(self, name: str) -> float:
        return float(self.data["thresholds"][name])

    def default_region(self) -> str:
        return self.data.get("calendar", {}).get("default_region", "EnglandAndWales")

    def channels(self) -> dict:
        return dict(self.data.get("channels", {}))

    def negative_signals(self) -> dict:
        return dict(self.data.get("negative_salary_signals", {}))

    def salary_hints_keywords(self) -> dict:
        return dict(self.data.get("salary_hints", {}))

    def candidate_generation(self) -> dict:
        return dict(self.data.get("candidate_generation", {}))

    def amount_config(self) -> dict:
        return dict(self.data.get("amount", {}))

    def bonuses(self) -> dict:
        return dict(self.data.get("bonuses", {}))

    def penalties(self) -> dict:
        return dict(self.data.get("penalties", {}))

    def versions(self) -> dict:
        return dict(self.data.get("versions", {}))

    # --- Generic accessor for unmapped fields ----------------------------

    def get(self, *keys, default=None):
        cursor: Any = self.data
        for key in keys:
            if not isinstance(cursor, dict):
                return default
            if key not in cursor:
                return default
            cursor = cursor[key]
        return cursor


# --- Public loader ---------------------------------------------------------


def load_config(
    *,
    overrides: Optional[dict] = None,
    file_overrides: Optional[Iterable[Path]] = None,
    source_bank: Optional[str] = None,
) -> AppConfig:
    """Build a fully merged ``AppConfig``.

    ``overrides`` is the highest-priority caller dict.
    ``file_overrides`` is an iterable of YAML paths applied in order between
    defaults and ``overrides``.
    ``source_bank`` triggers the matching ``bank_specific_mappings`` YAML to
    be merged in if it exists.
    """
    merged: dict = _load_defaults()

    if source_bank:
        bank_yaml = _load_bank_specific(source_bank)
        if bank_yaml is not None:
            _merge_bank_overlay(merged, bank_yaml)
            merged["source_bank"] = source_bank

    for path in file_overrides or ():
        merged = deep_merge(merged, _load_yaml(Path(path)))

    if overrides:
        merged = deep_merge(merged, overrides)

    _validate_schema(merged)
    return AppConfig(data=merged)


# --- YAML helpers ----------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigLoadError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigLoadError(f"{path} top-level YAML must be a mapping")
    return loaded


def _resource_yaml(filename: str) -> dict:
    res = resources.files("salary_extractor.config") / filename
    with resources.as_file(res) as p:
        return _load_yaml(Path(p))


def _load_defaults() -> dict:
    """Compose every built-in YAML into one dict.

    Files are merged in deterministic order so the result is stable.
    """
    files = [
        "default_scoring.yaml",
        "salary_keywords.yaml",
        "negative_keywords.yaml",
        "generic_payment_code_mappings.yaml",
        "uk_bank_holidays.yaml",
    ]
    merged: dict = {}
    for f in files:
        merged = deep_merge(merged, _resource_yaml(f))
    return merged


def _load_bank_specific(source_bank: str) -> Optional[dict]:
    name = source_bank.strip().lower()
    try:
        res = resources.files("salary_extractor.config.bank_specific_mappings") / f"{name}.yaml"
        if not res.is_file():
            return None
        with resources.as_file(res) as p:
            return _load_yaml(Path(p))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _merge_bank_overlay(base: dict, overlay: dict) -> None:
    """Apply bank-specific overlay in place.

    Overlay structure:

    ```yaml
    channels:
      bacs_or_direct_credit:
        positive_tokens: [BAC, BGC]
    negative_signals:
      government_benefits: [DWP, INT]
    ```

    Channel token lists are extended (de-dup later in code). The negative
    overlay extends the matching list under ``negative_salary_signals``.
    """
    overlay_channels = overlay.get("channels", {})
    base_channels = base.setdefault("channels", {})
    for channel, payload in overlay_channels.items():
        if not isinstance(payload, dict):
            continue
        base_channel = base_channels.setdefault(channel, {})
        tokens = payload.get("positive_tokens", [])
        if tokens:
            existing = list(base_channel.get("positive_tokens", []))
            for tok in tokens:
                if tok not in existing:
                    existing.append(tok)
            base_channel["positive_tokens"] = existing

    overlay_negative = overlay.get("negative_signals", {})
    base_negative = base.setdefault("negative_salary_signals", {})
    for category, tokens in overlay_negative.items():
        if not isinstance(tokens, list):
            continue
        existing = list(base_negative.get(category, []))
        for tok in tokens:
            if tok not in existing:
                existing.append(tok)
        base_negative[category] = existing


# --- Deep merge ------------------------------------------------------------


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — overlay wins on scalars, lists are concatenated.

    Returns a new dict; neither input is mutated.
    """
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise TypeError("deep_merge requires dict arguments")
    out: dict = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        elif key in out and isinstance(out[key], list) and isinstance(value, list):
            out[key] = list(out[key]) + [v for v in value if v not in out[key]]
        else:
            out[key] = value
    return out


# --- Schema validation -----------------------------------------------------

_CONFIG_VALIDATOR: Optional[Draft202012Validator] = None


def _config_validator() -> Draft202012Validator:
    global _CONFIG_VALIDATOR
    if _CONFIG_VALIDATOR is None:
        schema = json.loads(
            (resources.files("salary_extractor.schemas") / "config.schema.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        _CONFIG_VALIDATOR = Draft202012Validator(schema)
    return _CONFIG_VALIDATOR


def _validate_schema(data: dict) -> None:
    errors = sorted(_config_validator().iter_errors(data), key=lambda e: list(e.path))
    if not errors:
        return
    first = errors[0]
    raise ConfigLoadError(
        f"merged config failed schema validation at {list(first.path)}: {first.message}"
    )
