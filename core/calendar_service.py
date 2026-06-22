"""UK working-day calendar — Design §7.1.

Backed by the ``holidays`` PyPI library for canonical Gov.UK bank-holiday
data, with optional YAML overrides for cases where the library's data is
incomplete or wrong (Design §7.1).

Supported regions:

- ``EnglandAndWales``
- ``Scotland``
- ``NorthernIreland``
- ``UnknownDefaultEnglandAndWales`` — alias that emits a warning the first
  time it's used so the caller knows a default was applied (Requirement
  FR-004).

The ``WorkingDayCalendar`` instance is intentionally cheap to build —
``holidays.country_holidays(...)`` lazily computes dates as requested.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Optional

import holidays

logger = logging.getLogger(__name__)

# Region → holidays-lib subdivision code.
_REGION_TO_SUBDIV = {
    "EnglandAndWales": "ENG",   # Wales overlaps with England in the lib for our purposes
    "Scotland": "SCT",
    "NorthernIreland": "NIR",
}

_DEFAULT_REGION = "EnglandAndWales"
_UNKNOWN_REGION_ALIAS = "UnknownDefaultEnglandAndWales"


class UnknownRegionWarningEmittedError(Exception):
    """Internal sentinel — never raised, used for clarity if you trace logs."""


@dataclass
class CalendarOverrides:
    """Per-region add/remove date lists, parsed from uk_bank_holidays.yaml."""

    add: dict[str, set[date]] = field(default_factory=dict)
    remove: dict[str, set[date]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, region_block: dict | None) -> "CalendarOverrides":
        if not region_block:
            return cls()
        add: dict[str, set[date]] = {}
        remove: dict[str, set[date]] = {}
        for region, payload in region_block.items():
            if not isinstance(payload, dict):
                continue
            add[region] = {date.fromisoformat(d) for d in payload.get("add", [])}
            remove[region] = {date.fromisoformat(d) for d in payload.get("remove", [])}
        return cls(add=add, remove=remove)


class WorkingDayCalendar:
    """UK working-day calculator with weekend + bank-holiday awareness."""

    def __init__(
        self,
        *,
        overrides: Optional[CalendarOverrides] = None,
        warn_on_unknown_region: bool = True,
    ) -> None:
        self._overrides = overrides or CalendarOverrides()
        self._warn_on_unknown_region = warn_on_unknown_region
        self._unknown_region_warned = False
        self._cache: dict[tuple[str, int], set[date]] = {}

    # --- Public API ------------------------------------------------------

    def is_working_day(self, d: date, region: str) -> bool:
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return d not in self._holidays_for(region, d.year)

    def previous_working_day(self, d: date, region: str) -> date:
        candidate = d
        for _ in range(60):  # safety bound — no real-world year has 60 consecutive non-working days
            if self.is_working_day(candidate, region):
                return candidate
            candidate -= timedelta(days=1)
        raise RuntimeError(f"could not find a working day on or before {d} for region {region}")

    def next_working_day(self, d: date, region: str) -> date:
        candidate = d
        for _ in range(60):
            if self.is_working_day(candidate, region):
                return candidate
            candidate += timedelta(days=1)
        raise RuntimeError(f"could not find a working day on or after {d} for region {region}")

    def nearest_working_day(self, d: date, region: str) -> date:
        """Closest working day. Ties (equidistant) prefer the **previous** day —
        matches the more common UK payroll policy.
        """
        if self.is_working_day(d, region):
            return d
        for delta in range(1, 60):
            prev = d - timedelta(days=delta)
            nxt = d + timedelta(days=delta)
            if self.is_working_day(prev, region):
                return prev
            if self.is_working_day(nxt, region):
                return nxt
        raise RuntimeError(f"no nearest working day found for {d} region={region}")

    def last_working_day_of_month(self, year: int, month: int, region: str) -> date:
        last_day = monthrange(year, month)[1]
        return self.previous_working_day(date(year, month, last_day), region)

    def first_working_day_of_month(self, year: int, month: int, region: str) -> date:
        return self.next_working_day(date(year, month, 1), region)

    # --- Internal --------------------------------------------------------

    def _resolve_region(self, region: str) -> str:
        if region in _REGION_TO_SUBDIV:
            return region
        if region == _UNKNOWN_REGION_ALIAS:
            if self._warn_on_unknown_region and not self._unknown_region_warned:
                logger.warning(
                    "UK region is unknown — defaulting to %s holiday calendar (FR-004).",
                    _DEFAULT_REGION,
                )
                self._unknown_region_warned = True
            return _DEFAULT_REGION
        # Unrecognised region literal — same default + warning.
        if self._warn_on_unknown_region and not self._unknown_region_warned:
            logger.warning(
                "Unrecognised region %r — defaulting to %s holiday calendar.",
                region,
                _DEFAULT_REGION,
            )
            self._unknown_region_warned = True
        return _DEFAULT_REGION

    def _holidays_for(self, region: str, year: int) -> set[date]:
        canonical = self._resolve_region(region)
        cache_key = (canonical, year)
        if cache_key in self._cache:
            return self._cache[cache_key]

        subdiv = _REGION_TO_SUBDIV[canonical]
        lib_holidays = holidays.country_holidays("GB", subdiv=subdiv, years=[year])
        dates: set[date] = set(lib_holidays.keys())

        # Apply YAML overrides if present for this region.
        dates |= self._overrides.add.get(canonical, set())
        dates -= self._overrides.remove.get(canonical, set())

        self._cache[cache_key] = dates
        return dates

    # --- Diagnostic ------------------------------------------------------

    def list_holidays(self, region: str, year: int) -> list[date]:
        """Sorted list of bank holidays for the given region/year. Mostly used by tests."""
        return sorted(self._holidays_for(region, year))


def working_day_calendar_from_config(config_data: dict | None) -> WorkingDayCalendar:
    """Build a calendar from the merged config dict (Design §5)."""
    overrides_block = (config_data or {}).get("regions")
    return WorkingDayCalendar(overrides=CalendarOverrides.from_config(overrides_block))
