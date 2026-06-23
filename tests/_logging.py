"""Custom TRACE log level + helpers — Phase 14.

Python's stdlib ``logging`` defines DEBUG = 10 but no level below. The
salary detector emits *very* verbose diagnostics (per-transaction
features, per-edge scores) that we want to keep available but off by
default — TRACE = 5 is the right level for that.

Usage:

```python
from salary_extractor._logging import setup_trace, get_logger

setup_trace()                          # register TRACE level once
logger = get_logger(__name__)
logger.trace("very detailed event %s", payload)
```
"""
from __future__ import annotations

import logging
from typing import Any

TRACE = 5
_LEVEL_NAME = "TRACE"

# Register at import time so both logging.getLevelName("TRACE") and
# getattr(logging, "TRACE") work — pytest uses getattr() to resolve
# --log-cli-level=TRACE before any pytest_configure hook runs.
logging.addLevelName(TRACE, _LEVEL_NAME)
logging.TRACE = TRACE  # type: ignore[attr-defined]

_registered = False


def setup_trace() -> None:
    """Idempotent — registers TRACE = 5 and attaches a ``trace()`` method
    to ``logging.Logger`` so callers can write ``logger.trace(...)``."""
    global _registered
    if _registered:
        return
    logging.addLevelName(TRACE, _LEVEL_NAME)

    def _trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)

    # Attach as an instance method on the Logger class.
    logging.Logger.trace = _trace  # type: ignore[attr-defined]
    _registered = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger after ensuring TRACE is registered."""
    setup_trace()
    return logging.getLogger(name)


__all__ = ["TRACE", "get_logger", "setup_trace"]
