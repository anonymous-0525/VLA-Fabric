"""Non-paper condition range for pilot-model diagnostics."""

from __future__ import annotations


_DIAGNOSTIC_START = 9000
_DIAGNOSTIC_COUNT = 50


def diagnostic_condition_ids() -> tuple[int, ...]:
    """Return the frozen diagnostic range, disjoint from paper evaluation IDs."""

    return tuple(range(_DIAGNOSTIC_START, _DIAGNOSTIC_START + _DIAGNOSTIC_COUNT))
