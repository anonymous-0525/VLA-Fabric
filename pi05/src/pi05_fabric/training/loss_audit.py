"""Audit bilateral flow-matching losses emitted by native training smokes."""

from __future__ import annotations

import math
from pathlib import Path
import re
import statistics


_ROW = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<total>\S+)\s+"
    r"left_loss=(?P<left>\S+)\s+right_loss=(?P<right>\S+)"
)


def _side_summary(values: list[float], window: int) -> dict[str, float]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "first_window_median": statistics.median(values[:window]),
        "last_window_median": statistics.median(values[-window:]),
    }


def audit_loss_log(
    path: str | Path,
    *,
    expected_start: int,
    expected_end: int,
    trend_window: int = 20,
    max_window_growth: float = 4.0,
    max_side_ratio: float = 20.0,
) -> dict:
    """Validate loss observability and reject catastrophic unilateral behavior."""

    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        match = _ROW.search(line)
        if match is not None:
            rows.append(
                (
                    int(match["step"]),
                    float(match["total"]),
                    float(match["left"]),
                    float(match["right"]),
                )
            )
    if not rows:
        raise ValueError(f"no bilateral loss rows found in {path}")

    expected_steps = list(range(expected_start, expected_end + 1))
    actual_steps = [row[0] for row in rows]
    if actual_steps != expected_steps:
        raise ValueError(
            f"bilateral loss rows do not cover contiguous steps {expected_start}-{expected_end}: "
            f"got {actual_steps[:3]}...{actual_steps[-3:]}"
        )

    left_values, right_values = [], []
    maximum_ratio = 1.0
    for step, total, left, right in rows:
        if not all(math.isfinite(value) for value in (total, left, right)):
            raise ValueError(f"non-finite bilateral loss at step {step}")
        if left <= 0 or right <= 0:
            raise ValueError(f"non-positive bilateral loss at step {step}")
        tolerance = max(5e-5, abs(total) * 5e-4)
        if abs(total - (left + right)) > tolerance:
            raise ValueError(f"loss decomposition mismatch at step {step}")
        ratio = max(left, right) / min(left, right)
        maximum_ratio = max(maximum_ratio, ratio)
        if ratio > max_side_ratio:
            raise ValueError(f"left/right loss ratio {ratio:.3f} exceeds {max_side_ratio} at step {step}")
        left_values.append(left)
        right_values.append(right)

    window = min(max(1, trend_window), len(rows))
    left_summary = _side_summary(left_values, window)
    right_summary = _side_summary(right_values, window)
    for name, summary in (("left", left_summary), ("right", right_summary)):
        first = summary["first_window_median"]
        last = summary["last_window_median"]
        if last > first * max_window_growth:
            raise ValueError(
                f"{name} loss growth {last / first:.3f} exceeds {max_window_growth}"
            )

    return {
        "status": "PASS",
        "log": str(path),
        "start_step": expected_start,
        "end_step": expected_end,
        "steps": len(rows),
        "trend_window": window,
        "max_window_growth": max_window_growth,
        "max_side_ratio": maximum_ratio,
        "left": left_summary,
        "right": right_summary,
    }
