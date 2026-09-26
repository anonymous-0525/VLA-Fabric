"""Auditing and paired summaries for fixed no-retraining diagnostics."""

from __future__ import annotations

from typing import Iterable, Mapping

from pi05_fabric.evaluation.native_handover import diagnostic_validation_ids
from pi05_fabric.evaluation.protocol import audit_rollout_rows


REFERENCE = "full_exec20_reference"
REQUIRED_PROFILES = (
    REFERENCE,
    "full_exec10",
    "full_exec5",
    "core_exec20",
    "common_only_exec20",
)


def _profile_metrics(rows: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    successes = tuple(row for row in rows if bool(int(row["success"])))
    return {
        "trials": len(rows),
        "successes": len(successes),
        "success_rate": 100.0 * len(successes) / len(rows),
        "mean_success_steps": (
            sum(float(row["steps"]) for row in successes) / len(successes)
            if successes
            else None
        ),
        "mean_plan_calls": sum(float(row["plan_calls"]) for row in rows) / len(rows),
        "mean_plan_seconds": sum(float(row["mean_plan_seconds"]) for row in rows) / len(rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "errors": sum(bool(str(row["error"])) for row in rows),
    }


def summarize_diagnostic_profiles(
    profiles: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    expected_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    missing = tuple(name for name in REQUIRED_PROFILES if name not in profiles)
    if missing:
        raise ValueError(f"missing diagnostic profiles: {missing}")

    if expected_ids is None:
        expected_ids = diagnostic_validation_ids()
    frozen: dict[str, tuple[Mapping[str, object], ...]] = {}
    outcomes: dict[str, dict[int, int]] = {}
    metrics: dict[str, dict[str, object]] = {}
    for name in REQUIRED_PROFILES:
        rows = tuple(profiles[name])
        audit_rollout_rows(rows, expected_ids=expected_ids)
        if any(not bool(int(row["finite_all"])) for row in rows):
            raise ValueError(f"{name} contains non-finite rollout output")
        if any(int(row["timeout"]) or str(row["error"]) for row in rows):
            raise ValueError(f"{name} contains timeout or error rows")
        frozen[name] = rows
        outcomes[name] = {int(row["condition_id"]): int(row["success"]) for row in rows}
        metrics[name] = _profile_metrics(rows)

    reference = outcomes[REFERENCE]
    paired = {}
    for name in REQUIRED_PROFILES[1:]:
        alternative = outcomes[name]
        paired[name] = {
            "both_success": sum(reference[i] and alternative[i] for i in expected_ids),
            "reference_only": sum(reference[i] and not alternative[i] for i in expected_ids),
            "alternative_only": sum(not reference[i] and alternative[i] for i in expected_ids),
            "both_fail": sum(not reference[i] and not alternative[i] for i in expected_ids),
        }

    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "purpose": "no_retraining_diagnostic",
        "condition_ids": list(expected_ids),
        "profiles": metrics,
        "paired_vs_reference": paired,
    }
