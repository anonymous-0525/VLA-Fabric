import numpy as np

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.evaluation.robotwin2_diagnostic import action_support_summary
from pi05_fabric.evaluation.robotwin2_diagnostic import diagnostic_inference_mode
from pi05_fabric.evaluation.robotwin2_diagnostic import select_diagnostic_condition_ids
from pi05_fabric.evaluation.robotwin2_policy import resolve_robotwin_evaluation_mode
from pi05_fabric.training.stages import StageName


def _row(condition_id, *, success=0, lifted=0, middle=0, target=0):
    return {
        "condition_id": str(condition_id),
        "success": str(success),
        "block_lifted": str(lifted),
        "middle_reached": str(middle),
        "target_region_reached": str(target),
    }


def test_selects_balanced_diagnostic_conditions_deterministically():
    rows = []
    rows.extend(_row(i, success=1, lifted=1, middle=1, target=1) for i in range(0, 12))
    rows.extend(_row(i, lifted=1, middle=1, target=1) for i in range(20, 32))
    rows.extend(_row(i, lifted=1, middle=1) for i in range(40, 52))

    selection = select_diagnostic_condition_ids(rows, per_stratum=8)

    assert selection.counts == {
        "strict_success": 8,
        "target_without_success": 8,
        "pretarget_failure": 8,
    }
    assert len(selection.condition_ids) == 24
    assert selection == select_diagnostic_condition_ids(tuple(reversed(rows)), per_stratum=8)


def test_rejects_incomplete_diagnostic_strata():
    rows = [_row(i, success=1, lifted=1, middle=1, target=1) for i in range(8)]

    try:
        select_diagnostic_condition_ids(rows, per_stratum=8)
    except ValueError as exc:
        assert "target_without_success" in str(exc)
    else:
        raise AssertionError("incomplete strata must be rejected")


def test_maps_v2_diagnostic_profiles_without_changing_weights():
    assert diagnostic_inference_mode("full") is DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION
    assert diagnostic_inference_mode("core") is DualPi05Mode.PI_NATIVE_V2_CORE
    assert diagnostic_inference_mode("common_only") is DualPi05Mode.PI_NATIVE_V2_RAW_COMMON_ONLY


def test_resolves_optional_robotwin_diagnostic_profile_override():
    stage = StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2

    assert resolve_robotwin_evaluation_mode(stage, None) is DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION
    assert resolve_robotwin_evaluation_mode(stage, "core") is DualPi05Mode.PI_NATIVE_V2_CORE


def test_action_support_summary_counts_values_outside_q01_q99():
    actions = np.zeros((2, 20), dtype=np.float32)
    actions[0, 0] = -2.0
    actions[1, 19] = 2.0
    statistics = {
        "left": {"action": {"q01": [-1.0] * 10, "q99": [1.0] * 10}},
        "right": {"action": {"q01": [-1.0] * 10, "q99": [1.0] * 10}},
    }

    summary = action_support_summary(actions, statistics)

    assert summary["values"] == 40
    assert summary["below_q01"] == 1
    assert summary["above_q99"] == 1
    assert summary["outside_fraction"] == 0.05
