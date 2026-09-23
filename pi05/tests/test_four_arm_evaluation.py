from __future__ import annotations

import numpy as np
import pytest

from pi05_fabric.evaluation.four_arm import FourArmProgress
from pi05_fabric.evaluation.four_arm import RolePolicyRequest
from pi05_fabric.evaluation.four_arm import dispatch_role_actions
from pi05_fabric.evaluation.four_arm import evaluation_seeds
from pi05_fabric.evaluation.four_arm import strict_success


def test_dispatch_requires_four_finite_h50_chunks_and_returns_e25() -> None:
    chunks = tuple(np.full((50, 7), role, dtype=np.float32) for role in range(4))
    dispatched = dispatch_role_actions(chunks, execution_horizon=25)

    assert len(dispatched) == 4
    assert all(chunk.shape == (25, 7) for chunk in dispatched)
    assert [float(chunk[0, 0]) for chunk in dispatched] == [0.0, 1.0, 2.0, 3.0]


def test_dispatch_rejects_missing_role_and_nonfinite_actions() -> None:
    valid = np.zeros((50, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="four roles"):
        dispatch_role_actions((valid, valid, valid), execution_horizon=25)
    invalid = valid.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="role 2"):
        dispatch_role_actions((valid, valid, invalid, valid), execution_horizon=25)


def test_role_request_contains_only_shared_and_role_local_observation() -> None:
    request = RolePolicyRequest(
        seed=5100,
        planning_round=0,
        role_index=2,
        global_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        wrist_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(9, dtype=np.float32),
        instruction="insert the frame",
    )

    assert request.role_index == 2
    assert not hasattr(request, "peer_wrist_rgb")
    assert not hasattr(request, "peer_state")


def test_task_success_and_seed_splits_are_explicit() -> None:
    assert strict_success(FourArmProgress("frame_insertion", {"success": True}))
    assert not strict_success(FourArmProgress("arch_assembly", {"success": False}))
    assert evaluation_seeds("frame_insertion", "validation") == tuple(range(5100, 5300))
    assert evaluation_seeds("arch_assembly", "fresh") == tuple(range(6400, 6600))
