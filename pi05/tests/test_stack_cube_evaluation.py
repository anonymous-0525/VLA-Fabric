from pathlib import Path
import numpy as np
import pytest
import socket
import tempfile

from pi05_fabric.evaluation.stack_cube_protocol import StackCubeProgress
from pi05_fabric.evaluation.stack_cube_protocol import dispatch_role_actions
from pi05_fabric.evaluation.stack_cube_protocol import evaluation_seeds
from pi05_fabric.evaluation.stack_cube_protocol import strict_success
from pi05_fabric.evaluation.stack_cube_rollout import extract_role_observation
from pi05_fabric.evaluation.stack_cube_rollout import rollout_record
from pi05_fabric.evaluation.stack_cube_ipc import receive_pickle
from pi05_fabric.evaluation.stack_cube_ipc import send_pickle
from pi05_fabric.evaluation.stack_cube_ipc import short_environment_socket_path


def test_h50_predictions_dispatch_exactly_first_25_actions_per_role():
    actions = tuple(
        np.full((50, 8), role, dtype=np.float32) for role in range(3)
    )

    dispatched = dispatch_role_actions(actions, execution_horizon=25)

    assert len(dispatched) == 3
    assert all(action.shape == (25, 8) for action in dispatched)
    assert [float(action[0, 0]) for action in dispatched] == [0.0, 1.0, 2.0]


def test_nonfinite_action_from_one_role_rejects_the_whole_team_step():
    actions = [np.zeros((50, 8), dtype=np.float32) for _ in range(3)]
    actions[2][4, 1] = np.nan

    with pytest.raises(FloatingPointError, match="role 2"):
        dispatch_role_actions(tuple(actions), execution_horizon=25)


def test_strict_success_requires_correct_release_owner_for_every_cube():
    complete = StackCubeProgress(
        cube_b_on_a=True,
        cube_c_on_b=True,
        cube_b_in_goal=True,
        cube_c_in_goal=True,
        role_grasping_own_cube=(False, False, False),
    )
    wrong_release = StackCubeProgress(
        cube_b_on_a=True,
        cube_c_on_b=True,
        cube_b_in_goal=True,
        cube_c_in_goal=True,
        role_grasping_own_cube=(False, False, True),
    )

    assert strict_success(complete)
    assert not strict_success(wrong_release)


def test_validation_and_fresh_seed_sets_do_not_overlap():
    validation = evaluation_seeds("validation")
    fresh = evaluation_seeds("fresh")

    assert validation == tuple(range(40000, 40200))
    assert fresh == tuple(range(40200, 40400))
    assert set(validation).isdisjoint(fresh)


def test_environment_payload_is_split_into_shared_and_role_local_inputs():
    payload = {
        "images": {
            "head_camera_global": np.full((1, 4, 4, 3), 9, dtype=np.uint8),
            "wrist_camera_agent0": np.full((1, 4, 4, 3), 10, dtype=np.uint8),
            "wrist_camera_agent1": np.full((1, 4, 4, 3), 11, dtype=np.uint8),
            "wrist_camera_agent2": np.full((1, 4, 4, 3), 12, dtype=np.uint8),
        },
        "qpos": [np.full((1, 9), role, dtype=np.float32) for role in range(3)],
    }

    role = extract_role_observation(payload, role_index=2)

    assert role["global_rgb"].shape == (4, 4, 3)
    assert role["wrist_rgb"][0, 0, 0] == 12
    assert role["state"].shape == (9,)
    assert np.all(role["state"] == 2)


def test_rollout_record_preserves_strict_diagnostics():
    response = {
        "success": True,
        "elapsed_steps": 375,
        "task_diagnostics": {
            "strict_success": True,
            "official_success": True,
            "cubeB_on_cubeA": True,
            "cubeC_on_cubeB": True,
            "cubeB_in_goal": True,
            "cubeC_in_goal": True,
            "cubeA_grasped_by_agent0": False,
            "cubeB_grasped_by_agent1": False,
            "cubeC_grasped_by_agent2": False,
        },
        "initial_condition": {"initial_qpos_sha256": "q", "initial_images_sha256": "i"},
    }

    record = rollout_record(seed=40000, planning_rounds=15, response=response)

    assert record["strict_success"] is True
    assert record["planning_rounds"] == 15
    assert record["elapsed_steps"] == 375
    assert record["initial_condition"]["initial_images_sha256"] == "i"


def test_environment_pickle_transport_roundtrips_numpy_payloads():
    sender, receiver = socket.socketpair()
    try:
        payload = {"command": "step_chunk", "actions": np.ones((3, 25, 8))}
        send_pickle(sender, payload)
        restored = receive_pickle(receiver)
    finally:
        sender.close()
        receiver.close()

    assert restored["command"] == "step_chunk"
    np.testing.assert_array_equal(restored["actions"], payload["actions"])


def test_environment_socket_path_stays_within_unix_limit_for_long_outputs(tmp_path):
    output = tmp_path / ("deep_evaluation_directory_" * 8)

    socket_path = short_environment_socket_path(output)

    assert len(str(socket_path).encode()) < 108
    assert socket_path.parent == Path(tempfile.gettempdir())
    assert socket_path == short_environment_socket_path(output)
    assert socket_path != short_environment_socket_path(output / "another_run")
