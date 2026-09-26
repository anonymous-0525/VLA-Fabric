import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pi05_fabric.evaluation.robotwin2_adapter import observation_to_policy_payload
from pi05_fabric.evaluation.robotwin2_adapter import policy_r6_chunk_to_robotwin_ee
from pi05_fabric.evaluation.robotwin2_adapter import quaternion_wxyz_to_r6
from pi05_fabric.evaluation.robotwin2_adapter import r6_to_quaternion_wxyz
from pi05_fabric.evaluation.robotwin2_adapter import validate_policy_payload


def _observation():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image + 1},
            "right_camera": {"rgb": image + 2},
        },
        "endpose": {
            "left_endpose": np.asarray([0.1, 0.2, 0.3, 1, 0, 0, 0]),
            "left_gripper": 0.4,
            "right_endpose": np.asarray([-0.1, 0.5, 0.2, 1, 0, 0, 0]),
            "right_gripper": 0.6,
        },
    }


def test_quaternion_r6_round_trip_preserves_rotation():
    generator = np.random.default_rng(7)
    quaternions_xyzw = Rotation.random(20, random_state=generator).as_quat()
    for xyzw in quaternions_xyzw:
        wxyz = np.asarray([xyzw[3], *xyzw[:3]])
        restored = r6_to_quaternion_wxyz(quaternion_wxyz_to_r6(wxyz))
        expected_matrix = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
        actual_matrix = Rotation.from_quat(restored, scalar_first=True).as_matrix()
        np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-6)


def test_observation_payload_maps_images_and_two_local_states():
    payload = observation_to_policy_payload(
        _observation(),
        instruction="handover the red block",
        condition_id=11,
        planning_call=3,
        request_id="11:3",
    )
    validate_policy_payload(payload)

    assert payload["eef_state"].shape == (20,)
    np.testing.assert_allclose(payload["eef_state"][:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(payload["eef_state"][10:13], [-0.1, 0.5, 0.2])
    assert int(payload["left_wrist_image"][0, 0, 0]) == 1
    assert int(payload["right_wrist_image"][0, 0, 0]) == 2


def test_action_adapter_maps_each_arm_to_xyz_quaternion_gripper():
    actions = np.zeros((2, 20), dtype=np.float32)
    actions[:, 0:3] = [1, 2, 3]
    actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
    actions[:, 9] = 0.25
    actions[:, 10:13] = [4, 5, 6]
    actions[:, 13:19] = [1, 0, 0, 0, 1, 0]
    actions[:, 19] = 0.75

    converted = policy_r6_chunk_to_robotwin_ee(actions)

    assert converted.shape == (2, 16)
    np.testing.assert_allclose(converted[:, 0:3], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(converted[:, 3:7], [[1, 0, 0, 0], [1, 0, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(converted[:, 7], 0.25)
    np.testing.assert_allclose(converted[:, 8:11], [[4, 5, 6], [4, 5, 6]])
    np.testing.assert_allclose(converted[:, 11:15], [[1, 0, 0, 0], [1, 0, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(converted[:, 15], 0.75)


def test_adapter_rejects_non_uint8_image_and_bad_action_shape():
    observation = _observation()
    observation["observation"]["head_camera"]["rgb"] = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        observation_to_policy_payload(
            observation,
            instruction="task",
            condition_id=0,
            planning_call=0,
            request_id="0:0",
        )
    with pytest.raises(ValueError, match="shape"):
        policy_r6_chunk_to_robotwin_ee(np.zeros((50, 18), dtype=np.float32))
