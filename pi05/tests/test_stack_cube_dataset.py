import h5py
import numpy as np
import pytest

from pi05_fabric.data.stack_cube_multiagent import compute_role_quantiles
from pi05_fabric.data.stack_cube_multiagent import compute_dataset_statistics
from pi05_fabric.data.stack_cube_multiagent import normalize_quantile
from pi05_fabric.data.stack_cube_multiagent import pad_action_horizon
from pi05_fabric.data.stack_cube_multiagent import StackCubeMultiAgentDataset
from pi05_fabric.data.stack_cube_multiagent import unnormalize_quantile
from pi05_fabric.data.stack_cube_multiagent import validate_stack_cube_h5
from pi05_fabric.data.stack_cube_pi05 import StackCubeQuantileNormalization
from pi05_fabric.data.stack_cube_pi05 import build_role_policy_observation
from pi05_fabric.data.stack_cube_pi05 import build_role_training_sample
from pi05_fabric.data.stack_cube_pi05 import stack_role_training_samples


class _Tokenizer:
    _max_len = 6

    class _Inner:
        @staticmethod
        def encode(_text, add_bos):
            assert add_bos
            return [1, 2, 3]

    _tokenizer = _Inner()

    def tokenize(self, _prompt, _state):
        return (
            np.asarray([1, 2, 3, 4, 0, 0], dtype=np.int32),
            np.asarray([True, True, True, True, False, False]),
        )


def test_action_horizon_repeats_terminal_action():
    actions = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)

    chunk = pad_action_horizon(actions, start=2, horizon=5)

    assert chunk.shape == (5, 8)
    np.testing.assert_array_equal(chunk[:2], actions[2:])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[-1:], 3, axis=0))


def test_role_quantile_roundtrip_is_finite():
    values = np.asarray([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float32)
    stats = compute_role_quantiles(values)

    normalized = normalize_quantile(values, stats)
    restored = unnormalize_quantile(normalized, stats)

    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(restored, values, atol=1e-5)


def test_stack_cube_schema_requires_three_aligned_roles(tmp_path):
    path = tmp_path / "stack.h5"
    with h5py.File(path, "w") as handle:
        traj = handle.create_group("traj_0")
        actions = traj.create_group("actions")
        obs = traj.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        sensors.create_group("head_camera_global").create_dataset(
            "rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8)
        )
        for role in range(3):
            actions.create_dataset(f"panda-{role}", data=np.zeros((2, 8), dtype=np.float32))
            agents.create_group(f"panda-{role}").create_dataset(
                "qpos", data=np.zeros((3, 9), dtype=np.float32)
            )
            sensors.create_group(f"wrist_camera_agent{role}").create_dataset(
                "rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8)
            )

    audit = validate_stack_cube_h5(path, expected_trajectories=1)

    assert audit["trajectory_count"] == 1
    assert audit["agent_count"] == 3
    assert audit["state_dimension"] == 9
    assert audit["action_dimension"] == 8


def test_stack_cube_schema_rejects_missing_role(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("traj_0")

    with pytest.raises(ValueError, match="panda-0"):
        validate_stack_cube_h5(path, expected_trajectories=1)


def test_dataset_returns_aligned_role_local_h50_sample(tmp_path):
    path = tmp_path / "stack.h5"
    with h5py.File(path, "w") as handle:
        traj = handle.create_group("traj_0")
        actions = traj.create_group("actions")
        obs = traj.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        sensors.create_group("head_camera_global").create_dataset(
            "rgb", data=np.full((4, 3, 3, 3), 9, dtype=np.uint8)
        )
        for role in range(3):
            actions.create_dataset(
                f"panda-{role}", data=np.full((3, 8), role + 1, dtype=np.float32)
            )
            agents.create_group(f"panda-{role}").create_dataset(
                "qpos", data=np.full((4, 9), role + 2, dtype=np.float32)
            )
            sensors.create_group(f"wrist_camera_agent{role}").create_dataset(
                "rgb", data=np.full((4, 3, 3, 3), role + 3, dtype=np.uint8)
            )

    dataset = StackCubeMultiAgentDataset(path, action_horizon=50)
    sample = dataset.get(role_index=2, trajectory_index=0, step=1)

    assert dataset.sample_count == 3
    assert sample["role"] == "panda-2"
    assert sample["global_rgb"].shape == (3, 3, 3)
    assert sample["wrist_rgb"][0, 0, 0] == 5
    assert sample["state"].shape == (9,)
    assert sample["actions"].shape == (50, 8)
    assert np.all(sample["actions"] == 3)

    statistics = compute_dataset_statistics(path)
    assert tuple(statistics) == ("panda-0", "panda-1", "panda-2")
    assert statistics["panda-0"]["state"].q01.shape == (9,)
    assert statistics["panda-0"]["action"].q99.shape == (8,)


def test_role_local_pi05_batch_exposes_only_global_and_own_wrist(tmp_path):
    stats_path = tmp_path / "stats.npz"
    arrays = {}
    for role in range(3):
        arrays[f"panda-{role}_state_q01"] = np.zeros(9, dtype=np.float32)
        arrays[f"panda-{role}_state_q99"] = np.full(9, 10, dtype=np.float32)
        arrays[f"panda-{role}_action_q01"] = np.zeros(8, dtype=np.float32)
        arrays[f"panda-{role}_action_q99"] = np.full(8, 10, dtype=np.float32)
    np.savez(stats_path, **arrays)
    normalization = StackCubeQuantileNormalization.from_npz(stats_path)
    raw = {
        "role_index": 2,
        "global_rgb": np.full((4, 4, 3), 9, dtype=np.uint8),
        "wrist_rgb": np.full((4, 4, 3), 7, dtype=np.uint8),
        "state": np.full(9, 5, dtype=np.float32),
        "actions": np.full((50, 8), 5, dtype=np.float32),
    }

    sample = build_role_training_sample(
        raw,
        normalization=normalization,
        tokenizer=_Tokenizer(),
        instruction="stack the three cubes",
        image_tokens_per_view=2,
    )

    assert sample.observation.state.shape == (1, 32)
    assert sample.actions.shape == (1, 50, 32)
    assert np.isclose(
        sample.observation.images["base_0_rgb"][0, 0, 0, 0],
        9 / 127.5 - 1,
    )
    assert np.isclose(
        sample.observation.images["left_wrist_0_rgb"][0, 0, 0, 0],
        7 / 127.5 - 1,
    )
    assert not sample.observation.image_masks["right_wrist_0_rgb"][0]
    assert sample.ownership.shape == (1, 12)

    stacked = stack_role_training_samples([sample, sample])
    assert stacked.observation.state.shape == (2, 32)
    assert stacked.actions.shape == (2, 50, 32)
    assert stacked.ownership.shape == (2, 12)


def test_rollout_observation_uses_the_same_role_local_boundary(tmp_path):
    stats_path = tmp_path / "stats.npz"
    arrays = {}
    for role in range(3):
        arrays[f"panda-{role}_state_q01"] = np.zeros(9, dtype=np.float32)
        arrays[f"panda-{role}_state_q99"] = np.full(9, 10, dtype=np.float32)
        arrays[f"panda-{role}_action_q01"] = np.zeros(8, dtype=np.float32)
        arrays[f"panda-{role}_action_q99"] = np.full(8, 10, dtype=np.float32)
    np.savez(stats_path, **arrays)

    observation, ownership = build_role_policy_observation(
        role_index=1,
        global_rgb=np.full((4, 4, 3), 11, dtype=np.uint8),
        wrist_rgb=np.full((4, 4, 3), 13, dtype=np.uint8),
        state=np.full(9, 5, dtype=np.float32),
        normalization=StackCubeQuantileNormalization.from_npz(stats_path),
        tokenizer=_Tokenizer(),
        instruction="stack the three cubes",
        image_tokens_per_view=2,
    )

    assert observation.state.shape == (1, 32)
    assert observation.images["base_0_rgb"].shape == (1, 4, 4, 3)
    assert observation.images["left_wrist_0_rgb"].shape == (1, 4, 4, 3)
    assert not observation.image_masks["right_wrist_0_rgb"][0]
    assert ownership.shape == (1, 12)
