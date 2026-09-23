from pathlib import Path

import h5py
import numpy as np

from pi05_fabric.data.four_arm_tasks import FourArmQuantileNormalization
from pi05_fabric.data.four_arm_tasks import FourArmTaskDataset
from pi05_fabric.data.four_arm_tasks import QuantileStats
from pi05_fabric.data.four_arm_tasks import compute_four_arm_quantiles
from pi05_fabric.data.four_arm_tasks import convert_four_arm_release
from pi05_fabric.data.four_arm_tasks import normalize_quantile
from pi05_fabric.data.four_arm_tasks import unnormalize_quantile


def _write_source(path: Path, *, include_frontview: bool = True) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["num_agents"] = 4
        handle.attrs["training_ready"] = True
        handle.attrs["preview_only"] = False
        handle.attrs["environment"] = "multiarm_sim.FourArmArchAssembly"
        trajectory = handle.create_group("trajectory_000000")
        trajectory.attrs["success"] = True
        trajectory.attrs["complete"] = True
        trajectory.attrs["seed"] = 6000
        trajectory.attrs["instruction"] = "assemble the arch"
        trajectory.attrs["local_cameras_json"] = (
            '["robot0_eye_in_hand", "robot1_eye_in_hand", '
            '"robot2_eye_in_hand", "robot3_eye_in_hand"]'
        )
        actions = trajectory.create_group("actions")
        obs = trajectory.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        sensors.create_group("agentview").create_dataset(
            "rgb", data=np.full((4, 6, 6, 3), 11, dtype=np.uint8)
        )
        if include_frontview:
            sensors.create_group("frontview").create_dataset(
                "rgb", data=np.full((4, 6, 6, 3), 99, dtype=np.uint8)
            )
        for role in range(4):
            actions.create_dataset(
                f"panda-{role}",
                data=np.arange(4 * 7, dtype=np.float32).reshape(4, 7) + role,
            )
            agents.create_group(f"panda-{role}").create_dataset(
                "qpos",
                data=np.arange(4 * 9, dtype=np.float32).reshape(4, 9) + role,
            )
            sensors.create_group(f"robot{role}_eye_in_hand").create_dataset(
                "rgb", data=np.full((4, 4, 4, 3), role + 1, dtype=np.uint8)
            )


def test_same_team_index_aligns_all_four_roles(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    converted = tmp_path / "converted.h5"
    _write_source(source)
    convert_four_arm_release(source, converted)

    dataset = FourArmTaskDataset(converted, action_horizon=50)
    team = dataset.team_sample(trajectory_index=0, step=2)

    assert tuple(sample.role_index for sample in team.roles) == (0, 1, 2, 3)
    assert len({sample.team_id for sample in team.roles}) == 1
    assert all(sample.step == 2 for sample in team.roles)
    assert all(sample.actions.shape == (50, 7) for sample in team.roles)
    assert all(np.array_equal(sample.actions[-1], sample.actions[-2]) for sample in team.roles)


def test_conversion_keeps_agentview_and_local_wrists_but_not_frontview(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    converted = tmp_path / "converted.h5"
    _write_source(source, include_frontview=True)

    report = convert_four_arm_release(source, converted)

    assert report.trajectories == 1
    with h5py.File(converted, "r") as handle:
        trajectory = handle["trajectory_000000"]
        assert "global_rgb" in trajectory
        assert "frontview" not in trajectory
        assert tuple(trajectory["global_rgb"].shape) == (4, 6, 6, 3)
        assert tuple(trajectory["wrist_rgb/role_3"].shape) == (4, 4, 4, 3)
        assert tuple(trajectory["qpos/role_3"].shape) == (4, 9)
        assert tuple(trajectory["actions/role_3"].shape) == (4, 7)


def test_role_quantiles_are_finite_and_roundtrip_actions(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    converted = tmp_path / "converted.h5"
    stats_path = tmp_path / "stats.npz"
    _write_source(source)
    convert_four_arm_release(source, converted)

    normalization = compute_four_arm_quantiles(converted, stats_path)
    loaded = FourArmQuantileNormalization.from_npz(stats_path)
    actions = np.asarray([[0, 1, 2, 3, 4, 5, 6]], dtype=np.float32)
    normalized = normalize_quantile(actions, loaded.for_role(0).action)
    restored = unnormalize_quantile(normalized, normalization.for_role(0).action)

    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(restored, actions, atol=1e-5)


def test_constant_action_dimensions_follow_openpi_epsilon_contract() -> None:
    values = np.zeros((4, 3), dtype=np.float32)
    stats = QuantileStats(q01=np.zeros(3, dtype=np.float32), q99=np.zeros(3, dtype=np.float32))

    normalized = normalize_quantile(values, stats)
    restored = unnormalize_quantile(normalized, stats)

    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(restored, values, atol=1e-5)
