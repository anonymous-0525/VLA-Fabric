from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi05_fabric.data.four_arm_tasks import FourArmTaskDataset
from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.agent_parallel import batch_configuration
from pi05_fabric.training.agent_parallel import role_rng
from pi05_fabric.training.multiagent_engine import finite_gate_reference
from pi05_fabric.training.stages import fork_stage2
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import TrainingSnapshot


def test_four_rank_topology_owns_one_complete_role_per_rank() -> None:
    topology = AgentParallelTopology.create(world_size=4, agent_count=4)

    assert topology.agent_groups == ((0, 1, 2, 3),)
    assert topology.role_data_parallel_groups == ((0,), (1,), (2,), (3,))
    assert [topology.role_for_rank(rank) for rank in range(4)] == [0, 1, 2, 3]


def test_b6a3_one_team_is_global_batch_18() -> None:
    batch = batch_configuration(
        team_microbatch=6,
        accumulation=3,
        data_parallel_teams=1,
    )

    assert batch.global_team_batch == 18


def test_finite_gate_observes_all_four_roles() -> None:
    assert finite_gate_reference([True, True, True, True])
    assert not finite_gate_reference([True, True, False, True])


def test_four_arm_stage2_fork_resets_optimizer_step_and_requires_new_rng() -> None:
    stage1 = TrainingSnapshot(
        stage=StageName.PI05_FOUR_ARM_COMMON_STAGE1,
        step=10_000,
        schedule_step=10_000,
        params={"w": jnp.asarray([1.0])},
        opt_state={"step": jnp.asarray(10_000)},
        rng=jax.random.key(1),
        parent_model_sha256=None,
        model_seed=17,
        training_seed=31,
    )

    stage2 = fork_stage2(
        stage1,
        target=StageName.PI05_FOUR_ARM_FULL_STAGE2,
        optimizer_init=lambda params: {"fresh": jax.tree.map(jnp.zeros_like, params)},
        rng=jax.random.key(2),
        training_seed=37,
    )

    assert stage2.step == stage2.schedule_step == 0
    assert stage2.training_seed == 37
    with pytest.raises(ValueError, match="must differ"):
        fork_stage2(
            stage1,
            target=StageName.PI05_FOUR_ARM_FULL_STAGE2,
            optimizer_init=lambda params: params,
            rng=jax.random.key(2),
            training_seed=31,
        )


def test_four_arm_dataset_sampling_is_deterministic_and_role_aligned(tmp_path: Path) -> None:
    path = tmp_path / "converted.h5"
    with h5py.File(path, "w") as handle:
        for trajectory_index, steps in enumerate((3, 5)):
            trajectory = handle.create_group(f"trajectory_{trajectory_index:06d}")
            trajectory.create_dataset("global_rgb", data=np.zeros((steps, 2, 2, 3), dtype=np.uint8))
            for group_name, width in (("wrist_rgb", None), ("qpos", 9), ("actions", 7)):
                group = trajectory.create_group(group_name)
                for role in range(4):
                    shape = (steps, 2, 2, 3) if width is None else (steps, width)
                    group.create_dataset(f"role_{role}", data=np.zeros(shape, dtype=np.float32))
    dataset = FourArmTaskDataset(path)

    location_a = dataset.sample_location(np.random.default_rng(9))
    location_b = dataset.sample_location(np.random.default_rng(9))
    team = dataset.team_sample(trajectory_index=location_a[0], step=location_a[1])

    assert location_a == location_b
    assert tuple(role.role_index for role in team.roles) == (0, 1, 2, 3)


def test_four_agent_flow_noise_is_deterministic_but_role_local() -> None:
    keys = [
        role_rng(seed=17, step=4, microstep=2, team_index=0, role_index=role)
        for role in range(4)
    ]

    assert len({tuple(np.asarray(jax.random.key_data(key)).tolist()) for key in keys}) == 4
    repeated = role_rng(seed=17, step=4, microstep=2, team_index=0, role_index=2)
    np.testing.assert_array_equal(jax.random.key_data(keys[2]), jax.random.key_data(repeated))
