import jax.numpy as jnp
import pytest
import os
from pathlib import Path
import subprocess
import sys

from pi05_fabric.communication.multiagent_collectives import AgentGroupSpec
from pi05_fabric.communication.multiagent_collectives import pack_peer_contexts
from pi05_fabric.communication.multiagent_collectives import reference_fuse_common
from pi05_fabric.communication.multiagent_collectives import role_ordered_peer_indices
from pi05_fabric.communication.multiagent_collectives import select_role_ordered_peers
from pi05_fabric.communication.multiagent_gemma import residual_peer_contexts


def test_three_agent_peer_order_preserves_sender_identity_without_averaging():
    contexts = jnp.asarray([[[[1.0]]], [[[2.0]]], [[[3.0]]]])

    packed, valid = pack_peer_contexts(
        contexts, receiver_role=1, max_agents=4
    )

    assert packed.shape == (1, 1, 3)
    assert packed[0, 0].tolist() == [1.0, 3.0, 0.0]
    assert valid.tolist() == [True, True, False]


def test_four_agent_peer_order_excludes_receiver_and_keeps_all_peers():
    assert role_ordered_peer_indices(2, agent_count=4, max_agents=4) == (
        0,
        1,
        3,
    )


def test_two_agent_peer_packing_matches_single_peer_context():
    contexts = jnp.arange(2 * 1 * 2 * 2, dtype=jnp.float32).reshape(2, 1, 2, 2)

    packed, valid = pack_peer_contexts(
        contexts, receiver_role=0, max_agents=2
    )

    assert jnp.array_equal(packed, contexts[1])
    assert valid.tolist() == [True]


def test_common_reference_fuses_only_common_owned_tokens():
    values = jnp.asarray(
        [
            [[[1.0], [10.0], [3.0]]],
            [[[3.0], [20.0], [5.0]]],
            [[[5.0], [30.0], [7.0]]],
        ]
    )
    ownership = jnp.asarray([0, 1, 0])

    fused = reference_fuse_common(values, ownership, common_value=0)

    assert fused[:, 0, 0, 0].tolist() == [3.0, 3.0, 3.0]
    assert fused[:, 0, 1, 0].tolist() == [10.0, 20.0, 30.0]
    assert fused[:, 0, 2, 0].tolist() == [5.0, 5.0, 5.0]


def test_common_reference_broadcasts_over_attention_heads():
    values = jnp.arange(3 * 1 * 2 * 4 * 8, dtype=jnp.float32).reshape(
        3, 1, 2, 4, 8
    )
    ownership = jnp.asarray([[0, 1]])

    fused = reference_fuse_common(values, ownership, common_value=0)

    expected_common = jnp.mean(values[:, :, 0], axis=0)
    assert jnp.array_equal(fused[:, :, 0], jnp.broadcast_to(expected_common, (3, 1, 4, 8)))
    assert jnp.array_equal(fused[:, :, 1], values[:, :, 1])


def test_agent_groups_and_role_dp_groups_are_orthogonal():
    spec = AgentGroupSpec.from_world_size(world_size=6, agent_count=3)

    assert spec.agent_groups == ((0, 1, 2), (3, 4, 5))
    assert spec.role_data_parallel_groups == ((0, 3), (1, 4), (2, 5))


def test_world_size_must_contain_complete_agent_groups():
    with pytest.raises(ValueError, match="divisible"):
        AgentGroupSpec.from_world_size(world_size=4, agent_count=3)


def test_dynamic_peer_selection_zero_fills_the_unused_fourth_agent_slot():
    gathered = jnp.asarray([[[1.0]], [[2.0]], [[3.0]]])

    selected, valid = select_role_ordered_peers(
        gathered,
        receiver_role=jnp.asarray(1),
        agent_count=3,
        max_agents=4,
    )

    assert selected[:, 0, 0].tolist() == [1.0, 3.0, 0.0]
    assert valid.tolist() == [True, True, False]


def test_action_peer_contexts_use_independent_softmax_before_concatenation():
    # One query attends separately to two one-token peers. If peers were mixed
    # in one softmax, these exact per-peer values would not both survive.
    query = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)
    peer_keys = jnp.asarray(
        [
            [[[[1.0, 0.0]]]],
            [[[[0.0, 1.0]]]],
            [[[[0.0, 0.0]]]],
        ],
        dtype=jnp.float32,
    )
    peer_values = jnp.asarray(
        [
            [[[[2.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
            [[[[0.0, 0.0]]]],
        ],
        dtype=jnp.float32,
    )
    peer_valid = jnp.asarray(
        [
            [[True]],
            [[True]],
            [[False]],
        ]
    )

    contexts = residual_peer_contexts(
        query,
        peer_keys,
        peer_values,
        peer_valid,
        num_kv_heads=1,
    )

    assert contexts.shape == (1, 1, 6)
    assert contexts[0, 0].tolist() == [2.0, 3.0, 5.0, 7.0, 0.0, 0.0]


def test_forced_cpu_three_device_collectives_and_n2_regression():
    root = Path(__file__).resolve().parents[1]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    source_path = str(root / "src")
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=3",
        "PYTHONPATH": (
            f"{source_path}:{existing_pythonpath}"
            if existing_pythonpath
            else source_path
        ),
    }
    result = subprocess.run(
        [sys.executable, str(root / "scripts/smoke_cpu_three_agent.py")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"n2_regression": true' in result.stdout
