#!/usr/bin/env python3
"""Exercise three-agent collectives and gradients on forced CPU devices."""

from __future__ import annotations

import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import pi0_config

from pi05_fabric.agents.multiagent_pi05 import MultiAgentResidualActionBranch
from pi05_fabric.communication.multiagent_gemma import multiagent_gemma_forward
from pi05_fabric.communication.paired_gemma import paired_gemma_forward


def stack(values):
    return jax.tree.map(lambda *items: jnp.stack(items), *values)


def role_inputs(role):
    prefix = jax.random.normal(jax.random.key(20 + role), (1, 4, 64))
    suffix = jax.random.normal(jax.random.key(30 + role), (1, 2, 64))
    positions = jnp.arange(6, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 6, 6), dtype=bool).at[:, :4, 4:].set(False)
    cond = jax.random.normal(jax.random.key(40 + role), (1, 64))
    ownership = jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int8)
    return prefix, suffix, positions, mask, cond, ownership


def main():
    if jax.device_count() < 3:
        raise RuntimeError("three forced CPU devices are required")
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    configs = (gemma.get_config("dummy"), gemma.get_config("dummy"))
    action_config = configs[1]
    models = [config.create(jax.random.key(10 + role)) for role in range(3)]
    params = stack(
        [nnx.state(model).to_pure_dict()["PaliGemma"]["llm"] for model in models]
    )
    inputs = [role_inputs(role) for role in range(3)]
    prefix, suffix, positions, mask, cond, ownership = (
        jnp.stack([item[index] for item in inputs]) for index in range(6)
    )
    branches = [
        MultiAgentResidualActionBranch(
            depth=4,
            width=64,
            peer_width=action_config.num_heads * action_config.head_dim,
            max_agents=4,
            rngs=nnx.Rngs(100 + role),
        )
        for role in range(3)
    ]
    residual = stack([nnx.state(branch).to_pure_dict() for branch in branches])

    def local_forward(local_params, local_prefix, local_suffix, local_positions, local_mask, local_cond, local_owner, local_residual):
        return multiagent_gemma_forward(
            local_params,
            configs,
            (local_prefix, local_suffix),
            positions=local_positions,
            mask=local_mask,
            adarms_cond=(None, local_cond),
            ownership=local_owner,
            agent_count=3,
            max_agents=4,
            axis_name="agents",
            agent_groups=((0, 1, 2),),
            common_enabled=True,
            private_kv_enabled=True,
            residual_action_enabled=True,
            action_horizon=2,
            residual_params=local_residual,
        )

    mapped = jax.pmap(local_forward, axis_name="agents")
    outputs, trace = mapped(
        params,
        prefix,
        suffix,
        positions,
        mask,
        cond,
        ownership,
        residual,
    )
    jax.block_until_ready(outputs)
    assert outputs[0].shape == (3, 1, 4, 64)
    assert outputs[1].shape == (3, 1, 2, 64)
    assert np.asarray(trace.transmitted_queries).sum() == 0
    assert np.asarray(trace.transmitted_raw_observations).sum() == 0
    assert np.asarray(trace.residual_action_tokens).tolist() == [4, 4, 4]

    def objective(gates):
        changed = {**residual, "gate": gates}
        result, _ = mapped(
            params,
            prefix,
            suffix,
            positions,
            mask,
            cond,
            ownership,
            changed,
        )
        return jnp.sum(result[1])

    gate_gradient = jax.grad(objective)(residual["gate"])
    assert bool(jnp.isfinite(gate_gradient).all())
    assert float(jnp.linalg.norm(gate_gradient)) > 0

    # N=2 Common+Private must reduce to the established paired implementation.
    two_params = jax.tree.map(lambda value: value[:2], params)
    two_inputs = tuple(value[:2] for value in (prefix, suffix, positions, mask, cond, ownership))

    def local_core(local_params, local_prefix, local_suffix, local_positions, local_mask, local_cond, local_owner):
        return multiagent_gemma_forward(
            local_params,
            configs,
            (local_prefix, local_suffix),
            positions=local_positions,
            mask=local_mask,
            adarms_cond=(None, local_cond),
            ownership=local_owner,
            agent_count=2,
            max_agents=2,
            axis_name="agents",
            agent_groups=((0, 1),),
            common_enabled=True,
            private_kv_enabled=True,
            residual_action_enabled=False,
            action_horizon=2,
            residual_params=None,
        )[0]

    mapped_core = jax.pmap(local_core, axis_name="agents", devices=jax.devices()[:2])
    multi_out = mapped_core(two_params, *two_inputs)
    paired_left, paired_right, _ = paired_gemma_forward(
        jax.tree.map(lambda value: value[0], two_params),
        jax.tree.map(lambda value: value[1], two_params),
        configs,
        (prefix[0], suffix[0]),
        (prefix[1], suffix[1]),
        left_positions=positions[0],
        right_positions=positions[1],
        left_mask=mask[0],
        right_mask=mask[1],
        left_adarms_cond=(None, cond[0]),
        right_adarms_cond=(None, cond[1]),
        ownership=ownership[0],
        common_enabled=True,
        private_kv_enabled=True,
    )
    for actual, expected in zip(
        (multi_out[0][0], multi_out[1][0]),
        paired_left,
        strict=True,
    ):
        np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-3, rtol=0)
    for actual, expected in zip(
        (multi_out[0][1], multi_out[1][1]),
        paired_right,
        strict=True,
    ):
        np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-3, rtol=0)

    print(
        json.dumps(
            {
                "passed": True,
                "devices": jax.device_count(),
                "gate_gradient_norm": float(jnp.linalg.norm(gate_gradient)),
                "n2_regression": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
