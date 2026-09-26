import copy
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pi05_fabric.training.low_lr_extension import prepare_extension
from pi05_fabric.training.low_lr_extension import validate_extension_resume
from pi05_fabric.training.optimizer import OptimizerSettings, create_learning_rate_schedule
from pi05_fabric.training.stages import StageName, parameter_sha256


def fixture():
    params = {'x': jnp.ones(2)}
    source = {'horizon': 50, 'continuation': {'parent_model_sha256': 'original', 'sampler': 'fixed'}}
    target = copy.deepcopy(source)
    target['continuation']['low_lr_extension'] = {
        'source_step': 10000, 'end_step': 30000,
        'source_model_sha256': parameter_sha256(params),
    }
    snapshot = SimpleNamespace(
        params=params, protocol_metadata=source, step=10000, schedule_step=10000,
        training_seed=7, model_seed=8, parent_model_sha256='original',
        stage=StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
    )
    return snapshot, target


def test_initial_extension_and_subsequent_resume():
    snapshot, protocol = fixture()
    validate_extension_resume(snapshot, protocol=protocol, training_seed=7, model_seed=8)
    snapshot.protocol_metadata = protocol
    snapshot.step = snapshot.schedule_step = 10005
    validate_extension_resume(snapshot, protocol=protocol, training_seed=7, model_seed=8)


def test_expanded_extension_accepts_source_training_contract_change():
    params = {'x': jnp.ones(2)}
    source_protocol = {
        'horizon': 50,
        'initialization': {'kind': 'expanded_continuation'},
        'training_contract': {'optimizer_steps': 10000},
        'continuation': {
            'kind': 'scan_expanded_low_rewarm',
            'parent_model_sha256': 'original',
            'sample_step_offset': 20000,
        },
    }
    target = copy.deepcopy(source_protocol)
    target['training_contract'] = {'optimizer_steps': 40000}
    target['continuation']['low_lr_extension'] = {
        'source_stage': StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION.value,
        'source_step': 10000,
        'end_step': 40000,
        'source_model_sha256': parameter_sha256(params),
    }
    snapshot = SimpleNamespace(
        params=params,
        protocol_metadata=source_protocol,
        step=10000,
        schedule_step=10000,
        training_seed=7,
        model_seed=8,
        parent_model_sha256='original',
        stage=StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
    )
    validate_extension_resume(snapshot, protocol=target, training_seed=7, model_seed=8)


def test_prepare_expanded_extension_preserves_sampler_offset_and_all_rates(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    source_protocol = {
        'action_horizon': 50,
        'execution_horizon': 25,
        'flow_steps': 10,
        'normalization_file': 'normalization_q01_q99.json',
        'normalization': {'schema_version': 2},
        'training_contract': {'global_batch_size': 16},
        'continuation': {
            'kind': 'scan_expanded_low_rewarm',
            'parent_model_sha256': 'parent',
            'sample_step_offset': 20000,
        },
    }
    (source / 'manifest.json').write_text(json.dumps({
        'stage': StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION.value,
        'step': 10000,
        'schedule_step': 10000,
        'model_sha256': 'model',
        'state_sha256': 'state',
        'training_seed': 7,
        'model_seed': 8,
        'parent_model_sha256': 'parent',
        'protocol': source_protocol,
    }))
    spec = tmp_path / 'extension.json'
    spec.write_text(json.dumps({
        'kind': 'low_lr_extension',
        'source_checkpoint': str(source),
        'source_step': 10000,
        'source_model_sha256': 'model',
        'source_state_sha256': 'state',
        'parent_model_sha256': 'parent',
        'trainer_contract': {},
    }))
    config = SimpleNamespace(
        stage=StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
        weights_checkpoint=None,
        stage1_checkpoint=None,
        resume_checkpoint=source,
        resolved_training_seed=7,
        resolved_model_seed=8,
        global_batch_size=16,
        steps=40000,
        dataset=tmp_path,
    )
    args = SimpleNamespace(
        warmup_steps=0,
        schedule_offset=10000,
        action_expert_peak=5e-7,
        action_expert_final=1e-7,
        action_ffw_peak=1e-7,
        action_ffw_final=2e-8,
        paligemma_peak=3e-7,
        paligemma_final=5e-8,
        paligemma_qo_peak=5e-8,
        paligemma_qo_final=1e-8,
        learning_rate=5e-6,
        final_learning_rate=1e-6,
    )
    protocol = {key: source_protocol[key] for key in (
        'action_horizon', 'execution_horizon', 'flow_steps',
        'normalization_file', 'normalization')}
    sampler, contract = prepare_extension(spec, args, config, protocol)
    assert sampler is None
    assert contract['sample_step_offset'] == 20000
    assert contract['low_lr_extension']['source_stage'] == config.stage.value
    assert contract['low_lr_extension']['rates']['action_ffw'] == [1e-7, 2e-8]
    assert contract['low_lr_extension']['rates']['paligemma_qo'] == [5e-8, 1e-8]


@pytest.mark.parametrize('change', ['step', 'seed', 'params', 'sampler', 'schedule', 'lineage'])
def test_extension_rejects_incompatible_source(change):
    snapshot, protocol = fixture()
    if change == 'step': snapshot.step = 9999
    if change == 'seed': snapshot.training_seed = 9
    if change == 'params': snapshot.params = {'x': jnp.zeros(2)}
    if change == 'sampler': snapshot.protocol_metadata['continuation']['sampler'] = 'changed'
    if change == 'schedule': snapshot.schedule_step = 0
    if change == 'lineage': snapshot.parent_model_sha256 = 'other'
    with pytest.raises(ValueError):
        validate_extension_resume(snapshot, protocol=protocol, training_seed=7, model_seed=8)


@pytest.mark.parametrize('start,end', [(5e-7, 1e-7), (3e-7, 1e-7), (5e-6, 1e-6)])
def test_extension_decay_uses_offset_without_resetting_counter(start, end):
    schedule = create_learning_rate_schedule(OptimizerSettings(
        peak_learning_rate=start, final_learning_rate=end,
        warmup_steps=0, total_steps=30000, schedule_offset=10000))
    np.testing.assert_allclose([schedule(s) for s in (10000, 20000, 30000)],
                               [start, (start + end) / 2, end], rtol=1e-6)
    values = [float(schedule(s)) for s in range(10000, 30001, 500)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_nonzero_adam_checkpoint_keeps_moments_counter_rng_and_next_update(tmp_path):
    from pi05_fabric.training.checkpoint import save_training_state, restore_training_state
    from pi05_fabric.training.checkpoint import optimizer_state_dict, restore_optimizer_state
    from pi05_fabric.training.stages import TrainingSnapshot

    old = optax.adamw(lambda s: jnp.asarray(1e-3), b2=0.95)
    new = optax.adamw(lambda s: jnp.asarray(5e-7), b2=0.95)
    params = {'x': jnp.array([1., -2.])}
    state = old.init(params)
    for _ in range(3):
        updates, state = old.update({'x': jnp.array([.2, -.3])}, state, params)
        params = optax.apply_updates(params, updates)
    rng = jax.random.key(7)
    snapshot = TrainingSnapshot(StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
                                3, 3, params, optimizer_state_dict(state), rng, 'original')
    save_training_state(tmp_path / 'checkpoint', snapshot)
    loaded = restore_training_state(tmp_path / 'checkpoint')
    resumed = restore_optimizer_state(new.init(params), loaded.opt_state)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(a, b)
    assert loaded.step == loaded.schedule_step == 3
    np.testing.assert_array_equal(jax.random.key_data(loaded.rng), jax.random.key_data(rng))
    grad = {'x': jnp.array([.7, -.1])}
    expected, _ = new.update(grad, state, params)
    actual, _ = new.update(grad, resumed, loaded.params)
    np.testing.assert_array_equal(actual['x'], expected['x'])
