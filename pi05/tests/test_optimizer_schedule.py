import math

import jax.numpy as jnp

from pi05_fabric.training.optimizer import OptimizerSettings
from pi05_fabric.training.optimizer import create_learning_rate_schedule


def test_warmup_cosine_schedule_reaches_frozen_boundaries():
    settings = OptimizerSettings(
        peak_learning_rate=5e-5,
        final_learning_rate=5e-6,
        warmup_steps=2_000,
        total_steps=40_000,
    )
    schedule = create_learning_rate_schedule(settings)

    assert math.isclose(float(schedule(0)), 5e-5 / 2_001, rel_tol=1e-4)
    assert math.isclose(float(schedule(2_000)), 5e-5, rel_tol=1e-6)
    assert math.isclose(float(schedule(40_000)), 5e-6, rel_tol=1e-6)
    assert jnp.isfinite(schedule(20_000))


def test_optimizer_settings_match_selected_pi05_protocol():
    settings = OptimizerSettings(
        peak_learning_rate=5e-5,
        final_learning_rate=5e-6,
        warmup_steps=500,
        total_steps=10_000,
    )

    assert settings.beta1 == 0.9
    assert settings.beta2 == 0.95
    assert settings.epsilon == 1e-8
    assert settings.weight_decay == 1e-5
    assert settings.clip_gradient_norm == 1.0
