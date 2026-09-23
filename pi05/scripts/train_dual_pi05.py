#!/usr/bin/env python3
"""Train one frozen experiment2 stage; dry-run unless --execute is supplied."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer

from pi05_fabric.agents.load_pi05 import load_dual_pi05
from pi05_fabric.data.batch_stack import stack_training_pairs
from pi05_fabric.data.converted_dataset import ConvertedAlohaDataset
from pi05_fabric.data.pi05_batch import LocalNormalization
from pi05_fabric.data.pi05_batch import LocalQuantileNormalization
from pi05_fabric.data.pi05_batch import Pi05TrainingPair
from pi05_fabric.data.pi05_batch import build_training_pair
from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.distributed import process_sample_indices
from pi05_fabric.training.distributed import validate_replica_topology
from pi05_fabric.training.engine import create_engine
from pi05_fabric.training.engine import create_native_engine
from pi05_fabric.training.engine_checkpoint import apply_same_stage_snapshot
from pi05_fabric.training.engine_checkpoint import fork_from_stage1
from pi05_fabric.training.engine_checkpoint import write_engine_checkpoint
from pi05_fabric.training.launch_config import LaunchConfig
from pi05_fabric.training.optimizer import OptimizerSettings
from pi05_fabric.training.optimizer import GroupedOptimizerSettings
from pi05_fabric.training.optimizer import create_optimizer
from pi05_fabric.training.protocol import data_rng_for_sample
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import parameter_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=StageName, required=True, choices=list(StageName))
    parser.add_argument("--dataset", type=Path, default=Path("data/converted/aloha_handover_box"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("external/pi05_base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-device batch size")
    parser.add_argument("--device-count", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--final-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--parent-training-seed", type=int)
    parser.add_argument("--freeze-action-expert-ffw", action="store_true")
    parser.add_argument("--freeze-action-expert-attention", action="store_true")
    parser.add_argument("--freeze-paligemma-base-kv", action="store_true")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--distributed-process-count", type=int, default=1)
    parser.add_argument("--distributed-process-id", type=int, default=0)
    parser.add_argument("--distributed-coordinator-address")
    parser.add_argument("--preload-dataset", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _initialize_distributed(args: argparse.Namespace) -> bool:
    if args.distributed_process_count == 1:
        if args.distributed_process_id != 0 or args.distributed_coordinator_address is not None:
            raise ValueError("single-process mode cannot set a distributed rank or coordinator")
        return False
    if args.distributed_coordinator_address is None:
        raise ValueError("distributed_coordinator_address is required for multi-process training")
    if not 0 <= args.distributed_process_id < args.distributed_process_count:
        raise ValueError("distributed_process_id is outside the distributed world")
    jax.distributed.initialize(
        coordinator_address=args.distributed_coordinator_address,
        num_processes=args.distributed_process_count,
        process_id=args.distributed_process_id,
        local_device_ids=0,
    )
    return True


def _launch_config(args: argparse.Namespace) -> LaunchConfig:
    return LaunchConfig(
        stage=args.stage,
        dataset=args.dataset,
        base_checkpoint=args.base_checkpoint,
        output=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        warmup_steps=args.warmup_steps,
        device_count=args.device_count,
        gradient_accumulation=args.gradient_accumulation,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        model_seed=args.model_seed,
        training_seed=args.training_seed,
        parent_training_seed=args.parent_training_seed,
        stage1_checkpoint=args.stage1_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )


def _device_pair(pair):
    return Pi05TrainingPair(
        jax.tree.map(jnp.asarray, pair.left_observation),
        jax.tree.map(jnp.asarray, pair.right_observation),
        jnp.asarray(pair.left_actions),
        jnp.asarray(pair.right_actions),
        jnp.asarray(pair.ownership),
    )


def _host_scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _sync(name: str, *, distributed: bool) -> None:
    if distributed:
        multihost_utils.sync_global_devices(name)


def main() -> None:
    args = _parse_args()
    distributed = _initialize_distributed(args) if args.execute else False
    process_index = jax.process_index()
    primary = process_index == 0
    print(
        f"rank={process_index} process_pid={os.getpid()} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    config = _launch_config(args)
    config.validate_paths()
    selected_devices = tuple(jax.local_devices())
    action_horizon = 50 if config.mode.is_v2 else 20
    execution_horizon = 25 if config.mode.is_v2 else 10
    normalization_path = config.dataset / (
        "normalization_q01_q99.json" if config.mode.is_v2 else "normalization.json"
    )
    normalization_payload = json.loads(normalization_path.read_text())
    if config.mode.is_v2:
        manifest_bytes = (config.dataset / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if normalization_payload.get("dataset_manifest_sha256") != manifest_sha:
            raise ValueError("V2 quantile statistics do not match the dataset manifest")
        if normalization_payload.get("schema_version") != 2:
            raise ValueError("V2 requires q01/q99 normalization schema 2")
    protocol_metadata = {
        "action_horizon": action_horizon,
        "execution_horizon": execution_horizon,
        "flow_steps": 10,
        "normalization_file": normalization_path.name,
        "normalization": normalization_payload,
    }
    final_lr = config.learning_rate if config.final_learning_rate is None else config.final_learning_rate
    optimizer_settings = OptimizerSettings(
        peak_learning_rate=config.learning_rate,
        final_learning_rate=final_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    )
    plan = {
        "stage": config.stage.value,
        "mode": config.mode.value,
        "dataset": str(config.dataset.resolve()),
        "base_checkpoint": str(config.base_checkpoint.resolve()),
        "output": str(config.output.resolve()),
        "optimizer_steps": config.steps,
        "per_device_batch_size": config.batch_size,
        "global_batch_size": config.global_batch_size,
        "device_count": config.device_count,
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "gradient_accumulation": config.gradient_accumulation,
        "optimizer": {
            "name": "adamw",
            "peak_learning_rate": config.learning_rate,
            "final_learning_rate": final_lr,
            "warmup_steps": config.warmup_steps,
            "beta1": optimizer_settings.beta1,
            "beta2": optimizer_settings.beta2,
            "epsilon": optimizer_settings.epsilon,
            "weight_decay": optimizer_settings.weight_decay,
            "clip_gradient_norm": optimizer_settings.clip_gradient_norm,
            "scheduler": "warmup_cosine",
        },
        "checkpoint_every": config.checkpoint_every,
        "checkpoint_steps": list(config.checkpoint_steps),
        "model_seed": config.resolved_model_seed,
        "training_seed": config.resolved_training_seed,
        "parent_training_seed": config.parent_training_seed,
        "train_action_expert_ffw": not args.freeze_action_expert_ffw,
        "train_action_expert_attention": not args.freeze_action_expert_attention,
        "train_paligemma_base_kv": not args.freeze_paligemma_base_kv,
        "stage1_checkpoint": None if config.stage1_checkpoint is None else str(config.stage1_checkpoint.resolve()),
        "resume_checkpoint": None if config.resume_checkpoint is None else str(config.resume_checkpoint.resolve()),
        "topology": "one_process_per_gpu_replicated_dual_agent_graph",
        "preload_dataset": args.preload_dataset,
        "preload_workers": args.preload_workers,
        "protocol": protocol_metadata,
    }
    if config.mode.is_native:
        plan["optimizer"] = {
            "name": "grouped_adamw",
            "scheduler": "warmup_cosine",
            "warmup_steps": config.warmup_steps,
            "beta1": optimizer_settings.beta1,
            "beta2": optimizer_settings.beta2,
            "epsilon": optimizer_settings.epsilon,
            "weight_decay": optimizer_settings.weight_decay,
            "global_clip_gradient_norm": optimizer_settings.clip_gradient_norm,
            "groups": {
                "action_expert_base": {
                    "peak_learning_rate": 5e-6,
                    "final_learning_rate": 5e-7,
                },
                "paligemma_kv_norm": {
                    "peak_learning_rate": 3e-6,
                    "final_learning_rate": 3e-7,
                },
                "lora_and_projections": {
                    "peak_learning_rate": config.learning_rate,
                    "final_learning_rate": final_lr,
                },
            },
        }
    if primary:
        print(json.dumps(plan, indent=2))
    if not args.execute:
        if primary:
            print("DRY_RUN_ONLY: add --execute to start training")
        if distributed:
            jax.distributed.shutdown()
        return

    validate_replica_topology(
        requested_device_count=config.device_count,
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
    )

    _sync("validated_launch", distributed=distributed)
    if primary:
        config.output.mkdir(parents=True, exist_ok=config.resume_checkpoint is not None)
        manifest_name = "resume_launch_manifest.json" if config.resume_checkpoint is not None else "launch_manifest.json"
        (config.output / manifest_name).write_text(json.dumps(plan, indent=2) + "\n")
    _sync("created_output", distributed=distributed)

    normalization = (
        LocalQuantileNormalization.from_json(normalization_path)
        if config.mode.is_v2
        else LocalNormalization.from_json(normalization_path)
    )
    tokenizer = PaligemmaTokenizer(200)

    load_started = time.monotonic()
    _, model = load_dual_pi05(
        config.base_checkpoint,
        seed=config.resolved_model_seed,
        mode=config.mode,
        train_action_ffw=not args.freeze_action_expert_ffw,
        train_action_attention=not args.freeze_action_expert_attention,
        train_paligemma_kv=not args.freeze_paligemma_base_kv,
        action_horizon=action_horizon,
    )
    load_seconds = time.monotonic() - load_started
    _sync("loaded_models", distributed=distributed)
    print(f"rank={process_index} dual_model_load_seconds={load_seconds:.3f}")
    if config.mode.is_native:
        grouped_settings = GroupedOptimizerSettings(
            total_steps=config.steps,
            warmup_steps=config.warmup_steps,
            action_expert_peak=5e-6,
            action_expert_final=5e-7,
            paligemma_peak=3e-6,
            paligemma_final=3e-7,
            adapter_peak=config.learning_rate,
            adapter_final=final_lr,
        )
        engine = create_native_engine(
            model,
            mode=config.mode,
            settings=grouped_settings,
            train_action_ffw=not args.freeze_action_expert_ffw,
            train_action_attention=not args.freeze_action_expert_attention,
            train_paligemma_kv=not args.freeze_paligemma_base_kv,
        )
    else:
        engine = create_engine(
            model,
            mode=config.mode,
            tx=create_optimizer(optimizer_settings),
        )
    del model
    gc.collect()
    rng = jax.random.key(config.resolved_training_seed)
    parent_hash = None
    if config.stage1_checkpoint is not None:
        stage1 = restore_training_state(config.stage1_checkpoint)
        if (
            config.mode.is_native
            and stage1.training_seed is not None
            and stage1.training_seed == config.resolved_training_seed
        ):
            raise ValueError("Stage 2 training seed must differ from Stage 1")
        forked = fork_from_stage1(
            engine,
            stage1,
            target=config.stage,
            rng=rng,
            model_seed=config.resolved_model_seed,
            training_seed=config.resolved_training_seed,
            protocol_metadata=protocol_metadata,
        )
        parent_hash = forked.parent_model_sha256
        del stage1
        del forked
        gc.collect()
    elif config.resume_checkpoint is not None:
        resumed = restore_training_state(config.resume_checkpoint)
        apply_same_stage_snapshot(engine, resumed, expected_stage=config.stage)
        rng = resumed.rng
        parent_hash = resumed.parent_model_sha256
        del resumed
        gc.collect()

    if config.device_count > 1:
        engine.enable_data_parallel(selected_devices)
    _sync("allocated_training_state", distributed=distributed)

    dataset_started = time.monotonic()
    dataset = ConvertedAlohaDataset(
        config.dataset,
        action_horizon=action_horizon,
        preload=args.preload_dataset,
        preload_workers=args.preload_workers,
    )
    dataset_seconds = time.monotonic() - dataset_started
    _sync("loaded_dataset", distributed=distributed)
    print(
        f"rank={process_index} dataset_load_seconds={dataset_seconds:.3f} "
        f"preloaded_gib={dataset.preloaded_bytes / 2**30:.3f}"
    )
    checkpoint_targets = set(config.checkpoint_steps)
    samples_per_process = (
        config.batch_size
        * jax.local_device_count()
        * config.gradient_accumulation
    )
    sample_indices = process_sample_indices(
        process_index=process_index,
        samples_per_process=samples_per_process,
    )
    for optimizer_step in range(engine.current_step(), config.steps):
        pairs = []
        for sample_index in sample_indices:
            data_rng = data_rng_for_sample(
                seed=config.resolved_training_seed,
                step=optimizer_step,
                sample_index=sample_index,
            )
            episode_id, episode_step = dataset.sample_index(data_rng)
            pairs.append(
                build_training_pair(
                    dataset.get(episode_id, episode_step),
                    normalization=normalization,
                    tokenizer=tokenizer,
                )
            )
        pair = _device_pair(stack_training_pairs(pairs))
        rng, step_rng = jax.random.split(rng)
        engine.state, info = engine.step(
            engine.state,
            step_rng,
            pair.left_observation,
            pair.right_observation,
            pair.left_actions,
            pair.right_actions,
            pair.ownership,
            gradient_accumulation=config.gradient_accumulation,
        )
        loss = _host_scalar(info.loss)
        left_loss = _host_scalar(info.left_loss)
        right_loss = _host_scalar(info.right_loss)
        grad_norm = _host_scalar(info.grad_norm)
        local_loss = _host_scalar(info.local_loss)
        local_left_loss = _host_scalar(info.local_left_loss)
        local_right_loss = _host_scalar(info.local_right_loss)
        local_grad_norm = _host_scalar(info.local_grad_norm)
        finite = bool(np.asarray(jax.device_get(info.finite)).all())
        current_step = engine.current_step()
        print(
            f"rank_step={current_step} local_loss={local_loss:.6f} "
            f"local_left_loss={local_left_loss:.6f} "
            f"local_right_loss={local_right_loss:.6f} "
            f"local_grad_norm={local_grad_norm:.6f}"
        )
        if not finite:
            print(f"rank={process_index} non_finite_local_step={current_step}")
            raise FloatingPointError(f"non-finite step rejected at step {current_step}")
        if primary:
            print(
                f"step={current_step} loss={loss:.6f} "
                f"left_loss={left_loss:.6f} right_loss={right_loss:.6f} "
                f"grad_norm={grad_norm:.6f}"
            )

        if current_step in checkpoint_targets:
            if primary:
                checkpoint_started = time.monotonic()
                checkpoint = config.output / "checkpoints" / f"step_{current_step:08d}"
                write_engine_checkpoint(
                    checkpoint,
                    engine,
                    stage=config.stage,
                    rng=rng,
                    parent_model_sha256=parent_hash,
                    model_seed=config.resolved_model_seed,
                    training_seed=config.resolved_training_seed,
                    protocol_metadata=protocol_metadata,
                )
                print(
                    f"CHECKPOINT={checkpoint} "
                    f"save_seconds={time.monotonic() - checkpoint_started:.3f}"
                )
            _sync(f"checkpoint_{current_step}", distributed=distributed)

    final_step = engine.current_step()
    final_state = engine.host_state()
    final_digest = parameter_sha256(engine.selected_params(final_state).to_pure_dict())
    if distributed:
        multihost_utils.assert_equal(
            np.frombuffer(bytes.fromhex(final_digest), dtype=np.uint8),
            "final trainable parameter digests differ across ranks",
        )
    print(f"rank={process_index} final_step={final_step} final_model_sha256={final_digest}")
    memory_stats = selected_devices[0].memory_stats() or {}
    print(
        "JAX_MEMORY_STATS "
        + json.dumps(
            {
                key: int(value)
                for key, value in memory_stats.items()
                if isinstance(value, (int, np.integer))
            },
            sort_keys=True,
        )
    )
    final_checkpoint = config.output / "checkpoints" / f"step_{final_step:08d}"
    if primary and not final_checkpoint.exists():
        write_engine_checkpoint(
            final_checkpoint,
            engine,
            stage=config.stage,
            rng=rng,
            parent_model_sha256=parent_hash,
            model_seed=config.resolved_model_seed,
            training_seed=config.resolved_training_seed,
            protocol_metadata=protocol_metadata,
        )
    _sync("final_checkpoint", distributed=distributed)
    if primary:
        print(f"FINAL_CHECKPOINT={final_checkpoint}")
    if distributed:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
