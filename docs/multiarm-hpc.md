# Multi-Arm pi0.5 Workflows on a Single-Node HPC

The pi0.5 VLA-Fabric multi-arm workflows assign one complete policy process per arm on a single host. Each agent rank owns one GPU, local observations, one role checkpoint shard, and one local action chunk. Cross-rank collectives carry the configured Common, Private Prefix K/V, and peer-action representations.

## Task Configurations

| Team | Task | Task config |
|---|---|---|
| 3 agents | ThreeRobotsStackCube | `pi05/configs/multiarm/three_arm_stack_cube.yaml` |
| 4 agents | Frame Insertion | `pi05/configs/multiarm/four_arm_frame_insertion.yaml` |
| 4 agents | Arch Assembly | `pi05/configs/multiarm/four_arm_arch_assembly.yaml` |

## Environment

Install OpenPI and this package as described in `docs/installation.md`. Use `pi05/scripts/multiarm/env.example` to configure the dataset, model, simulator, and output paths. Set `GPU_IDS` explicitly to the devices allocated to the run.

```bash
export OPENPI_ROOT=/path/to/openpi
export PYTHON_BIN=/path/to/openpi-env/bin/python
export PI05_BASE_CHECKPOINT=/path/to/pi05_base
export OUTPUT_ROOT="$PWD/outputs"
export GPU_IDS=1,2,3
```

The generic launcher accepts exactly three or four comma-separated device IDs and exposes one device to each rank:

```bash
pi05/scripts/multiarm/launch_agent_ranks.sh \
  "$GPU_IDS" 29731 "$OUTPUT_ROOT/logs" -- COMMAND [ARGS...]
```

It disables JAX preallocation, appends the distributed process arguments, propagates a failure to the remaining ranks, and writes one log per rank.

## Three-Agent StackCube

Audit and convert the source release before training:

```bash
$PYTHON_BIN pi05/scripts/multiarm/convert_stack_cube.py \
  --source "$STACK_CUBE_DATASET" \
  --output /path/to/stack_cube_converted \
  --expected-trajectories 150
export STACK_CUBE_QUANTILES=/path/to/stack_cube_converted/role_quantiles.npz
```

The paper protocol predicts H50 and executes E25 with ten flow steps. Stage 1 trains Common interaction for 10k optimizer updates. Stage 2 restores the complete Stage-1 team checkpoint, resets optimizer/schedule/RNG, enables Full interaction, and trains for 50k updates.

```bash
pi05/scripts/multiarm/train_three_arm_stage1.sh
export STAGE1_CHECKPOINT=$OUTPUT_ROOT/stack_cube_stage1_common_10k/checkpoints/step_00010000
export COORDINATOR_PORT=29732
pi05/scripts/multiarm/train_three_arm_stage2.sh
```

Strict evaluation uses disjoint validation seeds 40000--40199 and fresh seeds 40200--40399. Configure the external RoboFactory interpreter, root, IPC server, and environment YAML, then run:

```bash
export CHECKPOINT=/path/to/complete/team/checkpoint
export OUTPUT=$OUTPUT_ROOT/stack_cube_fresh200
export SEED_START=40200
export SEED_COUNT=200
pi05/scripts/multiarm/evaluate_three_arm.sh
```

## Four-Agent Tasks

Convert a training-ready, success-only four-agent HDF5 release and generate role-specific quantiles:

```bash
$PYTHON_BIN pi05/scripts/multiarm/convert_four_arm_tasks.py \
  --source "$FOUR_ARM_SOURCE" --output "$FOUR_ARM_DATASET" \
  --manifest /path/to/conversion_manifest.json
$PYTHON_BIN pi05/scripts/multiarm/generate_four_arm_quantiles.py \
  --dataset "$FOUR_ARM_DATASET" --output "$FOUR_ARM_QUANTILES" \
  --audit /path/to/quantile_audit.json
```

Create a SHA-256 manifest whose relative paths are resolved from the manifest directory:

```bash
cd "$(dirname "$FOUR_ARM_DATASET")"
sha256sum "$(basename "$FOUR_ARM_DATASET")" "$(basename "$FOUR_ARM_QUANTILES")" \
  > converted_sha256.txt
export FOUR_ARM_AUDIT_MANIFEST=$PWD/converted_sha256.txt
```

Choose four allocated devices explicitly. The example uses team microbatch 6
and accumulation 3 (effective team batch 18). Check memory use on the target
machine before a long run; smaller equivalent-batch configurations are listed
in `pi05/configs/multiarm/four_gpu_agent_parallel.yaml`.

```bash
export GPU_IDS=1,2,3,4
export TEAM_MICROBATCH=6
export ACCUMULATION=3
pi05/scripts/multiarm/train_four_arm_stage1.sh
export STAGE1_CHECKPOINT=$OUTPUT_ROOT/four_arm_stage1_common_10k/checkpoints/step_00010000
pi05/scripts/multiarm/train_four_arm_stage2.sh
```

For evaluation, set `FOUR_ARM_ENV_PYTHON`, `MULTIARM_ENV_ROOT`, and `ROBOSUITE_ROOT`. The `four_arm_env_server.py` entry point connects policy evaluation to the configured task environment through local IPC.

```bash
export CHECKPOINT=/path/to/complete/team/checkpoint
export OUTPUT=$OUTPUT_ROOT/four_arm_fresh200
export SEED_START=5300  # Frame Insertion; use 6400 for Arch Assembly.
export SEED_COUNT=200
pi05/scripts/multiarm/evaluate_four_arm.sh
```

## CPU Verification

Run the tests with CPU-backed JAX devices:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
$PYTHON_BIN -m pytest -q pi05/tests
```
