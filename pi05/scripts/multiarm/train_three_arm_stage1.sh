#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${GPU_IDS:?set exactly three GPU IDs, for example 1,2,3}"
: "${STACK_CUBE_DATASET:?set STACK_CUBE_DATASET}"
: "${STACK_CUBE_QUANTILES:?set STACK_CUBE_QUANTILES}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/multiarm}"
RUN_NAME="${RUN_NAME:-stack_cube_stage1_common_10k}"
PORT="${COORDINATOR_PORT:-29731}"
"$ROOT/scripts/multiarm/launch_agent_ranks.sh" "$GPU_IDS" "$PORT" "$OUTPUT_ROOT/$RUN_NAME/logs" -- \
  "$PYTHON_BIN" "$ROOT/scripts/multiarm/train_stack_cube.py" \
  --stage pi05_stackcube_3a_common_stage1 \
  --dataset "$STACK_CUBE_DATASET" --quantiles "$STACK_CUBE_QUANTILES" \
  --base-checkpoint "$PI05_BASE_CHECKPOINT" --output "$OUTPUT_ROOT/$RUN_NAME" \
  --steps 10000 --checkpoint-every 10000 --warmup-steps 1000 \
  --team-microbatch 6 --gradient-accumulation 3 --training-seed 1701 --execute
