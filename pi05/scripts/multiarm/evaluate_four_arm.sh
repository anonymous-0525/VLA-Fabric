#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${GPU_IDS:?set exactly four GPU IDs selected by the scheduler}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${FOUR_ARM_TASK:?set frame_insertion or arch_assembly}"
: "${FOUR_ARM_QUANTILES:?set FOUR_ARM_QUANTILES}"
: "${FOUR_ARM_INSTRUCTION:?set FOUR_ARM_INSTRUCTION}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT}"
: "${FOUR_ARM_ENV_PYTHON:?set FOUR_ARM_ENV_PYTHON}"
: "${MULTIARM_ENV_ROOT:?set MULTIARM_ENV_ROOT}"
: "${ROBOSUITE_ROOT:?set ROBOSUITE_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:?set OUTPUT}"
SEED_START="${SEED_START:?set the task validation or fresh seed start}"
SEED_COUNT="${SEED_COUNT:-200}"
PORT="${COORDINATOR_PORT:-29841}"
"$ROOT/scripts/multiarm/launch_agent_ranks.sh" "$GPU_IDS" "$PORT" "$OUTPUT/logs" -- \
  "$PYTHON_BIN" "$ROOT/scripts/multiarm/evaluate_four_arm.py" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --seed-start "$SEED_START" \
  --seed-count "$SEED_COUNT" --task "$FOUR_ARM_TASK" \
  --quantiles "$FOUR_ARM_QUANTILES" --base-checkpoint "$PI05_BASE_CHECKPOINT" \
  --environment-python "$FOUR_ARM_ENV_PYTHON" --environment-root "$MULTIARM_ENV_ROOT" \
  --robosuite-root "$ROBOSUITE_ROOT" --instruction "$FOUR_ARM_INSTRUCTION"
