#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${GPU_IDS:?set exactly four GPU IDs selected by the scheduler}"
: "${FOUR_ARM_TASK:?set frame_insertion or arch_assembly}"
: "${FOUR_ARM_DATASET:?set FOUR_ARM_DATASET}"
: "${FOUR_ARM_QUANTILES:?set FOUR_ARM_QUANTILES}"
: "${FOUR_ARM_AUDIT_MANIFEST:?set FOUR_ARM_AUDIT_MANIFEST}"
: "${FOUR_ARM_INSTRUCTION:?set FOUR_ARM_INSTRUCTION}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT}"
: "${STAGE1_CHECKPOINT:?set the complete Stage 1 team checkpoint}"
: "${TEAM_MICROBATCH:?set after target-HPC readiness testing}"
: "${ACCUMULATION:?set after target-HPC readiness testing}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/multiarm}"
RUN_NAME="${RUN_NAME:-four_arm_stage2_full_50k}"
PORT="${COORDINATOR_PORT:-29742}"
"$ROOT/scripts/multiarm/launch_agent_ranks.sh" "$GPU_IDS" "$PORT" "$OUTPUT_ROOT/$RUN_NAME/logs" -- \
  "$PYTHON_BIN" "$ROOT/scripts/multiarm/train_four_arm.py" \
  --task "$FOUR_ARM_TASK" --instruction "$FOUR_ARM_INSTRUCTION" \
  --dataset "$FOUR_ARM_DATASET" --quantiles "$FOUR_ARM_QUANTILES" \
  --audit-manifest "$FOUR_ARM_AUDIT_MANIFEST" --base-checkpoint "$PI05_BASE_CHECKPOINT" \
  --stage pi05_four_arm_full_stage2 --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --output "$OUTPUT_ROOT/$RUN_NAME" --steps 50000 --checkpoint-every 10000 \
  --warmup-steps 2000 --team-microbatch "$TEAM_MICROBATCH" \
  --gradient-accumulation "$ACCUMULATION" --training-seed 2701 --execute
