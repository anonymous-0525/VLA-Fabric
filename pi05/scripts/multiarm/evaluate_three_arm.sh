#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${GPU_IDS:?set exactly three GPU IDs}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${STACK_CUBE_QUANTILES:?set STACK_CUBE_QUANTILES}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT}"
: "${ROBOFACTORY_PYTHON:?set ROBOFACTORY_PYTHON}"
: "${ROBOFACTORY_ROOT:?set ROBOFACTORY_ROOT}"
: "${ROBOFACTORY_ENV_SERVER:?set ROBOFACTORY_ENV_SERVER}"
: "${ROBOFACTORY_ENV_CONFIG:?set ROBOFACTORY_ENV_CONFIG}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:?set OUTPUT}"
SEED_START="${SEED_START:-40200}"
SEED_COUNT="${SEED_COUNT:-200}"
PORT="${COORDINATOR_PORT:-29831}"
"$ROOT/scripts/multiarm/launch_agent_ranks.sh" "$GPU_IDS" "$PORT" "$OUTPUT/logs" -- \
  "$PYTHON_BIN" "$ROOT/scripts/multiarm/evaluate_stack_cube.py" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --seed-start "$SEED_START" \
  --seed-count "$SEED_COUNT" --quantiles "$STACK_CUBE_QUANTILES" \
  --base-checkpoint "$PI05_BASE_CHECKPOINT" --robofactory-python "$ROBOFACTORY_PYTHON" \
  --robofactory-root "$ROBOFACTORY_ROOT" --environment-server "$ROBOFACTORY_ENV_SERVER" \
  --environment-config "$ROBOFACTORY_ENV_CONFIG"
