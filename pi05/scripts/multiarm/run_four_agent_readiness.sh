#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -v GPU_IDS || -z "$GPU_IDS" ]]; then
  echo "set GPU_IDS to exactly four scheduler-assigned devices" >&2
  exit 2
fi
: "${GPU_IDS:?set GPU_IDS to exactly four scheduler-assigned devices}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four scheduler-assigned devices" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -v "SEEN_GPUS[$gpu]" ]]; then
    echo "GPU_IDS must contain four distinct integer device IDs" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${FOUR_ARM_TASK:?set FOUR_ARM_TASK to frame_insertion or arch_assembly}"
: "${FOUR_ARM_DATASET:?set FOUR_ARM_DATASET}"
: "${FOUR_ARM_QUANTILES:?set FOUR_ARM_QUANTILES}"
: "${FOUR_ARM_AUDIT_MANIFEST:?set FOUR_ARM_AUDIT_MANIFEST}"
: "${FOUR_ARM_INSTRUCTION:?set FOUR_ARM_INSTRUCTION}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to an HPC scratch directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
READINESS_SCOPE="${READINESS_SCOPE:-training}"
if [[ "$READINESS_SCOPE" != "training" && "$READINESS_SCOPE" != "full" ]]; then
  echo "READINESS_SCOPE must be training or full" >&2
  exit 2
fi
if [[ "$READINESS_SCOPE" == "full" ]]; then
  : "${FOUR_ARM_ENV_PYTHON:?set FOUR_ARM_ENV_PYTHON for full readiness}"
  : "${MULTIARM_ENV_ROOT:?set MULTIARM_ENV_ROOT for full readiness}"
  : "${ROBOSUITE_ROOT:?set ROBOSUITE_ROOT for full readiness}"
fi

TEAM_MICROBATCH="${TEAM_MICROBATCH:-6}"
ACCUMULATION="${ACCUMULATION:-3}"
STAGE_STEPS="${READINESS_STAGE_STEPS:-5}"
RESUME_STEPS=$((STAGE_STEPS + 1))
WARMUP_STEPS="${READINESS_WARMUP_STEPS:-2}"
MAX_EXISTING_MIB="${MAX_EXISTING_MIB:-4096}"
MEMORY_GATE_MIB="${MEMORY_GATE_MIB:-44236.8}"
BASE_PORT="${COORDINATOR_PORT:-29830}"
READINESS_ROOT="${READINESS_ROOT:-$OUTPUT_ROOT/four_agent_readiness}"
LAUNCHER="$ROOT/scripts/multiarm/launch_agent_ranks.sh"
TRAIN="$ROOT/scripts/multiarm/train_four_arm.py"
EVALUATE="$ROOT/scripts/multiarm/evaluate_four_arm.py"
MONITOR="$ROOT/scripts/multiarm/monitor_gpu_memory.sh"
AUDITOR="$ROOT/scripts/multiarm/audit_four_agent_memory.py"
ACTIVE_PIDS=()

cleanup_active() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}

terminate_active() {
  cleanup_active
  exit 130
}

trap cleanup_active EXIT
trap terminate_active INT TERM

for path in \
  "$FOUR_ARM_DATASET" \
  "$FOUR_ARM_QUANTILES" \
  "$FOUR_ARM_AUDIT_MANIFEST"; do
  [[ -f "$path" ]] || { echo "required file is missing: $path" >&2; exit 2; }
done
[[ -d "$PI05_BASE_CHECKPOINT" ]] || {
  echo "base checkpoint directory is missing: $PI05_BASE_CHECKPOINT" >&2
  exit 2
}
[[ ! -e "$READINESS_ROOT" ]] || {
  echo "refusing to overwrite readiness output: $READINESS_ROOT" >&2
  exit 4
}

gpu_used_mib() {
  nvidia-smi -i "$1" --query-compute-apps=used_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

check_gpu_group() {
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required for target-HPC readiness" >&2
    exit 2
  }
  local check gpu used
  for check in 1 2 3; do
    for gpu in "${GPU_ARRAY[@]}"; do
      used="$(gpu_used_mib "$gpu")"
      if (( used >= MAX_EXISTING_MIB )); then
        echo "GPU $gpu is occupied (${used} MiB >= ${MAX_EXISTING_MIB} MiB)" >&2
        exit 3
      fi
    done
    sleep "${GPU_CHECK_INTERVAL_SECONDS:-2}"
  done
}

monitor_watchdog() {
  local launch_pid="$1" monitor_pid="$2"
  while kill -0 "$launch_pid" 2>/dev/null; do
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
      echo "memory monitor exited before the rank launcher" >&2
      kill "$launch_pid" 2>/dev/null || true
      return 1
    fi
    sleep 1
  done
}

run_stage() {
  local port="$1" log_dir="$2" audit_dir="$3"
  local launch_pid monitor_pid watchdog_pid
  local launch_status monitor_status
  shift 3
  mkdir -p "$audit_dir"
  "$LAUNCHER" "$GPU_IDS" "$port" "$log_dir" -- "$@" &
  launch_pid=$!
  "$MONITOR" "$GPU_IDS" "$launch_pid" "$audit_dir/memory.csv" &
  monitor_pid=$!
  monitor_watchdog "$launch_pid" "$monitor_pid" &
  watchdog_pid=$!
  ACTIVE_PIDS=("$launch_pid" "$monitor_pid" "$watchdog_pid")
  set +e
  wait "$launch_pid"
  launch_status=$?
  wait "$monitor_pid"
  monitor_status=$?
  kill "$watchdog_pid" 2>/dev/null
  wait "$watchdog_pid" 2>/dev/null
  ACTIVE_PIDS=()
  set -e
  if (( launch_status != 0 || monitor_status != 0 )); then
    return 1
  fi
  "$PYTHON_BIN" "$AUDITOR" \
    --log-dir "$log_dir" \
    --memory-csv "$audit_dir/memory.csv" \
    --output "$audit_dir/audit.json" \
    --gpu-ids "$GPU_IDS" \
    --max-peak-mib "$MEMORY_GATE_MIB"
}

run_closed_loop() {
  local port="$1" log_dir="$2" evaluation_pid evaluation_status
  shift 2
  "$LAUNCHER" "$GPU_IDS" "$port" "$log_dir" -- "$@" &
  evaluation_pid=$!
  ACTIVE_PIDS=("$evaluation_pid")
  set +e
  wait "$evaluation_pid"
  evaluation_status=$?
  ACTIVE_PIDS=()
  set -e
  return "$evaluation_status"
}

printf -v STAGE_CHECKPOINT_DIR 'step_%08d' "$STAGE_STEPS"
printf -v RESUME_CHECKPOINT_DIR 'step_%08d' "$RESUME_STEPS"
check_gpu_group
mkdir -p "$READINESS_ROOT"

STAGE1="$READINESS_ROOT/stage1"
run_stage "$BASE_PORT" "$STAGE1/logs" "$STAGE1/audit" \
  "$PYTHON_BIN" "$TRAIN" \
  --task "$FOUR_ARM_TASK" --instruction "$FOUR_ARM_INSTRUCTION" \
  --dataset "$FOUR_ARM_DATASET" --quantiles "$FOUR_ARM_QUANTILES" \
  --audit-manifest "$FOUR_ARM_AUDIT_MANIFEST" \
  --base-checkpoint "$PI05_BASE_CHECKPOINT" \
  --stage pi05_four_arm_common_stage1 --output "$STAGE1/run" \
  --steps "$STAGE_STEPS" --checkpoint-every "$STAGE_STEPS" \
  --warmup-steps "$WARMUP_STEPS" --team-microbatch "$TEAM_MICROBATCH" \
  --gradient-accumulation "$ACCUMULATION" --training-seed 9961 \
  --allow-nonformal-steps --execute
STAGE1_CHECKPOINT="$STAGE1/run/checkpoints/$STAGE_CHECKPOINT_DIR"

STAGE2="$READINESS_ROOT/stage2"
run_stage "$((BASE_PORT + 1))" "$STAGE2/logs" "$STAGE2/audit" \
  "$PYTHON_BIN" "$TRAIN" \
  --task "$FOUR_ARM_TASK" --instruction "$FOUR_ARM_INSTRUCTION" \
  --dataset "$FOUR_ARM_DATASET" --quantiles "$FOUR_ARM_QUANTILES" \
  --audit-manifest "$FOUR_ARM_AUDIT_MANIFEST" \
  --base-checkpoint "$PI05_BASE_CHECKPOINT" \
  --stage pi05_four_arm_full_stage2 --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --output "$STAGE2/run" --steps "$STAGE_STEPS" \
  --checkpoint-every "$STAGE_STEPS" --warmup-steps "$WARMUP_STEPS" \
  --team-microbatch "$TEAM_MICROBATCH" --gradient-accumulation "$ACCUMULATION" \
  --training-seed 9962 --allow-nonformal-steps --execute

run_stage "$((BASE_PORT + 2))" "$STAGE2/restore_logs" "$STAGE2/restore_audit" \
  "$PYTHON_BIN" "$TRAIN" \
  --task "$FOUR_ARM_TASK" --instruction "$FOUR_ARM_INSTRUCTION" \
  --dataset "$FOUR_ARM_DATASET" --quantiles "$FOUR_ARM_QUANTILES" \
  --audit-manifest "$FOUR_ARM_AUDIT_MANIFEST" \
  --base-checkpoint "$PI05_BASE_CHECKPOINT" \
  --stage pi05_four_arm_full_stage2 \
  --resume-checkpoint "$STAGE2/run/checkpoints/$STAGE_CHECKPOINT_DIR" \
  --output "$STAGE2/run" --steps "$RESUME_STEPS" \
  --checkpoint-every "$STAGE_STEPS" --warmup-steps "$WARMUP_STEPS" \
  --team-microbatch "$TEAM_MICROBATCH" --gradient-accumulation "$ACCUMULATION" \
  --training-seed 9962 --allow-nonformal-steps --execute

CLOSED_LOOP_STATUS="NOT_RUN"
if [[ "$READINESS_SCOPE" == "full" ]]; then
  CLOSED_LOOP="$READINESS_ROOT/closed_loop"
  run_closed_loop "$((BASE_PORT + 3))" "$CLOSED_LOOP/logs" \
    "$PYTHON_BIN" "$EVALUATE" \
    --task "$FOUR_ARM_TASK" --quantiles "$FOUR_ARM_QUANTILES" \
    --base-checkpoint "$PI05_BASE_CHECKPOINT" \
    --environment-python "$FOUR_ARM_ENV_PYTHON" \
    --environment-root "$MULTIARM_ENV_ROOT" --robosuite-root "$ROBOSUITE_ROOT" \
    --instruction "$FOUR_ARM_INSTRUCTION" \
    --checkpoint "$STAGE2/run/checkpoints/$RESUME_CHECKPOINT_DIR" \
    --output "$CLOSED_LOOP/evaluation" \
    --seed-start "${READINESS_SEED_START:-5100}" --seed-count 2 \
    --maximum-steps "${READINESS_MAXIMUM_STEPS:-50}"
  "$PYTHON_BIN" - "$CLOSED_LOOP/evaluation/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    summary.get("status") != "COMPLETE"
    or summary.get("trials") != 2
    or not summary.get("finite_actions")
):
    raise SystemExit(f"invalid closed-loop smoke summary: {summary}")
PY
  CLOSED_LOOP_STATUS="PASS"
fi

"$PYTHON_BIN" - \
  "$READINESS_ROOT/readiness_summary.json" \
  "$STAGE1/audit/audit.json" \
  "$STAGE2/audit/audit.json" \
  "$STAGE2/restore_audit/audit.json" \
  "$READINESS_SCOPE" "$CLOSED_LOOP_STATUS" "$GPU_IDS" \
  "$TEAM_MICROBATCH" "$ACCUMULATION" "$MEMORY_GATE_MIB" <<'PY'
import json
from pathlib import Path
import sys

output, stage1_path, stage2_path, restore_path = map(Path, sys.argv[1:5])
scope, closed_loop, gpu_ids = sys.argv[5:8]
team_microbatch, accumulation = map(int, sys.argv[8:10])
memory_gate = float(sys.argv[10])
audits = [json.loads(path.read_text(encoding="utf-8")) for path in (
    stage1_path, stage2_path, restore_path
)]
summary = {
    "status": "PASS_FULL" if scope == "full" else "PASS_TRAINING_PATH",
    "scope": scope,
    "gpu_ids": [int(value) for value in gpu_ids.split(",")],
    "team_microbatch": team_microbatch,
    "gradient_accumulation": accumulation,
    "global_team_batch": team_microbatch * accumulation,
    "stage1_to_stage2_fork": True,
    "same_stage_resume": True,
    "closed_loop_smoke": closed_loop,
    "memory_gate_mib": memory_gate,
    "maximum_process_peak_mib": max(
        audit["maximum_process_peak_mib"] for audit in audits
    ),
}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
