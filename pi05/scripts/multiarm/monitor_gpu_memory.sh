#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_IDS WATCHED_PID OUTPUT_CSV" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required for the four-agent memory audit" >&2
  exit 2
}

IFS=',' read -r -a GPUS <<< "$1"
WATCHED_PID="$2"
OUTPUT="$3"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four devices" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
printf 'timestamp,gpu,pid,process_memory_used_mib,gpu_memory_used_mib\n' > "$OUTPUT"
while kill -0 "$WATCHED_PID" 2>/dev/null; do
  timestamp="$(date --iso-8601=seconds)"
  for gpu in "${GPUS[@]}"; do
    total="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    rows="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_gpu_memory \
      --format=csv,noheader,nounits || true)"
    if [[ -z "$rows" ]]; then
      printf '%s,%s,,0,%s\n' "$timestamp" "$gpu" "$total" >> "$OUTPUT"
      continue
    fi
    while IFS=',' read -r process_pid process_used; do
      process_pid="${process_pid// /}"
      process_used="${process_used// /}"
      printf '%s,%s,%s,%s,%s\n' \
        "$timestamp" "$gpu" "$process_pid" "$process_used" "$total" >> "$OUTPUT"
    done <<< "$rows"
  done
  sleep "${MEMORY_POLL_SECONDS:-1}"
done
