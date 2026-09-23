#!/usr/bin/env python3
"""Audit four-rank training logs and process-specific GPU memory peaks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


PID_RE = re.compile(
    r"rank=(\d+) role=(\d+) pid=(\d+) cuda_visible_devices=(\d+)"
)
STEP_RE = re.compile(
    r"rank=(\d+) step=(\d+) team_loss=([0-9.eE+-]+) "
    r"local_loss=([0-9.eE+-]+) role_losses=\[([^]]+)] "
    r"grad_norm=([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-peak-mib", type=float, default=43.2 * 1024)
    return parser.parse_args()


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("gpu IDs must be comma-separated integers") from exc
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("exactly four distinct gpu IDs are required")
    return gpu_ids


def jax_peak_mib(memory: dict) -> float:
    for key in ("peak_bytes_in_use", "peak_bytes_reserved"):
        if key in memory:
            return float(memory[key]) / (1024 * 1024)
    raise RuntimeError("JAX memory statistics do not contain a peak byte field")


def finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def main() -> None:
    args = parse_args()
    expected_gpus = parse_gpu_ids(args.gpu_ids)
    ranks: dict[int, dict] = {}
    for rank, expected_gpu in enumerate(expected_gpus):
        log = args.log_dir / f"rank{rank}.log"
        text = log.read_text(encoding="utf-8")
        pid_match = PID_RE.search(text)
        steps = STEP_RE.findall(text)
        memory_rows = [
            json.loads(line.removeprefix("JAX_MEMORY_STATS "))
            for line in text.splitlines()
            if line.startswith("JAX_MEMORY_STATS ")
        ]
        if pid_match is None or not steps or not memory_rows:
            raise RuntimeError(f"incomplete rank evidence: {log}")

        reported_rank, role, pid, gpu = map(int, pid_match.groups())
        if reported_rank != rank or role != rank:
            raise RuntimeError(f"rank/role mismatch in {log}")
        if gpu != expected_gpu:
            raise RuntimeError(
                f"rank {rank} ran on GPU {gpu}, expected scheduler device {expected_gpu}"
            )

        last = steps[-1]
        role_losses = [float(value.strip()) for value in last[4].split(",")]
        scalars = [float(last[2]), float(last[3]), *role_losses, float(last[5])]
        if len(role_losses) != 4 or not finite(scalars):
            raise RuntimeError(f"rank {rank} reported invalid optimization values")
        ranks[rank] = {
            "pid": pid,
            "gpu": gpu,
            "steps": len(steps),
            "final_step": int(last[1]),
            "final_team_loss": float(last[2]),
            "final_local_loss": float(last[3]),
            "final_role_losses": role_losses,
            "final_grad_norm": float(last[5]),
            "jax_peak_mib": jax_peak_mib(memory_rows[-1]),
            "nvml_process_peak_mib": 0,
        }

    pids = {info["pid"]: info for info in ranks.values()}
    if len(pids) != 4:
        raise RuntimeError("rank logs do not contain four distinct process IDs")
    with args.memory_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row["pid"]:
                continue
            info = pids.get(int(row["pid"]))
            if info is not None:
                info["nvml_process_peak_mib"] = max(
                    info["nvml_process_peak_mib"],
                    int(row["process_memory_used_mib"]),
                )

    for rank, info in ranks.items():
        jax_peak = info["jax_peak_mib"]
        if jax_peak > args.max_peak_mib:
            raise RuntimeError(
                f"rank {rank} exceeds JAX memory gate: {jax_peak:.1f} > "
                f"{args.max_peak_mib:.1f} MiB"
            )
        peak = info["nvml_process_peak_mib"]
        if peak <= 0:
            raise RuntimeError(f"rank {rank} has no process-specific NVML sample")
        if peak > args.max_peak_mib:
            raise RuntimeError(
                f"rank {rank} exceeds memory gate: {peak} > "
                f"{args.max_peak_mib:.1f} MiB"
            )

    result = {
        "status": "PASS",
        "gpu_ids": expected_gpus,
        "memory_gate_mib": args.max_peak_mib,
        "maximum_process_peak_mib": max(
            info["nvml_process_peak_mib"] for info in ranks.values()
        ),
        "maximum_jax_peak_mib": max(info["jax_peak_mib"] for info in ranks.values()),
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
