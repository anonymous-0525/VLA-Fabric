from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "multiarm"


def _write_rank_evidence(log_dir: Path, gpu_ids: tuple[int, ...]) -> None:
    _write_rank_evidence_with_jax_peak(log_dir, gpu_ids, jax_peak=19_000)


def _write_rank_evidence_with_jax_peak(
    log_dir: Path,
    gpu_ids: tuple[int, ...],
    *,
    jax_peak: int,
) -> None:
    log_dir.mkdir(parents=True)
    for rank, gpu in enumerate(gpu_ids):
        payload = {
            "peak_bytes_in_use": (jax_peak + rank) * 1024 * 1024,
        }
        (log_dir / f"rank{rank}.log").write_text(
            "\n".join(
                [
                    f"rank={rank} role={rank} pid={4100 + rank} "
                    f"cuda_visible_devices={gpu}",
                    f"rank={rank} step=5 team_loss=0.08 local_loss=0.02 "
                    "role_losses=[0.01, 0.02, 0.03, 0.04] grad_norm=1.25",
                    "JAX_MEMORY_STATS " + json.dumps(payload),
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _write_memory_samples(path: Path, gpu_ids: tuple[int, ...], peak: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "timestamp",
                "gpu",
                "pid",
                "process_memory_used_mib",
                "gpu_memory_used_mib",
            ),
        )
        writer.writeheader()
        for rank, gpu in enumerate(gpu_ids):
            writer.writerow(
                {
                    "timestamp": "2026-09-02T17:00:00+08:00",
                    "gpu": gpu,
                    "pid": 4100 + rank,
                    "process_memory_used_mib": peak - rank,
                    "gpu_memory_used_mib": peak - rank,
                }
            )


def test_readiness_launcher_requires_explicit_four_gpu_allocation() -> None:
    launcher = SCRIPTS / "run_four_agent_readiness.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        env={key: value for key, value in os.environ.items() if key != "GPU_IDS"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "set GPU_IDS to exactly four scheduler-assigned devices" in result.stderr


def test_readiness_launcher_covers_training_restore_memory_and_optional_rollout() -> None:
    text = (SCRIPTS / "run_four_agent_readiness.sh").read_text(encoding="utf-8")

    assert 'GPU_IDS:?set GPU_IDS' in text
    assert "GPU_IDS:-" not in text
    assert "pi05_four_arm_common_stage1" in text
    assert "pi05_four_arm_full_stage2" in text
    assert "--stage1-checkpoint" in text
    assert "--resume-checkpoint" in text
    assert "--audit-manifest" in text
    assert "monitor_gpu_memory.sh" in text
    assert "audit_four_agent_memory.py" in text
    assert 'READINESS_SCOPE="${READINESS_SCOPE:-training}"' in text
    assert "evaluate_four_arm.py" in text
    assert "trap cleanup_active EXIT" in text
    assert "trap terminate_active INT TERM" in text
    assert "monitor_watchdog" in text
    assert "run_closed_loop" in text
    assert 'ACTIVE_PIDS=("$evaluation_pid")' in text


def test_memory_auditor_accepts_complete_four_rank_evidence(tmp_path: Path) -> None:
    gpu_ids = (1, 2, 3, 4)
    log_dir = tmp_path / "logs"
    memory_csv = tmp_path / "memory.csv"
    output = tmp_path / "audit.json"
    _write_rank_evidence(log_dir, gpu_ids)
    _write_memory_samples(memory_csv, gpu_ids, peak=33_328)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_four_agent_memory.py"),
            "--log-dir",
            str(log_dir),
            "--memory-csv",
            str(memory_csv),
            "--output",
            str(output),
            "--gpu-ids",
            "1,2,3,4",
            "--max-peak-mib",
            "44236.8",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["maximum_process_peak_mib"] == 33_328
    assert [audit["ranks"][str(rank)]["gpu"] for rank in range(4)] == list(gpu_ids)


def test_memory_auditor_rejects_a_process_above_the_gate(tmp_path: Path) -> None:
    gpu_ids = (1, 2, 3, 4)
    log_dir = tmp_path / "logs"
    memory_csv = tmp_path / "memory.csv"
    _write_rank_evidence(log_dir, gpu_ids)
    _write_memory_samples(memory_csv, gpu_ids, peak=45_000)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_four_agent_memory.py"),
            "--log-dir",
            str(log_dir),
            "--memory-csv",
            str(memory_csv),
            "--output",
            str(tmp_path / "audit.json"),
            "--gpu-ids",
            "1,2,3,4",
            "--max-peak-mib",
            "44236.8",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "exceeds memory gate" in result.stderr


def test_memory_auditor_rejects_a_jax_peak_above_the_gate(tmp_path: Path) -> None:
    gpu_ids = (1, 2, 3, 4)
    log_dir = tmp_path / "logs"
    memory_csv = tmp_path / "memory.csv"
    _write_rank_evidence_with_jax_peak(log_dir, gpu_ids, jax_peak=45_000)
    _write_memory_samples(memory_csv, gpu_ids, peak=33_000)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_four_agent_memory.py"),
            "--log-dir",
            str(log_dir),
            "--memory-csv",
            str(memory_csv),
            "--output",
            str(tmp_path / "audit.json"),
            "--gpu-ids",
            "1,2,3,4",
            "--max-peak-mib",
            "44236.8",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "JAX memory gate" in result.stderr


def test_memory_auditor_does_not_treat_device_capacity_as_a_peak(
    tmp_path: Path,
) -> None:
    gpu_ids = (1, 2, 3, 4)
    log_dir = tmp_path / "logs"
    memory_csv = tmp_path / "memory.csv"
    _write_rank_evidence(log_dir, gpu_ids)
    for log in log_dir.glob("rank*.log"):
        text = log.read_text(encoding="utf-8")
        text = text.replace('"peak_bytes_in_use":', '"bytes_limit":')
        log.write_text(text, encoding="utf-8")
    _write_memory_samples(memory_csv, gpu_ids, peak=33_000)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_four_agent_memory.py"),
            "--log-dir",
            str(log_dir),
            "--memory-csv",
            str(memory_csv),
            "--output",
            str(tmp_path / "audit.json"),
            "--gpu-ids",
            "1,2,3,4",
            "--max-peak-mib",
            "44236.8",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "peak byte field" in result.stderr


def test_four_gpu_config_separates_local_evidence_from_target_hpc_readiness() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/multiarm/four_gpu_agent_parallel.yaml").read_text(
            encoding="utf-8"
        )
    )
    operating_point = config["candidate_operating_point"]
    evidence = operating_point["local_server_validation"]

    assert operating_point["validation_status"] == "local_training_path_validated"
    assert operating_point["target_hpc_readiness_required"] is True
    assert operating_point["tested_on_target_hpc"] is False
    assert evidence["four_rank_training"] is True
    assert evidence["stage1_to_stage2_fork"] is True
    assert evidence["maximum_process_peak_mib"] == 33_328
    assert evidence["same_stage_resume"] == "not_run"
    assert evidence["closed_loop_rollout"] == "not_run"
