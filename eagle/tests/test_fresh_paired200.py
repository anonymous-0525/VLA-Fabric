import json
import threading
import time
from pathlib import Path

import pytest

from commvla.evaluation.fresh_paired import (
    COMM_GPU_PAIRS,
    FRESH_IDS,
    TASK_RUN_IDS,
    build_safe_comm_waves,
    build_tail_twin_jobs,
    build_rolling_jobs,
    cleanup_smoke_workspace,
    selection_to_entry,
    validate_devices,
    validate_comm_summary,
    validate_frozen_manifest,
    validate_twin_summary,
    run_rolling_workers,
)


def _model(run_id: str, task: str = "aloha_handover_box") -> dict:
    return {
        "kind": "commvla",
        "run_id": run_id,
        "task": task,
        "checkpoint": f"/checkpoints/{run_id}",
        "selection_file": f"/selection/{run_id}.json",
        "selected_tag": "step40000",
        "checkpoint_sha256": "a" * 64,
    }


def test_safe_comm_waves_never_share_a_gpu_within_a_wave() -> None:
    models = [_model(f"M{index:02d}") for index in range(16)]
    waves = build_safe_comm_waves(models, port_base=51000)

    assert [len(wave) for wave in waves] == [3] * 10 + [2]
    jobs = [job for wave in waves for job in wave]
    assert len(jobs) == 32
    assert all(job.trials == 100 for job in jobs)
    for wave in waves:
        assert len({job.devices for job in wave}) == len(wave)
        flattened = [gpu for job in wave for gpu in job.devices]
        assert len(flattened) == len(set(flattened))
        assert set(flattened) <= set(range(6))
    for model in models:
        assert [job.rollout_offset for job in jobs if job.run_id == model["run_id"]] == [200, 300]


def test_tail_twin_jobs_use_the_two_gpus_left_free_by_comm_tail() -> None:
    jobs = build_tail_twin_jobs({"kind": "twinvla"})
    assert jobs == ((200, (4,)), (300, (5,)))


def test_rolling_job_list_covers_all_6800_rollouts() -> None:
    entries = []
    for task in ("aloha_handover_box", "aloha_shoes_table"):
        entries.extend(_model(f"{task}-{index:02d}", task) for index in range(16))
        entries.append(
            {
                "kind": "twinvla",
                "run_id": f"TwinVLA-{task}",
                "task": task,
                "checkpoint": f"/checkpoints/TwinVLA-{task}",
                "checkpoint_sha256": "b" * 64,
            }
        )

    comm, twin = build_rolling_jobs(entries)

    assert len(comm) == 64
    assert len(twin) == 4
    assert sum(job.trials for job in comm + twin) == 6800
    assert all(job.rollout_offset in (200, 300) for job in comm + twin)


def test_rolling_workers_dispatch_next_job_before_slowest_slot_finishes() -> None:
    jobs = ["slow", "fast-1", "fast-2", "fast-3"]
    events = []
    lock = threading.Lock()

    def worker(job: str, slot: str) -> str:
        with lock:
            events.append(("start", job, slot, time.monotonic()))
        time.sleep(0.12 if job == "slow" else 0.01)
        with lock:
            events.append(("end", job, slot, time.monotonic()))
        return job

    completed = run_rolling_workers(jobs, ("slot-a", "slot-b"), worker)

    assert set(completed) == set(jobs)
    slow_end = next(at for event, job, _slot, at in events if event == "end" and job == "slow")
    assert any(event == "start" and job == "fast-2" and at < slow_end for event, job, _slot, at in events)


def test_device_validation_rejects_reserved_or_invalid_gpus() -> None:
    assert validate_devices((0, 1)) == (0, 1)
    assert validate_devices((4, 5)) == (4, 5)

    with pytest.raises(ValueError, match="reserved"):
        validate_devices((5, 6))
    with pytest.raises(ValueError, match="GPU 0--5"):
        validate_devices((-1, 0))


def test_manifest_requires_two_tasks_sixteen_comm_models_and_task_specific_twinvla() -> None:
    entries = []
    for task in ("aloha_handover_box", "aloha_shoes_table"):
        entries.extend(_model(f"{task}-{index:02d}", task) for index in range(16))
        entries.append(
            {
                "kind": "twinvla",
                "run_id": f"TwinVLA-{task}",
                "task": task,
                "checkpoint": f"/checkpoints/TwinVLA-{task}",
                "checkpoint_sha256": "b" * 64,
            }
        )

    validate_frozen_manifest({"fresh_ids": [200, 399], "seed": 1501, "entries": entries}, check_paths=False)

    entries.pop()
    with pytest.raises(ValueError, match="one TwinVLA"):
        validate_frozen_manifest({"fresh_ids": [200, 399], "seed": 1501, "entries": entries}, check_paths=False)


def test_summary_audit_requires_exact_fresh_half_and_frozen_protocol(tmp_path: Path) -> None:
    summary = {
        "checkpoint": "/checkpoints/A",
        "task_name": "aloha_handover_box",
        "trials": 100,
        "rollout_ids": list(range(200, 300)),
        "plan_seed_offset": 0,
        "execution_mode": "chunk",
        "action_len": 20,
        "dtype": "bfloat16",
        "cfg": 1.1,
        "num_denoising_steps": 10,
        "rows": [
            {"rollout_id": value, "finite_all": 1, "timeout": 0, "success": 0}
            for value in range(200, 300)
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    validate_comm_summary(path, _model("A"), rollout_offset=200)

    summary["rollout_ids"][-1] = 300
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout IDs"):
        validate_comm_summary(path, _model("A"), rollout_offset=200)


def test_smoke_cleanup_removes_workspace_but_retains_compact_audit(tmp_path: Path) -> None:
    workspace = tmp_path / "smoke"
    (workspace / "outputs").mkdir(parents=True)
    (workspace / "outputs" / "rollouts.csv").write_text("temporary", encoding="utf-8")
    audit = tmp_path / "manifests" / "smoke_audit.json"

    cleanup_smoke_workspace(workspace, audit, {"status": "pass", "peak_memory_mib": 30000})

    assert not workspace.exists()
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "pass"
    assert list(tmp_path.rglob("rollouts.csv")) == []


def test_fresh_id_constant_is_exactly_200_through_399() -> None:
    assert FRESH_IDS == tuple(range(200, 400))


def test_task_run_ids_preserve_task_specific_avg_names() -> None:
    assert len(TASK_RUN_IDS["aloha_handover_box"]) == 16
    assert len(TASK_RUN_IDS["aloha_shoes_table"]) == 16
    assert "D-I3" in TASK_RUN_IDS["aloha_handover_box"]
    assert "S-I3" in TASK_RUN_IDS["aloha_handover_box"]
    assert "SHOES-D-I3-AVG" in TASK_RUN_IDS["aloha_shoes_table"]
    assert "SHOES-S-I3-AVG" in TASK_RUN_IDS["aloha_shoes_table"]
    assert "SHOES-D-I3" not in TASK_RUN_IDS["aloha_shoes_table"]


def test_selection_entry_uses_only_confirmed_cleaned_best(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    selection = tmp_path / "checkpoint_selection.json"
    selection.write_text(
        json.dumps(
            {
                "run_id": "S-I3-RAW",
                "recommended_best": "step40000",
                "cleanup_performed": True,
                "candidate_results": [
                    {"tag": "step40000", "checkpoint": str(checkpoint), "selection_rank": 1},
                    {"tag": "step30000", "checkpoint": "/wrong", "selection_rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    entry = selection_to_entry(selection, "aloha_handover_box", checkpoint_sha256="c" * 64)
    assert entry["checkpoint"] == str(checkpoint)
    assert entry["selected_tag"] == "step40000"

    data = json.loads(selection.read_text(encoding="utf-8"))
    data["cleanup_performed"] = False
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="cleanup"):
        selection_to_entry(selection, "aloha_handover_box", checkpoint_sha256="c" * 64)


def test_twin_summary_uses_exact_task_checkpoint_and_fresh_ids(tmp_path: Path) -> None:
    entry = {
        "kind": "twinvla",
        "run_id": "TwinVLA-box",
        "task": "aloha_handover_box",
        "checkpoint": "/checkpoints/TwinVLA-box",
        "checkpoint_sha256": "d" * 64,
    }
    summary = {
        "kind": "twinvla",
        "checkpoint": entry["checkpoint"],
        "task_name": entry["task"],
        "trials": 100,
        "rollout_ids": list(range(300, 400)),
        "seed": 1501,
        "plan_seed_offset": 0,
        "cfg": 1.1,
        "num_denoising_steps": 10,
        "dtype": "bfloat16",
        "execution_mode": "chunk",
        "action_len": 20,
        "rows": [
            {"rollout_id": value, "finite_all": 1, "timeout": 0, "success": 0}
            for value in range(300, 400)
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    validate_twin_summary(path, entry, rollout_offset=300)
