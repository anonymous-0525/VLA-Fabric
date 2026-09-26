"""Frozen conditions and checkpoint-selection contract for Scan Object."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from pi05_fabric.evaluation.robotwin2_protocol import audit_robotwin_rollout_rows
from pi05_fabric.evaluation.robotwin2_protocol import file_sha256
from pi05_fabric.evaluation.robotwin2_protocol import load_condition_csv

TASK = "robotwin_scan_object"
EVALUATION_PROTOCOL = {
    "prediction_horizon": 50,
    "execution_horizon": 25,
    "flow_steps": 10,
    "maximum_steps": 500,
    "official_step_limit": 500,
}


def _read(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_conditions(root: str | Path):
    root = Path(root)
    identities = {}
    conditions = {}
    for split, first in (("validation", 0), ("fresh", 200)):
        directory = root / split
        manifest = _read(directory / "manifest.json")
        csv_path = directory / "conditions.csv"
        digest = file_sha256(csv_path)
        if (
            manifest.get("task") != TASK
            or manifest.get("split") != split
            or manifest.get("condition_count") != 200
            or manifest.get("frozen_sha256") != digest
            or not manifest.get("expert_validated")
        ):
            raise ValueError(f"{split} Scan Object conditions failed provenance audit")
        conditions[split] = load_condition_csv(
            csv_path, expected_ids=tuple(range(first, first + 200))
        )
        identities[split] = {
            "csv_sha256": digest,
            "manifest_sha256": file_sha256(directory / "manifest.json"),
        }
    validation_seeds = {condition.env_seed for condition in conditions["validation"]}
    fresh_seeds = {condition.env_seed for condition in conditions["fresh"]}
    if validation_seeds & fresh_seeds:
        raise ValueError("Scan Object validation and fresh seeds overlap")
    return identities, conditions


def snapshot_conditions(source: str | Path, target: str | Path):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    identity, _ = load_conditions(source)
    for split in identity:
        directory = target / split
        directory.mkdir(parents=True)
        for filename in ("conditions.csv", "manifest.json"):
            shutil.copy2(Path(source) / split / filename, directory / filename)
    load_snapshot(target, identity)
    return identity


def load_snapshot(root: str | Path, identity):
    actual, conditions = load_conditions(root)
    if actual != identity:
        raise ValueError("Scan Object condition snapshot changed")
    return conditions


def checkpoint_identity(checkpoint: str | Path, runs_root: str | Path):
    checkpoint = Path(checkpoint).resolve()
    runs_root = Path(runs_root).resolve()
    run = next((parent for parent in checkpoint.parents if parent.parent == runs_root), None)
    if run is None or not run.name.startswith(("formal_", "tail_")):
        raise ValueError("checkpoint is not from an audited Scan Object run")
    training = _read(run / "training_run.json")
    manifest = _read(checkpoint / "manifest.json")
    config = training.get("config", {})
    shared_valid = (
        training.get("four_gpu_gate_passed")
        and training.get("global_batch") == 16
        and config.get("task") == TASK
    )
    stage = manifest.get("stage")
    step = manifest.get("step")
    if stage == "pi_native_v2_full_direct":
        contract_valid = (
            config.get("steps") == 50000
            and step in (10000, 20000, 30000, 40000, 50000)
        )
        cumulative_step = step
    elif stage == "pi_native_v2_expanded_continuation" and run.name.startswith("tail_"):
        parent_hash = manifest.get("parent_model_sha256")
        initialization = manifest.get("protocol", {}).get("initialization", {})
        continuation = manifest.get("protocol", {}).get("continuation", {})
        extension = continuation.get("low_lr_extension", {})
        contract_valid = (
            training.get("status") == "COMPLETE"
            and config.get("stage") == stage
            and config.get("steps") == 40000
            and config.get("source_step") == 10000
            and config.get("parent_step", 20000) == 20000
            and config.get("cumulative_target_step") == 60000
            and config.get("source_model_sha256") == extension.get("source_model_sha256")
            and config.get("parent_model_sha256") == parent_hash
            and isinstance(step, int)
            and step in (20000, 30000, 35000, 40000)
            and manifest.get("schedule_step") == step
            and checkpoint.name == f"step_{step:08d}"
            and isinstance(parent_hash, str)
            and initialization.get("kind") == "expanded_continuation"
            and initialization.get("parent_model_sha256") == parent_hash
            and continuation.get("kind") == "scan_expanded_low_rewarm"
            and continuation.get("parent_model_sha256") == parent_hash
            and continuation.get("sample_step_offset") == 20000
            and extension.get("source_stage") == stage
            and extension.get("source_step") == 10000
            and extension.get("end_step") == 40000
        )
        cumulative_step = 20000 + step if isinstance(step, int) else None
    elif stage == "pi_native_v2_expanded_continuation" and run.name.startswith("formal_"):
        parent_hash = manifest.get("parent_model_sha256")
        initialization = manifest.get("protocol", {}).get("initialization", {})
        continuation = manifest.get("protocol", {}).get("continuation", {})
        contract_valid = (
            config.get("stage") == stage
            and config.get("steps") == 10000
            and config.get("parent_step") == 20000
            and config.get("cumulative_target_step") == 30000
            and step in (2000, 5000, 8000, 10000)
            and isinstance(parent_hash, str)
            and initialization.get("kind") == "expanded_continuation"
            and initialization.get("parent_model_sha256") == parent_hash
            and continuation.get("kind") == "scan_expanded_low_rewarm"
            and continuation.get("parent_model_sha256") == parent_hash
            and continuation.get("sample_step_offset") == 20000
        )
        cumulative_step = 20000 + step if isinstance(step, int) else None
    else:
        contract_valid = False
        cumulative_step = None
    if not shared_valid or not contract_valid:
        raise ValueError("formal Scan Object checkpoint contract mismatch")
    for seed_name in ("model_seed", "training_seed"):
        if config.get(seed_name) != manifest.get(seed_name):
            raise ValueError("formal Scan Object seed mismatch")
    state = checkpoint / "state.msgpack"
    if file_sha256(state) != manifest.get("state_sha256"):
        raise ValueError("Scan Object checkpoint state changed")
    identity = {
        "checkpoint": str(checkpoint),
        "run": str(run),
        "step": manifest["step"],
        "model_sha256": manifest["model_sha256"],
        "state_sha256": manifest["state_sha256"],
        "manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "training_run_sha256": file_sha256(run / "training_run.json"),
        "protocol": manifest["protocol"],
    }
    if stage == "pi_native_v2_expanded_continuation":
        identity["cumulative_step"] = cumulative_step
        identity["parent_model_sha256"] = manifest.get("parent_model_sha256")
    return identity


def _selection(evaluations, runs_root):
    candidates = []
    shared = None
    for evaluation in evaluations:
        directory = Path(evaluation).resolve()
        record = _read(directory / "evaluation_manifest.json")
        actual = checkpoint_identity(record["checkpoint"]["checkpoint"], runs_root)
        if actual != record["checkpoint"] or record.get("task") != TASK:
            raise ValueError("Scan Object evaluation checkpoint identity changed")
        if record.get("split") != "validation" or record.get("purpose") != "checkpoint_screen":
            raise ValueError("checkpoint selection must use the frozen validation screen")
        all_conditions = load_snapshot(
            directory / "conditions_snapshot", record["conditions"]
        )["validation"]
        selected_ids = tuple(int(value) for value in record.get("condition_ids", ()))
        if selected_ids != tuple(range(0, 200, 5)):
            raise ValueError("checkpoint screen must use the fixed 40-condition subset")
        by_id = {condition.condition_id: condition for condition in all_conditions}
        conditions = tuple(by_id[condition_id] for condition_id in selected_ids)
        protocol = record["evaluation_protocol"]
        if any(protocol.get(key) != value for key, value in EVALUATION_PROTOCOL.items()):
            raise ValueError("Scan Object evaluation protocol mismatch")
        comparable = {
            "run": actual["run"],
            "training_run_sha256": actual["training_run_sha256"],
            "conditions": record["conditions"],
            "evaluation_protocol": protocol,
        }
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError("Scan Object candidate evaluations are not comparable")
        with (directory / "merged/rollouts.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        audit_robotwin_rollout_rows(rows, conditions=conditions)
        condition_by_id = {condition.condition_id: condition for condition in conditions}
        if any(
            row.get("task") != TASK
            or row.get("instruction") != condition_by_id[int(row["condition_id"])].instruction
            for row in rows
        ):
            raise ValueError("Scan Object task or instruction mismatch")
        successes = sum(int(row["success"]) for row in rows)
        summary = _read(directory / "COMPLETE.json")
        if (
            summary.get("status") != "PASS"
            or summary.get("trials") != 40
            or summary.get("successes") != successes
            or summary.get("errors")
            or summary.get("timeouts")
        ):
            raise ValueError("incomplete Scan Object development evidence")
        candidate = {
            "checkpoint": actual,
            "successes": successes,
            "evaluation": str(directory),
            "rollouts_sha256": file_sha256(directory / "merged/rollouts.csv"),
        }
        if actual["protocol"].get("initialization", {}).get("kind") == "expanded_continuation":
            candidate["phase_metrics"] = {
                key: int(summary.get(key, 0))
                for key in ("block_lifted", "middle_reached", "target_region_reached")
            }
        candidates.append(candidate)
    stages = {candidate["checkpoint"]["protocol"].get("initialization", {}).get("kind")
              for candidate in candidates}
    tail_runs = {Path(candidate["checkpoint"]["run"]).name.startswith("tail_") for candidate in candidates}
    if stages == {"base_direct"} and tail_runs == {False}:
        expected_steps = [10000, 20000, 30000, 40000, 50000]
    elif stages == {"expanded_continuation"} and tail_runs == {False}:
        expected_steps = [2000, 5000, 8000, 10000]
    elif stages == {"expanded_continuation"} and tail_runs == {True}:
        expected_steps = [20000, 30000, 35000, 40000]
    else:
        raise ValueError("checkpoint selection mixes incompatible Scan Object lineages")
    if sorted(candidate["checkpoint"]["step"] for candidate in candidates) != expected_steps:
        raise ValueError("checkpoint selection does not match the frozen checkpoint set")
    candidates.sort(key=lambda candidate: candidate["checkpoint"]["step"])
    selected = min(candidates, key=lambda candidate: (-candidate["successes"], candidate["checkpoint"]["step"]))
    return {
        "selected_checkpoint": selected["checkpoint"],
        "candidates": candidates,
        "conditions": shared["conditions"],
        "evaluation_protocol": shared["evaluation_protocol"],
    }


def freeze_selection(evaluations, runs_root: str | Path, output: str | Path):
    result = _selection(evaluations, runs_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def select_expanded_candidate(candidates):
    baseline = {
        "block_lifted": 40,
        "middle_reached": 21,
        "target_region_reached": 14,
    }
    eligible = [
        candidate
        for candidate in candidates
        if candidate["successes"] >= 13
        and all(candidate.get("phase_metrics", {}).get(key, -1) >= value
                for key, value in baseline.items())
    ]
    eligible.sort(key=lambda candidate: (-candidate["successes"], candidate["checkpoint"]["step"]))
    return (eligible[0] if eligible else None), eligible


def freeze_expanded_selection(evaluations, runs_root: str | Path, output: str | Path):
    result = _selection(evaluations, runs_root)
    if any(candidate["checkpoint"].get("cumulative_step") is None for candidate in result["candidates"]):
        raise ValueError("expanded selection requires expanded continuation checkpoints")
    selected, eligible = select_expanded_candidate(result["candidates"])
    result["selected_checkpoint"] = None if selected is None else selected["checkpoint"]
    result["screen_gate"] = {
        "status": "PASS" if selected is not None else "NO_ELIGIBLE_CANDIDATE",
        "minimum_successes": 13,
        "baseline_phase_metrics": {
            "block_lifted": 40,
            "middle_reached": 21,
            "target_region_reached": 14,
        },
        "eligible_steps": [candidate["checkpoint"]["step"] for candidate in eligible],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def validate_frozen_selection(path, checkpoint, conditions, evaluation_protocol, runs_root):
    frozen = _read(path)
    actual = _selection([candidate["evaluation"] for candidate in frozen["candidates"]], runs_root)
    if (
        actual != frozen
        or checkpoint != frozen["selected_checkpoint"]
        or conditions != frozen["conditions"]
        or evaluation_protocol != frozen["evaluation_protocol"]
    ):
        raise ValueError("frozen Scan Object selection identity mismatch")
