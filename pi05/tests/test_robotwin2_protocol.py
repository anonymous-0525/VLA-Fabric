import csv
import json

import pytest

from pi05_fabric.evaluation.robotwin2_protocol import RobotwinCondition
from pi05_fabric.evaluation.robotwin2_protocol import audit_robotwin_rollout_rows
from pi05_fabric.evaluation.robotwin2_protocol import freeze_condition_split
from pi05_fabric.evaluation.robotwin2_protocol import load_condition_csv
from pi05_fabric.evaluation.robotwin2_protocol import merge_condition_csvs
from pi05_fabric.evaluation.robotwin2_protocol import merge_rollout_shards


def _write_conditions(path, *, first_id, count, seed_start=900_000):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("condition_id", "env_seed", "instruction"))
        writer.writeheader()
        for offset in range(count):
            writer.writerow(
                {
                    "condition_id": first_id + offset,
                    "env_seed": seed_start + offset,
                    "instruction": f"handover instruction {offset}",
                }
            )


def test_load_condition_csv_requires_exact_expected_range(tmp_path):
    source = tmp_path / "conditions.csv"
    _write_conditions(source, first_id=200, count=200)

    conditions = load_condition_csv(source, expected_ids=tuple(range(200, 400)))

    assert len(conditions) == 200
    assert conditions[0] == RobotwinCondition(200, 900_000, "handover instruction 0")
    assert conditions[-1].condition_id == 399


def test_load_condition_csv_rejects_duplicate_seed_and_blank_instruction(tmp_path):
    source = tmp_path / "conditions.csv"
    _write_conditions(source, first_id=0, count=200)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[1]["env_seed"] = rows[0]["env_seed"]
    rows[2]["instruction"] = " "
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="unique"):
        load_condition_csv(source, expected_ids=tuple(range(200)))


def test_freeze_condition_split_records_source_and_frozen_hashes(tmp_path):
    source = tmp_path / "source.csv"
    _write_conditions(source, first_id=0, count=200)

    manifest_path = freeze_condition_split(
        source,
        output_dir=tmp_path / "frozen",
        split="validation",
        expected_ids=tuple(range(200)),
        expert_validated=True,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["condition_count"] == 200
    assert manifest["expert_validated"] is True
    assert len(manifest["source_sha256"]) == 64
    assert len(manifest["frozen_sha256"]) == 64
    assert (tmp_path / "frozen" / "conditions.csv").is_file()
    with pytest.raises(FileExistsError):
        freeze_condition_split(
            source,
            output_dir=tmp_path / "frozen",
            split="validation",
            expected_ids=tuple(range(200)),
            expert_validated=True,
        )


def test_rollout_audit_requires_complete_rows_and_finite_timing():
    conditions = tuple(
        RobotwinCondition(index, 100_000 + index, f"instruction {index}")
        for index in range(3)
    )
    rows = [
        {
            "condition_id": index,
            "env_seed": 100_000 + index,
            "success": index % 2,
            "steps": 25,
            "planning_calls": 1,
            "mean_plan_seconds": 0.1,
            "p95_plan_seconds": 0.1,
            "episode_seconds": 1.0,
            "timeout": 0,
            "error": "",
        }
        for index in range(3)
    ]
    audit_robotwin_rollout_rows(rows, conditions=conditions)

    with pytest.raises(ValueError, match="missing"):
        audit_robotwin_rollout_rows(rows[:-1], conditions=conditions)
    bad = [dict(row) for row in rows]
    bad[0]["env_seed"] = -1
    with pytest.raises(ValueError, match="seed"):
        audit_robotwin_rollout_rows(bad, conditions=conditions)


def test_merge_condition_csvs_requires_contiguous_nonoverlapping_ranges(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.csv"
    _write_conditions(first, first_id=200, count=100)
    _write_conditions(second, first_id=300, count=100, seed_start=1_000_000)

    merged = merge_condition_csvs(
        (first, second), output=output, expected_ids=tuple(range(200, 400))
    )

    assert len(merged) == 200
    assert load_condition_csv(output, expected_ids=tuple(range(200, 400))) == merged


def test_merge_rollout_shards_audits_and_summarizes_exact_conditions(tmp_path):
    conditions = tuple(
        RobotwinCondition(index, 100_000 + index, f"instruction {index}")
        for index in range(4)
    )
    fieldnames = (
        "condition_id",
        "env_seed",
        "instruction",
        "success",
        "steps",
        "planning_calls",
        "mean_plan_seconds",
        "p95_plan_seconds",
        "episode_seconds",
        "block_lifted",
        "middle_reached",
        "target_region_reached",
        "timeout",
        "error",
    )
    sources = []
    for shard_index, ids in enumerate(((0, 1), (2, 3))):
        source = tmp_path / f"shard_{shard_index}.csv"
        sources.append(source)
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for condition_id in ids:
                writer.writerow(
                    {
                        "condition_id": condition_id,
                        "env_seed": 100_000 + condition_id,
                        "instruction": f"instruction {condition_id}",
                        "success": condition_id % 2,
                        "steps": 50,
                        "planning_calls": 2,
                        "mean_plan_seconds": 1.0,
                        "p95_plan_seconds": 1.5,
                        "episode_seconds": 10.0,
                        "block_lifted": 1,
                        "middle_reached": 0,
                        "target_region_reached": 0,
                        "timeout": 0,
                        "error": "",
                    }
                )

    summary = merge_rollout_shards(
        sources,
        output_dir=tmp_path / "merged",
        conditions=conditions,
        purpose="development_validation",
    )

    assert summary["status"] == "PASS"
    assert summary["successes"] == 2
    assert summary["trials"] == 4
    assert summary["mean_planning_calls"] == 2.0
    rows = list(csv.DictReader((tmp_path / "merged" / "rollouts.csv").open()))
    assert [int(row["condition_id"]) for row in rows] == [0, 1, 2, 3]
