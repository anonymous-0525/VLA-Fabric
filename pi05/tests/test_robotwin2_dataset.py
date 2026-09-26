import hashlib
import json

import numpy as np
import pytest

from pi05_fabric.data.robotwin2_dataset import EXPECTED_TASK
from pi05_fabric.data.robotwin2_dataset import audit_converted_dataset
from pi05_fabric.data.robotwin2_dataset import validate_source_metadata


def test_source_metadata_pins_task_episode_and_shard_counts(tmp_path):
    info = {
        "name": EXPECTED_TASK,
        "splits": [
            {
                "name": "train",
                "numBytes": "702464531",
                "shardLengths": ["6", "6", "7", "6", "6", "7", "6", "6"],
            }
        ],
    }
    (tmp_path / "dataset_info.json").write_text(json.dumps(info))

    result = validate_source_metadata(tmp_path)

    assert result == {
        "task": EXPECTED_TASK,
        "episode_count": 50,
        "shard_count": 8,
        "declared_bytes": 702464531,
    }


def test_source_metadata_rejects_wrong_task(tmp_path):
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"name": "other", "splits": []})
    )

    with pytest.raises(ValueError, match="expected source task"):
        validate_source_metadata(tmp_path)


def test_converted_audit_checks_paired_shapes_hashes_and_images(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    path = episodes / "episode_00000.npz"
    np.savez_compressed(
        path,
        global_image=np.zeros((3, 240, 320, 3), dtype=np.uint8),
        left_wrist_image=np.ones((3, 240, 320, 3), dtype=np.uint8),
        right_wrist_image=np.full((3, 240, 320, 3), 2, dtype=np.uint8),
        proprioception=np.zeros((3, 20), dtype=np.float32),
        action=np.ones((3, 20), dtype=np.float32),
        instruction=np.asarray(["handover the red block", "pass the red block"]),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "task": EXPECTED_TASK,
        "episodes": [
            {"id": 0, "path": "episodes/episode_00000.npz", "length": 3, "sha256": digest}
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = audit_converted_dataset(tmp_path, require_full_dataset=False)

    assert result["status"] == "PASS"
    assert result["episode_count"] == 1
    assert result["step_count"] == 3
    assert result["joint_dim"] == 20
    assert result["local_dim"] == 10
    assert result["image_shapes"]["global_image"] == [[240, 320, 3]]


def test_converted_audit_rejects_left_right_shape_mismatch(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    path = episodes / "episode_00000.npz"
    np.savez_compressed(
        path,
        global_image=np.zeros((3, 8, 8, 3), dtype=np.uint8),
        left_wrist_image=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        right_wrist_image=np.zeros((3, 8, 8, 3), dtype=np.uint8),
        proprioception=np.zeros((3, 20), dtype=np.float32),
        action=np.zeros((3, 20), dtype=np.float32),
        instruction=np.asarray(["task"]),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": EXPECTED_TASK,
                "episodes": [
                    {"id": 0, "path": "episodes/episode_00000.npz", "length": 3, "sha256": digest}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="left_wrist_image"):
        audit_converted_dataset(tmp_path, require_full_dataset=False)
