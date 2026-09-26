import json

import numpy as np

from pi05_fabric.data.pi05_batch import LocalQuantileNormalization
from pi05_fabric.evaluation.robotwin2_policy import policy_payload_to_pair
from pi05_fabric.evaluation.robotwin2_policy import response_from_local_actions


class _InnerTokenizer:
    def encode(self, _text, *, add_bos):
        assert add_bos
        return [1, 2, 3]


class _Tokenizer:
    _max_len = 6
    _tokenizer = _InnerTokenizer()

    def tokenize(self, _prompt, _state):
        return np.asarray([1, 2, 3, 9, 0, 0]), np.asarray(
            [True, True, True, True, False, False]
        )


def _normalization(tmp_path):
    stats = {"q01": [0.0] * 10, "q99": [2.0] * 10, "count": 100}
    payload = {
        "schema_version": 2,
        "normalization": "q01_q99",
        "epsilon": 1e-6,
        "dataset_manifest_sha256": "abc123",
        "left": {"state": stats, "action": stats},
        "right": {"state": stats, "action": stats},
    }
    path = tmp_path / "normalization_q01_q99.json"
    path.write_text(json.dumps(payload))
    return LocalQuantileNormalization.from_json(path)


def _payload():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "kind": np.asarray("inference"),
        "request_id": np.asarray("3:4"),
        "condition_id": np.asarray(3, dtype=np.int64),
        "planning_call": np.asarray(4, dtype=np.int64),
        "instruction": np.asarray("handover the red block"),
        "global_image": image,
        "left_wrist_image": image + 1,
        "right_wrist_image": image + 2,
        "eef_state": np.ones(20, dtype=np.float32),
    }


def test_policy_payload_builds_two_local_agent_inputs_without_peer_wrist_leakage(tmp_path):
    pair = policy_payload_to_pair(
        _payload(), normalization=_normalization(tmp_path), tokenizer=_Tokenizer()
    )

    assert pair.left_actions.shape == (1, 50, 32)
    assert pair.right_actions.shape == (1, 50, 32)
    np.testing.assert_array_equal(pair.left_observation.image_masks["left_wrist_0_rgb"], [True])
    np.testing.assert_array_equal(pair.left_observation.image_masks["right_wrist_0_rgb"], [False])
    np.testing.assert_array_equal(pair.right_observation.image_masks["left_wrist_0_rgb"], [False])
    np.testing.assert_array_equal(pair.right_observation.image_masks["right_wrist_0_rgb"], [True])
    np.testing.assert_array_equal(
        pair.left_observation.images["left_wrist_0_rgb"][0], _payload()["left_wrist_image"].astype(np.float32) / 255.0 * 2.0 - 1.0
    )
    np.testing.assert_array_equal(
        pair.right_observation.images["right_wrist_0_rgb"][0], _payload()["right_wrist_image"].astype(np.float32) / 255.0 * 2.0 - 1.0
    )


def test_response_merges_unnormalized_local_actions_and_preserves_identity(tmp_path):
    normalization = _normalization(tmp_path)
    left = np.zeros((1, 50, 32), dtype=np.float32)
    right = np.full((1, 50, 32), 0.5, dtype=np.float32)

    response = response_from_local_actions(
        request_id="3:4",
        flow_seed=123,
        left_actions=left,
        right_actions=right,
        normalization=normalization,
    )

    assert response["request_id"].item() == "3:4"
    assert response["flow_seed"].item() == 123
    assert response["actions_r6"].shape == (50, 20)
    np.testing.assert_allclose(response["actions_r6"][:, :10], 1.0, atol=1e-6)


def test_checkpoint_dataset_binding_rejects_wrong_task_statistics(tmp_path):
    import hashlib
    import pytest
    from pi05_fabric.evaluation.robotwin2_policy import validate_checkpoint_dataset
    raw = json.dumps({'task': 'robotwin_handover_mic'}).encode()
    (tmp_path / 'manifest.json').write_bytes(raw)
    stats = {'schema_version': 2, 'dataset_manifest_sha256': hashlib.sha256(raw).hexdigest()}
    (tmp_path / 'normalization_q01_q99.json').write_text(json.dumps(stats))
    manifest = {'protocol': {'normalization': stats}}
    assert validate_checkpoint_dataset(manifest, tmp_path) == 'robotwin_handover_mic'
    with pytest.raises(ValueError, match='normalization'):
        validate_checkpoint_dataset({'protocol': {'normalization': {}}}, tmp_path)
    (tmp_path / 'manifest.json').write_text('{"task":"robotwin_handover_block"}')
    with pytest.raises(ValueError, match='dataset'):
        validate_checkpoint_dataset(manifest, tmp_path)
