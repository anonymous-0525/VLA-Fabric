import json

import numpy as np

from pi05_fabric.agents.load_pi05 import pi05_lora_config
from pi05_fabric.data.aloha_dual_agent import AgentObservation
from pi05_fabric.data.converted_dataset import PairedTrainingSample
from pi05_fabric.data.converted_dataset import build_action_chunk
from pi05_fabric.data import pi05_batch
from pi05_fabric.data.pi05_batch import build_training_pair
from pi05_fabric.evaluation.native_handover import next_action_index


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


def _quantile_file(tmp_path):
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
    return path


def test_quantile_normalization_matches_openpi_formula_and_roundtrips(tmp_path):
    normalization = pi05_batch.LocalQuantileNormalization.from_json(_quantile_file(tmp_path))
    values = np.asarray([[0.0] * 10, [1.0] * 10, [2.0] * 10], dtype=np.float32)

    normalized = normalization.left_state.normalize(values)
    expected = (values - 0.0) / (2.0 + 1e-6) * 2.0 - 1.0

    np.testing.assert_allclose(normalized, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        normalization.left_state.unnormalize(normalized), values, rtol=0, atol=2e-6
    )
    assert normalization.dataset_manifest_sha256 == "abc123"


def test_quantile_normalization_rejects_invalid_ranges(tmp_path):
    path = _quantile_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["left"]["state"]["q99"] = [0.0] * 10
    path.write_text(json.dumps(payload))

    try:
        pi05_batch.LocalQuantileNormalization.from_json(path)
    except ValueError as exc:
        assert "q99" in str(exc)
    else:
        raise AssertionError("degenerate q01/q99 range must be rejected")


def test_h50_pair_preserves_fifty_targets_and_pads_to_internal_action_dim(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    left_action = np.arange(50 * 10, dtype=np.float32).reshape(50, 10)
    right_action = left_action + 1000
    sample = PairedTrainingSample(
        0,
        0,
        AgentObservation("left", {"global": image, "wrist": image}, np.ones(10), "task"),
        AgentObservation("right", {"global": image, "wrist": image}, np.ones(10), "task"),
        left_action,
        right_action,
    )

    pair = build_training_pair(
        sample,
        normalization=pi05_batch.LocalQuantileNormalization.from_json(_quantile_file(tmp_path)),
        tokenizer=_Tokenizer(),
        image_tokens_per_view=4,
    )

    assert pair.left_actions.shape == (1, 50, 32)
    assert pair.right_actions.shape == (1, 50, 32)
    np.testing.assert_array_equal(pair.left_actions[0, :, 10:], 0.0)


def test_h50_terminal_chunk_repeats_last_source_action():
    actions = np.arange(6 * 20, dtype=np.float32).reshape(6, 20)

    chunk = build_action_chunk(actions, start=4, horizon=50)

    assert chunk.shape == (50, 20)
    np.testing.assert_array_equal(chunk[0], actions[4])
    np.testing.assert_array_equal(chunk[1], actions[5])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[5:6], 48, axis=0))


def test_v2_openpi_config_uses_native_h50():
    config = pi05_lora_config(action_horizon=50)

    assert config.action_horizon == 50
    assert config.action_dim == 32
    assert config.pi05 is True


def test_e25_replans_after_exactly_twenty_five_actions_from_h50_chunk():
    index = 0
    observed = []
    for _ in range(50):
        observed.append(index)
        index = next_action_index(
            index, execute_horizon=25, prediction_horizon=50
        )

    assert observed == list(range(25)) * 2
