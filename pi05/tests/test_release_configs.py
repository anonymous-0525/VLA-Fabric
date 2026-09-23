from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text())


def test_v2_interaction_config_matches_released_architecture() -> None:
    config = _load("configs/interactions/pi_native_v2_residual_action.yaml")

    assert config["mode"] == "pi_native_v2_residual_action"
    assert config["channels"]["raw_common"]["enabled"] is True
    assert config["channels"]["remote_private_kv"]["enabled"] is True
    assert config["channels"]["remote_action_residual"]["enabled"] is True
    assert config["channels"]["final_linear_action"]["enabled"] is False


def test_training_config_records_common_first_and_full_stages() -> None:
    config = _load("configs/training/pi_native_v2_h50_4gpu.yaml")

    assert config["global_batch"] == 12
    assert config["stage1"]["mode"] == "pi_native_v2_raw_common_stage1"
    assert config["stage1"]["steps"] == 10_000
    assert config["stage2"]["mode"] == "pi_native_v2_residual_action_stage2"
    assert config["stage2"]["steps"] == 40_000


def test_paired_protocol_separates_development_and_disjoint_ids() -> None:
    config = _load("configs/evaluation/paired200.yaml")

    assert config["validation"]["condition_ids"] == [0, 199]
    assert config["fresh"]["condition_ids"] == [200, 399]
    assert config["fresh"]["require_condition_disjoint_from_validation"] is True

