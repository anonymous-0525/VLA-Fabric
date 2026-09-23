from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_eagle_package_contains_only_dual_agent_scope() -> None:
    package = ROOT / "src" / "commvla"
    assert (package / "models" / "dual_arm.py").is_file()
    assert (package / "training" / "train_dual_arm.py").is_file()
    assert not (package / "models" / "multi_arm.py").exists()
    assert not (package / "communication" / "nspr.py").exists()
    assert not (package / "evaluation" / "robofactory_multi_arm.py").exists()
